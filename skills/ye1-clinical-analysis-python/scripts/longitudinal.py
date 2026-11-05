"""Independent-subject longitudinal Gaussian mixed model with random intercept."""
from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict

import numpy as np
import statsmodels.api as sm
from scipy import stats

from local_data import DataError


def need(ok, reason):
    if not ok:
        raise DataError(reason)


def missing(v):
    return v is None or str(v).strip() == ""


def number(v):
    try:
        x=float(v)
    except (ValueError,TypeError,OverflowError):
        raise DataError("INVALID_NUMERIC_VALUE") from None
    need(math.isfinite(x),"NONFINITE_NUMERIC_VALUE")
    return x


def analyze(frame, identifier, spec, model="longitudinal"):
    time, outcome = spec.get("time"),spec.get("outcome")
    need(isinstance(identifier,str) and identifier in frame, "LONGITUDINAL_SUBJECT_ID_REQUIRED")
    need(isinstance(time,str) and isinstance(outcome,str) and time in frame and outcome in frame
         and len({identifier,time,outcome}) == 3, "LONGITUDINAL_FIELDS_REQUIRED")
    need(isinstance(spec.get("time_unit"),str) and spec["time_unit"].strip()
         and isinstance(spec.get("time_zero_definition"),str) and spec["time_zero_definition"].strip(),
         "TIME_ORIGIN_AND_UNIT_REQUIRED")
    need(spec.get("missing") == "complete_case", "COMPLETE_CASE_MUST_BE_EXPLICIT")
    need(model in {"longitudinal", "gee"}, "LONGITUDINAL_METHOD_REQUIRED")
    if model == "longitudinal":
        need(spec.get("random") == "intercept", "ONLY_RANDOM_INTERCEPT_SUPPORTED")
        need(spec.get("fit_method") in {"reml", "ml"}, "REML_OR_ML_REQUIRED")
    else:
        need(spec.get("family") in {"gaussian", "binomial"}, "GEE_FAMILY_REQUIRED")
        need(spec.get("working_correlation") in {"independence", "exchangeable"}, "GEE_CORRELATION_REQUIRED")
        if spec["family"] == "binomial":
            event, non_event = spec.get("event"), spec.get("non_event")
            need(isinstance(event, str) and isinstance(non_event, str) and event and non_event and event != non_event,
                 "BINARY_EVENT_CODES_REQUIRED")
    covariates=spec.get("covariates",[])
    need(isinstance(covariates,list) and all(isinstance(c,dict) for c in covariates), "COVARIATE_SPEC_REQUIRED")
    names=[c.get("name") for c in covariates]
    need(all(isinstance(x,str) and x in frame for x in names) and len(set(names))==len(names)
         and not set(names)&{identifier,time,outcome},"COVARIATE_NAMES_INVALID")
    for cov in covariates:
        need(cov.get("type") in {"numeric","categorical"} and cov.get("available_at_time_zero") is True,
             "COVARIATE_TYPE_AND_TIMING_REQUIRED")
        if cov["type"] == "categorical":
            levels,ref=cov.get("levels"),cov.get("reference")
            need(isinstance(levels,list) and len(levels)>=2 and all(isinstance(v,str) and v for v in levels)
                 and len(set(levels))==len(levels) and ref in levels,"CATEGORY_LEVELS_REQUIRED")
    rows=frame.to_dict(orient="records")
    need(all(not missing(r[identifier]) and not missing(r[time]) for r in rows), "SUBJECT_AND_TIME_REQUIRED")
    seen=set()
    baseline_values=defaultdict(set)
    for row in rows:
        key=(str(row[identifier]),number(row[time]))
        need(key not in seen,"DUPLICATE_SUBJECT_TIME")
        seen.add(key)
        if not missing(row[outcome]):
            if model == "gee" and spec["family"] == "binomial":
                need(str(row[outcome]) in {event, non_event}, "UNDECLARED_OUTCOME_CODE")
            else:
                number(row[outcome])
        for cov in covariates:
            v=row[cov["name"]]
            if not missing(v):
                if cov["type"]=="numeric": number(v)
                else: need(str(v) in cov["levels"],"UNDECLARED_CATEGORY")
                baseline_values[(str(row[identifier]),cov["name"])].add(number(v) if cov["type"]=="numeric" else str(v))
    need(all(len(v)<=1 for v in baseline_values.values()),"BASELINE_COVARIATE_CHANGED_WITHIN_SUBJECT")
    complete=[r for r in rows if not missing(r[outcome]) and all(not missing(r[n]) for n in names)]
    subjects={str(r[identifier]) for r in complete}
    need(len(subjects)>=(40 if model == "gee" else 20),
         "AT_LEAST_40_SUBJECTS_REQUIRED" if model == "gee" else "AT_LEAST_20_SUBJECTS_REQUIRED")
    counts=Counter(str(r[identifier]) for r in complete)
    need(all(n>=2 for n in counts.values()),"AT_LEAST_TWO_OBSERVATIONS_PER_SUBJECT_REQUIRED")
    t=np.asarray([number(r[time]) for r in complete])
    if model == "gee" and spec["family"] == "binomial":
        y=np.asarray([float(str(r[outcome]) == event) for r in complete])
        need(min(int(y.sum()), int(len(y)-y.sum())) >= 20, "AT_LEAST_20_OUTCOMES_PER_CLASS_REQUIRED")
    else:
        y=np.asarray([number(r[outcome]) for r in complete])
    columns=[np.ones(len(complete)),t]; labels=["intercept",time]
    for cov in covariates:
        n=cov["name"]
        if cov["type"]=="numeric":
            values=np.asarray([number(r[n]) for r in complete]); pieces=[(n,values)]
        else:
            pieces=[(n+"["+level+" vs "+cov["reference"]+"]",np.asarray([float(str(r[n])==level) for r in complete]))
                    for level in cov["levels"] if level != cov["reference"]]
        for label,values in pieces:
            columns.append(values);labels.append(label)
            if cov.get("interact_with_time") is True:
                columns.append(values*t);labels.append(label+" * "+time)
    x=np.column_stack(columns)
    need(len(complete)>x.shape[1]+2 and np.linalg.matrix_rank(x)==x.shape[1],"LONGITUDINAL_DESIGN_RANK_DEFICIENT")
    groups=[str(r[identifier]) for r in complete]
    if model == "gee":
        family = sm.families.Gaussian() if spec["family"] == "gaussian" else sm.families.Binomial()
        working = (sm.cov_struct.Independence() if spec["working_correlation"] == "independence"
                   else sm.cov_struct.Exchangeable())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GEE(y, x, groups=groups, family=family, cov_struct=working).fit(cov_type="robust")
        need(not caught and bool(fit.converged), "GEE_WARNING_OR_NONCONVERGENCE")
        beta=np.asarray(fit.params); se=np.asarray(fit.bse)
        need(np.isfinite(beta).all() and np.isfinite(se).all() and bool(np.all(se>0)), "INVALID_GEE_INFERENCE")
        terms=[]
        for label,b,s in zip(labels,beta,se):
            lo,hi=float(b-1.959963984540054*s),float(b+1.959963984540054*s)
            term={"term":label,"estimate":float(b),"standard_error":float(s),"ci95":[lo,hi],
                  "p":float(2*stats.norm.sf(abs(float(b/s))))}
            if spec["family"] == "binomial":
                need(max(abs(float(b)),abs(lo),abs(hi))<700,"GEE_ODDS_RATIO_OVERFLOW")
                term.update(odds_ratio=math.exp(float(b)),or_ci95=[math.exp(lo),math.exp(hi)])
            terms.append(term)
        return {"status":"COMPUTED_REVIEW_PENDING","method":"longitudinal_gee","n_total_rows":len(rows),
                "n_used_rows":len(complete),"n_excluded_missing":len(rows)-len(complete),
                "n_subjects":len(subjects),"time_unit":spec["time_unit"],"time_zero_definition":spec["time_zero_definition"],
                "family":spec["family"],"working_correlation":spec["working_correlation"],
                "covariance":"robust sandwich","terms":terms,
                "limitations":["Population-averaged effects; not subject-specific mixed-model effects.",
                               "Sandwich inference relies on adequate independent subjects; the minimum is only an engineering guard.",
                               "Missingness, covariate timing, functional form and working correlation require review."]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit=sm.MixedLM(y,x,groups=groups).fit(reml=spec["fit_method"]=="reml",method="lbfgs",disp=False)
    need(not caught and bool(fit.converged),"MIXED_MODEL_WARNING_OR_NONCONVERGENCE")
    beta=np.asarray(fit.fe_params); se=np.asarray(fit.bse_fe)
    need(np.isfinite(beta).all() and np.isfinite(se).all() and bool(np.all(se>0)),"INVALID_FIXED_EFFECT_INFERENCE")
    terms=[]
    for label,b,s in zip(labels,beta,se):
        z=float(b/s)
        terms.append({"term":label,"estimate":float(b),"standard_error":float(s),
                      "ci95":[float(b-1.959963984540054*s),float(b+1.959963984540054*s)],
                      "p":float(2*stats.norm.sf(abs(z)))})
    return {"status":"COMPUTED_REVIEW_PENDING","method":"longitudinal_mixedlm","n_total_rows":len(rows),
            "n_used_rows":len(complete),"n_excluded_missing":len(rows)-len(complete),
            "n_subjects":len(subjects),"time_unit":spec["time_unit"],"time_zero_definition":spec["time_zero_definition"],
            "fit_method":spec["fit_method"],"random_effect":"subject-specific random intercept",
            "random_intercept_variance":float(fit.cov_re[0,0]),"residual_variance":float(fit.scale),"terms":terms,
            "limitations":["Gaussian random-intercept model; not a latent trajectory class model.",
                           "Complete-case analysis and missing-at-random assumptions require review.",
                           "Covariate timing, functional form and within-subject covariance require scientific review."]}
