"""Limited linear mediation decomposition with subject bootstrap."""
from __future__ import annotations

import math

import numpy as np

from local_data import DataError


def need(ok,reason):
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


def _estimates(y,m,x,cov):
    med_design=np.column_stack([np.ones(len(x)),x,cov])
    outcome_design=np.column_stack([np.ones(len(x)),x,m,cov])
    total_design=np.column_stack([np.ones(len(x)),x,cov])
    if any(np.linalg.matrix_rank(z)<z.shape[1] for z in (med_design,outcome_design,total_design)):
        return None
    ma=np.linalg.lstsq(med_design,m,rcond=None)[0]
    oy=np.linalg.lstsq(outcome_design,y,rcond=None)[0]
    total=np.linalg.lstsq(total_design,y,rcond=None)[0]
    return {"a":float(ma[1]),"b":float(oy[2]),"indirect_ab":float(ma[1]*oy[2]),
            "direct_cprime":float(oy[1]),"total_c":float(total[1])}


def analyze(frame, spec):
    exposure=spec.get("exposure"); mediator=spec.get("mediator"); outcome=spec.get("outcome")
    need(isinstance(exposure,dict) and isinstance(mediator,str) and isinstance(outcome,str),"MEDIATION_FIELDS_REQUIRED")
    xname,treated,control=exposure.get("column"),exposure.get("treated"),exposure.get("control")
    need(all(isinstance(v,str) and v for v in [xname,treated,control,mediator,outcome]) and treated!=control,
         "BINARY_EXPOSURE_CODES_REQUIRED")
    covariates=spec.get("covariates",[])
    need(isinstance(covariates,list) and all(isinstance(c,dict) and c.get("type")=="numeric"
         and c.get("available_before_exposure") is True for c in covariates),"PRESPECIFIED_NUMERIC_CONFOUNDERS_REQUIRED")
    names=[c.get("name") for c in covariates]
    fields=[xname,mediator,outcome]+names
    need(all(isinstance(v,str) and v in frame for v in fields) and len(set(fields))==len(fields),"MEDIATION_FIELD_ROLE_CONFLICT")
    need(spec.get("missing")=="complete_case","COMPLETE_CASE_MUST_BE_EXPLICIT")
    reps,seed=spec.get("bootstrap_replicates"),spec.get("seed")
    need(type(reps) is int and 100<=reps<=2000 and type(seed) is int and 0<=seed<2**32,
         "BOOTSTRAP_100_TO_2000_AND_SEED_REQUIRED")
    rows=frame.to_dict(orient="records")
    for row in rows:
        if not missing(row[xname]):need(str(row[xname]) in {treated,control},"UNDECLARED_EXPOSURE_CODE")
        for n in [mediator,outcome]+names:
            if not missing(row[n]):number(row[n])
    complete=[r for r in rows if all(not missing(r[n]) for n in fields)]
    need(len(complete)>=max(30,5*(len(names)+3)),"INSUFFICIENT_COMPLETE_OBSERVATIONS")
    x=np.asarray([int(str(r[xname])==treated) for r in complete],dtype=float)
    m=np.asarray([number(r[mediator]) for r in complete]);y=np.asarray([number(r[outcome]) for r in complete])
    cov=np.asarray([[number(r[n]) for n in names] for r in complete],dtype=float).reshape(len(complete),len(names))
    need(min(int(x.sum()),int(len(x)-x.sum()))>=10,"INSUFFICIENT_EXPOSURE_GROUP_SIZE")
    point=_estimates(y,m,x,cov)
    need(point is not None and all(math.isfinite(v) for v in point.values()),"MEDIATION_MODEL_NOT_ESTIMABLE")
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(reps):
        ix=rng.integers(0,len(complete),size=len(complete))
        values=_estimates(y[ix],m[ix],x[ix],cov[ix])
        if values is not None and math.isfinite(values["indirect_ab"]):boot.append(values["indirect_ab"])
    need(len(boot)>=math.ceil(.9*reps),"BOOTSTRAP_TOO_MANY_UNESTIMABLE_REPLICATES")
    ci=np.quantile(boot,[.025,.975]).tolist()
    return {"status":"COMPUTED_REVIEW_PENDING","method":"linear_mediation_binary_exposure",
            "n_total":len(rows),"n_complete":len(complete),"n_excluded_missing":len(rows)-len(complete),
            "estimates":point,"indirect_percentile_ci95":ci,
            "bootstrap":{"requested":reps,"valid":len(boot),"seed":seed,"unit":"independent subject"},
            "limitations":["Binary exposure, continuous mediator and outcome; no exposure-mediator interaction.",
                           "The decomposition and bootstrap do not establish a causal mechanism.",
                           "Temporal order, confounding, mediator-outcome confounding and model assumptions require review."]}
