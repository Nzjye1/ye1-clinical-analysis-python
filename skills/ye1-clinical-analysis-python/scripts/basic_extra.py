"""Additional standard clinical statistics; no patient records are returned."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import optimize, stats

from baseline import is_missing, numeric, require
from local_data import DataError


def _paired_values(frame, fields):
    require(isinstance(fields, list) and len(set(fields)) == len(fields) and all(isinstance(x,str) and x in frame for x in fields),
            "UNIQUE_EXISTING_COLUMNS_REQUIRED")
    # Validate every observed cell, including values on an otherwise incomplete row.
    for field in fields:
        numeric(frame[field].tolist())
    complete = [[float(row[x]) for x in fields] for _, row in frame.iterrows()
                if all(not is_missing(row[x]) for x in fields)]
    return np.asarray(complete, dtype=float)


def correlation(frame, spec):
    x, y, method = spec.get("x"), spec.get("y"), spec.get("test")
    require(x != y and method in {"pearson", "spearman"}, "CORRELATION_SPEC_REQUIRED")
    values = _paired_values(frame, [x,y])
    require(len(values) >= 3, "AT_LEAST_THREE_COMPLETE_PAIRS_REQUIRED")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = stats.pearsonr(values[:,0],values[:,1]) if method == "pearson" else stats.spearmanr(values[:,0],values[:,1])
    require(not caught and math.isfinite(float(res.statistic)) and math.isfinite(float(res.pvalue)), "CORRELATION_NOT_ESTIMABLE")
    ci = None
    if method == "pearson" and len(values) >= 4:
        bound = res.confidence_interval(confidence_level=.95)
        if math.isfinite(float(bound.low)) and math.isfinite(float(bound.high)):
            ci = [float(bound.low),float(bound.high)]
    return {"status":"COMPUTED_REVIEW_PENDING","method":method+"_correlation","x":x,"y":y,"n_total":len(frame),
            "n_complete_pairs":len(values),"n_excluded_missing":len(frame)-len(values),
            "coefficient":float(res.statistic),"p":float(res.pvalue),"ci95":ci,
            "note":"Correlation does not establish causality; Spearman p uses the asymptotic method and ties require care."}


def normality(frame, spec):
    field = spec.get("column")
    require(isinstance(field,str) and field in frame, "NUMERIC_COLUMN_REQUIRED")
    values = numeric(frame[field].tolist())
    require(3 <= len(values) <= 5000, "SHAPIRO_REQUIRES_3_TO_5000_VALUES")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = stats.shapiro(values)
    require(not caught and math.isfinite(float(res.pvalue)), "NORMALITY_TEST_NOT_ESTIMABLE")
    return {"status":"COMPUTED_REVIEW_PENDING","method":"shapiro_wilk","column":field,"n_total":len(frame),
            "n_nonmissing":len(values),"missing":len(frame)-len(values),"statistic":float(res.statistic),"p":float(res.pvalue),
            "note":"Normality-test p alone must not determine the statistical method; assess design and distribution."}


def repeated(frame, spec):
    fields = spec.get("columns")
    require(spec.get("test") == "friedman" and isinstance(fields,list) and len(fields) >= 3, "THREE_REPEATED_COLUMNS_REQUIRED")
    require(spec.get("missing") == "complete_case", "COMPLETE_CASE_MUST_BE_EXPLICIT")
    values = _paired_values(frame,fields)
    require(len(values) >= 5, "AT_LEAST_FIVE_COMPLETE_REPEATED_UNITS_REQUIRED")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = stats.friedmanchisquare(*[values[:,i] for i in range(values.shape[1])])
    require(not caught and math.isfinite(float(res.pvalue)) and math.isfinite(float(res.statistic)), "FRIEDMAN_NOT_ESTIMABLE")
    return {"status":"COMPUTED_REVIEW_PENDING","method":"friedman_repeated","columns":fields,"n_total":len(frame),
            "n_complete_subjects":len(values),"n_excluded_missing":len(frame)-len(values),
            "statistic":float(res.statistic),"p":float(res.pvalue),
            "note":"Global repeated-measures comparison; no post-hoc comparisons or multiplicity correction."}


def _wilson(k,n):
    if not n:
        return {"estimate":None,"ci95":None,"numerator":k,"denominator":n}
    ci = stats.binomtest(int(k),int(n)).proportion_ci(confidence_level=.95,method="wilson")
    return {"estimate":k/n,"ci95":[float(ci.low),float(ci.high)],"numerator":k,"denominator":n}


def diagnostic(frame, spec):
    ref, index = spec.get("reference"), spec.get("index")
    require(isinstance(ref,dict) and isinstance(index,dict), "REFERENCE_AND_INDEX_REQUIRED")
    a,b = ref.get("column"),index.get("column")
    require(isinstance(a,str) and isinstance(b,str) and a != b and a in frame and b in frame, "REFERENCE_AND_INDEX_COLUMNS_REQUIRED")
    for item in (ref,index):
        require(isinstance(item.get("positive"),str) and isinstance(item.get("negative"),str)
                and item["positive"] != item["negative"], "POSITIVE_AND_NEGATIVE_CODES_REQUIRED")
        require(all(is_missing(v) or str(v) in {item["positive"],item["negative"]} for v in frame[item["column"]]),
                "UNDECLARED_DIAGNOSTIC_CODE")
    pairs = [(r,i) for r,i in zip(frame[a],frame[b]) if not is_missing(r) and not is_missing(i)]
    require(bool(pairs), "NO_COMPLETE_DIAGNOSTIC_PAIRS")
    tp = sum(r==ref["positive"] and i==index["positive"] for r,i in pairs)
    fn = sum(r==ref["positive"] and i==index["negative"] for r,i in pairs)
    fp = sum(r==ref["negative"] and i==index["positive"] for r,i in pairs)
    tn = sum(r==ref["negative"] and i==index["negative"] for r,i in pairs)
    return {"status":"COMPUTED_REVIEW_PENDING","method":"diagnostic_2x2","n_total":len(frame),
            "n_complete":len(pairs),"n_excluded_missing":len(frame)-len(pairs),"tp":tp,"fn":fn,"fp":fp,"tn":tn,
            "sensitivity":_wilson(tp,tp+fn),"specificity":_wilson(tn,tn+fp),
            "positive_predictive_value":_wilson(tp,tp+fp),"negative_predictive_value":_wilson(tn,tn+fn),
            "accuracy":_wilson(tp+tn,len(pairs)),
            "note":"Wilson 95% intervals; predictive values reflect this study sample and are prevalence-dependent."}


def diagnostic_roc(frame, spec):
    from sklearn.metrics import roc_auc_score,roc_curve
    ref=spec.get("reference");score=spec.get("score")
    require(isinstance(ref,dict) and isinstance(score,str) and score in frame,"REFERENCE_AND_SCORE_REQUIRED")
    field=ref.get("column");positive=ref.get("positive");negative=ref.get("negative")
    require(isinstance(field,str) and field in frame and field!=score and isinstance(positive,str) and isinstance(negative,str)
            and positive!=negative,"REFERENCE_CODES_REQUIRED")
    require(type(spec.get("higher_is_positive")) is bool,"SCORE_DIRECTION_REQUIRED")
    require(all(is_missing(v) or str(v) in {positive,negative} for v in frame[field]),"UNDECLARED_REFERENCE_CODE")
    numeric(frame[score].tolist())
    pairs=[(r,float(s)) for r,s in zip(frame[field],frame[score]) if not is_missing(r) and not is_missing(s)]
    labels=np.asarray([int(str(r)==positive) for r,_ in pairs]); values=np.asarray([s for _,s in pairs])
    require(len(labels)>0 and min(int(labels.sum()),int(len(labels)-labels.sum()))>=5,"AT_LEAST_FIVE_PER_REFERENCE_CLASS_REQUIRED")
    direction=1 if spec["higher_is_positive"] else -1
    z=direction*values
    fpr,tpr,threshold=roc_curve(labels,z,drop_intermediate=False)
    auc=float(roc_auc_score(labels,z))
    points=[{"fpr":float(f),"tpr":float(t),"threshold_original_scale":float(direction*q) if math.isfinite(float(q)) else None}
            for f,t,q in zip(fpr,tpr,threshold)]
    reps=spec.get("bootstrap_replicates",0); seed=spec.get("seed")
    require(type(reps) is int and (reps==0 or 100<=reps<=2000),"BOOTSTRAP_0_OR_100_TO_2000_REQUIRED")
    ci=None
    if reps:
        require(type(seed) is int and 0<=seed<2**32,"BOOTSTRAP_SEED_REQUIRED")
        rng=np.random.default_rng(seed);pos=np.where(labels==1)[0];neg=np.where(labels==0)[0];values_boot=[]
        for _ in range(reps):
            ix=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
            values_boot.append(float(roc_auc_score(labels[ix],z[ix])))
        ci=np.quantile(values_boot,[.025,.975]).tolist()
    return {"status":"COMPUTED_REVIEW_PENDING","method":"diagnostic_roc","n_total":len(frame),"n_complete":len(labels),
            "n_excluded_missing":len(frame)-len(labels),"reference_positive":int(labels.sum()),
            "reference_negative":int(len(labels)-labels.sum()),"auc":auc,"auc_ci95":ci,"roc":points,
            "decision_rule":"score >= threshold" if direction==1 else "score <= threshold",
            "bootstrap":{"replicates":reps,"seed":seed if reps else None,"method":"class-stratified subject percentile" if reps else "not requested"},
            "note":"AUC is estimated on this dataset; no optimal threshold selection or external validation is implied."}


def sample_size(spec):
    method = spec.get("method")
    require(method in {"mean_precision","proportion_precision","two_mean","two_proportion"}, "PLANNING_METHOD_REQUIRED")
    alpha, dropout = spec.get("alpha"),spec.get("dropout_fraction",0)
    require(isinstance(alpha,(int,float)) and not isinstance(alpha,bool) and 0 < alpha < 1, "ALPHA_REQUIRED")
    require(isinstance(dropout,(int,float)) and not isinstance(dropout,bool) and 0 <= dropout < 1, "DROPOUT_FRACTION_INVALID")
    z = float(stats.norm.ppf(1-alpha/2))
    def val(name):
        v=spec.get(name)
        require(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v), "PLANNING_PARAMETER_REQUIRED")
        return float(v)
    if method == "mean_precision":
        sd, margin = val("sd"),val("margin")
        require(sd>0 and margin>0, "POSITIVE_SD_AND_MARGIN_REQUIRED")
        n = math.ceil((z*sd/margin)**2)
        base = {"n":max(n,2)}
        assumption = "Normal-approximation precision for a mean with prespecified standard deviation."
    elif method == "proportion_precision":
        p, margin = val("proportion"),val("margin")
        require(0<p<1 and margin>0, "PROPORTION_AND_MARGIN_INVALID")
        n = math.ceil(z*z*p*(1-p)/(margin*margin))
        base = {"n":max(n,2)}
        assumption = "Normal-approximation precision for a proportion; sparse cases require another method."
    elif method == "two_mean":
        from statsmodels.stats.power import TTestIndPower
        sd,delta,power = val("sd"),abs(val("difference")),val("power")
        require(sd>0 and delta>0 and 0<power<1, "TWO_MEAN_PARAMETERS_INVALID")
        engine=TTestIndPower()
        fn=lambda n: engine.power(effect_size=delta/sd,nobs1=n,alpha=alpha,ratio=1,alternative="two-sided")-power
        initial=float(fn(2.))
        require(math.isfinite(initial), "POWER_CALCULATION_NOT_ESTIMABLE")
        if initial >= 0:
            root=2.0
        else:
            upper=4.0
            while upper <= 1_000_000:
                value=float(fn(upper))
                require(math.isfinite(value), "POWER_CALCULATION_NOT_ESTIMABLE")
                if value >= 0:
                    break
                upper*=2
            require(upper<=1_000_000, "REQUIRED_SAMPLE_SIZE_EXCEEDS_LIMIT")
            root=float(optimize.brentq(fn,2.,upper))
        n=max(2,math.ceil(root)); base={"n_per_group":n,"n_total":2*n}
        assumption = "Two-sided equal-size independent groups, common SD, pooled t test; not paired or Welch power."
    else:
        p1,p2,power=val("proportion_1"),val("proportion_2"),val("power")
        require(0<p1<1 and 0<p2<1 and p1!=p2 and 0<power<1, "TWO_PROPORTION_PARAMETERS_INVALID")
        avg=(p1+p2)/2; zb=float(stats.norm.ppf(power))
        require(zb>0, "POWER_MUST_EXCEED_HALF")
        n=math.ceil((z*math.sqrt(2*avg*(1-avg))+zb*math.sqrt(p1*(1-p1)+p2*(1-p2)))**2/(p1-p2)**2)
        base={"n_per_group":max(n,2),"n_total":2*max(n,2)}
        assumption = "Two-sided equal-size independent groups, normal approximation; sparse cases require another method."
    inflated={k:(math.ceil(v/(1-dropout)) if k!="n_total" else None) for k,v in base.items()}
    if "n_per_group" in inflated:
        inflated["n_total"]=2*inflated["n_per_group"]
    require(all(v is not None and v<=2_000_000 for v in inflated.values()), "REQUIRED_SAMPLE_SIZE_EXCEEDS_LIMIT")
    return {"status":"COMPUTED_REVIEW_PENDING","method":method,"assumptions":assumption,"alpha_two_sided":alpha,
            "dropout_fraction":dropout,"base":base,"after_dropout":inflated,
            "note":"Planning inputs are assumptions, not observed study results; clinical and methodological review is required."}
