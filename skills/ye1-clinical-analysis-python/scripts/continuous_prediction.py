"""Fixed regularized continuous-outcome prediction with an independent holdout."""
from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from clinical_methods import prediction as common
from local_data import DataError


def need(ok,reason):
    if not ok:raise DataError(reason)


def missing(v):
    return v is None or str(v).strip()==""


def number(v):
    try:x=float(v)
    except (ValueError,TypeError,OverflowError):raise DataError("INVALID_NUMERIC_VALUE") from None
    need(math.isfinite(x),"NONFINITE_NUMERIC_VALUE")
    return x


def analyze(rows,key,spec):
    outcome=spec.get("outcome");features=spec.get("features")
    need(isinstance(outcome,str) and outcome and all(outcome in r for r in rows),"CONTINUOUS_OUTCOME_REQUIRED")
    need(isinstance(spec.get("prediction_time"),str) and spec["prediction_time"].strip()
         and isinstance(spec.get("outcome_window"),str) and spec["outcome_window"].strip(),"PREDICTION_TIMING_REQUIRED")
    need(isinstance(features,list) and features and all(isinstance(f,dict) for f in features),"PREDICTION_FEATURES_REQUIRED")
    names=[f.get("name") for f in features]
    need(all(isinstance(n,str) and n for n in names) and len(set(names))==len(names)
         and not set(names)&{key,outcome},"PREDICTOR_ROLE_CONFLICT")
    need(all(all(n in row for n in names) for row in rows),"PREDICTOR_COLUMN_MISSING")
    need(spec.get("missing") in {"complete_case","train_median_mode"},"MISSING_METHOD_REQUIRED")
    for feature in features:
        need(feature.get("type") in {"number","category"} and feature.get("available_at_prediction") is True,
             "PREDICTOR_TYPE_AND_TIMING_REQUIRED")
        if feature["type"]=="category":
            levels=feature.get("levels")
            need(isinstance(levels,list) and len(levels)>=2 and all(isinstance(v,str) and v for v in levels)
                 and len(set(levels))==len(levels),"CATEGORY_LEVELS_REQUIRED")
    complete=[]; missing_outcome=0;excluded_features=0
    for row in rows:
        # Validate observed values even when an unrelated field is missing.
        if not missing(row[outcome]):number(row[outcome])
        record={}
        for feature in features:
            name=feature["name"];v=row[name]
            if missing(v):record[name]=None
            elif feature["type"]=="number":record[name]=number(v)
            else:
                need(str(v) in feature["levels"],"UNDECLARED_CATEGORY")
                record[name]=str(v)
        if missing(row[outcome]):missing_outcome+=1;continue
        if spec["missing"]=="complete_case" and any(v is None for v in record.values()):
            excluded_features+=1;continue
        record["__ye1_continuous_outcome"]=number(row[outcome]);complete.append(record)
    split=spec.get("split",{});seed=split.get("seed");fraction=split.get("test_fraction")
    need(split.get("method")=="random" and type(seed) is int and 0<=seed<2**32
         and isinstance(fraction,(int,float)) and not isinstance(fraction,bool) and .2<=fraction<=.5,
         "RANDOM_SPLIT_SEED_AND_FRACTION_REQUIRED")
    need(len(complete)>=50,"AT_LEAST_50_COMPLETE_OUTCOMES_REQUIRED")
    train_ix,test_ix=train_test_split(np.arange(len(complete)),test_size=float(fraction),random_state=seed)
    train=[complete[int(i)] for i in sorted(train_ix)];test=[complete[int(i)] for i in sorted(test_ix)]
    try:
        transforms=common._preprocess_fit(train,features)
        xtrain,term_names=common._transform(train,transforms);xtest,test_names=common._transform(test,transforms)
    except common.PredictionBlocked as exc:raise DataError(str(exc)) from None
    need(term_names==test_names and len(train)>max(20,5*xtrain.shape[1]) and len(test)>=10,
         "INSUFFICIENT_TRAINING_OR_HOLDOUT_INFORMATION")
    ytrain=np.asarray([r["__ye1_continuous_outcome"] for r in train]);ytest=np.asarray([r["__ye1_continuous_outcome"] for r in test])
    need(float(np.ptp(ytrain))>0,"CONSTANT_TRAINING_OUTCOME")
    algorithm=spec.get("algorithm");alpha=spec.get("alpha")
    need(algorithm in {"ridge","lasso","elasticnet"} and isinstance(alpha,(int,float))
         and not isinstance(alpha,bool) and math.isfinite(alpha) and alpha>0,"ALGORITHM_AND_POSITIVE_ALPHA_REQUIRED")
    if algorithm=="ridge":model=Ridge(alpha=float(alpha))
    elif algorithm=="lasso":model=Lasso(alpha=float(alpha),max_iter=5000,tol=1e-7)
    else:
        ratio=spec.get("l1_ratio")
        need(isinstance(ratio,(int,float)) and not isinstance(ratio,bool) and 0<ratio<1,"ELASTICNET_MIXING_RATIO_REQUIRED")
        model=ElasticNet(alpha=float(alpha),l1_ratio=float(ratio),max_iter=5000,tol=1e-7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning);model.fit(xtrain,ytrain)
    need(not caught and np.isfinite(model.coef_).all(),"PREDICTION_MODEL_WARNING_OR_NONCONVERGENCE")
    scores=model.predict(xtest)
    need(bool(np.isfinite(scores).all()),"NONFINITE_PREDICTION")
    metrics={"rmse":float(math.sqrt(mean_squared_error(ytest,scores))),"mae":float(mean_absolute_error(ytest,scores)),
             "r_squared":float(r2_score(ytest,scores)) if float(np.ptp(ytest))>0 else None}
    private={"classification":"LOCAL_ONLY_MODEL_BUNDLE","schema_version":"ye1-continuous-prediction/1.0",
             "algorithm":algorithm,"alpha":float(alpha),"l1_ratio":spec.get("l1_ratio") if algorithm=="elasticnet" else None,
             "features":features,"transforms":transforms,"term_names":term_names,
             "coefficients":model.coef_.tolist(),"intercept":float(model.intercept_),
             "prediction_time":spec["prediction_time"],"outcome_window":spec["outcome_window"]}
    result={"status":"ESTIMATED","method":"continuous_prediction","algorithm":algorithm,"n_total":len(rows),
            "missing_outcome":missing_outcome,"excluded_missing_features":excluded_features,
            "n_training":len(train),"n_holdout":len(test),"metrics":metrics,
            "private_model":"private/model.json; do not send to a model or external services",
            "limitations":["Fixed model and parameters; only internal random holdout, no tuning or external validation.",
                           "No bounds or clipping of predictions; clinical outcome scale and suitability require review.",
                           "Predictor timing and missing-data assumptions require review."]}
    return result,private
