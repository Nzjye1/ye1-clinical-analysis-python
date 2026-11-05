"""Fixed binary random forest with training-only transforms and local frozen trees."""
from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from clinical_methods import prediction as common
from local_data import DataError


def need(value, reason):
    if not value:
        raise DataError(reason)


def _frozen_scores(x, trees):
    # scikit-learn trees cast inference features to float32 before splitting.
    x = np.asarray(x, dtype=np.float32)
    scores = np.zeros(len(x), dtype=float)
    for tree in trees:
        for i, row in enumerate(x):
            node = 0
            while tree["left"][node] != -1:
                node = tree["left"][node] if row[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
            scores[i] += tree["probability"][node]
    return scores/len(trees)


def analyze(rows, spec):
    try:
        data, counts = common._rows(rows, spec, training=True)
        train_ix, test_ix = common._split(data, spec)
        train, test = [data[i] for i in train_ix], [data[i] for i in test_ix]
        y = np.asarray([r["_ye1_outcome"] for r in train], dtype=int)
        need(len(y) >= 40 and min(int(y.sum()), int(len(y)-y.sum())) >= 10, "INSUFFICIENT_TRAINING_CLASSES")
        transforms = common._preprocess_fit(train, spec["features"])
        x_train, names = common._transform(train, transforms)
        x_test, test_names = common._transform(test, transforms)
        need(names == test_names, "VALIDATION_DESIGN_MISMATCH")
        parameters = spec.get("forest")
        need(isinstance(parameters,dict), "FOREST_PARAMETERS_REQUIRED")
        n_trees, depth, leaf = parameters.get("n_estimators"),parameters.get("max_depth"),parameters.get("min_samples_leaf")
        need(type(n_trees) is int and 20 <= n_trees <= 300, "TREE_COUNT_OUTSIDE_20_TO_300")
        need(type(depth) is int and 2 <= depth <= 10, "MAX_DEPTH_OUTSIDE_2_TO_10")
        need(type(leaf) is int and 2 <= leaf <= 100, "MIN_LEAF_OUTSIDE_2_TO_100")
        forest=RandomForestClassifier(n_estimators=n_trees,max_depth=depth,min_samples_leaf=leaf,
                                      max_features="sqrt",random_state=spec["split"]["seed"],n_jobs=1)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            forest.fit(x_train,y)
        need(not caught and list(forest.classes_) == [0,1], "FOREST_TRAINING_FAILED")
        p = forest.predict_proba(x_test)[:,1]
        metrics = common._metrics(test,p,spec)
        trees=[]
        for estimator in forest.estimators_:
            t=estimator.tree_
            weights=t.value[:,0,:]
            denominators=weights.sum(axis=1)
            need(bool(np.all(denominators>0)), "FOREST_NODE_WEIGHTS_INVALID")
            probabilities=(weights[:,1]/denominators).tolist()
            trees.append({"left":t.children_left.tolist(),"right":t.children_right.tolist(),
                          "feature":t.feature.tolist(),"threshold":t.threshold.tolist(),"probability":probabilities})
        frozen=_frozen_scores(x_test,trees)
        need(bool(np.allclose(frozen,p,atol=1e-12,rtol=1e-12)), "FROZEN_FOREST_REPLAY_MISMATCH")
        counts.update(training_n=len(train), training_events=int(y.sum()), training_nonevents=int(len(y)-y.sum()),
                      holdout_n=len(test), holdout_events=sum(r["_ye1_outcome"] for r in test),
                      holdout_nonevents=sum(1-r["_ye1_outcome"] for r in test))
        private={"classification":"LOCAL_ONLY_MODEL_BUNDLE","schema_version":"ye1-forest-binary/1.0",
                 "features":spec["features"],"transforms":transforms,"feature_names":names,
                 "parameters":parameters,"trees":trees,"prediction_time":spec["prediction_time"],
                 "outcome_window":spec["outcome_window"],"event":spec["event"],"non_event":spec["non_event"]}
        report={"status":"ESTIMATED","schema_version":"clinical-prediction/1.0.0",
                "algorithm":"fixed_random_forest","counts":counts,"validation":"internal held-out "+spec["split"]["method"],
                "metrics":metrics,"forest_parameters":parameters,
                "feature_importance":[{"term":n,"importance":float(v)} for n,v in zip(names,forest.feature_importances_)],
                "private_model":"private/model.json; do not send to a model or external services",
                "limitations":["One fixed parameter set; no automatic hyperparameter search or external validation.",
                               "Feature importance reflects model behavior and may be biased; it is not causal effect.",
                               "No evidence of clinical usefulness or deployment approval."]}
        return report,private
    except common.PredictionBlocked as exc:
        raise DataError(str(exc)) from None


def analyze_hgb(rows, spec):
    """Fit one fixed HGB; report holdout metrics without exporting an unsafe pickle."""
    try:
        data,counts=common._rows(rows,spec,training=True)
        train_ix,test_ix=common._split(data,spec)
        train,test=[data[i] for i in train_ix],[data[i] for i in test_ix]
        y=np.asarray([r["_ye1_outcome"] for r in train],dtype=int)
        need(len(y)>=40 and min(int(y.sum()),int(len(y)-y.sum()))>=10,"INSUFFICIENT_TRAINING_CLASSES")
        transforms=common._preprocess_fit(train,spec["features"])
        x_train,names=common._transform(train,transforms);x_test,test_names=common._transform(test,transforms)
        need(names==test_names,"VALIDATION_DESIGN_MISMATCH")
        parameters=spec.get("hgb")
        need(isinstance(parameters,dict),"HGB_PARAMETERS_REQUIRED")
        iterations,leaves,minimum=parameters.get("max_iter"),parameters.get("max_leaf_nodes"),parameters.get("min_samples_leaf")
        rate=parameters.get("learning_rate")
        need(type(iterations) is int and 20<=iterations<=300,"HGB_ITERATIONS_OUTSIDE_20_TO_300")
        need(type(leaves) is int and 2<=leaves<=31,"HGB_LEAVES_OUTSIDE_2_TO_31")
        need(type(minimum) is int and 2<=minimum<=100,"MIN_LEAF_OUTSIDE_2_TO_100")
        need(isinstance(rate,(int,float)) and not isinstance(rate,bool) and 0<rate<=.3,"LEARNING_RATE_OUTSIDE_0_TO_0_3")
        model=HistGradientBoostingClassifier(max_iter=iterations,max_leaf_nodes=leaves,min_samples_leaf=minimum,
                                             learning_rate=float(rate),early_stopping=False,random_state=spec["split"]["seed"])
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always");model.fit(x_train,y)
        need(not caught and list(model.classes_)==[0,1],"HGB_TRAINING_FAILED")
        p=model.predict_proba(x_test)[:,1]
        metrics=common._metrics(test,p,spec)
        counts.update(training_n=len(train),training_events=int(y.sum()),training_nonevents=int(len(y)-y.sum()),
                      holdout_n=len(test),holdout_events=sum(r["_ye1_outcome"] for r in test),
                      holdout_nonevents=sum(1-r["_ye1_outcome"] for r in test))
        report={"status":"ESTIMATED","schema_version":"clinical-prediction/1.0.0","algorithm":"fixed_hist_gradient_boosting",
                "counts":counts,"validation":"internal held-out "+spec["split"]["method"],"metrics":metrics,
                "hgb_parameters":parameters,
                "limitations":["One fixed parameter set; no automatic tuning or external validation.",
                               "This prototype does not export a frozen HGB model; later inference is not supported.",
                               "No evidence of causality, clinical usefulness or deployment approval."]}
        return report,None
    except common.PredictionBlocked as exc:
        raise DataError(str(exc)) from None
