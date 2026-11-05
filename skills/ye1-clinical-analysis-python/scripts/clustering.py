"""Exploratory K-means/Gaussian mixture grouping; not formal latent trajectory modeling."""
from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from local_data import DataError


def need(ok,reason):
    if not ok:
        raise DataError(reason)


def missing(v):
    return v is None or str(v).strip() == ""


def number(v):
    try:x=float(v)
    except (ValueError,TypeError,OverflowError):raise DataError("INVALID_NUMERIC_VALUE") from None
    need(math.isfinite(x),"NONFINITE_NUMERIC_VALUE")
    return x


def analyze(rows,key,spec):
    features=spec.get("features"); algorithm=spec.get("algorithm"); k=spec.get("n_clusters");seed=spec.get("seed")
    need(algorithm in {"kmeans","gaussian_mixture"},"CLUSTERING_ALGORITHM_REQUIRED")
    need(isinstance(features,list) and len(features)>=2 and all(isinstance(n,str) and n for n in features)
         and len(set(features))==len(features) and key not in features,"AT_LEAST_TWO_UNIQUE_FEATURES_REQUIRED")
    need(type(k) is int and 2<=k<=10 and type(seed) is int and 0<=seed<2**32,"CLUSTER_COUNT_AND_SEED_REQUIRED")
    need(spec.get("missing")=="complete_case","COMPLETE_CASE_MUST_BE_EXPLICIT")
    need(all(all(n in row for n in features) for row in rows),"CLUSTER_FEATURE_MISSING")
    for row in rows:
        for n in features:
            if not missing(row[n]):number(row[n])
    complete=[r for r in rows if all(not missing(r[n]) for n in features)]
    need(len(complete)>=max(30,k*10) and len(complete)<=5000,"CLUSTER_SAMPLE_OUTSIDE_30_TO_5000")
    x=np.asarray([[number(r[n]) for n in features] for r in complete])
    mean=x.mean(axis=0);scale=x.std(axis=0)
    need(bool(np.all(scale>1e-12)),"CONSTANT_CLUSTER_FEATURE")
    z=(x-mean)/scale
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        if algorithm=="kmeans":
            model=KMeans(n_clusters=k,n_init=10,max_iter=300,random_state=seed).fit(z)
            converged=model.n_iter_<300
            labels=model.labels_
        else:
            covariance=spec.get("covariance")
            need(covariance in {"full","diag"},"MIXTURE_COVARIANCE_REQUIRED")
            model=GaussianMixture(n_components=k,covariance_type=covariance,n_init=5,max_iter=300,
                                  random_state=seed,reg_covar=1e-6).fit(z)
            converged=model.converged_
            labels=model.predict(z)
    need(not caught and converged,"CLUSTER_MODEL_WARNING_OR_NONCONVERGENCE")
    counts=np.bincount(labels,minlength=k)
    need(bool(np.all(counts>=5)),"CLUSTER_WITH_FEWER_THAN_FIVE_SUBJECTS")
    silhouette=float(silhouette_score(z,labels))
    need(math.isfinite(silhouette),"SILHOUETTE_NOT_ESTIMABLE")
    groups=[]
    for label in range(k):
        subset=x[labels==label]
        groups.append({"cluster":label+1,"n":int(len(subset)),
                       "feature_means":{n:float(v) for n,v in zip(features,subset.mean(axis=0))}})
    report={"status":"COMPUTED_REVIEW_PENDING","method":"exploratory_"+algorithm,"n_total":len(rows),
            "n_used":len(complete),"n_excluded_missing":len(rows)-len(complete),"clusters":groups,
            "silhouette":silhouette,"seed":seed,
            "limitations":["Exploratory grouping, not formal LPA or longitudinal latent trajectory analysis.",
                           "Number of clusters and features were prespecified; no causal, diagnostic or clinical validation claim.",
                           "Complete cases and standardization affect the grouping."]}
    private={"classification":"LOCAL_ONLY_CLUSTER_ASSIGNMENTS",
             "assignments":[{"id":str(row[key]),"cluster":int(label)+1} for row,label in zip(complete,labels)]}
    return report,private
