"""Prespecified 1:1 greedy propensity matching and paired outcome summaries."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats
import statsmodels.api as sm

from clinical_methods.association import AssociationBlocked, _separation_check
from local_data import DataError


def need(value, reason):
    if not value:
        raise DataError(reason)


def missing(value):
    return value is None or str(value).strip() == ""


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        raise DataError("INVALID_NUMERIC_VALUE") from None
    need(math.isfinite(value), "NONFINITE_NUMERIC_VALUE")
    return value


def _smd(treated, controls, denominator=None):
    a, b = np.asarray(treated, dtype=float), np.asarray(controls, dtype=float)
    if denominator is None:
        denominator = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0
    if not denominator:
        return (0.0 if a.mean() == b.mean() else None), denominator
    return float((a.mean()-b.mean())/denominator), denominator


def _match(logits, group, identifiers, caliper):
    """Treated descending logit; nearest available control; no replacement."""
    treated = [i for i, v in enumerate(group) if v == 1]
    controls = [i for i, v in enumerate(group) if v == 0]
    need(len(treated)*len(controls) <= 5_000_000, "MATCHING_RESOURCE_LIMIT")
    treated.sort(key=lambda i: (-logits[i], str(identifiers[i])))
    available = set(controls)
    pairs = []
    for i in treated:
        if not available:
            break
        j = min(available, key=lambda k: (abs(float(logits[i]-logits[k])), str(identifiers[k])))
        if abs(float(logits[i]-logits[j])) <= caliper:
            pairs.append((i, j))
            available.remove(j)
    return pairs


def _outcome(rows, pairs, outcome):
    if outcome is None:
        return None
    need(isinstance(outcome, dict) and outcome.get("type") in {"continuous", "binary"}, "OUTCOME_SPEC_INVALID")
    field = outcome.get("column")
    need(isinstance(field, str) and all(field in row for row in rows), "OUTCOME_COLUMN_REQUIRED")
    event, non_event = outcome.get("event"), outcome.get("non_event")
    if outcome["type"] == "binary":
        need(isinstance(event, str) and isinstance(non_event, str) and event != non_event, "OUTCOME_CODES_REQUIRED")
        need(all(missing(r[field]) or str(r[field]) in {event, non_event} for r in rows), "UNDECLARED_OUTCOME_CODE")
    else:
        for row in rows:
            if not missing(row[field]):
                number(row[field])
    complete = [(i, j) for i, j in pairs if not missing(rows[i][field]) and not missing(rows[j][field])]
    need(len(complete) >= 2, "MATCHED_OUTCOME_REQUIRES_TWO_COMPLETE_PAIRS")
    if outcome["type"] == "continuous":
        differences = np.asarray([number(rows[i][field])-number(rows[j][field]) for i, j in complete])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stats.ttest_1samp(differences, 0., alternative="two-sided")
        need(not caught and math.isfinite(float(result.pvalue)), "MATCHED_OUTCOME_NOT_ESTIMABLE")
        half = float(stats.t.ppf(.975, len(differences)-1)*stats.sem(differences))
        return {"type": "continuous", "n_complete_pairs": len(complete), "n_pairs_with_missing_outcome": len(pairs)-len(complete),
                "mean_difference_treated_minus_control": float(differences.mean()),
                "ci95": [float(differences.mean()-half), float(differences.mean()+half)],
                "method": "paired_t_test", "p": float(result.pvalue)}
    differences = np.asarray([(int(rows[i][field] == event)-int(rows[j][field] == event)) for i, j in complete], dtype=float)
    b, c = int(np.sum(differences == 1)), int(np.sum(differences == -1))
    p = float(stats.binomtest(b, b+c, .5).pvalue) if b+c else 1.0
    sem = float(stats.sem(differences))
    interval = None
    if sem > 0:
        half = float(stats.t.ppf(.975, len(differences)-1)*sem)
        interval = [float(differences.mean()-half), float(differences.mean()+half)]
    return {"type": "binary", "n_complete_pairs": len(complete), "n_pairs_with_missing_outcome": len(pairs)-len(complete),
            "risk_difference_treated_minus_control": float(differences.mean()), "risk_difference_t_ci95": interval,
            "discordant_treated_only": b, "discordant_control_only": c,
            "method": "exact_mcnemar_two_sided", "p": p,
            "note": "Approximate paired t interval; zero observed variance leaves the interval unestimated."}


def analyze_matching(rows, key, spec):
    treatment = spec.get("treatment")
    need(isinstance(treatment, dict), "TREATMENT_SPEC_REQUIRED")
    field, treated, control = treatment.get("column"), treatment.get("treated"), treatment.get("control")
    need(isinstance(field, str) and all(field in row for row in rows), "TREATMENT_COLUMN_REQUIRED")
    need(isinstance(treated, str) and isinstance(control, str) and treated and control and treated != control,
         "DISTINCT_TREATMENT_CODES_REQUIRED")
    features = spec.get("confounders")
    need(isinstance(features, list) and features, "PRESPECIFIED_CONFOUNDERS_REQUIRED")
    names = [f.get("column") for f in features if isinstance(f, dict)]
    outcome = spec.get("outcome")
    outcome_field = outcome.get("column") if isinstance(outcome, dict) else None
    need(outcome_field is None or outcome_field not in {field, key}, "OUTCOME_ROLE_CONFLICT")
    need(len(names) == len(features) and len(set(names)) == len(names) and all(isinstance(n, str) and n for n in names),
         "UNIQUE_CONFOUNDER_NAMES_REQUIRED")
    need(not set(names) & {field, key, outcome_field}, "CONFOUNDER_ROLE_CONFLICT")
    need(all(all(n in row for n in names) for row in rows), "CONFOUNDER_COLUMN_MISSING")
    need(spec.get("missing") == "complete_case", "COMPLETE_CASE_MUST_BE_EXPLICIT")
    caliper = spec.get("caliper_logit")
    need(isinstance(caliper, (int, float)) and not isinstance(caliper, bool) and math.isfinite(caliper) and caliper > 0,
         "PRESPECIFIED_ABSOLUTE_LOGIT_CALIPER_REQUIRED")
    need(all(missing(r[field]) or str(r[field]) in {treated, control} for r in rows), "UNDECLARED_TREATMENT_CODE")
    for feature in features:
        need(feature.get("available_before_treatment") is True, "CONFOUNDER_TIMING_REQUIRED")
        need(feature.get("type") in {"numeric", "categorical"}, "CONFOUNDER_TYPE_REQUIRED")
        if feature["type"] == "categorical":
            levels, ref = feature.get("levels"), feature.get("reference")
            need(isinstance(levels, list) and len(levels) >= 2 and all(isinstance(x, str) and x for x in levels)
                 and len(set(levels)) == len(levels) and ref in levels, "CONFOUNDER_LEVELS_REQUIRED")
        for r in rows:
            value = r[feature["column"]]
            if not missing(value):
                if feature["type"] == "numeric":
                    number(value)
                else:
                    need(str(value) in feature["levels"], "UNDECLARED_CONFOUNDER_LEVEL")
    complete = [r for r in rows if not missing(r[field]) and all(not missing(r[n]) for n in names)]
    need(len(complete) >= 10, "INSUFFICIENT_COMPLETE_ROWS")
    group = np.asarray([1 if str(r[field]) == treated else 0 for r in complete], dtype=int)
    need(min(int(group.sum()), int(len(group)-group.sum())) >= 5, "INSUFFICIENT_GROUP_SIZE")
    cols, labels = [], []
    for feature in features:
        n = feature["column"]
        if feature["type"] == "numeric":
            cols.append([number(r[n]) for r in complete]); labels.append(n)
        else:
            for level in feature["levels"]:
                if level != feature["reference"]:
                    cols.append([int(str(r[n]) == level) for r in complete]); labels.append(n+"="+level)
    x = np.column_stack(cols).astype(float)
    need(np.all(x.std(axis=0) > 0), "CONSTANT_CONFOUNDER")
    x_standard = (x-x.mean(axis=0))/x.std(axis=0)
    design = np.column_stack([np.ones(len(x)), x_standard])
    need(np.linalg.matrix_rank(design) == design.shape[1] and np.linalg.cond(design) < 1e8, "UNSTABLE_PROPENSITY_DESIGN")
    need(min(int(group.sum()), int(len(group)-group.sum())) / x.shape[1] >= 10, "SPARSE_TREATMENT_GROUP_PER_PARAMETER")
    try:
        _separation_check(design, group)
    except AssociationBlocked as exc:
        raise DataError(str(exc)) from None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.Logit(group, design).fit(disp=False, maxiter=200, tol=1e-10)
    need(not caught and bool(fit.mle_retvals.get("converged")) and np.isfinite(fit.params).all(), "PROPENSITY_MODEL_NOT_CONVERGED")
    logits = design @ fit.params
    ids = [str(r[key]) for r in complete]
    pairs = _match(logits, group, ids, float(caliper))
    need(len(pairs) >= 2, "FEWER_THAN_TWO_MATCHED_PAIRS")
    ti, ci = [i for i, _ in pairs], [j for _, j in pairs]
    balance = []
    for j, label in enumerate(labels):
        before, denominator = _smd(x[group == 1, j], x[group == 0, j])
        after, _ = _smd(x[ti, j], x[ci, j], denominator)
        balance.append({"term": label, "smd_before": before, "smd_after": after})
    matched_outcome = _outcome(complete, pairs, outcome)
    result = {"status": "COMPUTED_REVIEW_PENDING", "method": "1:1_greedy_propensity_matching_without_replacement",
              "n_total": len(rows), "n_complete_for_matching": len(complete), "n_excluded_missing": len(rows)-len(complete),
              "treated_before": int(group.sum()), "controls_before": int(len(group)-group.sum()),
              "matched_pairs": len(pairs), "unmatched_treated": int(group.sum())-len(pairs),
              "caliper_logit_absolute": float(caliper), "matching_order": "descending treated logit; ties by local identifier",
              "balance": balance, "outcome": matched_outcome,
              "limitations": ["Independent subjects; complete-case selection and propensity-model specification need review.",
                              "Covariate balance does not rule out unmeasured confounding or prove causality.",
                              "Greedy matching is order-dependent; paired outcome inference applies only to the matched set."]}
    private = {"classification": "LOCAL_ONLY_MATCHED_IDENTIFIERS", "pairs": [{"treated_id": ids[i], "control_id": ids[j]} for i, j in pairs]}
    return result, private
