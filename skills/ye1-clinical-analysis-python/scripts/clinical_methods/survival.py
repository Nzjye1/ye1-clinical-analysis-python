"""Explicit, independent-subject right-censored survival and competing risks.

The association entry point returns aggregate curves and model summaries. The
separate private prediction entry point also returns a frozen model and subject
predictions for protected local artifacts. Data access, project consent and
artifact privacy belong to the caller. Computation is not clinical approval.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter
if __package__:
    from . import resource_budget as budget_rules
else:
    import resource_budget as budget_rules


class SurvivalBlocked(ValueError):
    """A stable applicability code, never a patient value or row."""


def _need(condition, code):
    if not condition:
        raise SurvivalBlocked(code)


def _resource_limits(resource_budget, method_id):
    """Resolve allocation guards only; statistical applicability stays below."""
    try:
        return budget_rules.validate_limits(resource_budget, method_id) if resource_budget is not None else budget_rules.request_limits({}, method_id)
    except budget_rules.BudgetError as exc:
        raise SurvivalBlocked(exc.code) from None


def _check_resource_budget(rows, spec, horizons, resource_budget, method_id, operation):
    limits = _resource_limits(resource_budget, method_id)
    try:
        budget_rules.check_rows(len(rows), limits)
    except budget_rules.BudgetError as exc:
        raise SurvivalBlocked(exc.code) from None
    columns = max(1, sum(max(1, len(cov.get("levels", [])) - 1) if cov.get("type") == "category" else 1
                         for cov in spec.get("covariates", [])))
    _need(columns <= limits["max_design_columns"], "RESOURCE_DESIGN_BUDGET_EXCEEDED")
    n = len(rows)
    work = n * columns + n * max(1, len(horizons))
    if operation == "train":
        work += n * n * columns + n * columns * columns
    elif operation == "evaluate":
        work += n * n * max(1, len(horizons))
    _need(work <= limits["max_work_units"], "RESOURCE_WORK_BUDGET_EXCEEDED")
    return limits


def _missing(value):
    return value is None or value == ""


def _number(value):
    _need(not isinstance(value, bool), "INVALID_FINITE_NUMBER")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise SurvivalBlocked("INVALID_FINITE_NUMBER") from None
    _need(math.isfinite(number), "INVALID_FINITE_NUMBER")
    return number


def _code(value):
    # CSV strings and JSON numeric codes are intentionally distinct. This avoids
    # silently merging "01" and 1 or true and 1; exact protocol coding is required.
    _need(isinstance(value, (str, int, float)) and not isinstance(value, bool), "INVALID_EVENT_CODE")
    if isinstance(value, float):
        _need(math.isfinite(value), "INVALID_EVENT_CODE")
    return ("str" if isinstance(value, str) else "number", value)


def _coefficient_term(covariate, level=None):
    """Display labels share one checked namespace; never use them as row keys."""
    if covariate["type"] == "numeric":
        return covariate["name"]
    return covariate["name"] + "[" + str(level) + " vs " + str(covariate["reference"]) + "]"


def _validate_spec(spec):
    _need(isinstance(spec, dict), "SPEC_REQUIRED")
    method = spec.get("method")
    _need(method in {"kaplan_meier", "cox_ph", "aalen_johansen", "cause_specific_cox"}, "UNSUPPORTED_SURVIVAL_METHOD")
    for unsupported in ("left_truncation", "time_varying_covariates", "recurrent_events", "interval_censoring", "weights", "strata", "cluster"):
        _need(not spec.get(unsupported), "UNSUPPORTED_" + unsupported.upper())
    _need(spec.get("independent_subjects") is True, "INDEPENDENT_SUBJECTS_REQUIRED")
    _need(spec.get("missing_policy") == "complete_case", "EXPLICIT_COMPLETE_CASE_REQUIRED")
    for field in ("subject_id", "time", "status", "time_unit", "time_zero_definition", "event_definition", "censoring_definition"):
        _need(isinstance(spec.get(field), str) and bool(spec[field].strip()), "MISSING_" + field.upper())
    keys = [spec["subject_id"], spec["time"], spec["status"]]
    _need(len(set(keys)) == 3, "TIME_STATUS_AND_ID_MUST_DIFFER")
    _need("event_code" in spec and "censor_code" in spec, "EVENT_AND_CENSOR_CODES_REQUIRED")
    target = _code(spec["event_code"])
    censor = _code(spec["censor_code"])
    _need(target != censor, "EVENT_CENSOR_CODE_COLLISION")
    competing = spec.get("competing_codes", [])
    _need(isinstance(competing, list), "COMPETING_CODES_LIST_REQUIRED")
    competing = [_code(v) for v in competing]
    _need(len(set(competing)) == len(competing) and target not in competing and censor not in competing, "COMPETING_CODE_COLLISION")
    if method in {"aalen_johansen", "cause_specific_cox"}:
        _need(bool(competing), "COMPETING_EVENT_CODES_REQUIRED")
        definitions = spec.get("competing_event_definitions")
        _need(isinstance(definitions, list) and len(definitions) == len(competing), "COMPETING_EVENT_DEFINITIONS_REQUIRED")
        _need(all(isinstance(v, str) and v.strip() for v in definitions), "COMPETING_EVENT_DEFINITIONS_REQUIRED")
    else:
        _need(not competing, "USE_COMPETING_RISK_METHOD_FOR_COMPETING_EVENTS")
    covariates = spec.get("covariates", [])
    _need(isinstance(covariates, list), "COVARIATE_LIST_REQUIRED")
    if method in {"cox_ph", "cause_specific_cox"}:
        _need(bool(covariates), "COX_COVARIATES_REQUIRED")
        _need(spec.get("ties", "breslow") == "breslow", "ONLY_BRESLOW_TIES_SUPPORTED")
    else:
        _need(not covariates, "COVARIATES_REQUIRE_COX_METHOD")
    covariate_names = []
    for cov in covariates:
        _need(isinstance(cov, dict) and isinstance(cov.get("name"), str) and cov["name"].strip(), "COVARIATE_SPEC_REQUIRED")
        _need(cov.get("type") in {"numeric", "category"}, "UNSUPPORTED_COVARIATE_TYPE")
        _need(cov.get("available_at_time_zero") is True, "COVARIATE_TIMING_UNCONFIRMED")
        covariate_names.append(cov["name"])
        if cov["type"] == "category":
            levels = cov.get("levels")
            _need(isinstance(levels, list) and len(levels) >= 2, "CATEGORY_LEVELS_REQUIRED")
            encoded_levels = [_code(v) for v in levels]
            _need(len(set(encoded_levels)) == len(levels), "DUPLICATE_CATEGORY_LEVEL")
            _need("reference" in cov and _code(cov["reference"]) in encoded_levels, "CATEGORY_REFERENCE_REQUIRED")
    _need(len(set(covariate_names)) == len(covariate_names), "DUPLICATE_COVARIATE")
    _need(not set(covariate_names).intersection(keys), "OUTCOME_TIME_OR_ID_AS_COVARIATE")
    # User names and category values can contain our display punctuation. Also,
    # declared numeric and string levels remain distinct but may render alike.
    # Reject ambiguous coefficient labels before reading data or fitting; never
    # overwrite a caller column or silently merge two design columns by name.
    coefficient_terms = []
    for cov in covariates:
        if cov["type"] == "numeric":
            coefficient_terms.append(_coefficient_term(cov))
        else:
            coefficient_terms.extend(_coefficient_term(cov, level) for level in cov["levels"] if _code(level) != _code(cov["reference"]))
    _need(len(set(coefficient_terms)) == len(coefficient_terms), "AMBIGUOUS_COEFFICIENT_TERM_NAMES")
    columns = keys + covariate_names
    return method, target, censor, competing, covariates, columns


def _validate(rows, spec):
    method, target, censor, competing, covariates, columns = _validate_spec(spec)
    _need(isinstance(rows, list) and bool(rows), "EMPTY_DATA")
    _need(all(isinstance(row, dict) and all(k in row for k in columns) for row in rows), "REQUIRED_COLUMN_ABSENT")
    identifiers = [row[spec["subject_id"]] for row in rows]
    _need(all(not _missing(v) for v in identifiers), "MISSING_SUBJECT_ID")
    try:
        _need(len(set((_code(v)) for v in identifiers)) == len(rows), "REPEATED_SUBJECT_UNSUPPORTED")
    except TypeError:
        raise SurvivalBlocked("INVALID_SUBJECT_ID") from None
    kept, missing_by_field = [], Counter()
    allowed_codes = {target, censor, *competing}
    for row in rows:
        absent = [k for k in columns[1:] if _missing(row[k])]
        # Validate every observed event/time/category even on an otherwise
        # incomplete row; malformed source values are not silently deleted.
        if not _missing(row[spec["time"]]):
            _need(_number(row[spec["time"]]) > 0, "FOLLOWUP_MUST_BE_POSITIVE")
        if not _missing(row[spec["status"]]):
            _need(_code(row[spec["status"]]) in allowed_codes, "UNDECLARED_EVENT_STATUS")
        for cov in covariates:
            value = row[cov["name"]]
            if not _missing(value):
                if cov["type"] == "numeric":
                    _number(value)
                else:
                    _need(_code(value) in {_code(v) for v in cov["levels"]}, "UNDECLARED_CATEGORY")
        missing_by_field.update(absent)
        if not absent:
            kept.append(row)
    _need(bool(kept), "NO_COMPLETE_OBSERVATIONS")
    time = [_number(row[spec["time"]]) for row in kept]
    status = [0 if _code(row[spec["status"]]) == censor else 1 if _code(row[spec["status"]]) == target else 2 + competing.index(_code(row[spec["status"]])) for row in kept]
    grid = spec.get("evaluation_times")
    _need(isinstance(grid, list) and bool(grid), "PRESPECIFIED_EVALUATION_TIMES_REQUIRED")
    grid = [_number(t) for t in grid]
    _need(grid == sorted(set(grid)) and all(0 <= t <= max(time) for t in grid), "EVALUATION_GRID_OUTSIDE_OBSERVED_SUPPORT")
    return method, kept, time, status, grid, covariates, dict(missing_by_field)


def _curve(time, status, grid, cause_count, confidence=0.95):
    """KM/AJ event-before-censor convention at tied observed times."""
    import numpy as np
    from scipy.stats import norm

    time = np.asarray(time, dtype=float)
    status = np.asarray(status, dtype=int)
    event_times = np.unique(time[status > 0])
    survival, greenwood = 1.0, 0.0
    cif = np.zeros(cause_count)
    points = []
    critical = float(norm.ppf((1 + confidence) / 2))
    for current in event_times:
        at_risk = int(np.sum(time >= current))
        counts = np.array([np.sum((time == current) & (status == cause)) for cause in range(1, cause_count + 1)], dtype=float)
        all_events = float(counts.sum())
        before = survival
        cif += before * counts / at_risk
        survival *= 1.0 - all_events / at_risk
        if at_risk > all_events:
            greenwood += all_events / (at_risk * (at_risk - all_events))
        points.append((float(current), survival, cif.copy(), greenwood))
    result = []
    cursor = -1
    for current in grid:
        while cursor + 1 < len(points) and points[cursor + 1][0] <= current:
            cursor += 1
        if cursor < 0:
            value, cumulative, variance_term = 1.0, np.zeros(cause_count), 0.0
        else:
            _, value, cumulative, variance_term = points[cursor]
        if value <= 0 or value >= 1:
            interval = [float(value), float(value)]
        else:
            loglog = math.log(-math.log(value))
            se = math.sqrt(variance_term) / abs(math.log(value))
            interval = [math.exp(-math.exp(loglog + critical * se)), math.exp(-math.exp(loglog - critical * se))]
        result.append({
            "time": float(current), "at_risk_before_time": int(np.sum(time >= current)),
            "event_free_probability": float(value),
            "event_free_ci95": interval,
            "cumulative_incidence": [float(v) for v in cumulative],
        })
    return result


def _aj_bootstrap(time, status, grid, cause_count, spec):
    import numpy as np

    repeats = spec.get("bootstrap_replicates", 0)
    _need(isinstance(repeats, int) and not isinstance(repeats, bool) and (repeats == 0 or 20 <= repeats <= 2000), "BOOTSTRAP_REPLICATES_MUST_BE_ZERO_OR_20_TO_2000")
    if not repeats:
        return {"method": "none", "requested": 0, "attempted": 0, "valid": 0, "intervals": None}
    _need(isinstance(spec.get("seed"), int) and not isinstance(spec.get("seed"), bool) and spec["seed"] >= 0, "BOOTSTRAP_SEED_REQUIRED")
    _need(len(time) >= 2, "BOOTSTRAP_REQUIRES_TWO_INDEPENDENT_SUBJECTS")
    rng = np.random.default_rng(spec["seed"])
    time, status = np.asarray(time), np.asarray(status)
    values = []
    for _ in range(repeats):
        selected = rng.integers(0, len(time), size=len(time))
        curves = _curve(time[selected], status[selected], grid, cause_count)
        values.append([point["cumulative_incidence"] for point in curves])
    intervals = np.quantile(np.asarray(values), [0.025, 0.975], axis=0)
    return {
        "method": "independent_subject_percentile_bootstrap_pointwise_95pct",
        "requested": repeats, "attempted": repeats, "valid": repeats,
        "seed": spec["seed"], "intervals": intervals.tolist(),
        "limitations": "Exploratory pointwise percentile intervals; no simultaneous coverage or sparse-event reliability guarantee. No resampling until success.",
    }


def _design_matrix(rows, covariates):
    """Declared original-scale columns; no learning from prediction subjects."""
    import numpy as np

    columns, names, term_map = [], [], []
    for cov in covariates:
        first = len(columns)
        if cov["type"] == "numeric":
            columns.append([_number(row[cov["name"]]) for row in rows])
            names.append(_coefficient_term(cov))
        else:
            reference = _code(cov["reference"])
            for level in cov["levels"]:
                if _code(level) != reference:
                    columns.append([float(_code(row[cov["name"]]) == _code(level)) for row in rows])
                    names.append(_coefficient_term(cov, level))
        term_map.append({"term": cov["name"], "indices": list(range(first, len(columns)))})
    return np.asarray(columns, dtype=float).T, names, term_map


def _fit_cox(kept, time, status, covariates, spec):
    """The sole Cox fit. Its private state serves association, PH and prediction."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.stats import spearmanr
    from statsmodels.duration.hazard_regression import PHReg

    events = int(sum(v == 1 for v in status))
    original_x, names, term_map = _design_matrix(kept, covariates)
    x = original_x
    parameters = x.shape[1]
    counts = {"n_used": len(kept), "target_events": events, "parameters": parameters,
              "events_per_parameter": events / parameters, "event_per_parameter_formula": "target_events / non_intercept_design_parameters"}

    def blocked(reason):
        return {**counts, "status": "NON_ESTIMABLE", "reason": reason, "terms": [], "ph_assumption_verified": False}, None

    guard = spec.get("min_events_per_parameter", 10)
    _need(isinstance(guard, (int, float)) and not isinstance(guard, bool) and math.isfinite(guard) and guard >= 10, "COX_EVENT_GUARD_AT_LEAST_10_REQUIRED")
    if len(kept) <= parameters + 2 or events < max(20, guard * parameters):
        return blocked("INSUFFICIENT_EVENTS_OR_RESIDUAL_INFORMATION")
    scale = x.std(axis=0)
    if np.any(scale <= 0):
        return blocked("CONSTANT_OR_UNOBSERVED_COVARIATE_LEVEL")
    center = x.mean(axis=0)
    x = (x - center) / scale
    if np.linalg.matrix_rank(x) != parameters:
        return blocked("RANK_DEFICIENT_DESIGN")
    t, event = np.asarray(time), (np.asarray(status) == 1).astype(int)
    # Cox finite-MLE existence: a nonzero bounded direction with every
    # event-minus-risk-set contrast >= 0 and some > 0 gives monotone likelihood.
    # Use exact contrasts for moderate clinical tables; larger problems require
    # another checked solver rather than trusting convergence alone.
    event_indices = np.flatnonzero(event)
    contrast_n = sum(int(np.sum(t >= t[i])) for i in event_indices)
    if contrast_n > 2_000_000:
        return blocked("MLE_EXISTENCE_CHECK_RESOURCE_LIMIT")
    contrasts = np.vstack([x[i] - x[t >= t[i]] for i in event_indices])
    objective = contrasts.sum(axis=0)
    norm = float(np.max(np.abs(objective)))
    if norm > 1e-10:
        existence = linprog(-objective / norm, A_ub=-contrasts, b_ub=np.zeros(len(contrasts)), bounds=[(-1, 1)] * parameters, method="highs")
        if not existence.success:
            return blocked("MLE_EXISTENCE_CHECK_INCONCLUSIVE")
        if -float(existence.fun) > 1e-7:
            return blocked("MONOTONE_LIKELIHOOD")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PHReg(t, x, status=event, ties="breslow", missing="raise")
            fit = model.fit(disp=0, maxiter=200, tol=1e-9)
        if caught:
            return blocked("COX_FIT_WARNING")
        beta, se = np.asarray(fit.params), np.asarray(fit.bse)
        information = -model.hessian(beta)
        score = model.score(beta)
        if not np.isfinite(beta).all() or not np.isfinite(se).all() or np.any(se <= 0) or np.linalg.cond(information) > 1e10 or np.max(np.abs(score)) > 1e-5:
            return blocked("UNSTABLE_INFORMATION_OR_NONCONVERGENCE")
        beta, se = beta / scale, se / scale
        if np.any(np.abs(beta) + 1.959963984540054 * se > 700):
            return blocked("HAZARD_RATIO_INTERVAL_OVERFLOW")
        terms = []
        for name, coefficient, standard_error, pvalue in zip(names, beta, se, fit.pvalues):
            terms.append({"term": name, "log_hazard_ratio": float(coefficient), "standard_error": float(standard_error),
                          "hazard_ratio": float(np.exp(coefficient)), "hr_ci95": [float(np.exp(coefficient - 1.959963984540054 * standard_error)), float(np.exp(coefficient + 1.959963984540054 * standard_error))],
                          "p": float(pvalue), "unit": "per original covariate unit or declared category contrast"})
        residuals = np.asarray(fit.schoenfeld_residuals)
        trends = []
        for j, name in enumerate(names):
            observed = np.isfinite(residuals[:, j]) & (event == 1)
            rho = None
            if observed.sum() >= 3 and len(np.unique(t[observed])) >= 2 and np.std(residuals[observed, j]) > 0:
                rho = float(spearmanr(np.log(t[observed]), residuals[observed, j]).statistic)
                if not math.isfinite(rho):
                    rho = None
            trends.append({"term": name, "schoenfeld_log_time_spearman": rho})
        record = {**counts, "status": "ESTIMABLE_REVIEW_PENDING", "ties": "breslow", "covariance": "model_based_observed_information_independent_subjects",
                "terms": terms, "max_abs_score": float(np.max(np.abs(score))), "finite_mle_check": "exact_event_risk_set_contrast_linear_program",
                "ph_trend_diagnostics": trends, "ph_assumption_verified": False,
                "ph_diagnostic_note": "Descriptive correlation only, not a formal PH test or proof of proportional hazards. Inspect scientific time patterns and residual plots; prespecify alternatives when PH is doubtful.",
                "estimand": "cause_specific_hazard_ratio" if spec["method"] == "cause_specific_cox" else "hazard_ratio",
                "limitations": "HR is conditional association, not a risk ratio, CIF ratio, treatment effect, or causal effect. Functional form and independent noninformative censoring remain assumptions."}
        state = {"x": original_x, "standardized_x": x, "time": t, "event": event,
                 "center": center, "scale": scale, "beta": beta,
                 "standardized_beta": np.asarray(fit.params), "fit": fit,
                 "names": names, "term_map": term_map}
        return record, state
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, RuntimeError):
        return blocked("COX_FIT_FAILED")


def _cox(kept, time, status, covariates, spec):
    # Preserve the legacy dictionary and all numerical operations without a
    # ph_test request. Private fit state never enters an aggregate result.
    return _fit_cox(kept, time, status, covariates, spec)[0]


def _ph_config(value):
    _need(isinstance(value, dict) and set(value) == {"transform", "terms", "singledf", "global"}, "PH_TEST_SPEC_INVALID")
    _need(isinstance(value["transform"], str) and value["transform"] in {"km", "identity"} and value["terms"] is True
          and value["singledf"] is False and value["global"] is True, "PH_TEST_SPEC_INVALID")
    return value


def _checked_information(matrix, code):
    """Reject singular/ill-conditioned information; never ridge or pseudoinvert."""
    import numpy as np

    _need(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
          and np.isfinite(matrix).all(), code)
    _need(np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12), code)
    matrix = (matrix + matrix.T) / 2
    values = np.linalg.eigvalsh(matrix)
    _need(len(values) > 0 and values[0] > 0 and values[-1] / values[0] <= 1e10, code)
    return matrix


def _ph_score_test(time, event, x, beta, term_map, transform):
    """Breslow cox.zph score test with every beta retained as a nuisance.

    This internal mathematical helper takes an already fixed beta. The public
    caller supplies only the sole fitter's guarded converged state. Exact ties
    use T >= t, and g is the event-time transform for everyone in that risk set.
    """
    import numpy as np
    from scipy.special import softmax
    from scipy.stats import chi2

    t, event = np.asarray(time, dtype=float), np.asarray(event, dtype=int)
    x, beta = np.asarray(x, dtype=float), np.asarray(beta, dtype=float)
    p = x.shape[1]
    entries = [{"term": entry["term"], "indices": entry["indices"]} for entry in term_map]
    entries.append({"term": "GLOBAL", "indices": list(range(p))})

    def unavailable(reason):
        return [{"term": entry["term"], "chisq": None, "df": len(entry["indices"]),
                 "p_value": None, "status": "unavailable", "reason": reason} for entry in entries]

    try:
        _need(transform in {"km", "identity"}, "PH_TEST_SPEC_INVALID")
        _need(np.isfinite(t).all() and np.isfinite(x).all() and np.isfinite(beta).all(), "PH_NONFINITE_INPUT")
        times = np.unique(t[event == 1])
        _need(len(times) >= 2, "PH_INSUFFICIENT_DISTINCT_EVENT_TIMES")
        deaths = np.asarray([np.sum((t == current) & (event == 1)) for current in times], dtype=float)
        if transform == "identity":
            g = times.copy()
        else:
            survival, values = 1.0, []
            for current, d in zip(times, deaths):
                values.append(1 - survival)
                survival *= 1 - d / np.sum(t >= current)
            g = np.asarray(values)
        g -= np.dot(g, deaths) / deaths.sum()
        A, B, C, U = np.zeros((p, p)), np.zeros((p, p)), np.zeros((p, p)), np.zeros(p)
        for current, d, transformed in zip(times, deaths, g):
            risk_x = x[t >= current]
            weight = softmax(risk_x @ beta)
            mean = weight @ risk_x
            centered = risk_x - mean
            variance = centered.T @ (weight[:, None] * centered)
            residual = x[(t == current) & (event == 1)].sum(axis=0) - d * mean
            A += d * variance
            B += d * transformed * variance
            C += d * transformed * transformed * variance
            U += transformed * residual
        A = _checked_information(A, "PH_NUISANCE_INFORMATION_SINGULAR_OR_UNSTABLE")
        _need(np.isfinite(U).all(), "PH_NONFINITE_SCORE")
        table = []
        for entry in entries:
            ix = entry["indices"]
            row = {"term": entry["term"], "chisq": None, "df": len(ix), "p_value": None, "status": "unavailable"}
            try:
                # This is the restricted interaction hypothesis with ALL beta
                # nuisance columns, not a subblock of the global inverse.
                cross = B[:, ix]
                efficient = C[np.ix_(ix, ix)] - cross.T @ np.linalg.solve(A, cross)
                efficient = _checked_information(efficient, "PH_EFFICIENT_INFORMATION_SINGULAR_OR_UNSTABLE")
                score = U[ix]
                statistic = float(score @ np.linalg.solve(efficient, score))
                _need(math.isfinite(statistic) and statistic >= 0, "PH_NONFINITE_OR_NEGATIVE_STATISTIC")
                probability = float(chi2.sf(statistic, len(ix)))
                _need(math.isfinite(probability) and 0 <= probability <= 1, "PH_NONFINITE_P_VALUE")
                row.update(chisq=statistic, p_value=probability, status="completed")
            except (SurvivalBlocked, np.linalg.LinAlgError) as exc:
                row["reason"] = str(exc) if isinstance(exc, SurvivalBlocked) else "PH_INFORMATION_SOLVE_FAILED"
            table.append(row)
        return table
    except (SurvivalBlocked, np.linalg.LinAlgError) as exc:
        return unavailable(str(exc) if isinstance(exc, SurvivalBlocked) else "PH_INFORMATION_SOLVE_FAILED")


def _formal_ph(state, config, term_map, reason=None):
    if state is None:
        table = [{"term": entry["term"], "chisq": None, "df": len(entry["indices"]),
                  "p_value": None, "status": "unavailable", "reason": reason or "COX_FIT_NOT_ESTIMABLE"}
                 for entry in term_map + [{"term": "GLOBAL", "indices": [j for entry in term_map for j in entry["indices"]]}]]
    else:
        table = _ph_score_test(state["time"], state["event"], state["standardized_x"],
                               state["standardized_beta"], term_map, config["transform"])
    return {"status": "completed" if all(row["status"] == "completed" for row in table) else "unavailable",
            "method": "cox_score", **config, "table": table}


COX_MODEL_SCHEMA = "clinical-frozen-survival-cox@0.5.0-candidate1"
COX_CORE_VERSION = "0.5.0-candidate1"
_SEMANTIC_FIELDS = ("time_unit", "time_zero_definition", "event_definition", "censoring_definition", "event_code", "censor_code")
_FEATURE_SEMANTIC_FIELDS = {"name", "dtype", "role", "unit", "dimension", "concept", "occurred_at", "available_at", "levels", "reference"}


def _model_digest(model):
    import hashlib
    import json

    payload = {key: value for key, value in model.items() if key != "model_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _frozen_number(value):
    # CSV input coercion is intentional in _number. Frozen JSON fields must be
    # actual JSON numbers, so strings and booleans cannot masquerade as numbers.
    _need(type(value) in (int, float), "MODEL_NUMBER_INVALID")
    return _number(value)


def _feature_semantics(value, covariates):
    _need(isinstance(value, list) and len(value) == len(covariates), "FEATURE_SEMANTICS_INVALID")
    for entry, cov in zip(value, covariates):
        _need(isinstance(entry, dict) and set(entry) == _FEATURE_SEMANTIC_FIELDS, "FEATURE_SEMANTICS_INVALID")
        _need(entry["name"] == cov["name"], "FEATURE_SEMANTICS_ORDER_MISMATCH")
        _need(all(isinstance(entry[key], str) and entry[key].strip() for key in
                  ("name", "dtype", "role", "unit", "dimension", "concept")), "FEATURE_SEMANTICS_INVALID")
        _need(entry["dtype"] == ("number" if cov["type"] == "numeric" else "category")
              and entry["role"] in {"exposure", "covariate", "group"}, "FEATURE_SEMANTICS_INVALID")
        _need(type(entry["occurred_at"]) in (int, float) and type(entry["available_at"]) in (int, float), "FEATURE_SEMANTICS_TIMING_INVALID")
        _need(_number(entry["occurred_at"]) <= _number(entry["available_at"]) <= 0, "FEATURE_SEMANTICS_TIMING_INVALID")
        _need(entry["levels"] == cov.get("levels", []) and entry["reference"] == cov.get("reference"), "FEATURE_SEMANTICS_CATEGORY_MISMATCH")
    return value


def _prediction_data(rows, spec, require_features):
    method, target, censor, competing, covariates, columns = _validate_spec(spec)
    _need(method == "cox_ph" and not competing and "ph_test" not in spec, "PREDICTION_REQUIRES_PLAIN_COX_PH")
    _need(isinstance(rows, list) and bool(rows) and all(isinstance(row, dict) for row in rows), "EMPTY_DATA")
    _need(all(all(key in row for key in columns) for row in rows), "REQUIRED_COLUMN_ABSENT")
    identifiers = [row[spec["subject_id"]] for row in rows]
    _need(all(isinstance(value, str) and value.strip() for value in identifiers), "CANONICAL_STRING_SUBJECT_ID_REQUIRED")
    _need(len(set(identifiers)) == len(identifiers), "REPEATED_SUBJECT_UNSUPPORTED")
    for row in rows:
        if not _missing(row[spec["time"]]):
            _need(_number(row[spec["time"]]) > 0, "FOLLOWUP_MUST_BE_POSITIVE")
        if not _missing(row[spec["status"]]):
            _need(_code(row[spec["status"]]) in {target, censor}, "UNDECLARED_EVENT_STATUS")
        for cov in covariates:
            value = row[cov["name"]]
            if _missing(value):
                _need(not require_features, "PREDICTION_FEATURE_MISSING")
            elif cov["type"] == "numeric":
                _number(value)
            else:
                _need(_code(value) in {_code(level) for level in cov["levels"]}, "UNDECLARED_CATEGORY")
    return identifiers


def _prediction_config(prediction, spec):
    required = {"operation", "training_subjects", "holdout_subjects", "horizons",
                "canonical_identity_namespace", "feature_missing_policy"}
    _need(isinstance(prediction, dict) and set(prediction) == required, "COX_PREDICTION_SPEC_INVALID")
    _need(prediction["operation"] in {"train", "infer"} and prediction["feature_missing_policy"] == "complete_case", "COX_PREDICTION_SPEC_INVALID")
    _need(isinstance(prediction["canonical_identity_namespace"], str) and bool(prediction["canonical_identity_namespace"].strip()), "CANONICAL_IDENTITY_NAMESPACE_REQUIRED")
    for key in ("training_subjects", "holdout_subjects"):
        values = prediction[key]
        _need(isinstance(values, list) and all(isinstance(value, str) and value.strip() for value in values), "PARTITION_SUBJECT_IDS_INVALID")
        _need(len(set(values)) == len(values), "PARTITION_SUBJECT_IDS_DUPLICATED")
    horizons = prediction["horizons"]
    _need(isinstance(horizons, list) and bool(horizons), "PREDICTION_HORIZONS_REQUIRED")
    horizons = [_number(value) for value in horizons]
    _need(horizons == sorted(set(horizons)) and horizons[0] >= 0, "PREDICTION_HORIZONS_INVALID")
    _need(spec.get("evaluation_times") == prediction["horizons"], "PREDICTION_HORIZONS_SPEC_MISMATCH")
    if prediction["operation"] == "infer":
        _need(not prediction["training_subjects"] and not prediction["holdout_subjects"], "INFER_PARTITIONS_MUST_BE_EMPTY")
    return horizons


def _breslow_baseline(state):
    """Right-continuous H(t | training design mean), including events at t.

    Installed statsmodels 0.14.6 uses cumsum(increment)-increment in its
    baseline routine. That pre-event convention is deliberately not consumed.
    This calculation reuses the sole guarded fit; it never fits another model.
    """
    import numpy as np
    from scipy.special import logsumexp

    time, event = state["time"], state["event"]
    times = np.unique(time[event == 1])
    eta = (state["x"] - state["center"]) @ state["beta"]
    increments = []
    for current in times:
        log_increment = math.log(int(np.sum((time == current) & (event == 1)))) - float(logsumexp(eta[time >= current]))
        _need(math.isfinite(log_increment) and -745 < log_increment < 709, "BASELINE_HAZARD_NUMERIC_RANGE")
        increments.append(math.exp(log_increment))
    hazard = np.cumsum(increments)
    _need(np.isfinite(hazard).all() and np.all(np.diff(hazard) > 0) and hazard[0] > 0, "BASELINE_HAZARD_NOT_FINITE_INCREASING")
    return {"event_times": times.astype(float).tolist(), "cumulative_hazard": hazard.tolist(),
            "center": state["center"].tolist(), "convention": "right_continuous_breslow"}


def _freeze_cox(state, fit_record, spec, prediction, feature_semantics, provider_version, runtime, n_declared, missing, study_context=None):
    import copy
    import numpy as np

    scale = state["scale"]
    covariance = np.asarray(state["fit"].cov_params()) / np.outer(scale, scale)
    model = {
        "schema_version": COX_MODEL_SCHEMA, "method_id": "prediction.survival.cox", "core_version": COX_CORE_VERSION,
        "provider_version": provider_version, "runtime": copy.deepcopy(runtime),
        "test_only": True, "scientific_review": "pending", "export_status": "local_only",
        "identity": {"canonical_identity_namespace": prediction["canonical_identity_namespace"],
                     "training_subjects": list(prediction["training_subjects"])},
        "survival_semantics": {key: copy.deepcopy(spec[key]) for key in _SEMANTIC_FIELDS},
        "covariates": copy.deepcopy(spec["covariates"]), "feature_semantics": copy.deepcopy(feature_semantics),
        "feature_missing_policy": "complete_case", "coefficients": state["beta"].tolist(),
        "coefficient_covariance": covariance.tolist(),
        "design": {"columns": state["names"], "training_mean": state["center"].tolist(),
                   "training_std": scale.tolist(), "term_map": state["term_map"]},
        "baseline": _breslow_baseline(state),
        "support": {"last_event_time": float(max(state["time"][state["event"] == 1])),
                    "max_observed_time": float(max(state["time"])), "time_unit": spec["time_unit"]},
        "training_summary": {"n_declared": n_declared, "n_complete": len(state["time"]),
                             "n_events": int(sum(state["event"])), "n_censored": int(sum(state["event"] == 0)),
                             "n_excluded_missing": n_declared - len(state["time"]), "missing_by_field": dict(missing)},
        "fit": {"engine": "statsmodels.PHReg", "ties": "breslow",
                "finite_mle_check": fit_record["finite_mle_check"], "max_abs_score": fit_record["max_abs_score"],
                "parameters": fit_record["parameters"], "events_per_parameter": fit_record["events_per_parameter"],
                "min_events_per_parameter": spec.get("min_events_per_parameter", 10)},
    }
    if study_context is not None:
        fields = {"provenance_version", "data_origin", "task_purpose", "test_only", "scientific_review", "export_status"}
        _need(isinstance(study_context, dict) and set(study_context) == fields, "STUDY_CONTEXT_INVALID")
        model.update(copy.deepcopy(study_context))
    model["model_sha256"] = _model_digest(model)
    return model


def _validate_frozen_cox(model, spec, prediction, feature_semantics, provider_version, runtime, study_context=None):
    """Strict JSON artifact validation. No estimator or transform is fitted."""
    import numpy as np

    keys = {"schema_version", "method_id", "core_version", "provider_version", "runtime", "test_only", "scientific_review", "export_status",
            "identity", "survival_semantics", "covariates", "feature_semantics", "feature_missing_policy", "coefficients",
            "coefficient_covariance", "design", "baseline", "support", "training_summary", "fit", "model_sha256"}
    if study_context is not None and isinstance(model, dict) and model.get("provenance_version") is not None:
        keys |= {"provenance_version", "data_origin", "task_purpose"}
    _need(isinstance(model, dict) and set(model) == keys, "MODEL_SCHEMA_INVALID")
    _need(model["schema_version"] == COX_MODEL_SCHEMA and model["method_id"] == "prediction.survival.cox"
          and model["core_version"] == COX_CORE_VERSION, "MODEL_VERSION_MISMATCH")
    if study_context is None:
        _need(model["test_only"] is True and model["scientific_review"] == "pending" and model["export_status"] == "local_only", "MODEL_CLASSIFICATION_MISMATCH")
    else:
        if __package__:
            from . import real_family_admission as study_origin
        else:
            import real_family_admission as study_origin
        study_origin.frozen_context(study_context, model,
            lambda condition, code, stage="binding": _need(condition, code))
    _need(model["provider_version"] == provider_version and model["runtime"] == runtime, "MODEL_PROVIDER_RUNTIME_MISMATCH")
    _need(isinstance(model["model_sha256"], str) and model["model_sha256"] == _model_digest(model), "MODEL_INTERNAL_HASH_MISMATCH")
    _need(model["feature_missing_policy"] == "complete_case", "MODEL_MISSING_POLICY_MISMATCH")
    _need(model["survival_semantics"] == {key: spec[key] for key in _SEMANTIC_FIELDS}, "MODEL_SURVIVAL_SEMANTICS_MISMATCH")
    _need(model["covariates"] == spec["covariates"] and model["feature_semantics"] == feature_semantics, "MODEL_FEATURE_SEMANTICS_MISMATCH")
    _validate_spec({**spec, "covariates": model["covariates"]})
    _feature_semantics(model["feature_semantics"], model["covariates"])
    identity = model["identity"]
    _need(isinstance(identity, dict) and set(identity) == {"canonical_identity_namespace", "training_subjects"}, "MODEL_IDENTITY_INVALID")
    _need(identity["canonical_identity_namespace"] == prediction["canonical_identity_namespace"], "MODEL_IDENTITY_NAMESPACE_MISMATCH")
    ids = identity["training_subjects"]
    _need(isinstance(ids, list) and 0 < len(ids) <= budget_rules.ABSOLUTE_MAX_ROWS and all(isinstance(value, str) and value.strip() for value in ids)
          and len(ids) == len(set(ids)), "MODEL_TRAINING_IDENTITIES_INVALID")
    # Build just the declared column layout without reading subjects or learning.
    prototype = {cov["name"]: 0 if cov["type"] == "numeric" else cov["reference"] for cov in spec["covariates"]}
    _, names, term_map = _design_matrix([prototype], spec["covariates"])
    p = len(names)
    design = model["design"]
    _need(isinstance(design, dict) and set(design) == {"columns", "training_mean", "training_std", "term_map"}, "MODEL_DESIGN_INVALID")
    _need(design["columns"] == names and design["term_map"] == term_map, "MODEL_DESIGN_ORDER_MISMATCH")
    _need(all(isinstance(entry, dict) and set(entry) == {"term", "indices"}
              and isinstance(entry["indices"], list) and all(type(index) is int for index in entry["indices"])
              for entry in design["term_map"]), "MODEL_DESIGN_ORDER_MISMATCH")

    def vector(value, length):
        _need(isinstance(value, list) and len(value) == length, "MODEL_DIMENSION_INVALID")
        return np.asarray([_frozen_number(number) for number in value], dtype=float)

    vector(model["coefficients"], p)
    mean, scale = vector(design["training_mean"], p), vector(design["training_std"], p)
    _need(np.all(scale > 0), "MODEL_SCALE_INVALID")
    covariance = model["coefficient_covariance"]
    _need(isinstance(covariance, list) and len(covariance) == p, "MODEL_COVARIANCE_INVALID")
    covariance = np.asarray([vector(row, p) for row in covariance])
    _need(np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12), "MODEL_COVARIANCE_INVALID")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        raise SurvivalBlocked("MODEL_COVARIANCE_INVALID") from None
    baseline = model["baseline"]
    _need(isinstance(baseline, dict) and set(baseline) == {"event_times", "cumulative_hazard", "center", "convention"}, "MODEL_BASELINE_INVALID")
    _need(baseline["convention"] == "right_continuous_breslow" and baseline["center"] == design["training_mean"], "MODEL_BASELINE_CONVENTION_MISMATCH")
    _need(isinstance(baseline["event_times"], list) and bool(baseline["event_times"]), "MODEL_BASELINE_INVALID")
    times = vector(baseline["event_times"], len(baseline["event_times"]))
    hazard = vector(baseline["cumulative_hazard"], len(times))
    vector(baseline["center"], p)
    _need(times[0] > 0 and np.all(np.diff(times) > 0) and hazard[0] > 0 and np.all(np.diff(hazard) > 0), "MODEL_BASELINE_NOT_INCREASING")
    support = model["support"]
    _need(isinstance(support, dict) and set(support) == {"last_event_time", "max_observed_time", "time_unit"}, "MODEL_SUPPORT_INVALID")
    _need(_frozen_number(support["last_event_time"]) == times[-1] and _frozen_number(support["max_observed_time"]) >= times[-1]
          and support["time_unit"] == spec["time_unit"], "MODEL_SUPPORT_INVALID")
    summary = model["training_summary"]
    _need(isinstance(summary, dict) and set(summary) == {"n_declared", "n_complete", "n_events", "n_censored", "n_excluded_missing", "missing_by_field"}, "MODEL_TRAINING_SUMMARY_INVALID")
    _need(all(type(summary[key]) is int and summary[key] >= 0 for key in summary if key != "missing_by_field"), "MODEL_TRAINING_SUMMARY_INVALID")
    _need(summary["n_declared"] == len(ids) == summary["n_complete"] + summary["n_excluded_missing"]
          and summary["n_complete"] == summary["n_events"] + summary["n_censored"]
          and len(times) <= summary["n_events"], "MODEL_TRAINING_SUMMARY_INVALID")
    missing = summary["missing_by_field"]
    _need(isinstance(missing, dict) and set(missing) <= {spec["time"], spec["status"], *[cov["name"] for cov in spec["covariates"]]}
          and all(type(value) is int and 0 <= value <= summary["n_excluded_missing"] for value in missing.values())
          and sum(missing.values()) >= summary["n_excluded_missing"], "MODEL_MISSING_SUMMARY_INVALID")
    fit = model["fit"]
    _need(isinstance(fit, dict) and set(fit) == {"engine", "ties", "finite_mle_check", "max_abs_score", "parameters", "events_per_parameter", "min_events_per_parameter"}, "MODEL_FIT_METADATA_INVALID")
    guard = _frozen_number(fit["min_events_per_parameter"])
    _need(fit["engine"] == "statsmodels.PHReg" and fit["ties"] == "breslow"
          and fit["finite_mle_check"] == "exact_event_risk_set_contrast_linear_program"
          and type(fit["parameters"]) is int and fit["parameters"] == p
          and 0 <= _frozen_number(fit["max_abs_score"]) <= 1e-5 and guard >= 10
          and summary["n_complete"] > p + 2 and summary["n_events"] >= max(20, guard * p)
          and _frozen_number(fit["events_per_parameter"]) == summary["n_events"] / p, "MODEL_FIT_METADATA_INVALID")
    return model


def _frozen_predictions(rows, spec, model, horizons):
    import numpy as np

    _need(horizons[-1] <= model["support"]["last_event_time"], "PREDICTION_HORIZON_BEYOND_LAST_TRAINING_EVENT")
    x, names, _ = _design_matrix(rows, model["covariates"])
    _need(names == model["design"]["columns"], "MODEL_DESIGN_ORDER_MISMATCH")
    eta = (x - np.asarray(model["baseline"]["center"])) @ np.asarray(model["coefficients"])
    _need(np.isfinite(eta).all(), "PREDICTION_LINEAR_PREDICTOR_NONFINITE")
    event_times, baseline = np.asarray(model["baseline"]["event_times"]), np.asarray(model["baseline"]["cumulative_hazard"])
    predictions = []
    for row, linear in zip(rows, eta):
        for horizon in horizons:
            index = int(np.searchsorted(event_times, horizon, side="right")) - 1
            if index < 0:
                survival, risk = 1.0, 0.0
            else:
                log_hazard = math.log(float(baseline[index])) + float(linear)
                # Limiting probabilities avoid overflowing exp for extreme but
                # finite predictors; no clipping of the underlying coefficient.
                hazard = math.exp(log_hazard) if log_hazard < 709 else math.inf
                survival, risk = math.exp(-hazard), -math.expm1(-hazard)
            _need(math.isfinite(survival) and math.isfinite(risk) and 0 <= survival <= 1 and 0 <= risk <= 1, "PREDICTION_PROBABILITY_INVALID")
            predictions.append({"subject_id": row[spec["subject_id"]], "horizon": float(horizon),
                                "survival_probability": survival, "event_risk": risk})
    return predictions


def cox_prediction(rows, spec, prediction, feature_semantics, provider_version, runtime, model=None, study_context=None, resource_budget=None):
    """Private train/reload/inference payload; callers must protect model and rows."""
    import copy

    try:
        _need(isinstance(spec, dict), "SPEC_REQUIRED")
        horizons = _prediction_config(prediction, spec)
        _need(isinstance(rows, list) and bool(rows), "EMPTY_DATA")
        _check_resource_budget(rows, spec, horizons, resource_budget, "prediction.survival.cox", prediction["operation"])
        ids = _prediction_data(rows, spec, require_features=prediction["operation"] == "infer")
        _feature_semantics(feature_semantics, spec["covariates"])
        operation = prediction["operation"]
        if operation == "train":
            _need(model is None, "TRAIN_MUST_NOT_SUPPLY_MODEL")
            training_ids, holdout_ids = prediction["training_subjects"], prediction["holdout_subjects"]
            _need(bool(training_ids) and bool(holdout_ids) and not set(training_ids).intersection(holdout_ids)
                  and set(training_ids).union(holdout_ids) == set(ids), "EXPLICIT_COMPLETE_DISJOINT_PARTITION_REQUIRED")
            train = [row for row in rows if row[spec["subject_id"]] in set(training_ids)]
            predicted = [row for row in rows if row[spec["subject_id"]] in set(holdout_ids)]
            _prediction_data(predicted, spec, require_features=True)
            training_spec = {**spec, "evaluation_times": [0]}
            _, kept, time, status, _, covariates, missing = _validate(train, training_spec)
            fit_record, state = _fit_cox(kept, time, status, covariates, training_spec)
            _need(state is not None, fit_record.get("reason", "COX_FIT_NOT_ESTIMABLE"))
            model = _freeze_cox(state, fit_record, spec, prediction, feature_semantics, provider_version, runtime, len(train), missing, study_context=study_context)
        else:
            model = copy.deepcopy(model)
            predicted = rows
        _validate_frozen_cox(model, spec, prediction, feature_semantics, provider_version, runtime, study_context=study_context)
        _need(not {row[spec["subject_id"]] for row in predicted}.intersection(model["identity"]["training_subjects"]), "EXTERNAL_SUBJECT_OVERLAPS_TRAINING")
        individual = _frozen_predictions(predicted, spec, model, horizons)
        frozen = model["training_summary"]
        population = {
            "n_input": len(rows), "n_training_declared": frozen["n_declared"], "n_training_complete": frozen["n_complete"],
            "n_training_events": frozen["n_events"], "n_training_censored": frozen["n_censored"],
            "n_holdout": len(predicted) if operation == "train" else 0, "n_predicted": len(predicted),
            "n_predicted_events": sum(not _missing(row[spec["status"]]) and _code(row[spec["status"]]) == _code(spec["event_code"]) for row in predicted),
            "n_predicted_censored": sum(not _missing(row[spec["status"]]) and _code(row[spec["status"]]) == _code(spec["censor_code"]) for row in predicted),
            "n_predicted_missing_time": sum(_missing(row[spec["time"]]) for row in predicted),
            "n_predicted_missing_status": sum(_missing(row[spec["status"]]) for row in predicted),
            "n_dropped_training_missing": frozen["n_excluded_missing"], "training_missing_by_field": copy.deepcopy(frozen["missing_by_field"]),
        }
        summaries = []
        for horizon in horizons:
            selected = [row for row in individual if row["horizon"] == horizon]
            summary = {"horizon": float(horizon), "n": len(selected)}
            for field in ("survival_probability", "event_risk"):
                values = [row[field] for row in selected]
                summary.update({field + "_min": min(values), field + "_mean": math.fsum(values) / len(values), field + "_max": max(values)})
            summaries.append(summary)
        record = {
            "method_id": "prediction.survival.cox", "operation": operation, "analysis_population": population,
            "horizons": horizons, "prediction_summary": summaries, "model_sha256": model["model_sha256"],
            "support": {**model["support"], "horizons": horizons, "extrapolation": "forbidden_beyond_last_training_event"},
            "baseline_convention": "right_continuous_breslow", "fit_engine": "statsmodels.PHReg", "ties": "breslow",
            "extrapolation": "forbidden", "summary_estimand": "mean_frozen_predicted_absolute_event_risk",
            "uncertainty": "not_implemented", "warnings": [],
        }
        return {"ok": True, "record": record, "model_bundle": model, "individual_predictions": individual}
    except SurvivalBlocked as exc:
        return {"ok": False, "reason_code": str(exc)}


def analyze(rows: list[dict], spec: dict) -> dict:
    """Compute reviewed protocol-defined right-censored survival aggregates.

    Invalid design/data returns BLOCKED_METHOD with a stable reason. Unestimable
    Cox inference keeps valid nonparametric descriptions and exact event counts.
    No identifiers, raw records, per-person residuals or model predictions return.
    """
    try:
        if isinstance(spec, dict) and "ph_test" in spec:
            _ph_config(spec["ph_test"])
            _need(spec.get("method") == "cox_ph", "PH_TEST_REQUIRES_COX_PH")
        method, kept, time, status, grid, covariates, missing = _validate(rows, spec)
        if "ph_test" in spec:
            _need(all(cov["name"] != "GLOBAL" for cov in covariates), "PH_RESERVED_TERM_NAME")
        competing_count = len(spec.get("competing_codes", []))
        cause_count = 1 + competing_count
        counts = Counter(status)
        result = {
            "status": "COMPUTED_REVIEW_PENDING", "method": method,
            "n_total": len(rows), "n_used": len(kept), "n_excluded_missing": len(rows) - len(kept), "missing_by_field": missing,
            "target_events": counts[1], "censored": counts[0],
            "competing_events": [{"code": code, "definition": definition, "n": counts[i + 2]} for i, (code, definition) in enumerate(zip(spec.get("competing_codes", []), spec.get("competing_event_definitions", [])))],
            "time_unit": spec["time_unit"], "time_zero_definition": spec["time_zero_definition"],
            "event_definition": spec["event_definition"], "censoring_definition": spec["censoring_definition"],
            "missing_method": "complete_case_for_all_selected_analysis_fields_no_imputation",
            "curves": _curve(time, status, grid, cause_count),
            "curve_method": "Aalen-Johansen cumulative incidence and event-free probability" if competing_count else "Kaplan-Meier survival with Greenwood log-log pointwise 95% intervals",
            "ties": "events_processed_before_censoring_at_same_time_risk_set_includes_both",
            "curve_support": "prespecified_grid_within_observed_followup_no_extrapolation",
            "event_free_interval_note": "Pointwise log-log Greenwood interval for freedom from any declared event; boundary values give descriptive zero-width endpoints, not certainty.",
            "assumptions": ["one independent subject per row", "time zero and eligibility aligned", "right censoring only", "independent censoring conditional on modeled covariates when applicable", "event ascertainment and competing event definitions require clinical review"],
            "unsupported": ["Fine-Gray subdistribution hazards", "left truncation", "time-varying covariates", "recurrent events", "interval censoring", "survey or sampling weights", "patient or site clustered inference", "automatic causal interpretation"],
        }
        if competing_count:
            result["cumulative_incidence_order"] = [{"code": spec["event_code"], "definition": spec["event_definition"]}] + [{"code": code, "definition": definition} for code, definition in zip(spec["competing_codes"], spec["competing_event_definitions"])]
            result["cif_uncertainty"] = _aj_bootstrap(time, status, grid, cause_count, spec)
            result["competing_risk_note"] = "CIF accounts for all declared competing events; 1 minus a target-only KM is not reported as cumulative incidence. Cause-specific Cox removes competing events after their time and is not Fine-Gray."
        else:
            for point in result["curves"]:
                point.pop("cumulative_incidence")
        if method in {"cox_ph", "cause_specific_cox"}:
            if "ph_test" in spec:
                result["cox"], state = _fit_cox(kept, time, status, covariates, spec)
                term_map = state["term_map"] if state is not None else _design_matrix(kept, covariates)[2]
                result["ph_test"] = _formal_ph(state, spec["ph_test"], term_map, result["cox"].get("reason"))
            else:
                result["cox"] = _cox(kept, time, status, covariates, spec)
            if result["cox"]["status"] == "NON_ESTIMABLE":
                result["status"] = "DESCRIPTIVE_ONLY_MODEL_NOT_ESTIMABLE"
        if counts[1] == 0:
            result["target_event_note"] = "No target events observed; do not interpret the curve as proof of no future risk."
        return result
    except SurvivalBlocked as exc:
        return {"status": "BLOCKED_METHOD", "reason": str(exc), "computed": False, "patient_rows_returned": False}

