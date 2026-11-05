"""Independent-row clinical association; aggregate results, no I/O or clinical defaults.

Explicit plans only. Ordinary MLE is never silently replaced after separation.
Synthetic verification establishes software behavior, not research validity.
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.special import expit
from scipy.stats import chi2, f as f_distribution, t as t_distribution, norm
import statsmodels.api as sm


class AssociationBlocked(ValueError):
    """Fixed diagnostic code; do not include input values or patient identifiers."""


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise AssociationBlocked(code)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "") or (
        isinstance(value, (float, np.floating)) and math.isnan(float(value))
    )


def _token(value: Any) -> str:
    _need(isinstance(value, (str, int, float, bool, np.number)), "CATEGORY_SCALAR_REQUIRED")
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        _need(math.isfinite(float(value)), "NONFINITE_VALUE")
        return format(float(value), ".15g")
    return str(value)


def _number(value: Any) -> float:
    _need(not isinstance(value, (bool, np.bool_)), "NUMERIC_VALUE_REQUIRED")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise AssociationBlocked("NUMERIC_VALUE_REQUIRED") from None
    _need(math.isfinite(result), "NONFINITE_VALUE")
    return result


def _declared_category(config: dict) -> tuple[list[str], str]:
    _need(isinstance(config, dict) and set(config) == {"levels", "reference"}, "CATEGORY_CONTRACT_REQUIRED")
    _need(isinstance(config["levels"], list) and len(config["levels"]) >= 2, "CATEGORY_LEVELS_REQUIRED")
    levels = [_token(v) for v in config["levels"]]
    reference = _token(config["reference"])
    _need(len(levels) == len(set(levels)) and reference in levels, "CATEGORY_LEVELS_OR_REFERENCE")
    return levels, reference


def _validate_units(rows: list[dict], spec: dict) -> None:
    _need(isinstance(rows, list) and len(rows) > 0 and all(isinstance(r, dict) for r in rows), "NONEMPTY_TABLE_REQUIRED")
    _need(isinstance(spec.get("row_unit"), str) and bool(spec["row_unit"].strip()), "ROW_UNIT_REQUIRED")
    _need(spec.get("independent") is True, "INDEPENDENT_OBSERVATIONS_REQUIRED")
    key = spec.get("key")
    key = [key] if isinstance(key, str) else key
    _need(isinstance(key, list) and key and all(isinstance(k, str) and k for k in key), "RECORD_KEY_REQUIRED")
    _need(len(key) == len(set(key)), "RECORD_KEY_REQUIRED")
    identities = []
    for row in rows:
        _need(all(k in row and not _missing(row[k]) for k in key), "KEY_MISSING")
        identities.append(tuple(_token(row[k]) for k in key))
    _need(len(identities) == len(set(identities)), "DUPLICATE_RECORD_KEY")
    subject = spec.get("subject_id")
    if subject is not None:
        _need(isinstance(subject, str) and subject, "SUBJECT_FIELD_REQUIRED")
        _need(all(subject in r and not _missing(r[subject]) for r in rows), "SUBJECT_KEY_MISSING")
        _need(len({_token(r[subject]) for r in rows}) == len(rows), "REPEATED_SUBJECT_REQUIRES_OTHER_METHOD")
    _need(not any(k in spec for k in ("weights", "weight", "cluster", "psu", "strata", "repeated")), "COMPLEX_DESIGN_UNSUPPORTED")


def _rcs(x: np.ndarray, config: dict) -> tuple[list[np.ndarray], list[str], dict]:
    _need(isinstance(config, dict) and set(config) == {"method", "knots"}, "SPLINE_CONTRACT_REQUIRED")
    _need(config["method"] == "restricted_cubic", "SPLINE_METHOD_UNSUPPORTED")
    _need(isinstance(config["knots"], list) and 3 <= len(config["knots"]) <= 7, "SPLINE_KNOTS_REQUIRED")
    knots = np.asarray([_number(v) for v in config["knots"]])
    _need(bool(np.all(np.diff(knots) > 0)), "SPLINE_KNOTS_ORDER")
    _need(len(np.unique(x)) >= len(knots), "SPLINE_INSUFFICIENT_DISTINCT_VALUES")
    scale = (knots[-1] - knots[0]) ** 2
    columns = [x]
    with np.errstate(over="ignore", invalid="ignore"):
        for knot in knots[:-2]:
            col = (
                np.maximum(x - knot, 0) ** 3
                - np.maximum(x - knots[-2], 0) ** 3 * (knots[-1] - knot) / (knots[-1] - knots[-2])
                + np.maximum(x - knots[-1], 0) ** 3 * (knots[-2] - knot) / (knots[-1] - knots[-2])
            ) / scale
            columns.append(col)
    _need(all(np.isfinite(v).all() for v in columns), "SPLINE_NONFINITE_DESIGN")
    return columns, ["linear"] + [f"rcs_{i + 1}" for i in range(len(knots) - 2)], {
        "method": "restricted cubic spline; linear tails",
        "knots": knots.tolist(), "nonlinear_parameters": len(knots) - 2,
        "knot_selection": "prespecified; no significance-driven search",
    }


def _separation_check(x: np.ndarray, y: np.ndarray) -> dict:
    # Full-rank affine rescaling preserves separation geometry and avoids unit dependence.
    scaled = x.copy()
    if x.shape[1] > 1:
        scaled[:, 1:] = (x[:, 1:] - x[:, 1:].mean(axis=0)) / x[:, 1:].std(axis=0)
    signed = (2 * y - 1)[:, None] * scaled
    signed /= np.linalg.norm(signed, axis=1, keepdims=True)
    signed = np.unique(signed, axis=0)
    k = signed.shape[1]
    complete = linprog(
        np.r_[np.zeros(k), -1.0],
        A_ub=np.column_stack([-signed, np.ones(len(signed))]), b_ub=np.zeros(len(signed)),
        bounds=[(-1.0, 1.0)] * k + [(0.0, None)], method="highs",
    )
    _need(complete.success, "SEPARATION_CHECK_FAILED")
    if complete.x[-1] > 1e-8:
        raise AssociationBlocked("COMPLETE_SEPARATION")
    quasi = linprog(-signed.mean(axis=0), A_ub=-signed, b_ub=np.zeros(len(signed)),
                    bounds=[(-1.0, 1.0)] * k, method="highs")
    _need(quasi.success, "SEPARATION_CHECK_FAILED")
    _need(-quasi.fun <= 1e-8, "QUASI_COMPLETE_SEPARATION")
    return {"status": "OVERLAP_WITHIN_NUMERICAL_TOLERANCE", "method": "scaled_signed_design_linear_programs", "tolerance": 1e-8}


def _joint_wald(beta: np.ndarray, covariance: np.ndarray, indices: list[int], model: str, residual_df: int) -> dict:
    if not indices:
        return {"status": "NOT_REQUESTED"}
    cov = covariance[np.ix_(indices, indices)]
    _need(np.linalg.matrix_rank(cov) == len(indices) and np.linalg.eigvalsh(cov).min() > 0, "WALD_COVARIANCE_SINGULAR")
    statistic = float(beta[indices] @ np.linalg.solve(cov, beta[indices]))
    _need(math.isfinite(statistic) and statistic >= 0, "WALD_STATISTIC_INVALID")
    q = len(indices)
    if model == "linear":
        return {"method": "joint Wald F (HC3 approximate when selected)", "statistic": statistic / q,
                "df_num": q, "df_denom": residual_df, "p": float(f_distribution.sf(statistic / q, q, residual_df))}
    return {"method": "joint Wald chi-square", "statistic": statistic, "df": q, "p": float(chi2.sf(statistic, q))}


def _fit(rows: list[dict], spec: dict, reporting: dict | None = None) -> dict:
    counts: dict[str, Any] = {"n_total": len(rows)}
    try:
        model = spec.get("model")
        _need(isinstance(model, str) and model in {"linear", "logistic"}, "MODEL_REQUIRED")
        _need(spec.get("missing") == "complete_case", "EXPLICIT_COMPLETE_CASE_REQUIRED")
        outcome, exposure = spec.get("outcome"), spec.get("exposure")
        covariates = spec.get("covariates")
        _need(isinstance(outcome, str) and outcome and isinstance(exposure, str) and exposure, "OUTCOME_EXPOSURE_REQUIRED")
        _need(isinstance(covariates, list) and all(isinstance(v, str) and v for v in covariates), "COVARIATES_LIST_REQUIRED")
        predictors = [exposure] + list(covariates)
        _need(len(set(predictors)) == len(predictors) and outcome not in predictors, "DUPLICATE_OR_OUTCOME_PREDICTOR")
        categories = spec.get("categorical", {})
        _need(isinstance(categories, dict), "CATEGORY_CONTRACT_REQUIRED")
        category_contracts = {field: _declared_category(config) for field, config in categories.items()}
        subgroup = spec.get("subgroup")
        subgroup_field = None
        if subgroup is not None:
            _need(isinstance(subgroup, dict) and set(subgroup) == {"field", "levels", "reference"}, "SUBGROUP_CONTRACT_REQUIRED")
            subgroup_field = subgroup["field"]
            _need(isinstance(subgroup_field, str) and subgroup_field not in {outcome, exposure}, "SUBGROUP_FIELD_INVALID")
            contract = _declared_category({k: subgroup[k] for k in ("levels", "reference")})
            _need(subgroup_field not in category_contracts or category_contracts[subgroup_field] == contract, "SUBGROUP_CATEGORY_CONFLICT")
            category_contracts[subgroup_field] = contract
            if subgroup_field not in predictors:
                predictors.append(subgroup_field)
        # Source fields are never renamed or overwritten. Ambiguous model terms
        # must be resolved in a reviewed mapping before estimates reach plots.
        _need(all(field.strip().casefold() not in {"intercept", "(intercept)"} for field in predictors),
              "RESERVED_INTERCEPT_TERM_NAME")
        used = [outcome] + predictors
        _need(all(all(v in row for v in used) for row in rows), "MODEL_FIELD_MISSING")
        counts["missing_by_field"] = {v: sum(_missing(row[v]) for row in rows) for v in used}
        # Invalid observed values are not reclassified as missing or silently excluded.
        for row in rows:
            for field in predictors:
                if not _missing(row[field]):
                    if field in category_contracts:
                        _need(_token(row[field]) in category_contracts[field][0], "UNDECLARED_CATEGORY")
                    else:
                        _number(row[field])
        if model == "logistic":
            _need("event" in spec and "non_event" in spec, "BINARY_EVENT_CODING_REQUIRED")
            event, non_event = _token(spec["event"]), _token(spec["non_event"])
            _need(event != non_event, "BINARY_EVENT_CODING_REQUIRED")
            _need(all(_missing(r[outcome]) or _token(r[outcome]) in {event, non_event} for r in rows), "UNDECLARED_OUTCOME_CODE")
        else:
            for row in rows:
                if not _missing(row[outcome]):
                    _number(row[outcome])
        complete = [r for r in rows if all(not _missing(r[field]) for field in used)]
        n = len(complete)
        counts.update(n_used=n, n_excluded_missing=len(rows) - n,
                      missing_method="complete case; missingness assumptions remain unverified")
        _need(n >= 5, "INSUFFICIENT_COMPLETE_OBSERVATIONS")
        y = np.asarray([float(_token(r[outcome]) == event) if model == "logistic" else _number(r[outcome]) for r in complete])
        if reporting is not None:
            _validate_reporting_support(complete, spec, reporting)
        if reporting is not None and subgroup_field:
            group_counts = []
            for level in category_contracts[subgroup_field][0]:
                mask = np.asarray([_token(r[subgroup_field]) == level for r in complete])
                group_counts.append({"level": level, "n_used": int(mask.sum()),
                    "events": int(y[mask].sum()) if model == "logistic" else None,
                    "non_events": int(mask.sum() - y[mask].sum()) if model == "logistic" else None})
            counts["subgroup_counts"] = group_counts
            _need(all(g["n_used"] >= reporting["subgroup_min_n"] for g in group_counts), "SUBGROUP_TOO_SMALL")
            if model == "logistic":
                _need(all(g["events"] > 0 for g in group_counts), "SUBGROUP_NO_EVENTS")
                _need(all(g["non_events"] > 0 for g in group_counts), "SUBGROUP_NO_NONEVENTS")
        columns, names, ranges = [np.ones(n)], ["intercept"], {}
        exposure_indices, nonlinear_indices, subgroup_indices = [], [], []
        nonlinear = None
        for field in predictors:
            start = len(columns)
            if field in category_contracts:
                levels, reference = category_contracts[field]
                tokens = [_token(r[field]) for r in complete]
                _need(all(level in tokens for level in levels), "CATEGORY_LEVEL_ABSENT_FROM_ANALYSIS")
                for level in levels:
                    if level != reference:
                        columns.append(np.asarray([float(v == level) for v in tokens]))
                        names.append(f"{field}[{level} vs {reference}]")
            else:
                x = np.asarray([_number(r[field]) for r in complete])
                _need(float(np.ptp(x)) > 0, "CONSTANT_PREDICTOR")
                ranges[field] = {"min": float(x.min()), "max": float(x.max())}
                if field == exposure and spec.get("nonlinear") is not None:
                    new_columns, suffixes, nonlinear = _rcs(x, spec["nonlinear"])
                    columns.extend(new_columns)
                    names.extend([field if s == "linear" else f"{field}:{s}" for s in suffixes])
                    nonlinear_indices = list(range(start + 1, len(columns)))
                else:
                    columns.append(x)
                    names.append(field)
            if field == exposure:
                exposure_indices = list(range(start, len(columns)))
            if field == subgroup_field:
                subgroup_indices = list(range(start, len(columns)))
        _need(not (exposure in category_contracts and spec.get("nonlinear") is not None), "SPLINE_REQUIRES_NUMERIC_EXPOSURE")
        interaction_indices, nonlinear_interaction_indices = [], []
        for e in exposure_indices:
            for g in subgroup_indices:
                interaction_indices.append(len(columns))
                if e in nonlinear_indices:
                    nonlinear_interaction_indices.append(len(columns))
                columns.append(columns[e] * columns[g])
                names.append(names[e] + " * " + names[g])
        normalized_names = [name.strip().casefold() for name in names]
        _need(len(normalized_names) == len(set(normalized_names)), "DERIVED_TERM_NAME_COLLISION")
        x = np.column_stack(columns)
        p = x.shape[1]
        counts["predictor_parameters"] = p - 1
        if model == "logistic":
            counts.update(events=int(y.sum()), non_events=int(n - y.sum()),
                          effective_events_per_parameter=float(min(y.sum(), n - y.sum()) / (p - 1)))
        _need(n > p + 2, "INSUFFICIENT_RESIDUAL_DEGREES")
        scaled_design = x / np.linalg.norm(x, axis=0)
        _need(np.isfinite(x).all() and np.linalg.matrix_rank(scaled_design) == p, "RANK_DEFICIENT_DESIGN")
        _need(np.linalg.cond(scaled_design) < 1e8, "ILL_CONDITIONED_DESIGN")
        covariance_type = spec.get("covariance")
        separation = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model == "linear":
                _need(isinstance(covariance_type, str) and covariance_type in {"classical", "HC3"}, "LINEAR_COVARIANCE_REQUIRED")
                _need(float(np.ptp(y)) > 0, "CONSTANT_OUTCOME")
                fit = sm.OLS(y, x).fit(cov_type="nonrobust" if covariance_type == "classical" else "HC3", use_t=True)
                _need(float(fit.ssr) > 64 * np.finfo(float).eps * float(np.sum((y - y.mean()) ** 2)), "ZERO_RESIDUAL_VARIANCE")
                diagnostics = {"r_squared": float(fit.rsquared), "residual_sd": float(math.sqrt(fit.ssr / (n - p)))}
            else:
                _need(isinstance(covariance_type, str) and covariance_type in {"model", "HC0"}, "LOGISTIC_COVARIANCE_REQUIRED")
                _need(len(np.unique(y)) == 2, "SINGLE_OUTCOME_CLASS")
                separation = _separation_check(x, y)
                _need(counts["effective_events_per_parameter"] >= 10, "SPARSE_EFFECTIVE_EVENTS_PER_PARAMETER")
                fit = sm.Logit(y, x, check_rank=False).fit(disp=False, maxiter=200, tol=1e-10,
                                                        cov_type="nonrobust" if covariance_type == "model" else "HC0")
                _need(bool(fit.mle_retvals.get("converged")), "LOGISTIC_NONCONVERGENCE")
                probabilities = expit(x @ np.asarray(fit.params))
                information = x.T @ ((probabilities * (1 - probabilities))[:, None] * x)
                _need(np.isfinite(information).all() and np.linalg.cond(information) < 1e12, "UNSTABLE_LOGISTIC_INFORMATION")
                _need(float(np.max(np.abs(x.T @ (probabilities - y)))) < 1e-4, "LOGISTIC_GRADIENT_NOT_CONVERGED")
                diagnostics = {"max_abs_gradient": float(np.max(np.abs(x.T @ (probabilities - y)))), "separation_check": separation}
        _need(not caught, "ESTIMATOR_WARNING_REQUIRES_REVIEW")
        beta, se, ci, pv = (np.asarray(fit.params), np.asarray(fit.bse), np.asarray(fit.conf_int()), np.asarray(fit.pvalues))
        covariance = np.asarray(fit.cov_params())
        _need(all(np.isfinite(z).all() for z in (beta, se, ci, pv, covariance)) and bool(np.all(se > 0)), "INVALID_INFERENCE")
        coefficients = []
        for i, name in enumerate(names):
            record = {"term": name, "estimate": float(beta[i]), "standard_error": float(se[i]),
                      "ci95": ci[i].tolist(), "p": float(pv[i]),
                      "scale": "log odds" if model == "logistic" else "outcome units"}
            if model == "logistic":
                _need(ci[i, 1] < 700 and ci[i, 0] > -700, "ODDS_RATIO_INTERVAL_OVERFLOW")
                record.update(odds_ratio=float(np.exp(beta[i])), or_ci95=np.exp(ci[i]).tolist())
            coefficients.append(record)
        result = {"status": "COMPUTED_REVIEW_PENDING", "model": model, **counts,
                  "method": "ordinary binomial logit MLE" if model == "logistic" else "ordinary least squares",
                  "covariance": covariance_type, "coefficients": coefficients, "diagnostics": diagnostics,
                  "nonlinear": nonlinear, "nonlinearity_test": _joint_wald(beta, covariance, nonlinear_indices + nonlinear_interaction_indices, model, n - p),
                  "interaction_test": _joint_wald(beta, covariance, interaction_indices, model, n - p),
                  "exposure_test": _joint_wald(beta, covariance, exposure_indices + interaction_indices, model, n - p),
                  "numeric_analysis_ranges": ranges,
                  "limitations": ["Conditional association; no causal or clinical-validity claim.",
                                  "Independent observations, prespecified model and complete-case assumptions require research review.",
                                  "No automatic variable selection, repeated-measure, complex-sample or missing-data correction."]}
        if subgroup_field:
            result["subgroup"] = {"field": subgroup_field, "n_used_by_level": {
                level: sum(_token(r[subgroup_field]) == level for r in complete) for level in category_contracts[subgroup_field][0]},
                "interpretation": "Use the joint interaction test; different within-group P values do not establish heterogeneity."}
        if nonlinear_indices:
            result["nonlinearity_test"]["scope"] = "all main nonlinear exposure terms and nonlinear exposure-by-subgroup interactions"
        if model == "logistic":
            result["event_coding"] = {"event": spec["event"], "non_event": spec["non_event"]}
        if reporting is not None:
            result["reporting"] = _report_contrasts(complete, spec, reporting, beta, covariance,
                exposure_indices, subgroup_indices, interaction_indices, n - p, counts.get("subgroup_counts", []))
        return result
    except AssociationBlocked as exc:
        return {"status": "BLOCKED_ESTIMATION", "reason": str(exc), **counts, "estimates": None,
                "method_not_silently_replaced": True}
    except (np.linalg.LinAlgError, ArithmeticError, RuntimeError, ValueError):
        return {"status": "BLOCKED_ESTIMATION", "reason": "NUMERICAL_ESTIMATION_FAILED", **counts,
                "estimates": None, "method_not_silently_replaced": True}


def _subset(rows: list[dict], conditions: list[dict]) -> list[dict]:
    _need(isinstance(conditions, list) and conditions, "SENSITIVITY_FILTER_REQUIRED")
    selected = list(rows)
    for condition in conditions:
        _need(isinstance(condition, dict) and set(condition) == {"field", "op", "value"}, "SENSITIVITY_FILTER_SCHEMA")
        field, op, value = condition["field"], condition["op"], condition["value"]
        _need(isinstance(field, str) and all(field in row for row in rows), "SENSITIVITY_FIELD_MISSING")
        _need(isinstance(op, str) and op in {"eq", "ne", "lt", "le", "gt", "ge", "in"}, "SENSITIVITY_OPERATOR_UNSUPPORTED")
        if op in {"lt", "le", "gt", "ge"}:
            threshold = _number(value)
        elif op == "in":
            _need(isinstance(value, list) and bool(value), "SENSITIVITY_VALUES_REQUIRED")
            allowed = {_token(v) for v in value}
        def keep(row: dict) -> bool:
            if _missing(row[field]):
                return False
            if op == "eq": return _token(row[field]) == _token(value)
            if op == "ne": return _token(row[field]) != _token(value)
            if op == "in": return _token(row[field]) in allowed
            observed = _number(row[field])
            return {"lt": observed < threshold, "le": observed <= threshold,
                    "gt": observed > threshold, "ge": observed >= threshold}[op]
        selected = [row for row in selected if keep(row)]
    return selected


def analyze(rows: list[dict], spec: dict) -> dict:
    """Run a prespecified association and optional sensitivities; never mutate inputs."""
    try:
        _need(isinstance(spec, dict), "SPEC_OBJECT_REQUIRED")
        _validate_units(rows, spec)
        sensitivities = spec.get("sensitivity", [])
        _need(isinstance(sensitivities, list), "SENSITIVITY_LIST_REQUIRED")
        labels = []
        for item in sensitivities:
            _need(isinstance(item, dict) and set(item) <= {"label", "rationale", "where", "covariates", "nonlinear", "covariance"}, "SENSITIVITY_SCHEMA")
            _need(isinstance(item.get("label"), str) and item["label"].strip() and isinstance(item.get("rationale"), str) and item["rationale"].strip(), "SENSITIVITY_PROVENANCE_REQUIRED")
            labels.append(item["label"])
        _need(len(labels) == len(set(labels)), "SENSITIVITY_LABEL_DUPLICATE")
    except (AssociationBlocked, TypeError) as exc:
        return {"status": "BLOCKED_INPUT", "reason": str(exc) if isinstance(exc, AssociationBlocked) else "INVALID_SPEC_TYPE", "estimates": None}
    result = _fit(rows, spec)
    result["family"] = "association"
    result["row_unit"] = spec["row_unit"]
    result["patient_rows_returned"] = False
    result["sensitivity"] = []
    for item in sensitivities:
        try:
            subset = _subset(rows, item.get("where"))
            alternate = {**spec, **{k: item[k] for k in ("covariates", "nonlinear", "covariance") if k in item}}
            record = _fit(subset, alternate)
            record.update(label=item["label"], rationale=item["rationale"], n_before_filter=len(rows), n_after_filter=len(subset))
        except AssociationBlocked as exc:
            record = {"status": "BLOCKED_INPUT", "reason": str(exc), "label": item["label"], "estimates": None}
        result["sensitivity"].append(record)
    return result


def _validate_reporting_support(complete: list[dict], spec: dict, reporting: dict) -> None:
    """Prespecified contrast support, checked on the actual model population."""
    curve = reporting["curve"]
    if curve is None:
        return
    observed = np.asarray([_number(r[spec["exposure"]]) for r in complete])
    low, high = float(observed.min()), float(observed.max())
    knots = [_number(v) for v in spec["nonlinear"]["knots"]]
    _need(low <= knots[0] and knots[-1] <= high, "SPLINE_KNOTS_OUTSIDE_ANALYSIS_SUPPORT")
    lo, hi = curve["domain"]
    _need(low <= lo < hi <= high, "CURVE_DOMAIN_OUTSIDE_ANALYSIS_SUPPORT")
    _need(lo <= curve["reference"] <= hi, "CURVE_REFERENCE_OUTSIDE_DOMAIN")
    _need(all(lo <= v <= hi for v in curve["grid"]), "CURVE_GRID_OUTSIDE_DOMAIN")


def _contrast(beta: np.ndarray, covariance: np.ndarray, vector: np.ndarray,
              model: str, residual_df: int) -> dict:
    """Linear contrast using the fitted provider covariance, without a refit."""
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    _need(math.isfinite(variance) and variance >= -1e-12 * max(1.0, float(np.max(np.abs(covariance)))), "CONTRAST_VARIANCE_INVALID")
    standard_error = math.sqrt(max(0.0, variance))
    critical = float(t_distribution.ppf(.975, residual_df)) if model == "linear" else float(norm.ppf(.975))
    interval = [estimate - critical * standard_error, estimate + critical * standard_error]
    if standard_error == 0:
        _need(estimate == 0, "CONTRAST_ZERO_VARIANCE_NONZERO_ESTIMATE")
        p = 1.0
    else:
        statistic = abs(estimate / standard_error)
        p = float(2 * (t_distribution.sf(statistic, residual_df) if model == "linear" else norm.sf(statistic)))
    if model == "logistic":
        _need(interval[0] > -700 and interval[1] < 700, "ODDS_RATIO_INTERVAL_OVERFLOW")
    return {"estimate": math.exp(estimate) if model == "logistic" else estimate,
            "linear_contrast": estimate, "standard_error": standard_error,
            "ci95": [math.exp(v) for v in interval] if model == "logistic" else interval,
            "p": p, "effect_scale": "odds_ratio" if model == "logistic" else "mean_difference",
            "standard_error_scale": "log_odds" if model == "logistic" else "outcome_units"}


def _report_contrasts(complete: list[dict], spec: dict, reporting: dict,
                      beta: np.ndarray, covariance: np.ndarray, exposure_indices: list[int],
                      subgroup_indices: list[int], interaction_indices: list[int],
                      residual_df: int, group_counts: list[dict]) -> dict:
    model = spec["model"]
    output = {"curve": None, "subgroup_effects": [], "parameter_covariance": covariance.tolist(),
              "residual_df": residual_df,
              "ci_level": .95, "ci_method": "Wald t" if model == "linear" else "Wald normal on log odds, exponentiated"}
    curve = reporting["curve"]
    if curve is not None:
        observed = np.asarray([_number(r[spec["exposure"]]) for r in complete])
        # Reuse the exact original RCS basis. Append model observations so its
        # distinct-support check applies to the fitted data, not display grid size.
        evaluated = np.r_[observed, curve["grid"], curve["reference"]]
        basis, _, _ = _rcs(evaluated, spec["nonlinear"])
        points = []
        for index, value in enumerate(curve["grid"]):
            contrast = np.zeros(len(beta))
            for parameter, values in zip(exposure_indices, basis):
                contrast[parameter] = values[len(observed) + index] - values[-1]
            point = _contrast(beta, covariance, contrast, model, residual_df)
            point.pop("p")
            points.append({"x": value, **point})
        output["curve"] = {"reference": curve["reference"], "grid": list(curve["grid"]),
            "domain": list(curve["domain"]), "observed_domain": [float(observed.min()), float(observed.max())],
            "knots": list(spec["nonlinear"]["knots"]), "basis": "original restricted cubic spline; linear tails",
            "knot_selection": "prespecified; no significance-driven search", "ci_level": .95,
            "ci_method": output["ci_method"], "effect_scale": "odds_ratio" if model == "logistic" else "mean_difference",
            "standard_error_scale": "log_odds" if model == "logistic" else "outcome_units", "points": points,
            "interpretation": "Conditional exposure contrast against the fixed reference; covariates cancel; no causal claim."}
    subgroup = spec.get("subgroup")
    if subgroup is not None:
        levels, reference = _declared_category({k: subgroup[k] for k in ("levels", "reference")})
        other_levels = [v for v in levels if v != reference]
        for group in group_counts:
            contrast = np.zeros(len(beta))
            contrast[exposure_indices[0]] = 1.0
            if group["level"] != reference:
                contrast[interaction_indices[other_levels.index(group["level"])]] = 1.0
            output["subgroup_effects"].append({**group, "reference": reference,
                **_contrast(beta, covariance, contrast, model, residual_df),
                "interpretation": "Per-unit exposure slope contrast from the pooled interaction model; not a separately fitted subgroup regression."})
    return output


def _plan_used(spec: dict) -> list[str]:
    fields = [spec["outcome"], spec["exposure"]] + list(spec["covariates"])
    if spec.get("subgroup") is not None:
        fields.append(spec["subgroup"]["field"])
    return list(dict.fromkeys(fields))


def _validate_plan(rows: list[dict], plan: dict) -> list[str]:
    _need(isinstance(plan, dict) and set(plan) == {"sample_strategy", "multiplicity", "models"}, "ASSOCIATION_PLAN_SCHEMA")
    _need(plan["sample_strategy"] in {"per_model_complete_case", "common_complete_case"}, "ASSOCIATION_SAMPLE_STRATEGY_UNSUPPORTED")
    _need(plan["multiplicity"] == "unadjusted_prespecified", "ASSOCIATION_MULTIPLICITY_UNSUPPORTED")
    _need(isinstance(plan["models"], list) and 1 <= len(plan["models"]) <= 6, "ASSOCIATION_MODELS_REQUIRED")
    expected = {"model_id", "label", "rationale", "role", "where", "spec", "curve", "subgroup_min_n"}
    allowed_spec = {"model", "missing", "outcome", "exposure", "covariates", "categorical", "row_unit", "independent", "key", "subject_id", "covariance", "event", "non_event", "subgroup", "nonlinear", "sensitivity"}
    used = []
    identities = []
    primary = []
    first_spec = None
    for item in plan["models"]:
        _need(isinstance(item, dict) and set(item) == expected, "ASSOCIATION_MODEL_SCHEMA")
        _need(all(isinstance(item[k], str) and bool(item[k].strip()) for k in ("model_id", "label", "rationale")), "ASSOCIATION_MODEL_PROVENANCE_REQUIRED")
        identities.append(item["model_id"])
        _need(item["role"] in {"primary", "adjustment", "sensitivity"}, "ASSOCIATION_MODEL_ROLE_UNSUPPORTED")
        primary.append(item["role"] == "primary")
        spec = item["spec"]
        _need(isinstance(spec, dict) and set(spec) <= allowed_spec, "ASSOCIATION_SPEC_SCHEMA")
        _need(spec.get("model") in {"linear", "logistic"}, "MODEL_REQUIRED")
        _need(spec.get("missing") == "complete_case", "EXPLICIT_COMPLETE_CASE_REQUIRED")
        _need(spec.get("sensitivity", []) == [], "NESTED_SENSITIVITY_NOT_SUPPORTED")
        _need(isinstance(spec.get("outcome"), str) and isinstance(spec.get("exposure"), str) and isinstance(spec.get("covariates"), list), "OUTCOME_EXPOSURE_REQUIRED")
        _validate_units(rows, spec)
        if first_spec is None:
            first_spec = spec
        for field in ("model", "outcome", "exposure", "row_unit", "independent", "key", "subject_id", "event", "non_event"):
            _need(spec.get(field) == first_spec.get(field), "MODEL_SEQUENCE_TARGET_MISMATCH")
        _need(spec.get("covariance") == ("HC3" if spec["model"] == "linear" else "model"), "PLAN_COVARIANCE_NOT_SUPPORTED")
        _need(isinstance(item["where"], list), "SENSITIVITY_FILTER_SCHEMA")
        _need(not item["where"] or item["role"] == "sensitivity", "FILTER_REQUIRES_SENSITIVITY_ROLE")
        if item["where"]:
            _subset(rows, item["where"])
        subgroup = spec.get("subgroup")
        categories = spec.get("categorical", {})
        _need(isinstance(categories, dict), "CATEGORY_CONTRACT_REQUIRED")
        categories = dict(categories)
        if subgroup is not None:
            _need(isinstance(subgroup, dict) and set(subgroup) == {"field", "levels", "reference"}, "SUBGROUP_CONTRACT_REQUIRED")
            _need(spec.get("nonlinear") is None, "RCS_SUBGROUP_COMBINATION_NOT_SUPPORTED_V03")
            _need(spec["exposure"] not in categories, "SUBGROUP_NUMERIC_EXPOSURE_REQUIRED")
            _need(isinstance(item["subgroup_min_n"], int) and not isinstance(item["subgroup_min_n"], bool) and item["subgroup_min_n"] >= 5, "SUBGROUP_MINIMUM_REQUIRED")
            categories[subgroup["field"]] = {k: subgroup[k] for k in ("levels", "reference")}
        else:
            _need(item["subgroup_min_n"] is None, "SUBGROUP_MINIMUM_NOT_APPLICABLE")
        curve = item["curve"]
        if spec.get("nonlinear") is not None:
            _need(spec["exposure"] not in categories, "SPLINE_REQUIRES_NUMERIC_EXPOSURE")
            _need(isinstance(curve, dict) and set(curve) == {"reference", "grid", "domain"}, "RCS_CURVE_REQUIRED")
            _need(isinstance(curve["grid"], list) and 3 <= len(curve["grid"]) <= 101, "CURVE_GRID_INVALID")
            grid = [_number(v) for v in curve["grid"]]
            _need(all(a < b for a, b in zip(grid, grid[1:])), "CURVE_GRID_INVALID")
            _need(isinstance(curve["domain"], list) and len(curve["domain"]) == 2, "CURVE_DOMAIN_INVALID")
            lo, hi = [_number(v) for v in curve["domain"]]
            reference = _number(curve["reference"])
            _need(lo < hi, "CURVE_DOMAIN_INVALID")
            _need(lo <= reference <= hi, "CURVE_REFERENCE_OUTSIDE_DOMAIN")
            _need(all(lo <= v <= hi for v in grid), "CURVE_GRID_OUTSIDE_DOMAIN")
            nonlinear = spec["nonlinear"]
            _need(isinstance(nonlinear, dict) and set(nonlinear) == {"method", "knots"}, "SPLINE_CONTRACT_REQUIRED")
            # Original basis validates fixed knots before any model fitting.
            _rcs(np.linspace(lo, hi, 20), nonlinear)
        else:
            _need(curve is None, "CURVE_REQUIRES_RCS")
        fields = _plan_used(spec)
        _need(all(all(field in row for field in fields) for row in rows), "MODEL_FIELD_MISSING")
        declarations = {field: _declared_category(config) for field, config in categories.items()}
        for row in rows:
            for field in fields:
                value = row[field]
                if _missing(value):
                    continue
                if field == spec["outcome"] and spec["model"] == "logistic":
                    _need("event" in spec and "non_event" in spec and _token(spec["event"]) != _token(spec["non_event"]), "BINARY_EVENT_CODING_REQUIRED")
                    _need(_token(value) in {_token(spec["event"]), _token(spec["non_event"])}, "UNDECLARED_OUTCOME_CODE")
                elif field in declarations:
                    _need(_token(value) in declarations[field][0], "UNDECLARED_CATEGORY")
                else:
                    _number(value)
        used.extend(fields)
    _need(len(identities) == len(set(identities)), "ASSOCIATION_MODEL_ID_DUPLICATE")
    _need(primary[0] and sum(primary) == 1, "ONE_FIRST_PRIMARY_MODEL_REQUIRED")
    return list(dict.fromkeys(used))


def analyze_plan(rows: list[dict], plan: dict) -> dict:
    """Prespecified model sequence using only this module's original estimator.

    This numerical API does not own task execution, authorization or exports.
    Common complete cases are selected on the union of model fields before each
    sensitivity filter; filter-field missingness is a filter exclusion.
    """
    try:
        used = _validate_plan(rows, plan)
        common = [r for r in rows if all(not _missing(r[f]) for f in used)] if plan["sample_strategy"] == "common_complete_case" else None
        eligible = rows if common is None else common
        records = []
        for item in plan["models"]:
            subset = _subset(eligible, item["where"]) if item["where"] else list(eligible)
            filter_fields = {c["field"] for c in item["where"]}
            filter_missing = sum(any(_missing(r[f]) for f in filter_fields) for r in eligible)
            fit = _fit(subset, item["spec"], {"curve": item["curve"], "subgroup_min_n": item["subgroup_min_n"]})
            records.append({"model_id": item["model_id"], "label": item["label"], "rationale": item["rationale"],
                "role": item["role"], "where": item["where"], "flow": {"source_rows": len(rows),
                    "common_complete_n": None if common is None else len(common),
                    "excluded_common_missing": 0 if common is None else len(rows) - len(common),
                    "n_before_filter": len(eligible), "n_after_filter": len(subset),
                    "excluded_filter": len(eligible) - len(subset), "filter_missing_excluded": filter_missing,
                    "n_used": fit.get("n_used"), "n_excluded_missing": fit.get("n_excluded_missing"),
                    "sample_strategy": plan["sample_strategy"]}, "record": fit})
        computed = sum(r["record"]["status"] == "COMPUTED_REVIEW_PENDING" for r in records)
        return {"status": "COMPUTED_REVIEW_PENDING" if computed == len(records) else "PARTIAL_REVIEW_PENDING" if computed else "BLOCKED_ESTIMATION",
            "reason": None if computed == len(records) else "PRESPECIFIED_MODELS_PARTIALLY_BLOCKED" if computed else "ALL_PRESPECIFIED_MODELS_BLOCKED",
            "family": "association_plan", "sample_strategy": plan["sample_strategy"], "multiplicity": plan["multiplicity"],
            "source_rows": len(rows), "common_complete_n": None if common is None else len(common), "models": records,
            "patient_rows_returned": False, "limitations": ["Prespecified conditional associations; no causal or clinical-validity claim.",
                "All confidence intervals and P values are unadjusted for multiplicity; no significance-driven model selection.",
                "Common complete-case eligibility uses the union of model fields before model-specific sensitivity filters.",
                "Subgroup effects are pooled interaction-model contrasts; group P values do not replace the interaction test."]}
    except (AssociationBlocked, TypeError, KeyError) as exc:
        return {"status": "BLOCKED_INPUT", "reason": str(exc) if isinstance(exc, AssociationBlocked) else "ASSOCIATION_PLAN_SCHEMA",
            "family": "association_plan", "source_rows": len(rows) if isinstance(rows, list) else None,
            "models": [], "patient_rows_returned": False, "method_not_silently_replaced": True}
