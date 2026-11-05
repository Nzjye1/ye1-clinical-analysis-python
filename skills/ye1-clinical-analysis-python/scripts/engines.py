"""Bridge explicit, local analysis requests to fixed Python numerical providers."""
from __future__ import annotations

import copy
import math

import baseline
from local_data import DataError, rows_for_compute


def require(value, reason):
    if not value:
        raise DataError(reason)


def _observed_grid(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        try:
            n = float(value)
        except (ValueError, TypeError, OverflowError):
            raise DataError("INVALID_FOLLOWUP_TIME") from None
        require(math.isfinite(n) and n > 0, "FOLLOWUP_MUST_BE_POSITIVE")
        values.append(n)
    require(bool(values), "NO_VALID_FOLLOWUP_TIME")
    grid = [0.] + sorted(set(values))
    require(len(grid) <= 5001, "TOO_MANY_OBSERVED_TIMES_USE_PRESPECIFIED_GRID")
    return grid


def analyze(frame, task):
    require(isinstance(task, dict), "TASK_OBJECT_REQUIRED")
    method = task.get("method")
    unit = task.get("unit", {})
    require(isinstance(unit, dict), "UNIT_DESCRIPTION_REQUIRED")
    independent = unit.get("independent") is True
    require(independent, "CONFIRM_INDEPENDENT_SUBJECTS" if method in {"longitudinal", "gee"}
            else "CONFIRM_ONE_INDEPENDENT_UNIT_PER_ROW")
    if method in {"longitudinal", "gee"}:
        from longitudinal import analyze as longitudinal_analyze
        return longitudinal_analyze(frame, unit.get("id"), task.get("spec", {}), method), None
    rows, key = rows_for_compute(frame, unit.get("id"), independent)
    spec = copy.deepcopy(task.get("spec", {}))
    require(isinstance(spec, dict), "METHOD_SPEC_REQUIRED")
    if method == "baseline":
        spec["independent"] = True
        return baseline.analyze(frame, spec), None
    if method == "paired":
        spec["independent_pairs"] = True
        return baseline.paired(frame, spec), None
    if method in {"correlation", "normality", "repeated", "diagnostic", "diagnostic_roc"}:
        import basic_extra
        return getattr(basic_extra, method)(frame, spec), None
    if method == "mediation":
        from mediation import analyze as mediation_analyze
        return mediation_analyze(frame, spec), None
    if method == "association":
        from clinical_methods import association
        require(isinstance(unit.get("description"), str) and unit["description"].strip(), "STUDY_UNIT_DESCRIPTION_REQUIRED")
        spec.update(key=key, subject_id=key, row_unit=unit["description"], independent=True)
        result = association.analyze(rows, spec)
        return result, None
    if method == "survival":
        return _survival(rows, key, spec), None
    if method == "prediction":
        from clinical_methods import prediction
        spec["id"] = key
        spec["independent_subjects"] = True
        try:
            result = prediction.analyze(rows, spec)
        except prediction.PredictionBlocked as exc:
            return {"status": "BLOCKED_INPUT", "reason": str(exc), "patient_rows_returned": False}, None
        private = result.pop("model_bundle", None)
        if private:
            result["private_model"] = "private/model.json; do not send to model or external services"
        return result, private
    if method == "prediction_forest":
        from forest_prediction import analyze as forest_analyze
        spec["id"] = key
        spec["independent_subjects"] = True
        return forest_analyze(rows, spec)
    if method == "prediction_hgb":
        from forest_prediction import analyze_hgb
        spec["id"] = key
        spec["independent_subjects"] = True
        return analyze_hgb(rows, spec)
    if method == "continuous_prediction":
        from continuous_prediction import analyze as continuous_analyze
        return continuous_analyze(rows, key, spec)
    if method == "clustering":
        from clustering import analyze as clustering_analyze
        return clustering_analyze(rows, key, spec)
    if method == "matching":
        from matching import analyze_matching
        return analyze_matching(rows, key, spec)
    raise DataError("UNSUPPORTED_METHOD")


def _survival(rows, key, spec):
    from clinical_methods import survival
    spec["subject_id"] = key
    spec["independent_subjects"] = True
    method = spec.get("method")
    require(method in {"kaplan_meier", "cox_ph", "aalen_johansen", "cause_specific_cox"}, "SURVIVAL_METHOD_UNSUPPORTED")
    group = spec.pop("group", None)
    want_logrank = spec.pop("logrank", False)
    grid_rule = spec.pop("evaluation_grid", None)
    if group is None:
        require(not want_logrank, "LOGRANK_REQUIRES_TWO_GROUPS")
        if grid_rule == "observed":
            spec["evaluation_times"] = _observed_grid(rows, spec.get("time"))
        return survival.analyze(rows, spec)
    require(method in {"kaplan_meier", "aalen_johansen"} and isinstance(group, dict), "GROUP_CURVES_REQUIRE_KM_OR_AJ")
    require(not want_logrank or method == "kaplan_meier", "LOGRANK_NOT_VALID_FOR_COMPETING_RISK_CIF")
    field, levels = group.get("column"), group.get("levels")
    require(isinstance(field, str) and all(field in r for r in rows), "GROUP_COLUMN_REQUIRED")
    require(isinstance(levels, list) and len(levels) >= 2 and all(isinstance(v, str) and v for v in levels)
            and len(set(levels)) == len(levels), "GROUP_LEVELS_REQUIRED")
    actual = [str(r[field]) for r in rows if r[field] is not None]
    require(all(v in levels for v in actual), "UNDECLARED_GROUP_VALUE")
    groups = []
    for level in levels:
        subset = [r for r in rows if r[field] == level]
        require(bool(subset), "EMPTY_DECLARED_GROUP")
        part_spec = copy.deepcopy(spec)
        if grid_rule == "observed":
            part_spec["evaluation_times"] = _observed_grid(subset, spec.get("time"))
        result = survival.analyze(subset, part_spec)
        if result.get("status") != "COMPUTED_REVIEW_PENDING":
            return {"status": "BLOCKED_METHOD", "reason": result.get("reason", "GROUP_CURVE_BLOCKED"),
                    "patient_rows_returned": False}
        groups.append({"group": level, "result": result})
    compare = None
    if want_logrank:
        require(len(levels) == 2, "LOGRANK_REQUIRES_TWO_GROUPS")
        from lifelines.statistics import logrank_test
        def vectors(level):
            vals = [r for r in rows if r[field] == level and r.get(spec["time"]) is not None and r.get(spec["status"]) is not None]
            return ([float(r[spec["time"]]) for r in vals],
                    [int(r[spec["status"]] == spec["event_code"]) for r in vals])
        t1, e1 = vectors(levels[0]); t2, e2 = vectors(levels[1])
        comparison = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
        require(math.isfinite(float(comparison.p_value)) and math.isfinite(float(comparison.test_statistic)),
                "LOGRANK_NOT_ESTIMABLE")
        compare = {"method": "two_sided_logrank", "statistic": float(comparison.test_statistic),
                   "p": float(comparison.p_value)}
    return {"status": "COMPUTED_REVIEW_PENDING", "method": method+"_by_group",
            "n_total": len(rows), "missing_group": len(rows)-len(actual), "groups": groups,
            "comparison": compare, "patient_rows_returned": False}
