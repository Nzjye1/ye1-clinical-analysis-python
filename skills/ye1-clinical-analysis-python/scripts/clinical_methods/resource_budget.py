"""Finite project-request budgets; no fitting, sampling or host memory claims.

Legacy fixture caps are separate. These profiles bound transport and estimated
design work before invoking the unchanged numerical owner. They are not speed
guarantees or statistical sample-size rules.
"""
from __future__ import annotations
import json
from pathlib import Path

_CONFIG = json.loads((Path(__file__).resolve().parents[2] / "config/resource-budget-profiles.json").read_text(encoding="utf-8"))

MIB = 1024 * 1024
ABSOLUTE_MAX_ROWS = 100_000
ABSOLUTE_MAX_INPUT_BYTES = 256 * MIB
ABSOLUTE_MAX_DESIGN_COLUMNS = 1024
ABSOLUTE_MAX_WORK_UNITS = 10_000_000_000
BASE_LIMIT_KEYS = frozenset({"max_rows", "max_input_bytes", "max_design_columns", "max_work_units"})
LIMIT_KEYS = BASE_LIMIT_KEYS | {"max_candidate_pairs", "max_background_rows"}



class BudgetError(ValueError):
    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code, self.detail = code, detail or {}


def _integer(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise BudgetError("RESOURCE_BUDGET_INVALID", {"field": name, "allowed_maximum": maximum})
    return value


def profile(method_id):
    name = "tabular"
    if "psm" in method_id or "matching" in method_id: name = "pair_matching"
    elif "formal_ph" in method_id: name = "formal_ph"
    elif "tree_shap" in method_id: name = "tree_explanation"
    elif method_id in {"prediction.binary.random_forest", "prediction.binary.hist_gradient_boosting"}: name = "tree_training"
    elif "ipcw" in method_id: name = "survival_evaluation"
    elif "interpretation" in method_id: name = "linear_explanation"
    elif "prediction.survival.cox" in method_id: name = "survival_prediction"
    elif any(x in method_id for x in ("trajectory", "growth", "profile")): name = "latent_longitudinal"
    return {"profile": name, **_CONFIG["profiles"][name]}


def defaults(method_id):
    p = profile(method_id); d = _CONFIG["defaults"]
    return {"max_rows": p["default_rows"], "max_bytes": d["max_bytes"], "max_decoded_bytes": d["max_decoded_bytes"],
            "resource_budget": {"max_design_columns": d["max_design_columns"], "max_work_units": d["max_work_units"],
                                "max_candidate_pairs": d["max_candidate_pairs"],
                                "max_background_rows": p.get("default_background_rows", d["max_background_rows"])}}


def method(task):
    analysis = task.get("analysis", {})
    result = task.get("method_id", analysis.get("method_id", "baseline.table1"))
    if isinstance(analysis.get("survival"), dict) and "ph_test" in analysis["survival"]:
        result += ".formal_ph"
    return result


def resolve(execution, method_id):
    if not isinstance(execution, dict):
        raise BudgetError("RESOURCE_BUDGET_INVALID")
    d = defaults(method_id)
    extra = execution.get("resource_budget", {})
    if not isinstance(extra, dict) or set(extra) - {"max_design_columns", "max_work_units", "max_candidate_pairs", "max_background_rows"}:
        raise BudgetError("RESOURCE_BUDGET_UNKNOWN_FIELD")
    rows = _integer(execution.get("max_rows", d["max_rows"]), "max_rows", profile(method_id)["ceiling_rows"])
    raw = _integer(execution.get("max_bytes", d["max_bytes"]), "max_bytes", ABSOLUTE_MAX_INPUT_BYTES)
    decoded = _integer(execution.get("max_decoded_bytes", d["max_decoded_bytes"]), "max_decoded_bytes", ABSOLUTE_MAX_INPUT_BYTES)
    width = _integer(extra.get("max_design_columns", d["resource_budget"]["max_design_columns"]), "max_design_columns", ABSOLUTE_MAX_DESIGN_COLUMNS)
    work = _integer(extra.get("max_work_units", d["resource_budget"]["max_work_units"]), "max_work_units", ABSOLUTE_MAX_WORK_UNITS)
    pairs = _integer(extra.get("max_candidate_pairs", d["resource_budget"]["max_candidate_pairs"]), "max_candidate_pairs", _CONFIG["absolute"]["max_candidate_pairs"])
    background = _integer(extra.get("max_background_rows", d["resource_budget"]["max_background_rows"]), "max_background_rows", profile(method_id).get("ceiling_background_rows", _CONFIG["absolute"]["max_background_rows"]))
    return {"max_history_rows": ABSOLUTE_MAX_ROWS, "max_candidate_pairs": pairs, "max_background_rows": background, "max_rows": rows, "max_bytes": raw, "max_decoded_bytes": decoded,
            "max_input_bytes": min(ABSOLUTE_MAX_INPUT_BYTES, decoded * 4 + MIB),
            "max_design_columns": width, "max_work_units": work}


def request_limits(execution, method_id):
    values = resolve(execution, method_id)
    return {key: values[key] for key in LIMIT_KEYS}


def validate_limits(limits, method_id):
    if not isinstance(limits, dict) or not BASE_LIMIT_KEYS <= set(limits) or set(limits) - LIMIT_KEYS:
        raise BudgetError("RESOURCE_BUDGET_INVALID")
    bounds = {"max_rows": profile(method_id)["ceiling_rows"], "max_input_bytes": ABSOLUTE_MAX_INPUT_BYTES,
              "max_design_columns": ABSOLUTE_MAX_DESIGN_COLUMNS, "max_work_units": ABSOLUTE_MAX_WORK_UNITS,
              "max_candidate_pairs": _CONFIG["absolute"]["max_candidate_pairs"],
              "max_background_rows": profile(method_id).get("ceiling_background_rows", _CONFIG["absolute"]["max_background_rows"])}
    d = defaults(method_id)["resource_budget"]
    return {k: _integer(limits.get(k, d.get(k)), k, bounds[k]) for k in LIMIT_KEYS}


def check_rows(count, limits):
    if type(count) is not int or not 0 < count <= limits["max_rows"]:
        raise BudgetError("RESOURCE_ROW_BUDGET_EXCEEDED", {"actual_rows": count, "max_rows": limits["max_rows"]})


def owner_capacity(task):
    """Former fixture bounds are extracted; workload is now request-budgeted."""
    return None


def check_owner_capacity(task, count):
    # Compatibility hook retained for callers; no obsolete 200/500 sample gate.
    check_rows(count, resolve(task["execution"], method(task)))


def check_task(task, count, limits):
    check_rows(count, limits)
    # Conservative declared expansion, checked before allocating a design matrix.
    columns = 1 + sum(max(1, len(v.get("levels", [])) - 1) for v in task.get("variables", [])
                      if v.get("dtype") != "id")
    analysis = task.get("analysis", {})
    plan = analysis.get("association", {})
    models = plan.get("models", []) if isinstance(plan, dict) else []
    for item in models:
        spec = item.get("spec", {})
        spline = spec.get("nonlinear")
        if isinstance(spline, dict):
            columns += len(spline.get("knots", []))
        subgroup = spec.get("subgroup")
        if isinstance(subgroup, dict):
            columns *= max(1, len(subgroup.get("levels", [])))
    if columns > limits["max_design_columns"]:
        raise BudgetError("RESOURCE_DESIGN_BUDGET_EXCEEDED", {"declared_columns": columns, "max_design_columns": limits["max_design_columns"]})
    work = count * columns * columns * max(1, len(models))
    if "psm" in method(task) or "formal_ph" in method(task):
        work += count * count
    if work > limits["max_work_units"]:
        raise BudgetError("RESOURCE_WORK_BUDGET_EXCEEDED", {"estimated_work_units": work, "max_work_units": limits["max_work_units"]})
    return {"declared_design_columns": columns, "estimated_work_units": work}
