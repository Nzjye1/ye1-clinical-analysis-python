"""Aggregate-only deliverables; private model/pair artifacts remain local."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def safe_cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _sheet(book, name, columns, rows):
    sheet = book.create_sheet(name)
    sheet.append(columns)
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor="194A62")
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(wrap_text=True)
    for row in rows:
        sheet.append([safe_cell(row.get(col)) for col in columns])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for i, col in enumerate(columns, 1):
        max_length = max([len(str(col))] + [min(len(str(r.get(col, ""))), 32) for r in rows[:200]])
        sheet.column_dimensions[get_column_letter(i)].width = min(max(max_length+2, 12), 34)


def _table_rows(report):
    method = report.get("method", "")
    if method == "baseline_summary":
        result = []
        for var in report["variables"]:
            comparison = var.get("comparison") or {}
            for group, summary in [("Overall", var["overall"])] + [(g["group"], g["summary"]) for g in var["groups"]]:
                base = {"variable": var["variable"], "type": var["type"], "group": group,
                        "n_nonmissing": summary["n"], "missing": summary["missing"],
                        "test": comparison.get("method"), "p": comparison.get("p"),
                        "test_note": comparison.get("reason")}
                if var["type"] == "numeric":
                    result.append({**base, **{k: summary.get(k) for k in ("mean", "sd", "median", "q1", "q3")}})
                else:
                    result.extend({**base, **v} for v in summary["levels"])
        return ["variable", "type", "group", "level", "n_nonmissing", "missing", "count", "percent", "mean", "sd", "median", "q1", "q3", "test", "p", "test_note"], result
    if report.get("family") == "association":
        return ["term", "estimate", "standard_error", "ci95", "odds_ratio", "or_ci95", "p", "scale"], report.get("coefficients", [])
    if method in {"kaplan_meier", "cox_ph", "kaplan_meier_by_group", "aalen_johansen", "aalen_johansen_by_group", "cause_specific_cox"}:
        groups = report.get("groups") or [{"group": "Overall", "result": report}]
        result = []
        for group in groups:
            for point in group["result"].get("curves", []):
                result.append({"group": group["group"], "time": point["time"], "at_risk": point["at_risk_before_time"],
                               "survival": point["event_free_probability"], "ci95": point["event_free_ci95"],
                               "cumulative_incidence": point.get("cumulative_incidence")})
        return ["group", "time", "at_risk", "survival", "ci95", "cumulative_incidence"], result
    if method == "1:1_greedy_propensity_matching_without_replacement":
        return ["term", "smd_before", "smd_after"], report.get("balance", [])
    if method in {"longitudinal_mixedlm", "longitudinal_gee"}:
        return ["term", "estimate", "standard_error", "ci95", "odds_ratio", "or_ci95", "p"], report.get("terms", [])
    if method == "linear_mediation_binary_exposure":
        return ["metric", "value"], ([{"metric":k,"value":v} for k,v in report.get("estimates",{}).items()] +
                                      [{"metric":"indirect_percentile_ci95","value":report.get("indirect_percentile_ci95")}])
    if method == "diagnostic_2x2":
        return ["metric", "estimate", "ci95", "numerator", "denominator"], [
            {"metric":k,**report[k]} for k in ("sensitivity","specificity","positive_predictive_value","negative_predictive_value","accuracy")]
    if method == "diagnostic_roc":
        return ["fpr", "tpr", "threshold_original_scale"], report.get("roc",[])
    if method in {"mean_precision","proportion_precision","two_mean","two_proportion"}:
        return ["metric", "base", "after_dropout"], [{"metric":k,"base":v,"after_dropout":report["after_dropout"].get(k)}
                                                          for k,v in report.get("base",{}).items()]
    if method in {"exploratory_kmeans","exploratory_gaussian_mixture"}:
        return ["cluster", "n", "feature_means"], report.get("clusters",[])
    if method == "continuous_prediction":
        return ["metric", "value"], [{"metric":k,"value":v} for k,v in report.get("metrics",{}).items()]
    if report.get("schema_version") == "clinical-prediction/1.0.0":
        metrics = report.get("metrics") or {}
        return ["metric", "value"], [{"metric": key, "value": metrics.get(key)} for key in ("auc", "average_precision", "brier", "calibration")]
    return ["metric", "value"], [{"metric": k, "value": v} for k, v in report.items() if not isinstance(v, (dict, list))]


def write_report(target, report, private=None):
    target = Path(target)
    target.mkdir(parents=False, exist_ok=False)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    workbook = Workbook()
    workbook.remove(workbook.active)
    columns, rows = _table_rows(report)
    _sheet(workbook, "Results", columns, rows)
    if report.get("method") in {"cox_ph", "cause_specific_cox"} and report.get("cox"):
        _sheet(workbook, "Cox", ["term", "hazard_ratio", "hr_ci95", "p", "standard_error"], report["cox"].get("terms", []))
    if report.get("method") == "1:1_greedy_propensity_matching_without_replacement" and report.get("outcome"):
        _sheet(workbook, "Matched outcome", ["metric", "value"],
               [{"metric": k, "value": v} for k, v in report["outcome"].items()])
    if report.get("schema_version") == "clinical-prediction/1.0.0" and report.get("metrics"):
        _sheet(workbook, "ROC", ["threshold", "fpr", "tpr", "selected_n"], report["metrics"].get("aggregate_roc_pr", []))
    info = [{"item": "status", "value": report.get("status")},
            {"item": "n_total", "value": report.get("n_total")},
            {"item": "note", "value": "Aggregate output; review design, coding, missingness and estimates."}]
    _sheet(workbook, "Information", ["item", "value"], info)
    workbook.save(target / "results.xlsx")
    if private is not None:
        folder = target / "private"
        folder.mkdir(mode=0o700)
        name = ("matched_pairs.json" if private.get("classification") == "LOCAL_ONLY_MATCHED_IDENTIFIERS"
                else "cluster_assignments.json" if private.get("classification") == "LOCAL_ONLY_CLUSTER_ASSIGNMENTS"
                else "model.json")
        file = folder / name
        file.write_text(json.dumps(private, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        try:
            file.chmod(0o600)
        except OSError:
            pass
    _plot(target, report)


def _plot(target, report):
    # Do not write Matplotlib's user-global configuration or caches.
    with tempfile.TemporaryDirectory(prefix="ye1-plot-") as tmp:
        os.environ["MPLCONFIGDIR"] = tmp
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        available = {x.name for x in font_manager.fontManager.ttflist}
        for candidate in ("PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Heiti SC"):
            if candidate in available:
                plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False
        method = report.get("method")
        fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
        filename = None
        if method in {"kaplan_meier", "kaplan_meier_by_group"}:
            groups = report.get("groups") or [{"group": "Overall", "result": report}]
            for group in groups:
                points = group["result"].get("curves", [])
                if not points:
                    continue
                x = [p["time"] for p in points]; y = [p["event_free_probability"] for p in points]
                line = ax.step(x, y, where="post", label=group["group"])[0]
                ax.fill_between(x, [p["event_free_ci95"][0] for p in points],
                                [p["event_free_ci95"][1] for p in points], step="post", alpha=.14, color=line.get_color())
            ax.set(xlabel="Time (see report for units)", ylabel="Survival probability", ylim=(0, 1.03), title="Kaplan–Meier")
            ax.legend(); filename = "survival.png"
        elif method in {"aalen_johansen", "aalen_johansen_by_group"}:
            groups = report.get("groups") or [{"group": "Overall", "result": report}]
            for group in groups:
                points = group["result"].get("curves", [])
                if points:
                    ax.step([p["time"] for p in points], [p["cumulative_incidence"][0] for p in points],
                            where="post", label=group["group"])
            ax.set(xlabel="Time (see report for units)", ylabel="Target event cumulative incidence", ylim=(0, 1.03), title="Competing events: cumulative incidence")
            ax.legend(); filename = "cumulative_incidence.png"
        elif report.get("family") == "association" and report.get("status") == "COMPUTED_REVIEW_PENDING":
            terms = [r for r in report.get("coefficients", []) if r["term"] != "intercept"]
            if terms:
                logistic = report.get("model") == "logistic"
                estimates = [r["odds_ratio"] if logistic else r["estimate"] for r in terms]
                intervals = [r["or_ci95"] if logistic else r["ci95"] for r in terms]
                for i, (term, value, ci) in enumerate(zip(terms, estimates, intervals)):
                    ax.plot(ci, [i, i], color="#1f718d", linewidth=1.5)
                    ax.scatter([value], [i], color="#1f718d", s=25)
                ax.set_yticks(range(len(terms)), [r["term"] for r in terms]); ax.invert_yaxis()
                if logistic:
                    ax.set_xscale("log"); ax.axvline(1, color="0.5", linestyle="--"); ax.set_xlabel("Odds ratio (95% CI)")
                else:
                    ax.axvline(0, color="0.5", linestyle="--"); ax.set_xlabel("Coefficient (95% CI)")
                ax.set_title("Association estimates"); filename = "forest.png"
        elif report.get("schema_version") == "clinical-prediction/1.0.0" and report.get("metrics"):
            points = report["metrics"].get("aggregate_roc_pr", [])
            if points:
                ax.plot([p["fpr"] for p in reversed(points)], [p["tpr"] for p in reversed(points)], color="#1f718d")
                ax.plot([0, 1], [0, 1], "--", color="0.6")
                ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC", xlim=(0, 1), ylim=(0, 1))
                filename = "roc.png"
        elif method == "diagnostic_roc" and report.get("roc"):
            ax.plot([p["fpr"] for p in report["roc"]], [p["tpr"] for p in report["roc"]], color="#1f718d")
            ax.plot([0,1],[0,1],"--",color="0.6")
            ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Diagnostic ROC (study sample)",xlim=(0,1),ylim=(0,1))
            filename="diagnostic_roc.png"
        if filename:
            fig.savefig(target / filename, dpi=200)
        plt.close(fig)
