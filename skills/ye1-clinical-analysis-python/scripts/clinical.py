#!/usr/bin/env python3
"""Run local clinical summaries and models without printing patient records."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


METHOD_DEPS = {
    "prepare": ["pandas", "openpyxl"],
    "inspect": ["pandas", "openpyxl"],
    "baseline": ["numpy", "pandas", "scipy", "openpyxl", "matplotlib"],
    "paired": ["numpy", "pandas", "scipy", "openpyxl", "matplotlib"],
    "association": ["numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib"],
    "survival": ["numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib", "lifelines"],
    "prediction": ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "openpyxl", "matplotlib"],
    "prediction_forest": ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "openpyxl", "matplotlib"],
    "prediction_hgb": ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "openpyxl", "matplotlib"],
    "continuous_prediction": ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "matplotlib"],
    "matching": ["numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib"],
    "correlation": ["numpy", "pandas", "scipy", "openpyxl", "matplotlib"],
    "normality": ["numpy", "pandas", "scipy", "openpyxl", "matplotlib"],
    "repeated": ["numpy", "pandas", "scipy", "openpyxl", "matplotlib"],
    "diagnostic": ["numpy", "pandas", "scipy", "openpyxl", "matplotlib"],
    "diagnostic_roc": ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "matplotlib"],
    "sample_size": ["numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib"],
    "longitudinal": ["numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib"],
    "gee": ["numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib"],
    "mediation": ["numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib"],
    "clustering": ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "matplotlib"],
}
MODULES = {"scikit-learn": "sklearn"}


def doctor(method):
    required = METHOD_DEPS.get(method)
    if required is None:
        raise ValueError("UNKNOWN_METHOD")
    versions = {}
    for package in required:
        module = MODULES.get(package, package)
        if importlib.util.find_spec(module) is None:
            versions[package] = None
        else:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "UNKNOWN"
    missing = [key for key, value in versions.items() if value is None]
    return {"status": "DEPENDENCIES_FOUND" if not missing else "DEPENDENCIES_MISSING",
            "method": method, "python": sys.version.split()[0], "packages": versions, "missing": missing,
            "note": "Availability is not evidence that this environment passed numerical or host validation."}


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor")
    check.add_argument("--method", required=True, choices=METHOD_DEPS)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", required=True); prep.add_argument("--output", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--data", required=True); inspect.add_argument("--output", required=True); inspect.add_argument("--sheet")
    run = sub.add_parser("run")
    run.add_argument("--data", required=True); run.add_argument("--task", required=True)
    run.add_argument("--output", required=True); run.add_argument("--sheet")
    plan = sub.add_parser("plan")
    plan.add_argument("--task", required=True); plan.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        result = doctor(args.method); emit(result)
        return 0 if result["status"] == "DEPENDENCIES_FOUND" else 2
    task = None
    if args.command in {"run", "plan"}:
        task = json.loads(Path(args.task).read_text(encoding="utf-8"))
        if not isinstance(task, dict) or task.get("method") not in METHOD_DEPS or (args.command == "plan" and task.get("method") != "sample_size") or (args.command == "run" and task.get("method") == "sample_size"):
            emit({"status": "BLOCKED", "reason": "UNSUPPORTED_METHOD"}); return 2
    group = task["method"] if task is not None else args.command
    health = doctor(group)
    if health["status"] != "DEPENDENCIES_FOUND":
        emit(health); return 2
    if args.command == "plan":
        from basic_extra import sample_size
        from output import write_report
        result = sample_size(task.get("spec", {}))
        result["software"] = health
        result["privacy"] = {"patient_rows_returned": False, "analysis": "local planning without input data"}
        write_report(args.output, result)
        emit({"status": result["status"], "output": str(Path(args.output).resolve()), "report": "report.json", "table": "results.xlsx"})
        return 0
    from local_data import load, prepare, safe_inspect
    if args.command == "prepare":
        info = prepare(args.data, args.output)
        emit({"status": "LOCAL_WORKING_COPY_CREATED", "output": str(Path(args.output).resolve()), **info,
              "note": "Direct-column removal is not complete anonymization; original preserved."})
        return 0
    frame = load(args.data, args.sheet)
    if args.command == "inspect":
        path = Path(args.output)
        with path.open("x", encoding="utf-8") as out:
            json.dump(safe_inspect(frame), out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
        emit({"status": "AGGREGATE_METADATA_WRITTEN", "output": str(path.resolve())})
        return 0
    from engines import analyze
    from output import write_report
    result, private = analyze(frame, task)
    result["software"] = health
    result["privacy"] = {"patient_rows_returned": False, "analysis": "local", "original_file_modified": False,
                         "caution": "Raw and deidentified records and private artifacts remain sensitive."}
    write_report(args.output, result, private)
    status = result.get("status", "UNKNOWN")
    emit({"status": status, "output": str(Path(args.output).resolve()), "report": "report.json", "table": "results.xlsx"})
    return 2 if status.startswith("BLOCKED") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        # Error values may contain cell contents or paths. Only explicitly
        # authored stable codes are eligible for return to the model.
        try:
            from local_data import DataError
            reason = str(exc) if isinstance(exc, DataError) and str(exc).isupper() else "LOCAL_PROCESSING_FAILED"
        except Exception:
            reason = "LOCAL_PROCESSING_FAILED"
        emit({"status": "BLOCKED", "reason": reason, "patient_rows_returned": False})
        raise SystemExit(2)
