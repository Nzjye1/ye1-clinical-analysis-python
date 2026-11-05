#!/usr/bin/env python3
"""Small synthetic check for installation and basic analysis."""
import json
import sys

from clinical import doctor


def main():
    checks = [doctor("baseline"), doctor("survival")]
    missing = sorted({p for check in checks for p in check["missing"]})
    if missing:
        print(json.dumps({"status": "DEPENDENCIES_MISSING", "missing": missing}))
        return 2
    import pandas as pd
    from engines import analyze
    unit = {"id": "id", "independent": True, "description": "synthetic independent subject"}
    f = pd.DataFrame({"id": [str(i) for i in range(6)], "group": ["0"]*3+["1"]*3,
                      "x": ["1", "2", "3", "4", "5", "6"]})
    task = {"method": "baseline", "unit": unit, "spec": {"group": {"column": "group", "levels": ["0", "1"]},
            "variables": [{"column": "x", "type": "numeric", "test": "welch_t"}]}}
    baseline, _ = analyze(f, task)
    p = baseline["variables"][0]["comparison"]["p"]
    s = pd.DataFrame({"id": ["a", "b", "c", "d"], "time": ["1", "2", "2", "3"], "event": ["1", "1", "0", "1"]})
    spec = {"method": "kaplan_meier", "time": "time", "status": "event", "time_unit": "days",
            "time_zero_definition": "synthetic start", "event_definition": "synthetic event",
            "censoring_definition": "synthetic follow-up", "event_code": "1", "censor_code": "0",
            "missing_policy": "complete_case", "evaluation_grid": "observed", "covariates": []}
    survival, _ = analyze(s, {"method": "survival", "unit": unit, "spec": spec})
    actual = [v["event_free_probability"] for v in survival["curves"]]
    if abs(p - 0.021311641128756727) > 1e-10 or actual != [1.0, .75, .5, 0.0]:
        print(json.dumps({"status": "SYNTHETIC_CHECK_FAILED"}))
        return 2
    print(json.dumps({"status": "SYNTHETIC_BASELINE_AND_KM_PASSED", "scope": "synthetic baseline and survival"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(json.dumps({"status": "SYNTHETIC_CHECK_FAILED"}))
        sys.exit(2)
