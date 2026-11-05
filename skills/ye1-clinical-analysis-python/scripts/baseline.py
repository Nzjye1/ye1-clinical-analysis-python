"""Descriptive summaries and explicitly chosen comparisons, all in Python."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats

from local_data import DataError


def require(value, reason):
    if not value:
        raise DataError(reason)


def is_missing(v):
    return v is None or str(v).strip() == ""


def numeric(values):
    result = []
    for value in values:
        if is_missing(value):
            continue
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            raise DataError("NONNUMERIC_VALUE_IN_NUMERIC_VARIABLE") from None
        require(math.isfinite(number), "NONFINITE_NUMERIC_VALUE")
        result.append(number)
    return np.asarray(result, dtype=float)


def numeric_summary(values):
    x = numeric(values)
    if not len(x):
        return {"n": 0, "missing": len(values), "mean": None, "sd": None,
                "median": None, "q1": None, "q3": None}
    return {"n": len(x), "missing": len(values) - len(x), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "median": float(np.median(x)), "q1": float(np.quantile(x, .25)),
            "q3": float(np.quantile(x, .75))}


def category_summary(values, levels):
    observed = [str(v) for v in values if not is_missing(v)]
    require(all(v in levels for v in observed), "UNDECLARED_CATEGORY_VALUE")
    denominator = len(observed)
    return {"n": denominator, "missing": len(values) - denominator,
            "levels": [{"level": level, "count": observed.count(level),
                        "percent": 100 * observed.count(level) / denominator if denominator else None}
                       for level in levels]}


def _levels(levels):
    require(isinstance(levels, list) and len(levels) >= 2, "AT_LEAST_TWO_DECLARED_LEVELS_REQUIRED")
    require(all(isinstance(x, str) and x for x in levels) and len(set(levels)) == len(levels),
            "UNIQUE_STRING_LEVELS_REQUIRED")
    return levels


def _test_continuous(a, b, method):
    x, y = numeric(a), numeric(b)
    require(len(x) >= 2 and len(y) >= 2, "INSUFFICIENT_NONMISSING_GROUP_VALUES")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "welch_t":
            result = stats.ttest_ind(x, y, equal_var=False, alternative="two-sided")
            df = float(result.df)
            difference = float(x.mean() - y.mean())
            sem = math.sqrt(float(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y)))
            width = float(stats.t.ppf(.975, df) * sem)
            extra = {"difference_group1_minus_group2": difference, "difference_ci95": [difference-width, difference+width],
                     "degrees_of_freedom": df}
        elif method == "mann_whitney_asymptotic":
            result = stats.mannwhitneyu(x, y, method="asymptotic", use_continuity=True, alternative="two-sided")
            extra = {"note": "Two-sided Mann-Whitney; asymptotic tie correction and continuity correction."}
        else:
            raise DataError("UNSUPPORTED_CONTINUOUS_TEST")
    require(not caught and math.isfinite(float(result.pvalue)) and math.isfinite(float(result.statistic)),
            "COMPARISON_NOT_ESTIMABLE")
    return {"method": method, "statistic": float(result.statistic), "p": float(result.pvalue),
            "n_group1": len(x), "n_group2": len(y), **extra}


def _test_categorical(groups, levels, method):
    clean = [[str(x) for x in group if not is_missing(x)] for group in groups]
    require(all(v in levels for group in clean for v in group), "UNDECLARED_CATEGORY_VALUE")
    require(all(clean), "EMPTY_COMPARISON_GROUP")
    table = np.asarray([[group.count(level) for level in levels] for group in clean], dtype=int)
    require(not np.any(table.sum(axis=0) == 0), "UNOBSERVED_CATEGORY_IN_COMPARISON")
    if method == "chi_square":
        stat, p, df, expected = stats.chi2_contingency(table, correction=False)
        require(bool(np.all(expected >= 5)), "CHI_SQUARE_EXPECTED_COUNT_LT_5")
        return {"method": "pearson_chi_square_no_continuity_correction", "statistic": float(stat),
                "p": float(p), "degrees_of_freedom": int(df), "minimum_expected": float(expected.min())}
    if method == "fisher_2x2":
        require(table.shape == (2, 2), "FISHER_REQUIRES_TWO_BY_TWO")
        result = stats.fisher_exact(table, alternative="two-sided")
        return {"method": "fisher_exact_two_sided", "p": float(result.pvalue),
                "note": "Fisher p-value; no effect-size confidence interval is calculated here."}
    raise DataError("UNSUPPORTED_CATEGORICAL_TEST")


def _test_multiple_continuous(groups, method):
    samples = [numeric(group) for group in groups]
    require(len(samples) >= 2 and all(len(x) >= 2 for x in samples), "INSUFFICIENT_NONMISSING_GROUP_VALUES")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method in {"oneway_anova", "welch_anova"}:
            if method == "welch_anova":
                require(all(float(x.var(ddof=1)) > 0 for x in samples), "ZERO_GROUP_VARIANCE")
            res = stats.f_oneway(*samples, equal_var=method == "oneway_anova")
        elif method == "kruskal_wallis":
            res = stats.kruskal(*samples, nan_policy="raise")
        else:
            raise DataError("UNSUPPORTED_MULTIGROUP_TEST")
    require(not caught and math.isfinite(float(res.pvalue)) and math.isfinite(float(res.statistic)), "COMPARISON_NOT_ESTIMABLE")
    return {"method": method, "statistic": float(res.statistic), "p": float(res.pvalue),
            "group_nonmissing_n": [len(x) for x in samples],
            "note": "Global comparison; no post-hoc pairwise comparisons or multiplicity adjustment performed."}


def analyze(frame, spec):
    require(spec.get("independent") is True, "CONFIRM_INDEPENDENT_OBSERVATIONS")
    variables = spec.get("variables")
    require(isinstance(variables, list) and variables, "VARIABLE_LIST_REQUIRED")
    group = spec.get("group")
    if group is not None:
        require(isinstance(group, dict) and group.get("column") in frame, "GROUP_COLUMN_REQUIRED")
        group_column = group["column"]
        group_levels = _levels(group.get("levels"))
        observed = [str(v) for v in frame[group_column] if not is_missing(v)]
        require(all(v in group_levels for v in observed), "UNDECLARED_GROUP_VALUE")
        populations = [(level, frame[frame[group_column] == level]) for level in group_levels]
        missing_group = len(frame) - len(observed)
    else:
        group_levels, populations, missing_group = [], [], 0
    results = []
    seen = set()
    for var in variables:
        require(isinstance(var, dict), "VARIABLE_SPEC_REQUIRED")
        name, kind, method = var.get("column"), var.get("type"), var.get("test", "none")
        require(isinstance(name, str) and name in frame and name not in seen, "VALID_UNIQUE_VARIABLE_REQUIRED")
        seen.add(name)
        require(kind in {"numeric", "categorical"}, "VARIABLE_TYPE_REQUIRED")
        require(not group or name != group["column"], "GROUP_CANNOT_BE_A_SUMMARY_VARIABLE")
        vals = frame[name].tolist()
        levels = _levels(var.get("levels")) if kind == "categorical" else None
        summary = numeric_summary(vals) if kind == "numeric" else category_summary(vals, levels)
        groups = [{"group": label, "summary": numeric_summary(part[name].tolist()) if kind == "numeric"
                   else category_summary(part[name].tolist(), levels)} for label, part in populations]
        result = {"variable": name, "type": kind, "overall": summary, "groups": groups, "comparison": None}
        if method != "none":
            require(len(populations) >= 2, "COMPARISON_REQUIRES_GROUPS")
            try:
                sample_values = [part[name].tolist() for _, part in populations]
                if kind == "categorical":
                    result["comparison"] = _test_categorical(sample_values, levels, method)
                elif method in {"oneway_anova", "welch_anova", "kruskal_wallis"}:
                    result["comparison"] = _test_multiple_continuous(sample_values, method)
                else:
                    require(len(populations) == 2, "COMPARISON_REQUIRES_TWO_GROUPS")
                    result["comparison"] = _test_continuous(sample_values[0], sample_values[1], method)
            except DataError as exc:
                result["comparison"] = {"status": "NOT_ESTIMABLE", "reason": str(exc), "p": None}
        results.append(result)
    return {"status": "COMPUTED_REVIEW_PENDING", "method": "baseline_summary",
            "n_total": len(frame), "group": group, "missing_group": missing_group,
            "variables": results,
            "notes": ["Only blank cells are treated as missing; category codes and tests are explicitly declared.",
                      "Percentages use the nonmissing denominator in each population.",
                      "Group comparison p-values are descriptive and do not establish causality."]}


def paired(frame, spec):
    require(spec.get("independent_pairs") is True, "CONFIRM_INDEPENDENT_PAIRS")
    before, after = spec.get("before"), spec.get("after")
    require(isinstance(before, str) and isinstance(after, str) and before != after and before in frame and after in frame,
            "BEFORE_AND_AFTER_COLUMNS_REQUIRED")
    pairs = [(a, b) for a, b in zip(frame[before], frame[after]) if not is_missing(a) and not is_missing(b)]
    require(len(pairs) >= 2, "INSUFFICIENT_COMPLETE_PAIRS")
    left, right = numeric([p[0] for p in pairs]), numeric([p[1] for p in pairs])
    diff = right-left
    method = spec.get("test")
    require(method in {"paired_t", "wilcoxon"}, "PAIRED_TEST_REQUIRED")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = stats.ttest_rel(right, left, alternative="two-sided") if method == "paired_t" else stats.wilcoxon(diff, alternative="two-sided", method="auto")
    require(not caught and math.isfinite(float(result.pvalue)) and math.isfinite(float(result.statistic)), "PAIRED_TEST_NOT_ESTIMABLE")
    summary = {"mean_after_minus_before": float(diff.mean()), "median_after_minus_before": float(np.median(diff))}
    if method == "paired_t":
        half = float(stats.t.ppf(.975, len(diff)-1)*stats.sem(diff))
        summary["mean_difference_ci95"] = [float(diff.mean()-half), float(diff.mean()+half)]
    return {"status": "COMPUTED_REVIEW_PENDING", "method": method, "n_total": len(frame),
            "n_complete_pairs": len(pairs), "n_excluded_missing": len(frame)-len(pairs),
            "statistic": float(result.statistic), "p": float(result.pvalue), **summary}
