"""Fixed binary prediction with training-only transforms and frozen JSON validation.

No data acquisition, cloud calls, clinical schema defaults, or pickle execution.
The caller must enforce project authorization and retain all model bundles locally.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import warnings
from collections import Counter

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, TimeSeriesSplit, train_test_split

VERSION = 'clinical-prediction/1.0.0'


class PredictionBlocked(ValueError):
    """Method prerequisites are absent; messages contain no patient values."""


def _need(condition, code):
    if not condition:
        raise PredictionBlocked(code)


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def _id_digest(value):
    return _digest({'local_subject_identifier': str(value)})


def _missing(v):
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and math.isnan(v))


def _reject_reserved_names(names):
    # Internal outcome labels must never collide with any caller-controlled
    # role or source column. Reject rather than silently renaming source data.
    _need(not any(isinstance(name, str) and name.casefold().startswith('_ye1_') for name in names),
          'RESERVED_INTERNAL_NAMESPACE')


def _reserved_spec_guard(spec):
    features = spec.get('features', [])
    split = spec.get('split', {})
    names = [spec.get('id'), spec.get('outcome')]
    if isinstance(features, list):
        names.extend(f.get('name') for f in features if isinstance(f, dict))
    post_outcome_fields = spec.get('post_outcome_fields', [])
    if isinstance(post_outcome_fields, list):
        names.extend(post_outcome_fields)
    if isinstance(split, dict):
        names.extend(split.get(role) for role in ('group', 'time', 'label_available_time'))
    _reject_reserved_names(names)


def _threshold_policy(spec):
    """Resolve a DCA evaluation grid separately from a clinical operating point.

    ``decision_thresholds`` is retained solely as the legacy DCA-grid alias; it
    never chooses or implies a clinical decision threshold. Neither is searched.
    """
    grid = spec.get('dca_thresholds', spec.get('decision_thresholds', []))
    _need(isinstance(grid, list) and all(isinstance(t, (int, float)) and not isinstance(t, bool)
          and math.isfinite(t) and 0 < t < 1 for t in grid),
          'DCA_EVALUATION_GRID_MUST_BE_PREDECLARED_INSIDE_0_1')
    _need(len(grid) == len(set(grid)), 'DUPLICATE_DCA_EVALUATION_THRESHOLD')
    if 'dca_thresholds' in spec and 'decision_thresholds' in spec:
        _need(spec['dca_thresholds'] == spec['decision_thresholds'], 'CONFLICTING_DCA_GRID_ALIAS')
    clinical = spec.get('clinical_decision_threshold')
    _need(clinical is None or (isinstance(clinical, (int, float)) and not isinstance(clinical, bool)
          and math.isfinite(clinical) and 0 < clinical < 1),
          'CLINICAL_DECISION_THRESHOLD_MUST_BE_PRESPECIFIED_INSIDE_0_1')
    return [float(t) for t in grid], None if clinical is None else float(clinical)


def _validate_spec(spec):
    _reserved_spec_guard(spec)
    _need(spec.get('penalty', 'l2') in {'l1', 'l2', 'elasticnet'}, 'PENALTY_MUST_BE_L1_L2_OR_ELASTICNET')
    if spec.get('penalty', 'l2') == 'elasticnet':
        ratio = spec.get('l1_ratio')
        _need(isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and 0 < ratio < 1,
              'ELASTICNET_MIXING_RATIO_REQUIRED')
    for key in ('id', 'outcome', 'event', 'non_event', 'cohort_id', 'prediction_time', 'outcome_window'):
        _need(key in spec and spec[key] is not None and str(spec[key]).strip(), 'PREDICTION_DEFINITION_REQUIRED')
    _need(str(spec['event']) != str(spec['non_event']), 'DISTINCT_EVENT_ENCODING_REQUIRED')
    features = spec.get('features')
    _need(isinstance(features, list) and bool(features), 'EXPLICIT_FEATURE_DEFINITIONS_REQUIRED')
    names = [f.get('name') for f in features]
    _need(all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names), 'UNIQUE_FEATURE_NAMES_REQUIRED')
    prohibited = {spec['id'], spec['outcome'], *spec.get('post_outcome_fields', [])}
    split = spec.get('split', {})
    prohibited.update(x for x in (split.get('group'), split.get('time'), split.get('label_available_time')) if x)
    _need(not (set(names) & prohibited), 'LEAKAGE_ROLE_IN_FEATURES')
    for f in features:
        _need(f.get('available_at_prediction') is True, 'PREDICTOR_TIMING_NOT_CONFIRMED')
        _need(f.get('type') in {'number', 'category'}, 'UNSUPPORTED_FEATURE_TYPE')
        if f['type'] == 'category':
            levels = f.get('levels')
            _need(isinstance(levels, list) and len(levels) >= 2 and
                  len({str(v) for v in levels}) == len(levels) and all(not _missing(v) for v in levels),
                  'PREDECLARED_CATEGORY_LEVELS_REQUIRED')
    _need(spec.get('missing', 'train_median_mode') in {'train_median_mode', 'complete_case'}, 'MISSING_METHOD_UNSUPPORTED')
    _need(spec.get('selection', 'none') in {'none', 'train_univariate'}, 'SELECTION_METHOD_UNSUPPORTED')
    _threshold_policy(spec)
    return features


def _rows(rows, spec, *, training):
    features = _validate_spec(spec)
    _need(isinstance(rows, list) and bool(rows) and all(isinstance(r, dict) for r in rows), 'NONEMPTY_TABLE_REQUIRED')
    _reject_reserved_names(name for row in rows for name in row)
    required = {spec['id'], spec['outcome'], *(f['name'] for f in features)}
    split = spec.get('split', {})
    if training:
        required.update(x for x in (split.get('group'), split.get('time'), split.get('label_available_time')) if x)
    _need(all(required <= set(r) for r in rows), 'REQUIRED_COLUMN_ABSENT')
    ids = [r[spec['id']] for r in rows]
    _need(all(not _missing(i) for i in ids), 'IDENTIFIER_MISSING')
    _need(len({_id_digest(i) for i in ids}) == len(ids), 'DUPLICATE_SUBJECT_IN_INDEPENDENT_PREDICTION')
    out = []
    raw_events = raw_nonevents = missing_outcome = missing_features = 0
    for r in rows:
        v = r[spec['outcome']]
        if _missing(v):
            missing_outcome += 1
            continue
        _need(str(v) in {str(spec['event']), str(spec['non_event'])}, 'UNDECLARED_OUTCOME_CODE')
        y = int(str(v) == str(spec['event']))
        raw_events += y
        raw_nonevents += 1-y
        record = dict(r)
        any_missing = False
        for f in features:
            name = f['name']; value = record[name]
            if _missing(value):
                record[name] = None; any_missing = True
            elif f['type'] == 'number':
                try:
                    record[name] = float(value)
                except (ValueError, TypeError, OverflowError):
                    raise PredictionBlocked('INVALID_NUMERIC_FEATURE') from None
                _need(math.isfinite(record[name]), 'NONFINITE_FEATURE')
            else:
                record[name] = str(value)
                _need(record[name] in {str(v) for v in f['levels']}, 'UNKNOWN_CATEGORY')
        if spec.get('missing', 'train_median_mode') == 'complete_case' and any_missing:
            missing_features += 1
            continue
        record['_ye1_outcome'] = y
        out.append(record)
    counts = {'total_n': len(rows), 'total_events': raw_events, 'total_nonevents': raw_nonevents,
              'missing_outcome_n': missing_outcome, 'missing_feature_excluded_n': missing_features,
              'analyzable_n': len(out), 'analyzable_events': sum(r['_ye1_outcome'] for r in out),
              'analyzable_nonevents': sum(1-r['_ye1_outcome'] for r in out)}
    _need(bool(out), 'NO_ANALYZABLE_OUTCOMES')
    return out, counts


def _times(values):
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
    if numeric:
        return values
    _need(all(isinstance(v, str) and v for v in values), 'CONSISTENT_TIME_ENCODING_REQUIRED')
    from datetime import datetime
    try:
        parsed = [datetime.fromisoformat(v) for v in values]
        sorted(parsed)  # Reject mixed aware/naive times; no host timezone assumptions.
        return parsed
    except (ValueError, TypeError):
        raise PredictionBlocked('CONSISTENT_ISO8601_TIME_REQUIRED') from None


def _temporal_label_check(rows, spec, train, test):
    """A historical model may use only labels already known at validation start."""
    split = spec['split']
    if split['method'] != 'temporal':
        return
    label_field = split.get('label_available_time')
    _need(isinstance(label_field, str) and label_field.strip(), 'TEMPORAL_LABEL_AVAILABILITY_COLUMN_REQUIRED')
    _need(all(label_field in r and not _missing(r[label_field]) for r in rows),
          'TEMPORAL_LABEL_AVAILABILITY_MISSING')
    # Parse in one call so numeric versus ISO, and aware versus naive times,
    # cannot silently use incompatible scales or the host's local timezone.
    values = _times([r[split['time']] for r in rows] + [r[label_field] for r in rows])
    prediction_times, label_times = values[:len(rows)], values[len(rows):]
    _need(all(label >= prediction for prediction, label in zip(prediction_times, label_times)),
          'LABEL_AVAILABILITY_PRECEDES_PREDICTION_TIME')
    earliest_validation = min(prediction_times[int(i)] for i in test)
    _need(all(label_times[int(i)] <= earliest_validation for i in train),
          'TEMPORAL_TRAINING_LABELS_NOT_MATURE_AT_VALIDATION_START')


def _split(rows, spec):
    split = spec.get('split', {})
    method = split.get('method')
    _need(method in {'random', 'group', 'temporal'}, 'EXPLICIT_SPLIT_METHOD_REQUIRED')
    seed = split.get('seed')
    _need(isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**32, 'REPRODUCIBLE_SEED_REQUIRED')
    fraction = split.get('test_fraction')
    _need(isinstance(fraction, (int, float)) and .2 <= fraction <= .5, 'TEST_FRACTION_OUTSIDE_02_05')
    ix = np.arange(len(rows)); y = np.array([r['_ye1_outcome'] for r in rows])
    if method == 'random':
        _need(len(rows) >= 3, 'TOO_FEW_SUBJECTS_FOR_SPLIT')
        usable_strata = y if len(set(y)) == 2 and min(Counter(y).values()) >= 2 else None
        train, test = train_test_split(ix, test_size=fraction, random_state=seed, stratify=usable_strata)
    elif method == 'group':
        field = split.get('group'); _need(field and all(not _missing(r[field]) for r in rows), 'GROUP_DEFINITION_REQUIRED')
        groups = np.array([str(r[field]) for r in rows]); _need(len(set(groups)) >= 4, 'TOO_FEW_GROUPS_FOR_SPLIT')
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(ix, y, groups))
        _need(set(groups[train]).isdisjoint(set(groups[test])), 'GROUP_OVERLAP')
    else:
        field = split.get('time'); _need(field and all(not _missing(r[field]) for r in rows), 'TIME_DEFINITION_REQUIRED')
        # Datetime comparisons use an explicit ISO8601 representation; numeric time remains numeric.
        values = _times([r[field] for r in rows])
        ordered = sorted(ix, key=lambda i: values[i])
        cutoff = values[ordered[max(1, int(len(rows)*(1-fraction)))]]
        train = np.array([i for i in ordered if values[i] < cutoff], dtype=int)
        test = np.array([i for i in ordered if values[i] >= cutoff], dtype=int)
        _need(len(train) > 0 and len(test) > 0, 'TEMPORAL_BOUNDARY_EMPTY')
        _temporal_label_check(rows, spec, train, test)
    return sorted(map(int, train)), sorted(map(int, test))


def _preprocess_fit(rows, features):
    transforms = []
    for f in features:
        values = [r[f['name']] for r in rows if r[f['name']] is not None]
        _need(bool(values), 'ALL_MISSING_TRAINING_FEATURE')
        if f['type'] == 'number':
            median = float(np.median(values))
            filled = np.array([median if r[f['name']] is None else r[f['name']] for r in rows])
            mean = float(filled.mean()); scale = float(filled.std(ddof=0))
            _need(scale > 1e-12 and math.isfinite(scale), 'CONSTANT_TRAINING_FEATURE')
            transforms.append({'name': f['name'], 'type': 'number', 'impute': median, 'mean': mean, 'scale': scale})
        else:
            levels = [str(v) for v in f['levels']]
            counts = Counter(values)
            mode = max(levels, key=lambda v: (counts[v], -levels.index(v)))
            _need(len(counts) >= 2, 'CONSTANT_TRAINING_CATEGORY')
            _need(all(counts[level] > 0 for level in levels), 'TRAINING_CATEGORY_LEVEL_ABSENT')
            transforms.append({'name': f['name'], 'type': 'category', 'impute': mode, 'levels': levels})
    return transforms


def _transform(rows, transforms):
    columns = []; names = []
    for t in transforms:
        values = [t['impute'] if r[t['name']] is None else r[t['name']] for r in rows]
        if t['type'] == 'number':
            columns.append((np.array(values, dtype=float)-t['mean'])/t['scale']); names.append(t['name'])
        else:
            _need(all(v in t['levels'] for v in values), 'UNKNOWN_CATEGORY')
            for level in t['levels'][1:]:
                columns.append(np.array([float(v == level) for v in values])); names.append(t['name']+'='+level)
    x = np.column_stack(columns)
    _need(np.isfinite(x).all(), 'NONFINITE_DESIGN')
    return x, names


def _train(rows, spec, *, enforce_epv=True):
    features = spec['features']; y = np.array([r['_ye1_outcome'] for r in rows], dtype=int)
    _need(len(set(y)) == 2, 'TRAINING_SINGLE_CLASS')
    transforms = _preprocess_fit(rows, features); x, names = _transform(rows, transforms)
    _need(np.linalg.matrix_rank(np.column_stack([np.ones(len(x)), x])) == x.shape[1]+1,
          'RANK_DEFICIENT_TRAINING_DESIGN')
    candidate_count = x.shape[1]
    minimum = spec.get('min_epv', 10)
    _need(isinstance(minimum, (int, float)) and minimum >= 10, 'MIN_EPV_MUST_BE_AT_LEAST_10')
    if enforce_epv:
        _need(min(int(y.sum()), int((1-y).sum()))/candidate_count >= minimum, 'SPARSE_TRAINING_EPV')
    selected = list(range(candidate_count))
    if spec.get('selection', 'none') == 'train_univariate':
        k = spec.get('selection_k')
        _need(isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= candidate_count, 'SELECTION_K_INVALID')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            scores, _ = f_classif(x, y)
        _need(np.isfinite(scores).all(), 'UNESTIMABLE_TRAINING_SELECTION')
        selected = sorted(np.argsort(-scores, kind='stable')[:k].tolist())
    regularization = spec.get('C', 1.)
    _need(isinstance(regularization, (int, float)) and 0 < regularization <= 1e6, 'FIXED_RIDGE_C_REQUIRED')
    penalty = spec.get('penalty', 'l2')
    solver = 'liblinear' if penalty == 'l1' else 'saga' if penalty == 'elasticnet' else 'lbfgs'
    extras = {'l1_ratio': float(spec['l1_ratio'])} if penalty == 'elasticnet' else {}
    model = LogisticRegression(C=float(regularization), penalty=penalty, solver=solver, max_iter=2000,
                               random_state=spec['split']['seed'], tol=1e-9, **extras)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x[:, selected], y)
    _need(not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'MODEL_NOT_CONVERGED')
    return {'transforms': transforms, 'candidate_parameter_names': names, 'selected_indices': selected,
            'parameter_names': [names[i] for i in selected], 'coefficients': model.coef_[0].tolist(),
            'intercept': float(model.intercept_[0]), 'C': float(regularization), 'penalty': penalty,
            'l1_ratio': spec.get('l1_ratio') if penalty == 'elasticnet' else None}


def _probabilities(rows, model):
    x, names = _transform(rows, model['transforms'])
    _need(names == model['candidate_parameter_names'], 'FROZEN_DESIGN_MISMATCH')
    p = expit(float(model['intercept']) + x[:, model['selected_indices']] @ np.array(model['coefficients']))
    _need(np.isfinite(p).all(), 'NONFINITE_PREDICTION')
    return p


def _calibration(y, p):
    z = logit(np.clip(p, 1e-10, 1-1e-10))
    intercept_only = float(brentq(lambda a: np.sum(y-expit(z+a)), -60, 60))
    design = np.column_stack([np.ones(len(y)), z])
    if np.linalg.matrix_rank(design) != 2:
        return {'intercept_in_large': intercept_only, 'intercept': None, 'slope': None,
                'status': 'CALIBRATION_SLOPE_UNESTIMABLE'}
    # In one score dimension, ordered non-overlapping outcome ranges give
    # complete/quasi separation. Optimizer convergence at a large finite slope
    # does not establish existence of a finite joint calibration MLE.
    no_event, event = z[y == 0], z[y == 1]
    forward_gap = float(event.min()-no_event.max())
    reverse_gap = float(no_event.min()-event.max())
    if forward_gap >= 0 or reverse_gap >= 0:
        kind = 'COMPLETE' if forward_gap > 0 or reverse_gap > 0 else 'QUASI'
        return {'intercept_in_large': intercept_only, 'intercept': None, 'slope': None,
                'status': 'CALIBRATION_'+kind+'_SEPARATION',
                'note': 'Joint calibration intercept/slope have no finite MLE; calibration-in-the-large uses a fixed score offset.'}
    def fun(beta):
        eta = design @ beta
        return float(np.sum(np.logaddexp(0, eta)-y*eta))
    def jac(beta):
        return design.T @ (expit(design @ beta)-y)
    fit = minimize(fun, [0., 1.], jac=jac, method='BFGS', options={'gtol': 1e-7, 'maxiter': 1000})
    q = expit(design @ fit.x); info = design.T @ ((q*(1-q))[:, None]*design)
    if (not fit.success and np.linalg.norm(jac(fit.x)) > 1e-5) or np.linalg.cond(info) > 1e10 or np.any(np.abs(fit.x) > 100):
        return {'intercept_in_large': intercept_only, 'intercept': None, 'slope': None,
                'status': 'CALIBRATION_FIT_UNSTABLE_OR_SEPARATED'}
    return {'intercept_in_large': intercept_only, 'intercept': float(fit.x[0]),
            'slope': float(fit.x[1]), 'status': 'ESTIMABLE'}


def _point(y, p):
    _need(len(set(y)) == 2, 'VALIDATION_SINGLE_CLASS')
    return {'auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p)), 'calibration': _calibration(y, p)}


def _metrics(rows, p, spec):
    y = np.array([r['_ye1_outcome'] for r in rows], dtype=int)
    _need(min(int(y.sum()), int((1-y).sum())) >= 5, 'VALIDATION_REQUIRES_5_PER_CLASS')
    result = _point(y, p)
    reps = spec.get('bootstrap', 0)
    _need(isinstance(reps, int) and not isinstance(reps, bool) and (reps == 0 or 50 <= reps <= 2000), 'BOOTSTRAP_REPS_INVALID')
    _need(reps == 0 or spec.get('independent_subjects') is True,
          'INDEPENDENT_SUBJECTS_CONFIRMATION_REQUIRED_FOR_BOOTSTRAP')
    seed = spec.get('evaluation_seed', spec.get('split', {}).get('seed', 1729))
    rng = np.random.default_rng(seed)
    values = {'auc': [], 'average_precision': [], 'brier': []}
    for _ in range(reps):
        # Subject-level bootstrap: API requires one independent record per subject.
        ix = rng.integers(0, len(y), len(y))
        if len(set(y[ix])) != 2:
            continue
        values['auc'].append(float(roc_auc_score(y[ix], p[ix])))
        values['average_precision'].append(float(average_precision_score(y[ix], p[ix])))
        values['brier'].append(float(brier_score_loss(y[ix], p[ix])))
    result['bootstrap'] = {'requested': reps, 'successful': len(values['auc']),
                           'method': 'independent subject percentile bootstrap; model and preprocessing frozen',
                           'independent_subjects_explicitly_declared': spec.get('independent_subjects') is True,
                           'cluster_resampling': False,
                           'group_split_note': 'A grouped train/validation split does not provide group-cluster-robust confidence intervals.',
                           'intervals95': {k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in values.items()},
                           'calibration_intervals': None}
    bins = []
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 6)))
    labels = np.digitize(p, edges[1:-1], right=True)
    for b in sorted(set(labels)):
        ix = labels == b
        if int(ix.sum()) >= 5:
            bins.append({'n': int(ix.sum()), 'predicted_mean': float(p[ix].mean()), 'observed_fraction': float(y[ix].mean())})
    result['calibration_bins'] = bins
    # A fixed grid is an aggregate plotting contract, not a list of subject scores.
    curves = []
    for threshold in np.linspace(0, 1, 101):
        selected = p >= threshold
        tp = int(np.sum(selected & (y == 1))); fp = int(np.sum(selected & (y == 0)))
        curves.append({'threshold': float(threshold), 'tpr': float(tp/y.sum()),
                       'fpr': float(fp/(1-y).sum()), 'recall': float(tp/y.sum()),
                       'precision': float(tp/(tp+fp)) if tp+fp else None,
                       'selected_n': tp+fp})
    result['aggregate_roc_pr'] = curves
    result['plot_contract'] = 'fixed threshold aggregate grid and aggregate calibration bins; no individual probabilities or identifiers'
    thresholds, clinical_threshold = _threshold_policy(spec)
    result['dca_thresholds'] = thresholds
    result['decision_curve'] = []
    for t in thresholds:
        selected = p >= t; tp = int(np.sum(selected & (y == 1))); fp = int(np.sum(selected & (y == 0)))
        result['decision_curve'].append({'threshold': float(t), 'net_benefit': float((tp-fp*t/(1-t))/len(y)),
                                          'treat_all': float(y.mean()-(1-y.mean())*t/(1-t)), 'treat_none': 0.})
    result['decision_curve_note'] = ('Exploratory utility estimates on the frozen prespecified DCA evaluation grid; '
                                     'this grid does not select a clinical operating threshold or demonstrate clinical utility.')
    result['threshold_policy'] = {'dca_role': 'evaluation_grid_only',
                                  'clinical_decision_threshold': clinical_threshold,
                                  'threshold_search_performed': False,
                                  'external_policy': 'DCA grid and clinical operating point must exactly match the frozen development specification'}
    result['clinical_operating_point'] = None
    if clinical_threshold is not None:
        positive = p >= clinical_threshold
        tp = int(np.sum(positive & (y == 1))); fp = int(np.sum(positive & (y == 0)))
        tn = int(np.sum(~positive & (y == 0))); fn = int(np.sum(~positive & (y == 1)))
        result['clinical_operating_point'] = {
            'threshold': clinical_threshold, 'positive_rule': 'predicted_probability_greater_than_or_equal_to_threshold',
            'threshold_source': 'prespecified_and_frozen_before_model_fitting; no search or external optimization',
            'n': len(y), 'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn,
            'sensitivity': float(tp/(tp+fn)), 'specificity': float(tn/(tn+fp)),
            'positive_predictive_value': float(tp/(tp+fp)) if tp+fp else None,
            'negative_predictive_value': float(tn/(tn+fn)) if tn+fn else None,
            'accuracy': float((tp+tn)/len(y)),
            'clinical_use_note': 'Aggregate validation at a requested operating point; no deployment or utility approval is implied.'}
    return result


def _cv(rows, spec):
    folds = spec.get('cv_folds', 0)
    _need(isinstance(folds, int) and not isinstance(folds, bool) and (folds == 0 or 2 <= folds <= 10), 'CV_FOLDS_INVALID')
    if not folds:
        return {'status': 'NOT_REQUESTED', 'folds': 0}
    y = np.array([r['_ye1_outcome'] for r in rows], dtype=int)
    method = spec['split']['method']
    if method == 'group':
        groups = np.array([str(r[spec['split']['group']]) for r in rows])
        _need(len(set(groups)) >= folds, 'CV_GROUPS_INSUFFICIENT')
        indices = list(GroupKFold(n_splits=folds).split(np.zeros(len(rows)), y, groups))
    elif method == 'temporal':
        field = spec['split']['time']
        times = _times([r[field] for r in rows])
        order = sorted(range(len(rows)), key=lambda i: times[i])
        ordered_rows = [rows[i] for i in order]
        rows = ordered_rows
        times = [times[i] for i in order]
        unique_times = sorted(set(times))
        _need(len(unique_times) > folds, 'CV_TIMES_INSUFFICIENT')
        indices = []
        for a, b in TimeSeriesSplit(n_splits=folds).split(unique_times):
            train_times = {unique_times[int(i)] for i in a}; test_times = {unique_times[int(i)] for i in b}
            indices.append((np.array([i for i, value in enumerate(times) if value in train_times]),
                            np.array([i for i, value in enumerate(times) if value in test_times])))
    else:
        _need(min(Counter(y).values()) >= folds, 'CV_CLASSES_INSUFFICIENT')
        indices = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=spec['split']['seed']).split(np.zeros(len(rows)), y))
    results = []
    for train, test in indices:
        try:
            tr = [rows[int(i)] for i in train]; te = [rows[int(i)] for i in test]
            _temporal_label_check(rows, spec, train, test)
            model = _train(tr, spec)
            values = _point(np.array([r['_ye1_outcome'] for r in te]), _probabilities(te, model))
            results.append({'status': 'ESTIMATED', 'training_n': len(tr), 'validation_n': len(te),
                            'auc': values['auc'], 'brier': values['brier']})
        except PredictionBlocked as exc:
            results.append({'status': 'BLOCKED', 'reason': str(exc), 'training_n': len(train), 'validation_n': len(test)})
    ok = [r for r in results if r['status'] == 'ESTIMATED']
    return {'status': 'ESTIMATED' if len(ok) == folds else 'PARTIAL_OR_BLOCKED', 'folds': folds,
            'scope': 'training partition only; preprocessing and optional selection refitted within every fold; temporal labels must mature before fold validation starts; no hyperparameter tuning',
            'fold_results': results, 'mean_auc': float(np.mean([r['auc'] for r in ok])) if ok else None}


def analyze(rows: list[dict], spec: dict) -> dict:
    """Fit one prespecified L1, L2 or elastic-net logistic model and a local bundle.

    EPV is a conservative engine guard, not a universal adequacy criterion. If blocked
    after splitting, the returned counts and EPV remain available and no performance
    or model is fabricated. Invalid schema/timing/splits raise PredictionBlocked.
    """
    data, counts = _rows(rows, spec, training=True)
    train_ix, test_ix = _split(data, spec)
    train = [data[i] for i in train_ix]; test = [data[i] for i in test_ix]
    p_count = sum(1 if f['type'] == 'number' else len(f['levels'])-1 for f in spec['features'])
    counts.update(training_n=len(train), training_events=sum(r['_ye1_outcome'] for r in train),
                  training_nonevents=sum(1-r['_ye1_outcome'] for r in train),
                  holdout_n=len(test), holdout_events=sum(r['_ye1_outcome'] for r in test),
                  holdout_nonevents=sum(1-r['_ye1_outcome'] for r in test))
    numerator = min(counts['training_events'], counts['training_nonevents'])
    result = {'schema_version': VERSION, 'status': 'BLOCKED_ESTIMATION', 'counts': counts,
              'candidate_parameter_count': p_count, 'predictor_parameter_count': p_count,
              'epv': {'numerator': numerator, 'denominator': p_count, 'value': numerator/p_count,
                      'formula': f'{numerator}/{p_count}', 'scope': 'training only, intercept excluded; full candidate set before optional selection'},
              'validation': 'internal held-out '+spec['split']['method'], 'metrics': None,
              'limitations': ['Conservative EPV guard does not establish sample adequacy.',
                              'One fixed L1, L2 or elastic-net logistic model; no claim of causality, transportability or demonstrated clinical utility.',
                              'Outcome coding, timing and independence require researcher review; declarations cannot prove source semantics.']}
    try:
        model = _train(train, spec)
        metrics = _metrics(test, _probabilities(test, model), spec)
        internal_cv = _cv(train, spec)
    except PredictionBlocked as exc:
        result['reason'] = str(exc)
        return result
    frozen_spec = {k: copy.deepcopy(spec[k]) for k in ('id', 'outcome', 'event', 'non_event', 'prediction_time',
                                                      'outcome_window', 'features')}
    frozen_spec['missing'] = spec.get('missing', 'train_median_mode')
    frozen_spec['dca_thresholds'], frozen_spec['clinical_decision_threshold'] = _threshold_policy(spec)
    bundle = {'schema_version': VERSION, 'classification': 'LOCAL_ONLY_MODEL_BUNDLE',
              'development_cohort_id': spec['cohort_id'], 'definitions': frozen_spec,
              'development_id_digests': sorted(_id_digest(r[spec['id']]) for r in rows),
              'model': model, 'fit_scope': 'training partition only; heldout is never refitted or selected against',
              'training_n': len(train), 'training_events': counts['training_events']}
    bundle['sha256'] = _digest(bundle)
    result.update(status='ESTIMATED', metrics=metrics, internal_cv=internal_cv,
                  predictor_parameter_count=len(model['selected_indices']), model_bundle=bundle)
    return result


def validate_external(rows: list[dict], model_bundle: dict, spec: dict) -> dict:
    """Evaluate exactly frozen parameters in a distinct cohort, without any refit.

    Caller supplies the same explicit definitions and a new cohort_id. Development
    identifier digests stay inside the local bundle to reject subject overlap.
    """
    _reserved_spec_guard(spec)
    bundle = copy.deepcopy(model_bundle)
    expected = bundle.pop('sha256', None)
    _need(expected == _digest(bundle), 'MODEL_BUNDLE_HASH_MISMATCH')
    _need(bundle.get('schema_version') == VERSION and bundle.get('classification') == 'LOCAL_ONLY_MODEL_BUNDLE',
          'UNSUPPORTED_MODEL_BUNDLE')
    _need({'dca_thresholds', 'clinical_decision_threshold'} <= set(bundle.get('definitions', {})),
          'FROZEN_THRESHOLD_POLICY_REQUIRED')
    _need(spec.get('cohort_id') != bundle['development_cohort_id'], 'EXTERNAL_COHORT_MUST_DIFFER')
    external_grid, external_clinical_threshold = _threshold_policy(spec)
    for key, value in bundle['definitions'].items():
        if key == 'dca_thresholds':
            actual = external_grid
        elif key == 'clinical_decision_threshold':
            actual = external_clinical_threshold
        else:
            actual = spec.get(key, 'train_median_mode' if key == 'missing' else None)
        _need(actual == value, 'EXTERNAL_DEFINITIONS_DIFFER_FROM_FROZEN_MODEL')
    data, counts = _rows(rows, spec, training=False)
    development = set(bundle['development_id_digests'])
    _need(not ({_id_digest(r[spec['id']]) for r in rows} & development), 'EXTERNAL_SUBJECT_OVERLAP')
    probabilities = _probabilities(data, bundle['model'])
    metrics = _metrics(data, probabilities, spec)
    return {'schema_version': VERSION, 'status': 'ESTIMATED', 'validation': 'external frozen-model validation',
            'model_sha256': expected, 'model_refitted': False, 'preprocessing_refitted': False,
            'counts': counts, 'metrics': metrics,
            'limitations': ['External cohort independence, endpoint harmonization and timing remain researcher responsibilities.',
                            'External evaluation does not establish clinical usefulness or transportability to other settings.']}
