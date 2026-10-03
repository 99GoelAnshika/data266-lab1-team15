"""Complete binary metrics and paired inference on saved predictions only.

Definitions are read from the preserved pre-training evaluation protocol.
Implementation is independent of teammate evaluation code.
"""
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, matthews_corrcoef, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, auc)


def validate_predictions(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Labels and probabilities must be equal nonempty one-dimensional arrays.')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid binary labels or positive-class probabilities.')
    return y.astype(np.int8), p


def calibration_bins(labels, probabilities, bins=15, threshold=.5):
    y, p = validate_predictions(labels, probabilities)
    prediction = p >= threshold
    confidence = np.maximum(p, 1 - p)
    edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.clip(np.searchsorted(edges, confidence, side='right') - 1, 0, bins - 1)
    rows, ece = [], 0.
    for k in range(bins):
        mask = bin_ids == k
        count = int(mask.sum())
        accuracy = float(np.mean(prediction[mask] == y[mask])) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        gap = abs(accuracy - mean_confidence) if count else None
        if count:
            ece += count / len(y) * gap
        rows.append({'bin': k, 'left': float(edges[k]), 'right': float(edges[k + 1]),
                     'count': count, 'fraction': count / len(y), 'accuracy': accuracy,
                     'mean_confidence': mean_confidence, 'absolute_gap': gap})
    return float(ece), rows


def full_metrics(labels, probabilities, threshold=.5, bins=15):
    y, p = validate_predictions(labels, probabilities)
    predicted = (p >= threshold).astype(np.int8)
    out = {'accuracy': float(accuracy_score(y, predicted)),
           'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
           'mcc': float(matthews_corrcoef(y, predicted)),
           'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
           'pr_auc_average_precision': float(average_precision_score(y, p)) if np.any(y == 1) else None,
           'brier_score': float(brier_score_loss(y, p))}
    for average in ('macro', 'micro', 'weighted'):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predicted, labels=[0, 1], average=average, zero_division=0)
        out.update({f'precision_{average}': float(precision), f'recall_{average}': float(recall),
                    f'f1_{average}': float(f1)})
    precision, recall, _ = precision_recall_curve(y, p)
    out['pr_auc_trapezoidal'] = float(auc(recall, precision))
    out['ece_15_bins'], _ = calibration_bins(y, p, bins, threshold)
    return out


def counts_metrics(tn, fp, fn, tp):
    """Vectorized binary-count identities, including fixed-label macro-F1."""
    tn, fp, fn, tp = (np.asarray(a, dtype=np.float64) for a in (tn, fp, fn, tp))
    total = tn + fp + fn + tp
    safe = lambda numerator, denominator: np.divide(numerator, denominator,
                                                    out=np.zeros_like(numerator), where=denominator != 0)
    accuracy = safe(tn + tp, total)
    f1_negative = safe(2 * tn, 2 * tn + fp + fn)
    f1_positive = safe(2 * tp, 2 * tp + fp + fn)
    mcc = safe(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {'accuracy': accuracy, 'f1_macro': (f1_negative + f1_positive) / 2, 'mcc': mcc}


def bootstrap_intervals(labels, probabilities_by_model, protocol):
    """Identical explicit row draws for all models; no sklearn call per replicate."""
    y = np.asarray(labels, dtype=np.int8)
    threshold = protocol['classification_threshold']
    predictions = {}
    for name, p in probabilities_by_model.items():
        validate_predictions(y, p)
        predictions[name] = np.asarray(p) >= threshold
    settings = protocol['bootstrap']
    if len(y) != settings['replicate_size']:
        raise ValueError('Bootstrap replicate size must equal the entire evaluation population.')
    count = settings['replicates']
    rng = np.random.default_rng(settings['seed'])
    samples = {name: {metric: np.empty(count) for metric in ('accuracy', 'f1_macro', 'mcc')}
               for name in predictions}
    for start in range(0, count, 32):
        end = min(start + 32, count)
        # PCG64 draws, int64, in source-row order. Chunking does not change the RNG stream.
        indices = rng.integers(0, len(y), size=(end - start, len(y)), dtype=np.int64)
        positive = y[indices] == 1
        for name, prediction in predictions.items():
            q = prediction[indices]
            tn = np.count_nonzero(~positive & ~q, axis=1)
            fp = np.count_nonzero(~positive & q, axis=1)
            fn = np.count_nonzero(positive & ~q, axis=1)
            tp = np.count_nonzero(positive & q, axis=1)
            values = counts_metrics(tn, fp, fn, tp)
            for metric in values:
                samples[name][metric][start:end] = values[metric]
    alpha = (1 - settings['confidence_level']) / 2
    intervals = {}
    for name, prediction in predictions.items():
        point = full_metrics(y, probabilities_by_model[name], threshold)
        intervals[name] = {}
        for metric, values in samples[name].items():
            lower, upper = np.quantile(values, [alpha, 1 - alpha], method=settings['quantile_method'])
            intervals[name][metric] = {'lower': float(lower), 'upper': float(upper),
                                        'point_estimate': point[metric]}
    return {'method': settings['method'], 'replicates': count, 'seed': settings['seed'],
            'confidence_level': settings['confidence_level'], 'shared_row_resamples': True,
            'rng': 'numpy.default_rng PCG64; explicit int64 row draws', 'intervals': intervals}


def paired_mcnemar(labels, probabilities_by_model, threshold=.5):
    y = np.asarray(labels)
    base_correct = (np.asarray(probabilities_by_model['baseline']) >= threshold) == y
    rows = []
    for name in ('experiment_1', 'experiment_2'):
        correct = (np.asarray(probabilities_by_model[name]) >= threshold) == y
        b = int(np.count_nonzero(base_correct & ~correct))
        c = int(np.count_nonzero(~base_correct & correct))
        n = b + c
        statistic = (abs(b - c) - 1)**2 / n if n else 0.
        p = float(chi2.sf(statistic, 1)) if n else 1.
        rows.append({'model_a': 'baseline', 'model_b': name,
                     'a_correct_b_wrong': b, 'a_wrong_b_correct': c, 'discordant_pairs': n,
                     'chi_square_statistic': float(statistic), 'raw_p_value': p,
                     'continuity_correction': True})
    order = np.argsort([row['raw_p_value'] for row in rows])
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(rows) - rank) * rows[index]['raw_p_value']))
        rows[index]['holm_adjusted_p_value'] = running
        rows[index]['reject_after_holm_at_0_05'] = running < .05
    return rows


def robustness_metrics(labels, probabilities, masks, model, threshold=.5):
    y, p = validate_predictions(labels, probabilities)
    rows = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        count = int(mask.sum())
        if count:
            predicted = p[mask] >= threshold
            _, _, f1, _ = precision_recall_fscore_support(y[mask], predicted, labels=[0, 1],
                                                         average='macro', zero_division=0)
            error = float(np.mean(predicted != y[mask]))
        else:
            f1 = error = None
        rows.append({'model': model, 'slice': name, 'count': count, 'fraction': count / len(y),
                     'macro_f1': float(f1) if f1 is not None else None, 'error_rate': error})
    return rows
