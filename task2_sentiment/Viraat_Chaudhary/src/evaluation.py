"""Frozen final-test inference; repeat statistics reuse saved probabilities."""
import time
from pathlib import Path
import numpy as np
import torch
from data_runtime import encode_official_test, require_frozen_models
from models import build_model
from training import freeze_models, predict_arrays, sync
from metrics import full_metrics, bootstrap_intervals, paired_mcnemar, robustness_metrics, calibration_bins
from runtime_utils import (MODEL_NAMES, append_event, data_contract, fingerprint, read_json,
                           sha256, source_fingerprint, utc_now, write_csv, write_json)


def plot_evaluation(root, name, labels, probabilities, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    folder = Path(root) / 'outputs/plots'
    folder.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    image = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['negative', 'positive'],
           yticklabels=['negative', 'positive'], xlabel='Predicted', ylabel='True', title=name)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(folder / f'{name}_confusion_matrix.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    axes[0].plot(fpr, tpr, label=f"ROC AUC={metrics['roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate')
    axes[1].plot(recall, precision, label=f"Average precision={metrics['pr_auc_average_precision']:.4f}")
    axes[1].set(xlabel='Recall', ylabel='Precision', ylim=(0, 1.02))
    for ax in axes:
        ax.legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(folder / f'{name}_roc_pr.png', dpi=160)
    plt.close(fig)
    ece, bins = calibration_bins(labels, probabilities)
    write_csv(Path(root) / f'outputs/metrics/{name}_calibration_bins.csv', bins)
    occupied = [b for b in bins if b['count']]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.scatter([b['mean_confidence'] for b in occupied], [b['accuracy'] for b in occupied],
               s=[20 + 150 * b['fraction'] for b in occupied])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Predicted-class confidence', ylabel='Accuracy',
           title=f'{name}; ECE={ece:.4f}')
    fig.tight_layout()
    fig.savefig(folder / f'{name}_reliability.png', dpi=160)
    plt.close(fig)


def load_saved_predictions(root, name):
    path = Path(root) / f'outputs/predictions/{name}_official_test.npz'
    with np.load(path, allow_pickle=False) as bundle:
        return {k: bundle[k].copy() for k in bundle.files}


def evaluate_models(root, drive_base=None, cache_root=None, arrays=None):
    from yelp_data import slice_masks
    root = Path(root)
    frozen = freeze_models(root)
    protocol = read_json(root / 'configs/evaluation_protocol.json')
    if frozen['implementation_fingerprint'] != source_fingerprint(root):
        raise RuntimeError('The implementation changed after model selection.')
    seal_path = root / 'outputs/metrics/final_test_started.json'
    seal = {'frozen_models_sha256': sha256(root / 'outputs/metrics/frozen_models.json'),
            'evaluation_protocol_sha256': sha256(root / 'configs/evaluation_protocol.json'),
            'slice_config_sha256': sha256(root / 'configs/robustness_slices.json'),
            'data_contract': data_contract(root)}
    if seal_path.exists():
        if read_json(seal_path)['contract'] != seal:
            raise RuntimeError('Final test contract changed. Do not retune using test results.')
    else:
        write_json(seal_path, {'started_utc': utc_now(), 'contract': seal,
                               'test_policy': 'frozen checkpoints; cached final predictions reused on reruns'})
    # arrays argument exists only for explicit tiny synthetic verification, never notebook use.
    memberships_path = root / 'outputs/predictions/final_test_slice_memberships.npz'
    cached_masks = None
    if arrays is None and memberships_path.exists() and all(
            (root / f'outputs/predictions/{n}_official_test.npz').exists() for n in MODEL_NAMES):
        saved = load_saved_predictions(root, 'baseline')
        arrays = {'labels': saved['labels'], 'source_indices': saved['source_indices']}
        with np.load(memberships_path, allow_pickle=False) as bundle:
            cached_masks = {k: bundle[k].copy() for k in bundle.files}
        completed_path = root / 'outputs/metrics/final_evaluation_completed.json'
        if completed_path.exists() and sha256(memberships_path) != read_json(completed_path)['slice_memberships_sha256']:
            raise RuntimeError('Saved slice membership integrity failed.')
    elif arrays is None:
        arrays, _ = encode_official_test(root, drive_base, cache_root)
    if len(arrays['labels']) != protocol['bootstrap']['replicate_size']:
        raise RuntimeError('Final evaluation count differs from the recorded protocol.')
    expected_source_ids = np.arange(len(arrays['labels']), dtype=np.int32)
    if not np.array_equal(arrays['source_indices'], expected_source_ids):
        raise RuntimeError('Final test source rows must be in official ascending order.')
    labels = np.asarray(arrays['labels'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions, metrics, report_rows = {}, {}, []
    for name in MODEL_NAMES:
        record = frozen['models'][name]
        pred_path = root / f'outputs/predictions/{name}_official_test.npz'
        pred_manifest_path = root / f'outputs/metrics/{name}_test_inference.json'
        identity = fingerprint({'seal': seal, 'model': name, 'checkpoint': record['checkpoint_sha256']})
        if pred_path.exists():
            saved = load_saved_predictions(root, name)
            if str(saved['identity'].item()) != identity:
                raise RuntimeError('Saved test probabilities refer to a different checkpoint/contract.')
            if not np.array_equal(saved['source_indices'], expected_source_ids) or not np.array_equal(saved['labels'], labels):
                raise RuntimeError('Saved test row IDs or labels changed.')
            if pred_manifest_path.exists() and sha256(pred_path) != read_json(pred_manifest_path)['predictions_sha256']:
                raise RuntimeError('Saved predictions hash mismatch.')
            p, seconds = saved['probabilities'], float(saved['inference_seconds'].item())
            print(name + ': saved final predictions reused; no test inference repeated.')
        else:
            config = read_json(root / record['config'])
            checkpoint = torch.load(root / record['checkpoint'], map_location=device, weights_only=True)
            if checkpoint['identity'] != record['identity']:
                raise RuntimeError('Checkpoint identity mismatch.')
            model = build_model(config, checkpoint['vocabulary_size']).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            sync(device)
            start = time.perf_counter()
            p, _ = predict_arrays(model, arrays, config, device)
            sync(device)
            seconds = time.perf_counter() - start
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = pred_path.with_name(pred_path.name + '.part')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, labels=labels, source_indices=expected_source_ids,
                                    probabilities=p, identity=np.asarray(identity),
                                    inference_seconds=np.asarray(seconds))
            temporary.replace(pred_path)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        write_json(pred_manifest_path, {'identity': identity, 'checkpoint_sha256': record['checkpoint_sha256'],
                                        'predictions_sha256': sha256(pred_path),
                                        'rows': len(labels), 'inference_seconds': seconds,
                                        'test_examples_per_second': len(labels) / seconds})
        predictions[name] = p
        values = full_metrics(labels, p, protocol['classification_threshold'], protocol['ece']['bins'])
        values.update({'inference_seconds': seconds, 'test_examples_per_second': len(labels) / seconds})
        metrics[name] = values
        plot_evaluation(root, name, labels, p, values)
    ci = bootstrap_intervals(labels, predictions, protocol)
    tests = paired_mcnemar(labels, predictions, protocol['classification_threshold'])
    masks = cached_masks or slice_masks(arrays['lengths'], arrays['oov_rates'], arrays['contains_negation'],
                                        read_json(root / 'configs/robustness_slices.json'))
    if not np.all(sum(masks[k].astype(np.int8) for k in ('short_reviews', 'medium_reviews', 'long_reviews')) == 1):
        raise RuntimeError('Length slices do not partition official test rows.')
    slice_rows = [r for name in MODEL_NAMES for r in robustness_metrics(labels, predictions[name], masks, name)]
    np.savez_compressed(memberships_path, **masks)
    resource_keys = ('total_parameter_count', 'trainable_parameter_count', 'training_wall_seconds',
                     'training_phase_seconds', 'training_examples_per_second', 'peak_cuda_allocated_mib',
                     'peak_cuda_reserved_mib', 'cpu_process_peak_rss_mib', 'completed_epochs', 'selected_epoch')
    for name in MODEL_NAMES:
        training = frozen['models'][name]
        config = read_json(root / training['config'])
        row = {'member': 'Viraat_Chaudhary', 'model': name, 'architecture': config['architecture'],
               'evidence_status': 'computed_from_frozen_checkpoints',
               'hyperparameters': json_config(config), **metrics[name],
               **{k: training[k] for k in resource_keys},
               'cpu_model': training['hardware']['cpu_model'], 'gpu_model': training['hardware']['gpu_model'],
               'checkpoint': training['checkpoint'], 'checkpoint_sha256': training['checkpoint_sha256']}
        cm = row.pop('confusion_matrix')
        row.update({'tn': cm[0][0], 'fp': cm[0][1], 'fn': cm[1][0], 'tp': cm[1][1]})
        for metric, values in ci['intervals'][name].items():
            row[metric + '_ci_lower'], row[metric + '_ci_upper'] = values['lower'], values['upper']
        paired = next((t for t in tests if t['model_b'] == name), None)
        if paired:
            row.update({'mcnemar_vs_baseline_raw_p': paired['raw_p_value'],
                        'mcnemar_vs_baseline_holm_p': paired['holm_adjusted_p_value'],
                        'mcnemar_vs_baseline_statistic': paired['chi_square_statistic']})
        for slice_row in [r for r in slice_rows if r['model'] == name]:
            for key in ('macro_f1', 'error_rate', 'count', 'fraction'):
                row[slice_row['slice'] + '_' + key] = slice_row[key]
        report_rows.append(row)
    metric_dir = root / 'outputs/metrics'
    write_json(metric_dir / 'final_test_metrics.json', metrics)
    write_json(metric_dir / 'bootstrap_confidence_intervals.json', ci)
    write_json(metric_dir / 'mcnemar_tests.json', {'family': 'baseline versus two experiments',
                                                'adjustment': 'Holm', 'tests': tests})
    write_csv(metric_dir / 'mcnemar_tests.csv', tests)
    write_json(metric_dir / 'robustness_slice_metrics.json', slice_rows)
    write_csv(metric_dir / 'robustness_slice_metrics.csv', slice_rows)
    write_csv(root / 'metrics_report.csv', report_rows)
    write_csv(root / 'outputs/predictions/final_test_predictions.csv',
              ({'source_index': int(expected_source_ids[i]), 'label': int(labels[i]),
                **{name + '_probability_positive': float(predictions[name][i]) for name in MODEL_NAMES},
                **{name + '_prediction': int(predictions[name][i] >= .5) for name in MODEL_NAMES}}
               for i in range(len(labels))))
    write_json(metric_dir / 'checkpoint_result_mapping.json',
               {'frozen_models_sha256': seal['frozen_models_sha256'],
                'models': {name: {'checkpoint': frozen['models'][name]['checkpoint'],
                                  'checkpoint_sha256': frozen['models'][name]['checkpoint_sha256'],
                                  'config': frozen['models'][name]['config'],
                                  'config_sha256': frozen['models'][name]['config_sha256'],
                                  'validation_manifest': f'outputs/metrics/{name}_training_manifest.json',
                                  'test_metrics': f'outputs/metrics/final_test_metrics.json#{name}',
                                  'predictions': f'outputs/predictions/{name}_official_test.npz'}
                           for name in MODEL_NAMES}})
    append_event(root / 'logs/final_evaluation.jsonl', 'statistics_completed_from_saved_predictions',
                 rows=len(labels), bootstrap_replicates=protocol['bootstrap']['replicates'],
                 frozen_models_sha256=seal['frozen_models_sha256'], test_based_selection=False)
    write_json(metric_dir / 'final_evaluation_completed.json',
               {'status': 'completed', 'rows': len(labels), 'completed_utc': utc_now(),
                'slice_memberships_sha256': sha256(memberships_path),
                'manual_review_model': frozen['manual_review_model'],
                'selected_using': 'validation before test access', 'test_based_retraining': False})
    return report_rows


def json_config(config):
    import json
    return json.dumps(config, sort_keys=True, separators=(',', ':'))
