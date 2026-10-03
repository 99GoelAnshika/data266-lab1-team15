"""Traceable result tables, student-analysis slots and dataset-free exports."""
import json
import zipfile
from pathlib import Path
from runtime_utils import MODEL_NAMES, read_csv, read_json, sha256, write_csv, write_json, utc_now
from review import ANALYSIS_FIELDS, write_failure_analysis


def markdown_table(rows, fields):
    def value(v):
        if v is None or v == '':
            return '—'
        if isinstance(v, float):
            return f'{v:.6g}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |']
    lines += ['| ' + ' | '.join(value(row.get(k)) for k in fields) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def team_comparison(root):
    root = Path(root)
    own = read_csv(root / 'metrics_report.csv')
    reference = read_json(root / 'references/anshika_reported_evaluation.json')
    rows = list(own)
    for name, metrics in reference['metrics'].items():
        row = {'member': 'Anshika_Goel', 'model': name,
               'architecture': reference['model_configs'][name]['architecture'],
               'hyperparameters': json.dumps(reference['model_configs'][name], sort_keys=True),
               'evidence_status': 'reported_in_uploaded_executed_teammate_notebook',
               **metrics, 'cpu_model': None, 'gpu_model': None,
               'resource_comparison_note': 'Different/unsupplied exact hardware and timing aggregation; '
                                           'do not attribute cross-member speed differences to architecture alone.'}
        cm = row.pop('confusion_matrix')
        row.update({'tn': cm[0][0], 'fp': cm[0][1], 'fn': cm[1][0], 'tp': cm[1][1]})
        row['pr_auc_average_precision'] = row.pop('pr_auc')
        resource = reference['resources'][name]
        row.update({'total_parameter_count': resource['parameter_count'],
                    'training_wall_seconds': resource['training_seconds'],
                    'training_examples_per_second': resource['mean_train_examples_per_second'],
                    'peak_cuda_allocated_mib': resource['peak_gpu_allocated_mb'],
                    'peak_cuda_reserved_mib': resource['peak_gpu_reserved_mb']})
        for metric, item in reference['bootstrap_intervals'][name].items():
            metric = 'f1_macro' if metric == 'macro_f1' else metric
            row[metric + '_ci_lower'], row[metric + '_ci_upper'] = item['lower'], item['upper']
        for pair in reference['mcnemar_tests'].values():
            if pair['model_b'] == name:
                row.update({'mcnemar_vs_baseline_raw_p': pair['raw_p_value'],
                            'mcnemar_vs_baseline_holm_p': pair['holm_adjusted_p_value'],
                            'mcnemar_vs_baseline_statistic': pair['chi_square_statistic']})
        for s in reference['slice_metrics']:
            if s['model'] == name:
                for metric in ('count', 'fraction', 'macro_f1', 'error_rate'):
                    row[s['slice'] + '_' + metric] = s[metric]
        row['slice_comparison_note'] = 'Anshika tokenizer/slice membership differs; slice values are reported at displayed precision.'
        rows.append(row)
    # Optional student-supplied hardware details; never infer a teammate device.
    hardware_path = root / 'references/anshika_hardware.json'
    if hardware_path.exists():
        hardware = read_json(hardware_path)
        for row in rows:
            if row['member'] == 'Anshika_Goel' and row['model'] in hardware:
                for key in ('cpu_model', 'gpu_model'):
                    row[key] = hardware[row['model']].get(key)
    write_csv(root / 'outputs/metrics/team_comparison_metrics.csv', rows)
    return rows


def refresh_reports(root):
    root = Path(root)
    designs = read_json(root / 'student_design_notes.json') if (root / 'student_design_notes.json').exists() else {}
    analysis = read_json(root / 'student_analysis.json') if (root / 'student_analysis.json').exists() else {}
    text = '# Task 2 — Viraat Chaudhary\n\n'
    text += 'Yelp Polarity; embeddings and all classifier weights learned from scratch.\n\n'
    text += 'Status: ' + ('evaluated; student review/analysis status below' if (root / 'metrics_report.csv').exists()
                          else 'waiting for Colab training and final evaluation') + '.\n\n'
    text += '## Design choices\n\n'
    for name, key in [('baseline', 'baseline_reason'), ('experiment_1', 'experiment_1_reason'),
                      ('experiment_2', 'experiment_2_reason')]:
        config = read_json(root / f'configs/{name}.json')
        text += f'### {name}\n\n' + (designs.get(key) or 'PENDING student justification') + '\n\n'
        text += 'Configuration: ' + json.dumps(config, sort_keys=True) + '\n\n'
    for key in ('embedding_reason', 'hyperparameter_reason', 'comparison_question'):
        text += key.replace('_', ' ').capitalize() + ': ' + (designs.get(key) or 'PENDING student explanation') + '\n\n'
    text += '## Preprocessing evidence\n\n'
    a = read_json(root / 'outputs/metrics/processed_training_analysis.json')
    text += 'Verified development preprocessing statistics:\n\n' + '```json\n' + json.dumps(a, indent=2) + '\n```\n\n'
    if (root / 'metrics_report.csv').exists():
        own = read_csv(root / 'metrics_report.csv')
        text += '## Final official-test metrics\n\n'
        text += 'All fields, including precision/recall/F1 averages, confidence intervals, paired tests, slices and resources, are in metrics_report.csv.\n\n'
        text += markdown_table(own, ['model', 'accuracy', 'f1_macro', 'mcc', 'roc_auc',
                                    'pr_auc_average_precision', 'brier_score', 'ece_15_bins']) + '\n'
        text += markdown_table(own, ['model', 'total_parameter_count', 'training_wall_seconds',
                                    'training_examples_per_second', 'peak_cuda_allocated_mib',
                                    'cpu_model', 'gpu_model']) + '\n'
        text += 'Checkpoint-to-config/result provenance: outputs/metrics/checkpoint_result_mapping.json.\n\n'
        for name in MODEL_NAMES:
            text += f"- {name}: [training curves](outputs/plots/{name}_training_curves.png), "
            text += f"[confusion matrix](outputs/plots/{name}_confusion_matrix.png), "
            text += f"[ROC/PR](outputs/plots/{name}_roc_pr.png), "
            text += f"[calibration](outputs/plots/{name}_reliability.png), "
            text += f"[raw log](logs/{name}_training.jsonl).\n"
        text += '\n'
        combined = team_comparison(root)
        text += '## Team comparison\n\n'
        text += markdown_table(combined, ['member', 'model', 'accuracy', 'f1_macro', 'mcc',
                                         'roc_auc', 'pr_auc_average_precision', 'brier_score', 'ece_15_bins']) + '\n'
        text += 'Anshika values are transcribed from her uploaded executed evaluation notebook. Her code is not included. Overall official-test metrics share the population; tokenizer-dependent slices differ. Exact teammate CPU/GPU details require her hardware manifest. Cross-member runtime is not a controlled architecture benchmark.\n\n'
    text += '## Student-written findings\n\n'
    for key in ANALYSIS_FIELDS:
        text += '### ' + key.replace('_', ' ').capitalize() + '\n\n'
        text += (analysis.get(key) or 'PENDING: Viraat writes this after reviewing the actual evidence.') + '\n\n'
    text += '## Evidence and references\n\n'
    text += 'Original raw logs are preserved. Training checkpoints are selected on validation; test predictions are cached and never used for retuning. Twenty-error evidence and student fixes are in failure_analysis.md.\n\n'
    text += '- Bai, Kolter and Koltun (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling: https://arxiv.org/abs/1803.01271\n'
    text += '- PyTorch RNN/LSTM documentation: https://docs.pytorch.org/docs/2.11/generated/torch.nn.LSTM.html\n'
    text += '- Yelp source: https://huggingface.co/datasets/fancyzhx/yelp_polarity\n\n'
    text += 'Assistant assistance is disclosed in AI_use.md; student justifications and annotations are user-entered.\n'
    (root / 'results.md').write_text(text, encoding='utf-8')
    write_failure_analysis(root)
    (root / 'REPORT_TASK2_SECTION.md').write_text(text, encoding='utf-8')
    if (root / 'metrics_report.csv').exists():
        combined = read_csv(root / 'outputs/metrics/team_comparison_metrics.csv')
        snippet = '## Task 2 — Viraat findings\n\n'
        snippet += 'Append this section below the existing Anshika findings in the shared README.\n\n'
        snippet += markdown_table([r for r in combined if r['member'] == 'Viraat_Chaudhary'],
                                  ['model', 'accuracy', 'f1_macro', 'mcc', 'roc_auc',
                                   'pr_auc_average_precision', 'brier_score', 'ece_15_bins'])
        snippet += '\n' + (analysis.get('own_model_comparison') or 'PENDING student-written interpretation.') + '\n'
        (root / 'SHARED_README_APPEND.md').write_text(snippet, encoding='utf-8')


def submission_audit(root):
    root = Path(root)
    required = ['student_design_notes.json', 'student_analysis.json', 'metrics_report.csv',
                'failure_analysis.md', 'results.md', 'outputs/metrics/final_evaluation_completed.json',
                'outputs/metrics/checkpoint_result_mapping.json', 'outputs/metrics/bootstrap_confidence_intervals.json',
                'outputs/metrics/mcnemar_tests.json', 'outputs/metrics/robustness_slice_metrics.json',
                'outputs/errors/error_annotations_completed.csv', 'outputs/errors/manual_review_completion.json',
                'outputs/metrics/evaluator_smoke.json']
    required += [f'checkpoints/{n}_best.pt' for n in MODEL_NAMES]
    required += [f'logs/{n}_training.jsonl' for n in MODEL_NAMES]
    missing = [p for p in required if not (root / p).is_file()]
    notebook_missing = []
    errors = []
    for path in sorted((root / 'notebooks').glob('*.ipynb')):
        notebook = read_json(path)
        code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        if any(c.get('execution_count') is None for c in code):
            notebook_missing.append(path.name)
        if any(o.get('output_type') == 'error' for c in code for o in c.get('outputs', [])):
            errors.append(path.name)
    pending_analysis = not (root / 'student_analysis.json').exists()
    if not pending_analysis:
        analysis = read_json(root / 'student_analysis.json')
        pending_analysis = any(len(str(analysis.get(k, '')).strip()) < 25 for k in ANALYSIS_FIELDS)
    files_too_large = [str(p.relative_to(root)) for p in (root / 'checkpoints').glob('*_best.pt')
                      if p.stat().st_size >= 100 * 2**20]
    report = {'task2_individual_ready_for_review': not missing and not pending_analysis,
              'individual_submission_files_complete':
                  not missing and not pending_analysis and not notebook_missing and not errors and not files_too_large,
              'full_lab_submission_ready': False,
              'missing_required_files': missing, 'student_analysis_pending': pending_analysis,
              'executed_notebook_copies_still_needed': notebook_missing,
              'notebooks_with_saved_errors': errors, 'checkpoints_over_github_limit': files_too_large,
              'remaining_team_work': ['confirm teammate exact hardware/provenance',
                                      'jointly write task comparison and integrate combined lab PDF report'],
              'datasets_in_submission_bundle': False, 'created_utc': utc_now()}
    write_json(root / 'outputs/metrics/submission_audit.json', report)
    return report


def export_artifacts(root, label, include_checkpoints=False, destination=None):
    root = Path(root)
    destination = Path(destination) if destination else root / 'exports'
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f'Viraat_Task2_{label}.zip'
    manifest = []
    allowed_checkpoints = {f'checkpoints/{name}_best.pt' for name in MODEL_NAMES}
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in ('exports', '__pycache__', 'data_processed', 'preprocessing_cache') for part in relative.parts):
            continue
        if path.suffix in ('.pyc', '.part') or relative.parts[0] == 'data':
            continue
        if relative.parts[0] == 'checkpoints' and (not include_checkpoints or str(relative) not in allowed_checkpoints):
            continue
        files.append((path, relative))
        manifest.append({'path': str(relative), 'sha256': sha256(path), 'bytes': path.stat().st_size})
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as bundle:
        for path, relative in files:
            bundle.write(path, 'task2_sentiment/Viraat_Chaudhary/' + str(relative))
        bundle.writestr('EXPORT_MANIFEST.json', json.dumps({'label': label, 'created_utc': utc_now(),
                         'contains_best_checkpoints': include_checkpoints, 'contains_datasets': False,
                         'files': manifest}, indent=2) + '\n')
    return target
