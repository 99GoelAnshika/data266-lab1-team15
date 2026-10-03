"""Select twenty real errors and validate student-entered annotations."""
from pathlib import Path
import numpy as np
from evaluation import load_saved_predictions
from data_runtime import require_frozen_models, restore_raw
from runtime_utils import read_csv, read_json, write_csv, write_json, fingerprint, utc_now

GROUPS = ('confident_false_positive', 'confident_false_negative',
          'near_threshold_error', 'slice_specific_failure')
HUMAN_FIELDS = ('error_type', 'explanation', 'testable_fix')


def choose_errors(labels, probabilities, source_ids, masks):
    y, p, source_ids = np.asarray(labels), np.asarray(probabilities), np.asarray(source_ids)
    predicted = p >= .5
    errors = predicted != y
    chosen, used = [], set()

    def pick(group, candidates, priority, count=5, slice_name=''):
        ordered = sorted(np.flatnonzero(candidates), key=lambda i: (float(priority[i]), int(source_ids[i])))
        selected = [i for i in ordered if int(i) not in used][:count]
        if len(selected) != count:
            raise RuntimeError(f'Not enough unique real errors for {group}. '
                               'Report the actual counts; do not invent or duplicate examples.')
        for i in selected:
            used.add(int(i))
            chosen.append({'row': int(i), 'group': group, 'selected_slice': slice_name})

    pick(GROUPS[0], errors & (y == 0), -p)
    pick(GROUPS[1], errors & (y == 1), p)
    pick(GROUPS[2], errors, np.abs(p - .5))
    # Prefer one example per available frozen slice; fill from remaining slice errors if needed.
    for slice_name in ('high_oov_rate', 'contains_negation', 'long_reviews',
                       'medium_reviews', 'short_reviews'):
        available = errors & np.asarray(masks[slice_name])
        available = available.copy()
        if used:
            available[list(used)] = False
        if available.any():
            pick(GROUPS[3], available, -np.maximum(p, 1 - p), count=1, slice_name=slice_name)
    remaining = 5 - sum(c['group'] == GROUPS[3] for c in chosen)
    if remaining:
        available = errors & np.logical_or.reduce(list(masks.values()))
        start = len(chosen)
        pick(GROUPS[3], available, -np.maximum(p, 1 - p), count=remaining)
        for c in chosen[start:]:
            c['selected_slice'] = next(k for k, mask in masks.items() if mask[c['row']])
    assert len(chosen) == 20 and len(used) == 20
    return chosen


def prepare_review(root, drive_base=None, cache_root=None, test_dataset=None, masks=None):
    root = Path(root)
    if not (root / 'outputs/metrics/final_evaluation_completed.json').is_file():
        raise RuntimeError('Finish notebook 05 before reviewing official test errors.')
    frozen = require_frozen_models(root)
    name = frozen['manual_review_model']
    saved = load_saved_predictions(root, name)
    from runtime_utils import sha256
    record = read_json(root / f'outputs/metrics/{name}_test_inference.json')
    if sha256(root / f'outputs/predictions/{name}_official_test.npz') != record['predictions_sha256']:
        raise RuntimeError('Review prediction integrity failed.')
    if masks is None:
        from runtime_utils import sha256
        path = root / 'outputs/predictions/final_test_slice_memberships.npz'
        complete = read_json(root / 'outputs/metrics/final_evaluation_completed.json')
        if sha256(path) != complete['slice_memberships_sha256']:
            raise RuntimeError('Saved slice memberships changed.')
        with np.load(path, allow_pickle=False) as bundle:
            masks = {k: bundle[k].copy() for k in bundle.files}
    selected = choose_errors(saved['labels'], saved['probabilities'], saved['source_indices'], masks)
    if test_dataset is None:
        test_dataset = restore_raw(root, drive_base, cache_root)['test']
    rows = []
    for number, case in enumerate(selected, 1):
        i = case['row']
        source = int(saved['source_indices'][i])
        example = test_dataset[source]
        if int(example['label']) != int(saved['labels'][i]):
            raise RuntimeError('Review text and prediction labels are not aligned.')
        p = float(saved['probabilities'][i])
        rows.append({'error_id': number, 'model': name, 'group': case['group'],
                     'source_index': source, 'true_label': int(saved['labels'][i]),
                     'predicted_label': int(p >= .5), 'probability_positive': p,
                     'predicted_class_confidence': max(p, 1 - p),
                     'distance_from_threshold': abs(p - .5), 'selected_slice': case['selected_slice'],
                     'slice_memberships': ';'.join(k for k, m in masks.items() if m[i]),
                     'review_text': example['text'], **{field: '' for field in HUMAN_FIELDS}})
    selection_path = root / 'outputs/errors/error_selection.csv'
    template_path = root / 'outputs/errors/error_annotations_template.csv'
    if selection_path.exists():
        prior = read_csv(selection_path)
        # Saved source records are immutable. User annotations live in a separate file.
        if fingerprint(prior) != fingerprint([{k: str(v) for k, v in r.items()} for r in rows]):
            raise RuntimeError('Error selection changed; do not replace an existing review silently.')
    else:
        write_csv(selection_path, rows)
    if not template_path.exists():
        write_csv(template_path, rows)
    write_json(root / 'outputs/errors/selection_policy.json',
               {'model': name, 'model_selected_using': 'maximum validation macro-F1 before test',
                'groups': {g: 5 for g in GROUPS}, 'unique_source_ids': 20,
                'confidence_definition': 'five most confident errors within FP/FN groups',
                'near_threshold_definition': 'five closest to 0.5 among remaining errors',
                'slice_policy': 'available frozen slices, unique remaining errors',
                'text_source': 'official test rows matching saved source IDs',
                'human_annotations_generated': False})
    return rows


def import_annotations(root, uploaded_csv):
    root = Path(root)
    selected = read_csv(root / 'outputs/errors/error_selection.csv')
    annotated = read_csv(uploaded_csv)
    if len(annotated) != 20 or len({r['source_index'] for r in annotated}) != 20:
        raise ValueError('Exactly twenty unique source rows are required.')
    lookup = {r['error_id']: r for r in annotated}
    if set(lookup) != {r['error_id'] for r in selected}:
        raise ValueError('Error IDs changed.')
    cleaned = []
    for original in selected:
        row = lookup[original['error_id']]
        for field in original:
            if field not in HUMAN_FIELDS and row.get(field) != original[field]:
                raise ValueError('Do not edit the selected evidence column: ' + field)
        for field in HUMAN_FIELDS:
            text = row.get(field, '').strip()
            if len(text) < (3 if field == 'error_type' else 15) or text.upper().startswith(('TODO', 'TBD', 'WRITE', 'PENDING')):
                raise ValueError(f'Complete your own {field} for error {row["error_id"]}.')
            row[field] = text
        cleaned.append(row)
    if {g: sum(r['group'] == g for r in cleaned) for g in GROUPS} != {g: 5 for g in GROUPS}:
        raise ValueError('The required five-per-group partition changed.')
    write_csv(root / 'outputs/errors/error_annotations_completed.csv', cleaned)
    write_json(root / 'outputs/errors/manual_review_completion.json',
               {'status': 'completed', 'errors': 20, 'unique_rows': 20,
                'annotations_by': 'Viraat, uploaded manual annotations', 'completed_utc': utc_now()})
    write_failure_analysis(root)
    return cleaned


def write_failure_analysis(root):
    root = Path(root)
    complete = root / 'outputs/errors/error_annotations_completed.csv'
    selection = root / 'outputs/errors/error_selection.csv'
    if not selection.exists():
        text = '# Task 2 — Manual error analysis\n\nPending real errors from notebook 06.\n'
    else:
        rows = read_csv(complete if complete.exists() else selection)
        text = '# Task 2 — Manual error analysis\n\n'
        text += ('Completed student annotations.\n\n' if complete.exists()
                 else '**PENDING: enter your own error types, explanations and testable fixes.**\n\n')
        text += 'Twenty unique official test errors, five in each required group. The model was chosen using validation before test access.\n\n'
        for row in rows:
            text += f"## Error {row['error_id']} — {row['group']}\n\n"
            text += f"Model: {row['model']}; official source ID: {row['source_index']}. True={row['true_label']}, predicted={row['predicted_label']}, P(positive)={row['probability_positive']}.\n\n"
            text += 'Slice memberships: ' + row['slice_memberships'] + '\n\n'
            text += 'Actual review:\n\n' + '\n'.join('> ' + line for line in row['review_text'].splitlines()) + '\n\n'
            for field in HUMAN_FIELDS:
                text += field.replace('_', ' ').capitalize() + ': ' + (row[field] or 'PENDING student annotation') + '\n\n'
    (root / 'failure_analysis.md').write_text(text, encoding='utf-8')


ANALYSIS_FIELDS = ('preprocessing_findings', 'own_model_comparison', 'team_model_comparison',
                   'limitations', 'future_work', 'individual_contribution')


def create_analysis_template(root):
    path = Path(root) / 'student_analysis_template.json'
    if not path.exists():
        write_json(path, {k: '' for k in ANALYSIS_FIELDS})
    return path


def import_student_analysis(root, uploaded_json):
    value = read_json(uploaded_json)
    for key in ANALYSIS_FIELDS:
        if len(str(value.get(key, '')).strip()) < 25 or str(value[key]).upper().startswith(('TODO', 'TBD', 'WRITE', 'PENDING')):
            raise ValueError('Write your own evidence-based analysis: ' + key)
    write_json(Path(root) / 'student_analysis.json', {k: value[k].strip() for k in ANALYSIS_FIELDS})
    return value
