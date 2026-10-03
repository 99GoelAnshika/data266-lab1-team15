"""Restore approved data backups without downloading or refitting preprocessing."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
from runtime_utils import read_json, write_json, sha256, fingerprint, data_contract, MODEL_NAMES

FIELDS = ('input_ids', 'labels', 'source_indices', 'lengths', 'processed_lengths',
          'oov_counts', 'oov_rates', 'contains_negation')


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Unsafe ZIP path: ' + info.filename)
        if bundle.testzip() is not None:
            raise ValueError('ZIP CRC failed.')
        bundle.extractall(destination)


def verify_preprocessing(root):
    root = Path(root)
    m = read_json(root / 'outputs/metrics/data_artifact_manifest.json')
    paths = {'data_protocol_sha256': 'configs/data_protocol.json',
             'split_manifest_sha256': 'outputs/metrics/split_manifest.json',
             'vocabulary_sha256': 'outputs/metrics/vocabulary.json',
             'preprocessing_helper_sha256': 'src/yelp_data.py',
             'evaluation_protocol_sha256': 'configs/evaluation_protocol.json',
             'robustness_slices_sha256': 'configs/robustness_slices.json'}
    for key, relative in paths.items():
        if sha256(root / relative) != m[key]:
            raise RuntimeError('Frozen preprocessing file changed: ' + relative)
    c = read_json(root / 'configs/data_protocol.json')
    if c['preprocessing_revision'] != 2:
        raise RuntimeError('Use the approved preprocessing revision 2.')
    if sha256(root / 'configs/english_stopwords.json') != c['effective_stopwords_sha256']:
        raise RuntimeError('Stopwords snapshot changed.')
    complete = read_json(root / 'outputs/metrics/preprocessing_completion.json')
    if complete['status'] != 'PASS' or complete['official_test_evaluated']:
        raise RuntimeError('Unexpected preprocessing stage state.')
    return m


def locate_backup(drive_base, run_id, filename):
    run_folder = Path(drive_base) / run_id
    direct = run_folder / filename
    if direct.is_file():
        return direct
    matches = list(run_folder.rglob(filename)) if run_folder.exists() else []
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one backup named {filename} under {run_folder}. '
                                'Keep the existing preprocessing Drive folders intact.')
    return matches[0]


def copy_verified_archive(source, cache_root, record):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    local = cache_root / record['zip_filename']
    if not local.is_file() or sha256(local) != record['sha256']:
        shutil.copy2(source, local)
    if sha256(local) != record['sha256']:
        raise RuntimeError('Backup SHA-256 mismatch; do not train from this ZIP.')
    return local


def restore_development(root, drive_base, cache_root):
    root, cache_root = Path(root), Path(cache_root)
    manifest = verify_preprocessing(root)
    processed = cache_root / 'processed/Viraat_Chaudhary'
    if not (processed / 'train_input_ids.npy').exists():
        record = read_json(root / 'outputs/metrics/processed_dataset_backup.json')
        if record['data_artifact_manifest_sha256'] != data_contract(root):
            raise RuntimeError('Processed backup refers to different preprocessing.')
        source = locate_backup(drive_base, manifest['run_id'], record['zip_filename'])
        archive = copy_verified_archive(source, cache_root, record)
        safe_extract(archive, cache_root)
    for name in ('train', 'validation'):
        for filename, expected in manifest['encoded_splits'][name]['files_sha256'].items():
            if not (processed / filename).is_file() or sha256(processed / filename) != expected:
                raise RuntimeError('Processed array integrity failed: ' + filename)
    arrays = {name: load_arrays(processed, name) for name in ('train', 'validation')}
    split = read_json(root / 'outputs/metrics/split_manifest.json')
    for name, key in [('train', 'training_index_sha256'), ('validation', 'validation_index_sha256')]:
        a = arrays[name]
        expected = manifest['encoded_splits'][name]
        if list(a['input_ids'].shape) != expected['input_shape']:
            raise RuntimeError('Incorrect input shape: ' + name)
        if np.bincount(a['labels'], minlength=2).tolist() != expected['class_counts']:
            raise RuntimeError('Incorrect class counts: ' + name)
        digest = hashlib.sha256(np.ascontiguousarray(a['source_indices']).tobytes()).hexdigest()
        if digest != split[key] or not np.all(np.diff(a['source_indices']) > 0):
            raise RuntimeError('Incorrect source-row membership/order: ' + name)
        if not np.all((a['lengths'] >= 1) & (a['lengths'] <= a['input_ids'].shape[1])):
            raise RuntimeError('Invalid encoded lengths.')
    if np.intersect1d(arrays['train']['source_indices'], arrays['validation']['source_indices']).size:
        raise RuntimeError('Training and validation overlap.')
    return arrays, processed


def load_arrays(directory, name):
    return {field: np.load(Path(directory) / f'{name}_{field}.npy', mmap_mode='r') for field in FIELDS}


def batch_iterator(arrays, batch_size, device, seed=None):
    import torch
    n = len(arrays['labels'])
    order = np.arange(n) if seed is None else np.random.default_rng(seed).permutation(n)
    for start in range(0, n, batch_size):
        rows = order[start:start + batch_size]
        lengths = np.asarray(arrays['lengths'][rows], dtype=np.int64)
        width = min(arrays['input_ids'].shape[1], ((int(lengths.max()) + 7) // 8) * 8)
        ids = torch.from_numpy(np.asarray(arrays['input_ids'][rows, :width], dtype=np.int64))
        lens = torch.from_numpy(lengths)
        labels = torch.from_numpy(np.asarray(arrays['labels'][rows], dtype=np.float32))
        if device.type == 'cuda':
            ids, lens, labels = ids.pin_memory(), lens.pin_memory(), labels.pin_memory()
        yield rows, ids.to(device, non_blocking=True), lens.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def require_frozen_models(root):
    root = Path(root)
    frozen_path = root / 'outputs/metrics/frozen_models.json'
    if not frozen_path.is_file():
        raise RuntimeError('Freeze all three validation-selected checkpoints before opening test data.')
    frozen = read_json(frozen_path)
    for name in MODEL_NAMES:
        item = frozen['models'][name]
        if sha256(root / item['checkpoint']) != item['checkpoint_sha256']:
            raise RuntimeError('Frozen checkpoint changed: ' + name)
        if sha256(root / item['config']) != item['config_sha256']:
            raise RuntimeError('Frozen configuration changed: ' + name)
    if frozen['data_contract'] != data_contract(root):
        raise RuntimeError('Frozen model/data contract changed.')
    return frozen


def restore_raw(root, drive_base, cache_root):
    from datasets import load_from_disk
    root, cache_root = Path(root), Path(cache_root)
    manifest = verify_preprocessing(root)
    record = read_json(root / 'outputs/metrics/dataset_backup_manifest.json')
    destination = cache_root / 'raw'
    dataset_root = destination / 'yelp_polarity'
    if not (dataset_root / 'dataset_dict.json').exists():
        source = locate_backup(drive_base, manifest['raw_source_run_id'], record['zip_filename'])
        archive = copy_verified_archive(source, cache_root, record)
        safe_extract(archive, destination)
    marker = destination / 'restored_raw_integrity.json'
    if marker.exists():
        checks = read_json(marker)
        if checks['archive_sha256'] != record['sha256']:
            raise RuntimeError('Raw cache belongs to a different backup.')
        for relative, expected in checks['files_sha256'].items():
            if sha256(dataset_root / relative) != expected:
                raise RuntimeError('Raw restored file changed: ' + relative)
    else:
        checks = {str(p.relative_to(dataset_root)): sha256(p)
                  for p in dataset_root.rglob('*') if p.is_file()}
        write_json(marker, {'archive_sha256': record['sha256'], 'files_sha256': checks})
    dataset = load_from_disk(str(dataset_root))
    if len(dataset['train']) != 560000 or len(dataset['test']) != 38000:
        raise RuntimeError('Official raw split counts differ.')
    return dataset


def encode_official_test(root, drive_base, cache_root):
    from yelp_data import build_preprocessor, encode_tokens
    root, cache_root = Path(root), Path(cache_root)
    frozen = require_frozen_models(root)
    vocabulary = read_json(root / 'outputs/metrics/vocabulary.json')['token_to_index']
    protocol = read_json(root / 'configs/data_protocol.json')
    key = fingerprint({'frozen_data_contract': frozen['data_contract'], 'split': 'official_test'})
    output = cache_root / 'test_encoded'
    marker = output / 'test_encoding_manifest.json'
    if marker.is_file():
        saved = read_json(marker)
        if saved['encoding_key'] != key:
            raise RuntimeError('Test cache belongs to a different frozen preprocessing contract.')
        for filename, expected in saved['files_sha256'].items():
            if sha256(output / filename) != expected:
                raise RuntimeError('Test cache integrity failed: ' + filename)
        return load_arrays(output, 'test'), saved
    dataset = restore_raw(root, drive_base, cache_root)['test']
    snapshot = read_json(root / 'configs/english_stopwords.json')
    stopwords = snapshot['words'] if isinstance(snapshot, dict) else snapshot
    tokenize = build_preprocessor(stopwords)
    output.mkdir(parents=True, exist_ok=True)
    n, width = len(dataset), protocol['sequence_length']
    specifications = {'input_ids': ('int32', (n, width)), 'labels': ('int8', (n,)),
                      'source_indices': ('int32', (n,)), 'lengths': ('int32', (n,)),
                      'processed_lengths': ('int32', (n,)), 'oov_counts': ('int32', (n,)),
                      'oov_rates': ('float64', (n,)), 'contains_negation': ('bool', (n,))}
    arrays = {k: np.lib.format.open_memmap(output / f'test_{k}.npy', mode='w+', dtype=d, shape=s)
              for k, (d, s) in specifications.items()}
    row = 0
    for batch in dataset.iter(batch_size=2048):
        for text, label in zip(batch['text'], batch['label']):
            if not isinstance(text, str) or label not in (0, 1):
                raise ValueError('Unexpected malformed test row; do not silently drop it.')
            ids, features = encode_tokens(tokenize(text), vocabulary, width)
            arrays['input_ids'][row], arrays['labels'][row] = ids, label
            arrays['source_indices'][row] = row
            for field, value in features.items():
                arrays[field][row] = value
            row += 1
    if row != 38000 or np.bincount(arrays['labels'], minlength=2).tolist() != [19000, 19000]:
        raise RuntimeError('Official Yelp test count or label mapping is incorrect.')
    for a in arrays.values():
        a.flush()
        a._mmap.close()
    arrays.clear()
    saved = {'encoding_key': key, 'rows': n, 'frozen_models_sha256': sha256(root / 'outputs/metrics/frozen_models.json'),
             'data_contract': frozen['data_contract'], 'fit_performed': False,
             'files_sha256': {f'test_{k}.npy': sha256(output / f'test_{k}.npy') for k in FIELDS}}
    write_json(marker, saved)
    write_json(root / 'outputs/metrics/official_test_encoding_record.json', saved)
    return load_arrays(output, 'test'), saved
