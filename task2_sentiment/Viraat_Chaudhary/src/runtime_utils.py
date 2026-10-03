"""Run records and integrity utilities. Assistant-assisted implementation."""
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path

MODEL_NAMES = ('baseline', 'experiment_1', 'experiment_2')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':')).encode()).hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_event(path, event, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'time_utc': utc_now(), 'event': event, **fields},
                                ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + '.part')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def member_root():
    return Path(__file__).resolve().parent.parent


def source_fingerprint(root):
    names = ('models.py', 'training.py', 'data_runtime.py', 'metrics.py',
             'runtime_utils.py', 'yelp_data.py', 'evaluation.py', 'review.py', 'evaluator.py')
    return fingerprint({name: sha256(Path(root) / 'src' / name) for name in names})


def data_contract(root):
    return sha256(Path(root) / 'outputs/metrics/data_artifact_manifest.json')


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA algorithms/platforms can still differ numerically. Record the flags.


def environment():
    import torch
    cpu = platform.processor() or platform.machine()
    proc = Path('/proc/cpuinfo')
    if proc.exists():
        for line in proc.read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'nltk',
                 'datasets', 'matplotlib', 'tqdm'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not-installed'
    return {'captured_utc': utc_now(), 'python': platform.python_version(),
            'platform': platform.platform(), 'cpu_model': cpu,
            'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_total_memory_mib': (torch.cuda.get_device_properties(0).total_memory / 2**20
                                     if torch.cuda.is_available() else None),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'packages': packages}


def dependency_snapshot(path):
    packages = sorted({f"{d.metadata['Name']}=={d.version}"
                       for d in importlib.metadata.distributions() if d.metadata['Name']})
    Path(path).write_text('\n'.join(packages) + '\n', encoding='utf-8')


def cpu_peak_mib():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(value / (2**20 if sys.platform == 'darwin' else 1024))
    except ImportError:
        return None


def collect_design_notes(root, supplied=None):
    """Collect the student's words; never generate justifications or findings."""
    path = Path(root) / 'student_design_notes.json'
    if path.exists() and supplied is None:
        existing = read_json(path)
        required = ('baseline_reason', 'experiment_1_reason', 'experiment_2_reason',
                    'embedding_reason', 'hyperparameter_reason', 'comparison_question')
        if any(len(str(existing.get(k, '')).strip()) < 15 for k in required):
            raise ValueError('Complete the student design explanations before training.')
        return existing
    prompts = {
        'baseline_reason': 'Why did you select a plain RNN as your baseline?',
        'experiment_1_reason': 'What do you want the LSTM comparison to test?',
        'experiment_2_reason': 'What do you want the TCN comparison to test?',
        'embedding_reason': 'Why use these learned-from-scratch embeddings and their size?',
        'hyperparameter_reason': 'Explain your batch size, learning rate and stopping choices.',
        'comparison_question': 'What accuracy-versus-time question will your experiments answer?'}
    notes = dict(supplied or {})
    for key, prompt in prompts.items():
        if not str(notes.get(key, '')).strip():
            notes[key] = input(prompt + '\nYour explanation: ').strip()
        if len(notes[key].strip()) < 15 or notes[key].upper().startswith(('TODO', 'TBD', 'WRITE')):
            raise ValueError('Write a substantive explanation in your own words: ' + key)
    notes['written_by'] = 'Viraat; user-entered explanations'
    notes['captured_utc'] = utc_now()
    write_json(path, notes)
    return notes
