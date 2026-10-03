"""Benchmark, detach, inspect and export the lab run without stopping Jupyter."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

MEMBER = Path(__file__).resolve().parents[1]
ROOT = MEMBER.parents[1]
CONFIG = MEMBER / 'configs/cyclegan_viraat.json'
DEFAULT_RUN = 'viraat_resizeconv6_run001'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def matching_process(pid, run_id):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return (stat != 'Z' and os.fsencode(run_id) in args
                and os.fsencode(str(MEMBER / 'src/train.py')) in args)
    except (OSError, ValueError):
        return False


def benchmark():
    import torch
    import train
    from trainer import CycleGANTrainer

    config, digest = train.load_config(CONFIG)
    train.validate_canonical_config(config)
    device = train.resolve_device(mode='full', requested_device='cuda')
    print('GPU:', torch.cuda.get_device_name(device), flush=True)
    a, b = train.resolve_domain_directories(root=ROOT, config=config, mode='full',
                                           domain_a_override=None, domain_b_override=None)
    paths_a, paths_b = train.validate_dataset(a, b)
    expected = config['domains']
    if (len(paths_a), len(paths_b)) != (expected['expected_domain_a_images'], expected['expected_domain_b_images']):
        raise RuntimeError(f'Expected the shared 300 Monet / 7038 photo dataset, got {len(paths_a)} / {len(paths_b)}')
    torch.manual_seed(config['experiment']['seed'])
    networks = train.build_networks(config=config, device=device)
    pools = train.make_new_pools(config)
    trainer = CycleGANTrainer(networks=networks,
        optimizers=train.build_optimizer_mapping(config=config, networks=networks),
        losses=train.build_loss_bundle(config), fake_a_pool=pools[0], fake_b_pool=pools[1],
        mixed_precision=config['training']['mixed_precision'])
    factory, workers, pin = train.build_epoch_loader_factory(config=config,
        domain_a_dir=a, domain_b_dir=b, mode='full', device=device)
    loader = factory(1, config['experiment']['seed'])
    iterator = iter(loader)
    print('Actual BF16:', trainer.amp_enabled, '| batch size:', config['training']['batch_size'], flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for step in range(13):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        batch = next(iterator)
        trainer.train_step(real_a=batch['A'].to(device, non_blocking=True),
                           real_b=batch['B'].to(device, non_blocking=True))
        torch.cuda.synchronize(device)
        if step >= 3:
            timings.append(time.perf_counter() - start)
    seconds = sum(timings) / len(timings)
    steps_per_epoch = math.ceil(max(len(paths_a), len(paths_b)) / config['training']['batch_size'])
    hours = seconds * steps_per_epoch * config['training']['epochs'] / 3600
    result = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'config_sha256': digest,
        'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'bf16': trainer.amp_enabled, 'workers': workers,
        'seconds_per_step': seconds, 'steps_per_epoch': steps_per_epoch,
        'planned_epochs': config['training']['epochs'],
        'estimated_compute_hours': hours, 'estimate_with_15_percent_overhead_hours': hours * 1.15,
        'peak_cuda_memory_mib': torch.cuda.max_memory_allocated(device) / 1024**2,
        'purpose': 'short real-data training-path test; these model weights are discarded'}
    write_json(MEMBER / 'outputs/preflight/benchmark.json', result)
    print(json.dumps(result, indent=2), flush=True)
    print('Benchmark passed. Runtime is an estimate; epoch timings will refine it.', flush=True)


def launch(run_id):
    import train
    train.validate_run_id(run_id)
    manifest = MEMBER / 'logs' / (run_id + '_launcher.json')
    lock_path = MEMBER / 'logs' / (run_id + '_launcher.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            info = json.loads(manifest.read_text())
            if matching_process(info['pid'], run_id):
                print('This run is already running; PID', info['pid'])
                return
        config, digest = train.load_config(CONFIG)
        train.validate_canonical_config(config)
        preflight = MEMBER / 'outputs/preflight/benchmark.json'
        if not preflight.exists() or json.loads(preflight.read_text())['config_sha256'] != digest:
            raise RuntimeError('Run the GPU benchmark on this configuration before launching')
        latest = MEMBER / 'checkpoints' / run_id / 'latest.pt'
        existing_run = MEMBER / 'logs' / run_id
        if existing_run.exists() and not latest.exists():
            raise RuntimeError('An interrupted run has no completed-epoch checkpoint. Use a NEW run ID; preserve its logs.')
        command = [sys.executable, '-u', str(MEMBER / 'src/train.py'),
                   '--config', str(CONFIG), '--run-id', run_id, '--mode', 'full', '--device', 'cuda']
        if latest.exists():
            command.append('--resume')
        stdout_path = MEMBER / 'logs' / (run_id + '_console.log')
        with stdout_path.open('ab', buffering=0) as output, open(os.devnull, 'rb') as devnull:
            process = subprocess.Popen(command, cwd=ROOT, stdin=devnull,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        write_json(manifest, {'pid': process.pid, 'run_id': run_id,
            'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'console': str(stdout_path.relative_to(ROOT)), 'resume': latest.exists(),
            'config_sha256': digest, 'command': command})
        # Capture immediate failures instead of reporting a dead process as running.
        time.sleep(3)
        if process.poll() is not None:
            print(stdout_path.read_text(errors='replace')[-10000:])
            if process.returncode:
                raise RuntimeError(f'Training exited with code {process.returncode}')
        status(run_id)


def status(run_id):
    info_path = MEMBER / 'logs' / (run_id + '_launcher.json')
    if not info_path.exists():
        print('Run has not been launched.')
        return
    info = json.loads(info_path.read_text())
    print('PID:', info['pid'], '| running:', matching_process(info['pid'], run_id))
    console = ROOT / info['console']
    if console.exists():
        with console.open('rb') as f:
            f.seek(max(0, f.seek(0, 2) - 6000))
            print(f.read().decode('utf-8', errors='replace'))
    events = MEMBER / 'logs' / run_id / 'events.jsonl'
    if events.exists():
        completed = None
        last_event = None
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # The writer may still be appending its final line.
            last_event = event
            if event.get('event_type') == 'epoch_end':
                completed = event
        if completed:
            print('Last completed epoch event:', json.dumps(completed)[-2500:])
        if last_event:
            print('Latest event type:', last_event.get('event_type'))
    latest = MEMBER / 'checkpoints' / run_id / 'latest.pt'
    print('Completed-epoch checkpoint exists:', latest.exists())
    print('Keep Docker and the lab machine running; do not press Ctrl+C in its Jupyter server terminal.')


def export(run_id):
    info_path = MEMBER / 'logs' / (run_id + '_launcher.json')
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if matching_process(info['pid'], run_id):
            raise RuntimeError('For a consistent export, wait until the training process has finished')
    out = ROOT / 'Viraat_Task3_Run_Results.zip'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(MEMBER.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.lock':
                continue
            if 'checkpoints' in p.parts and p.name not in {'latest.pt', 'latest.pt.sha256', '.gitkeep'}:
                continue  # The latest complete checkpoint is enough to resume.
            z.write(p, p.relative_to(ROOT))
    print('Export:', out, '| MiB:', round(out.stat().st_size / 1024**2, 2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['benchmark', 'launch', 'status', 'export'])
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.action == 'benchmark':
        benchmark()
    elif args.action == 'launch':
        launch(args.run_id)
    elif args.action == 'status':
        status(args.run_id)
    else:
        export(args.run_id)
