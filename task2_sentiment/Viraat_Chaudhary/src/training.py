"""Validation-only training with epoch-boundary resume and immutable raw logs."""
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from data_runtime import batch_iterator
from models import build_model, parameter_counts
from metrics import full_metrics
from runtime_utils import (MODEL_NAMES, append_event, data_contract, dependency_snapshot,
                           environment, fingerprint, read_json, seed_everything, sha256,
                           source_fingerprint, utc_now, write_csv, write_json, cpu_peak_mib)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    torch.save(payload, temporary)
    temporary.replace(path)


def predict_arrays(model, arrays, config, device):
    model.eval()
    probabilities = np.empty(len(arrays['labels']), dtype=np.float64)
    weighted_loss = 0.
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    amp = bool(config['amp'] and device.type == 'cuda')
    with torch.inference_mode():
        for rows, ids, lengths, labels in batch_iterator(arrays, config['batch_size'], device):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(ids, lengths)
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite inference logits.')
            weighted_loss += float(criterion(logits.float(), labels).item())
            probabilities[rows] = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
    return probabilities, weighted_loss / len(probabilities)


def plot_training(root, name, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(x, [h['train_loss'] for h in history], label='train BCE')
    axes[0].plot(x, [h['validation_loss'] for h in history], label='validation BCE')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title=name)
    axes[0].legend()
    axes[1].plot(x, [h['validation_macro_f1'] for h in history], marker='o')
    axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1', title='Checkpoint selection')
    fig.tight_layout()
    path = Path(root) / f'outputs/plots/{name}_training_curves.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_model(root, name, arrays, allow_cpu=False):
    root = Path(root)
    if name not in MODEL_NAMES:
        raise ValueError('Unknown model name.')
    config_path = root / f'configs/{name}.json'
    config = read_json(config_path)
    data_manifest = read_json(root / 'outputs/metrics/data_artifact_manifest.json')
    for split in ('train', 'validation'):
        if len(arrays[split]['labels']) != data_manifest['encoded_splits'][split]['rows']:
            raise RuntimeError('Training/validation row count differs from the approved manifest.')
    if config['name'] != name or config['pretrained_weights']:
        raise ValueError('Incorrect model configuration.')
    if not (root / 'student_design_notes.json').is_file():
        raise RuntimeError('Enter your design explanations in notebook 02 before training.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not allow_cpu:
        raise RuntimeError('Select Colab A100 GPU before training. CPU is only for explicit smoke tests.')
    cfg_hash, data_hash, source_hash = sha256(config_path), data_contract(root), source_fingerprint(root)
    identity = fingerprint({'config': cfg_hash, 'data': data_hash, 'implementation': source_hash})
    manifest_path = root / f'outputs/metrics/{name}_training_manifest.json'
    best_path = root / f'checkpoints/{name}_best.pt'
    resume_path = root / f'checkpoints/resume/{name}_last.pt'
    log_path = root / f'logs/{name}_training.jsonl'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['identity'] != identity:
            raise RuntimeError('Existing run uses different settings/code/data. Use a new RUN_NAME.')
        if old['status'] == 'completed':
            if sha256(best_path) != old['checkpoint_sha256']:
                raise RuntimeError('Selected checkpoint integrity failed.')
            print(name + ': completed run restored; no training repeated.')
            return old
    if (root / 'outputs/metrics/frozen_models.json').exists():
        raise RuntimeError('Models are frozen for final test. Further training is disabled in this run.')
    seed_everything(config['seed'])
    vocabulary_size = len(read_json(root / 'outputs/metrics/vocabulary.json')['token_to_index'])
    model = build_model(config, vocabulary_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=0)
    amp = bool(config['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best_score, best_loss, best_epoch, stalled, start_epoch = [], -1., float('inf'), 0, 0, 1
    run_started = utc_now()
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        if saved['identity'] != identity:
            raise RuntimeError('Resume checkpoint belongs to different settings/code/data.')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        history = saved['history']
        # The JSON history records transfer timing measured after the durable checkpoint write.
        history_path = root / f'outputs/metrics/{name}_history.json'
        if history_path.exists():
            durable_history = read_json(history_path)
            if [h['epoch'] for h in durable_history] == [h['epoch'] for h in history]:
                history = durable_history
        best_score, best_loss, best_epoch = saved['best_score'], saved['best_loss'], saved['best_epoch']
        stalled, start_epoch, run_started = saved['stalled'], saved['epoch'] + 1, saved['run_started']
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        if device.type == 'cuda' and saved['cuda_rng_states']:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng_states']])
        append_event(log_path, 'resume_from_completed_epoch', completed_epoch=saved['epoch'])
    hardware = environment()
    write_json(root / f'environments/{name}.json', hardware)
    dependency_snapshot(root / f'environments/{name}_requirements.lock.txt')
    write_json(manifest_path, {'status': 'training', 'identity': identity,
                              'name': name, 'started_utc': run_started})
    append_event(log_path, 'training_session_started', model=name, identity=identity,
                 config=config, hardware=hardware, resumed_at_epoch=start_epoch,
                 official_test_used=False)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    criterion = nn.BCEWithLogitsLoss()
    peak_allocated = max([h.get('peak_cuda_allocated_mib', 0.) or 0. for h in history] or [0.])
    peak_reserved = max([h.get('peak_cuda_reserved_mib', 0.) or 0. for h in history] or [0.])
    for epoch in range(start_epoch, config['max_epochs'] + 1):
        if stalled >= config['early_stopping_patience']:
            break
        sync(device)
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, examples, skipped = 0., 0, 0
        train_start = time.perf_counter()
        batches = batch_iterator(arrays['train'], config['batch_size'], device,
                                 seed=config['seed'] + epoch)
        progress = tqdm(batches, total=math.ceil(len(arrays['train']['labels']) / config['batch_size']),
                        desc=f'{name} epoch {epoch}', disable=config.get('quiet', False))
        for rows, ids, lengths, labels in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(ids, lengths)
                loss = criterion(logits.float(), labels)
            if not torch.isfinite(loss):
                append_event(log_path, 'non_finite_loss', epoch=epoch, examples_completed=examples)
                raise RuntimeError('Non-finite loss; training stopped without fabricating results.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'],
                                           error_if_nonfinite=not amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            loss_sum += float(loss.detach().item()) * len(rows)
            examples += len(rows)
        sync(device)
        train_seconds = time.perf_counter() - train_start
        probabilities, validation_loss = predict_arrays(model, arrays['validation'], config, device)
        scores = full_metrics(arrays['validation']['labels'], probabilities)
        f1 = scores['f1_macro']
        improved = f1 > best_score or (f1 == best_score and validation_loss < best_loss)
        if improved:
            best_score, best_loss, best_epoch, stalled = f1, validation_loss, epoch, 0
            save_checkpoint(best_path, {'state_dict': model.state_dict(), 'config': config,
                                        'vocabulary_size': vocabulary_size, 'identity': identity,
                                        'data_contract': data_hash, 'config_sha256': cfg_hash,
                                        'implementation_fingerprint': source_hash,
                                        'epoch': epoch, 'validation_macro_f1': f1,
                                        'validation_loss': validation_loss})
        else:
            stalled += 1
        scheduler.step(validation_loss)
        sync(device)
        if device.type == 'cuda':
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device) / 2**20)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device) / 2**20)
        entry = {'epoch': epoch, 'train_loss': loss_sum / examples,
                 'validation_loss': validation_loss, 'validation_macro_f1': f1,
                 'validation_accuracy': scores['accuracy'], 'train_examples': examples,
                 'train_phase_seconds': train_seconds,
                 'epoch_wall_seconds_including_validation': time.perf_counter() - epoch_start,
                 'training_examples_per_second': examples / train_seconds,
                 'learning_rate_for_next_epoch': optimizer.param_groups[0]['lr'],
                 'selected_new_best': improved, 'amp_skipped_updates': skipped,
                 'peak_cuda_allocated_mib': peak_allocated if device.type == 'cuda' else None,
                 'peak_cuda_reserved_mib': peak_reserved if device.type == 'cuda' else None,
                 'cpu_process_peak_rss_mib': cpu_peak_mib()}
        history.append(entry)
        resume_write_start = time.perf_counter()
        save_checkpoint(resume_path, {'identity': identity, 'epoch': epoch,
                                      'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                      'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                                      'history': history, 'best_score': best_score, 'best_loss': best_loss,
                                      'best_epoch': best_epoch, 'stalled': stalled, 'run_started': run_started,
                                      'torch_rng_state': torch.get_rng_state(),
                                      'cuda_rng_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        entry['resume_checkpoint_seconds'] = time.perf_counter() - resume_write_start
        entry['epoch_wall_seconds_including_validation'] = time.perf_counter() - epoch_start
        append_event(log_path, 'epoch_completed', **entry)
        write_json(root / f'outputs/metrics/{name}_history.json', history)
        print(f'Epoch {epoch}: validation macro-F1={f1:.5f}; training phase={train_seconds:.1f}s; '
              f'best epoch={best_epoch}.')
    if not history or not best_path.exists():
        raise RuntimeError('No completed epoch/checkpoint is available.')
    # Use only the validation-selected checkpoint for stored validation predictions.
    selected = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(selected['state_dict'])
    probabilities, val_loss = predict_arrays(model, arrays['validation'], config, device)
    pred_path = root / f'outputs/predictions/{name}_validation.npz'
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, source_indices=np.asarray(arrays['validation']['source_indices']),
                        labels=np.asarray(arrays['validation']['labels']), probabilities=probabilities)
    summary = {'status': 'completed', 'model': name, 'identity': identity,
               'config': str(config_path.relative_to(root)), 'config_sha256': cfg_hash,
               'data_contract': data_hash, 'implementation_fingerprint': source_hash,
               'checkpoint': str(best_path.relative_to(root)), 'checkpoint_sha256': sha256(best_path),
               'selected_epoch': best_epoch, 'completed_epochs': len(history),
               'selection': 'maximum validation macro-F1; ties use lower validation BCE',
               'validation_macro_f1': best_score, 'validation_loss': val_loss,
               'validation_predictions_sha256': sha256(pred_path),
               'training_wall_seconds': sum(h['epoch_wall_seconds_including_validation'] for h in history),
               'training_phase_seconds': sum(h['train_phase_seconds'] for h in history),
               'training_examples_processed': sum(h['train_examples'] for h in history),
               'training_examples_per_second': sum(h['train_examples'] for h in history) /
                                                 sum(h['train_phase_seconds'] for h in history),
               'peak_cuda_allocated_mib': peak_allocated if device.type == 'cuda' else None,
               'peak_cuda_reserved_mib': peak_reserved if device.type == 'cuda' else None,
               'cpu_process_peak_rss_mib': cpu_peak_mib(), **parameter_counts(model),
               'hardware': hardware, 'amp_enabled': amp, 'started_utc': run_started,
               'completed_utc': utc_now(), 'official_test_used': False,
               'timing_scope': 'completed epochs, including validation and checkpoint writes; '
                               'excludes data restore and interrupted partial epochs. A disconnect immediately '
                               'after a durable checkpoint can leave its final transfer timing unavailable.',
               'log': str(log_path.relative_to(root))}
    write_json(manifest_path, summary)
    write_csv(root / f'outputs/metrics/{name}_history.csv', history)
    plot_training(root, name, history)
    append_event(log_path, 'training_completed', summary=summary)
    del model, optimizer, scheduler, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def freeze_models(root):
    root = Path(root)
    items = {}
    for name in MODEL_NAMES:
        record = read_json(root / f'outputs/metrics/{name}_training_manifest.json')
        if record['status'] != 'completed' or record['official_test_used']:
            raise RuntimeError('Complete all three validation-only training runs first.')
        if sha256(root / record['checkpoint']) != record['checkpoint_sha256']:
            raise RuntimeError('Selected checkpoint changed: ' + name)
        if sha256(root / record['config']) != record['config_sha256']:
            raise RuntimeError('Configuration changed: ' + name)
        if record['implementation_fingerprint'] != source_fingerprint(root):
            raise RuntimeError('Training implementation changed: ' + name)
        items[name] = record
    if len({read_json(root / items[name]['config'])['architecture'] for name in MODEL_NAMES}) != 3:
        raise RuntimeError('The three accepted architectures must remain distinct.')
    path = root / 'outputs/metrics/frozen_models.json'
    proposed = {'models': items, 'data_contract': data_contract(root),
                'evaluation_protocol_sha256': sha256(root / 'configs/evaluation_protocol.json'),
                'manual_review_model': max(MODEL_NAMES, key=lambda n: items[n]['validation_macro_f1']),
                'selection_population': 'validation only', 'implementation_fingerprint': source_fingerprint(root)}
    if path.exists():
        old = read_json(path)
        if fingerprint({k: v for k, v in old.items() if k != 'frozen_utc'}) != fingerprint(proposed):
            raise RuntimeError('Frozen models cannot be replaced after test access.')
        return old
    proposed['frozen_utc'] = utc_now()
    write_json(path, proposed)
    return proposed
