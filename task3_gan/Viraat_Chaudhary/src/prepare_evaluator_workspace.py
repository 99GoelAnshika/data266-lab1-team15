#!/usr/bin/env python3
"""Restore byte-identical instructor inputs from saved images; no model runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

MEMBER = Path(__file__).resolve().parents[1]
RUN_ID = 'viraat_resizeconv6_run001'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(repo_root):
    evidence = json.loads((MEMBER / 'configs/evaluator_input_manifest.json').read_text())
    output = MEMBER / 'outputs' / RUN_ID
    predictions = output / 'predictions'
    inference = json.loads((predictions / 'inference_manifest.json').read_text())
    if (inference['run_id'] != RUN_ID or inference['epoch'] != 60
            or inference['checkpoint_sha256'] != evidence['checkpoint_sha256']):
        raise ValueError('Prediction run/checkpoint identity mismatch')
    records = inference['records']
    if Counter(r['direction'] for r in records) != {'A2B': 300, 'B2A': 300}:
        raise ValueError('Exactly 300 predictions per direction are required')
    copies = []
    for domain, expected in evidence['domains'].items():
        folder = repo_root / 'task3_gan/data' / domain
        found = sorted(p.name for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if found[:300] != [r['filename'] for r in expected]:
            raise ValueError('First 300 sorted filenames differ in ' + domain)
        for record in expected:
            source = folder / record['filename']
            if source.stat().st_size != record['size'] or sha256(source) != record['sha256']:
                raise ValueError('Source image bytes differ: ' + str(source))
            copies.append((source, Path(domain) / source.name, record['sha256']))
    for direction in ('A2B', 'B2A'):
        selected = [r for r in records if r['direction'] == direction]
        domain = 'monet_jpg' if direction == 'A2B' else 'photo_jpg'
        expected_names = {r['filename'] for r in evidence['domains'][domain]}
        names = [r['prediction'] for r in selected]
        folder = predictions / ('pred_' + direction)
        if (len(set(names)) != 300 or set(names) != expected_names
                or any(r['source'] != r['prediction'] for r in selected)
                or {p.name for p in folder.iterdir() if p.is_file()} != expected_names):
            raise ValueError('Prediction/source filename mismatch: ' + direction)
        for record in selected:
            source = folder / record['prediction']
            if sha256(source) != record['prediction_sha256']:
                raise ValueError('Prediction bytes differ: ' + str(source))
            copies.append((source, Path('pred_' + direction) / source.name,
                           record['prediction_sha256']))
    workspace = output / 'evaluator_workspace'
    final = workspace / 'Part 3'
    if final.exists():
        actual = {p.relative_to(final / 'Data').as_posix()
                  for p in (final / 'Data').rglob('*') if p.is_file()}
        if actual != {rel.as_posix() for _, rel, _ in copies}:
            raise FileExistsError('Existing evaluator workspace differs; nothing changed')
        for _, rel, digest in copies:
            if sha256(final / 'Data' / rel) != digest:
                raise FileExistsError('Existing evaluator bytes differ; nothing changed')
        print('Existing evaluator workspace: all 1,200 image hashes match.')
        return
    workspace.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.restore_', dir=workspace))
    try:
        for source, rel, digest in copies:
            destination = temporary / 'Data' / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256(destination) != digest:
                raise OSError('Copied image failed verification: ' + str(rel))
        if final.exists():
            raise FileExistsError('Evaluator destination appeared during preparation')
        temporary.rename(final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print('Restored and verified 1,200 byte-identical evaluator images.')
    print('Instructor notebook working directory:', workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=MEMBER.parents[1])
    args = parser.parse_args()
    try:
        prepare(args.repo_root.expanduser().resolve())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, 'STOP: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
