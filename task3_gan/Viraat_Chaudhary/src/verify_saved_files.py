#!/usr/bin/env python3
"""Read-only verification of the prepared repository snapshot."""
import hashlib
import json
from pathlib import Path
import sys

MEMBER = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = json.loads((MEMBER / 'repository_manifest.json').read_text())
    problems = []
    for relative, record in manifest['files'].items():
        path = MEMBER / relative
        if not path.is_file():
            problems.append('MISSING: ' + relative)
        elif path.stat().st_size != record['size'] or sha256(path) != record['sha256']:
            problems.append('CHANGED: ' + relative)
    actual = {p.relative_to(MEMBER).as_posix() for p in MEMBER.rglob('*') if p.is_file()
              and p.name != '.DS_Store' and p.suffix != '.pyc'
              and '__pycache__' not in p.parts and '__MACOSX' not in p.parts
              and not p.relative_to(MEMBER).as_posix().startswith(
                  'outputs/viraat_resizeconv6_run001/evaluator_workspace/Part 3/')}
    expected = set(manifest['files']) | {'repository_manifest.json'}
    problems += ['EXTRA: ' + rel for rel in sorted(actual - expected)]
    if problems:
        print('\n'.join(problems))
        print('STOP: review these differences before staging. Later legitimate audit updates also change hashes.')
        return 1
    print('PASS: %d prepared files; all recorded SHA256 hashes match.' % len(expected))
    print('Final checkpoint, 600 predictions and all raw logs match the saved run.')
    print('Human ratings, Kaggle evidence, team PDF, root README and Git LFS remain separate checks.')
    print('This verifier does not run the model. The manifest describes the preparation snapshot.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(verify())
    except (OSError, ValueError, KeyError) as exc:
        sys.exit('STOP: ' + str(exc))
