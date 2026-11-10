"""Audit shipped evidence, or execute a fresh full run in an isolated ignored directory."""
import argparse
import csv
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from config import LOGS, TABLES, TRAINING_SOURCES
from data import read_json, save_json, digest, load_raw, make_split, fit_statistics
from check_rigor import validate_split, validate_predictions
from engine import classification_metrics, source_hashes
from test_invariants import deterministic_smoke


def audit(reference_environment=None):
    start = time.perf_counter()
    s = read_json(LOGS / 'split.json')
    validate_split(s)
    x, y = load_raw(True)
    tr, va = make_split(y)
    assert tr == s['train_indices'] and va == s['val_indices']
    mean, std = fit_statistics(x, tr)
    assert np.allclose(mean, s['mean'], atol=1e-12, rtol=0)
    assert np.allclose(std, s['std'], atol=1e-12, rtol=0)
    del x, y
    with (TABLES / 'test_predictions.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    m = read_json(LOGS / 'test_metrics.json')
    validate_predictions(rows, m)
    recalculated = classification_metrics(np.array([int(r['target']) for r in rows]),
                                         np.array([int(r['prediction']) for r in rows]))
    for key in ['accuracy', 'macro_f1', 'confusion_matrix', 'per_class', 'wilson_95']:
        assert recalculated[key] == m[key], key
    smoke = deterministic_smoke()
    reference = read_json(LOGS / 'smoke_reference.json')
    assert smoke == reference, 'short training replay differs'
    for k in 'ABCD':
        assert read_json(LOGS / f'run_{k}.json')['source_sha256'] == source_hashes(), 'source mismatch'
    independent = None
    if reference_environment is not None:
        reference = Path(reference_environment).resolve()
        assert (reference / 'conda-meta').is_dir(), 'Reference must be an existing conda environment'
        assert (reference / 'python.exe').is_file() or (reference / 'bin/python').is_file(), 'Reference Python missing'
        independent = (reference != Path(sys.prefix).resolve()
                       and Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()))
        assert independent, 'Audit must use the separately installed environment'
    save_json(LOGS / 'reproduction.json', dict(passed=True, seconds=time.perf_counter()-start,
              python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
              checks=['training data file hashes', 'split indices identical', 'training mean/std identical',
                      'test CSV metrics recalculated', 'training source fingerprints identical',
                      'four-step synthetic training losses and state SHA256 identical'],
              smoke=smoke, scope='data/metrics/source audit and short replay', independent_environment=independent,
              full_80_epoch_retraining_in_second_environment=False))
    print(f'Audit passed; separate environment verified={independent}; full training not repeated.', flush=True)


def full(use_local_data):
    # New destination every time: no existing evidence is deleted or overwritten.
    root = Path('../tmp/reproduction') / ('run-' + uuid.uuid4().hex[:10])
    dest = root / 'AI-Lab/Lab03/lab3-image-classification'
    dest.mkdir(parents=True)
    shutil.copy2('../../AGENTS.md', root / 'AI-Lab/AGENTS.md')
    for name in ['AGENTS.md', 'WORKFLOW.md', '当代人工智能作业三.pdf']:
        shutil.copy2(Path('..') / name, dest.parent / name)
    for path in list(Path('.').glob('*.py')) + [Path('environment.yml'), Path('environment-lock.txt')]:
        shutil.copy2(path, dest / path.name)
    if use_local_data:
        shutil.copytree('data/cifar-10-batches-py', dest / 'data/cifar-10-batches-py')
        shutil.copy2('data/cifar-10-python.tar.gz', dest / 'data/cifar-10-python.tar.gz')
    for stage in ['environment', 'prepare', 'verify', 'train', 'select', 'test', 'analyze']:
        print(f'Fresh run: {stage} in {dest.as_posix()}', flush=True)
        subprocess.run([sys.executable, '-X', 'utf8', 'run_pipeline.py', stage], cwd=dest, check=True)
    print(f'Fresh full run complete: {dest.as_posix()}. Review its figures before writing new report observations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['audit', 'full'], default='audit')
    parser.add_argument('--use-local-data', action='store_true')
    parser.add_argument('--reference-environment', help='Original environment prefix; verifies the audit uses a different install')
    args = parser.parse_args()
    from run_pipeline import read_instructions
    read_instructions()
    audit(args.reference_environment) if args.mode == 'audit' else full(args.use_local_data)
