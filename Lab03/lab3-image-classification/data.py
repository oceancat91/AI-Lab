"""Download verified CIFAR-10; keep test loading behind a frozen selection."""
import hashlib
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import DATA, LOGS, SPLIT_SEED, make_dirs

URL = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
ARCHIVE_MD5 = 'c58f30108f718f92721af3b95e74349a'
FILES_MD5 = {
    'data_batch_1': 'c99cafc152244af753f735de768cd75f',
    'data_batch_2': 'd4bba439e000b95fd0a9bffe97cbabec',
    'data_batch_3': '54ebc095f3ab1f0389bbae665268c751',
    'data_batch_4': '634d18415352ddfa80567beed471001a',
    'data_batch_5': '482c414d41f54cd18b22e5b47cb7c3cb',
    'test_batch': '40351d587109b95175f43aff81a1287e',
    'batches.meta': '5ff9c542aee3614f3951f8cda6e48888',
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def download():
    make_dirs()
    archive = DATA / 'cifar-10-python.tar.gz'
    if not archive.exists():
        print('Downloading official CIFAR-10 archive...', flush=True)
        partial = archive.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as stream:
            received = 0
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                stream.write(chunk)
                received += len(chunk)
                print(f'Downloaded {received / (1024 ** 2):.1f} MiB', flush=True)
        if digest(partial, 'md5') != ARCHIVE_MD5:
            raise ValueError('Archive MD5 mismatch; download not used')
        partial.replace(archive)
    assert digest(archive, 'md5') == ARCHIVE_MD5, 'Archive integrity failure'
    root = DATA / 'cifar-10-batches-py'
    if not all((root / n).exists() for n in FILES_MD5):
        with tarfile.open(archive) as tar:
            # Extract only verified official archive, regular files, and fixed names.
            for name in FILES_MD5:
                member = tar.getmember('cifar-10-batches-py/' + name)
                assert member.isfile()
                root.mkdir(exist_ok=True)
                with tar.extractfile(member) as src, (root / name).open('wb') as dst:
                    dst.write(src.read())
    files = {}
    for name, expected in FILES_MD5.items():
        assert digest(root / name, 'md5') == expected, f'{name}: MD5 mismatch'
        files[name] = dict(md5=expected, sha256=digest(root / name))
    save_json(LOGS / 'data_integrity.json', dict(source=URL, archive_md5=ARCHIVE_MD5,
              archive_sha256=digest(archive), files=files, verified=True,
              test_policy='byte-level integrity only before selection; test samples not loaded'))


def load_raw(train=True):
    if not train and not (LOGS / 'selection.json').exists():
        raise RuntimeError('Test set is sealed until validation-only selection exists')
    names = [f'data_batch_{i}' for i in range(1, 6)] if train else ['test_batch']
    xs, ys = [], []
    for name in names:
        path = DATA / 'cifar-10-batches-py' / name
        assert digest(path, 'md5') == FILES_MD5[name], 'Untrusted/corrupt pickle refused'
        with path.open('rb') as f:
            obj = pickle.load(f, encoding='bytes')
        xs.append(obj[b'data'].reshape(-1, 3, 32, 32))
        ys.extend(obj[b'labels'])
    return np.concatenate(xs), np.asarray(ys, dtype=np.int64)


def make_split(labels):
    rng = np.random.default_rng(SPLIT_SEED)
    tr, va = [], []
    for c in range(10):
        indices = rng.permutation(np.flatnonzero(labels == c))
        assert len(indices) == 5000
        va.extend(indices[:500].tolist())
        tr.extend(indices[500:].tolist())
    return sorted(tr), sorted(va)


def fit_statistics(images, indices):
    # Chunked float64 accumulation; caller supplies training indices only.
    sums, squares = np.zeros(3), np.zeros(3)
    for start in range(0, len(indices), 256):
        x = images[indices[start:start + 256]].astype(np.float64) / 255.0
        sums += x.sum(axis=(0, 2, 3))
        squares += np.square(x).sum(axis=(0, 2, 3))
    count = len(indices) * 32 * 32
    mean = sums / count
    std = np.sqrt(squares / count - mean ** 2)
    return mean.tolist(), std.tolist()


def prepare():
    download()
    x, y = load_raw(True)
    tr, va = make_split(y)
    mean, std = fit_statistics(x, tr)
    result = dict(seed=SPLIT_SEED, train_indices=tr, val_indices=va,
                  train_class_counts=np.bincount(y[tr], minlength=10).tolist(),
                  val_class_counts=np.bincount(y[va], minlength=10).tolist(),
                  statistics_fit_count=len(tr), mean=mean, std=std,
                  indices_sha256=hashlib.sha256(np.asarray(tr + va, dtype='<i8').tobytes()).hexdigest())
    if (LOGS / 'split.json').exists():
        assert read_json(LOGS / 'split.json') == result, 'Existing split changed'
    else:
        save_json(LOGS / 'split.json', result)
    print('Prepared train=45000, validation=5000; test remains sealed', flush=True)


def tensors(train=True):
    images, labels = load_raw(train)
    # Retain uint8 in RAM; convert one minibatch at a time.
    x, y = torch.from_numpy(images), torch.from_numpy(labels)
    s = read_json(LOGS / 'split.json')
    if train:
        tr, va = torch.tensor(s['train_indices']), torch.tensor(s['val_indices'])
        return x[tr], y[tr], x[va], y[va]
    return x, y


def transform(x, mean, std, augment=False, generator=None):
    x = x.float().div(255)
    if augment:
        if generator is None:
            raise ValueError('Augmentation requires its own random generator')
        n = len(x)
        x = F.pad(x, (4, 4, 4, 4), mode='reflect')
        top = torch.randint(0, 9, (n,), generator=generator)
        left = torch.randint(0, 9, (n,), generator=generator)
        rows = top[:, None, None] + torch.arange(32)[None, :, None]
        cols = left[:, None, None] + torch.arange(32)[None, None, :]
        x = x.permute(0, 2, 3, 1)[torch.arange(n)[:, None, None], rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(n, generator=generator) < 0.5
        x[flip] = x[flip].flip(-1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def normalization():
    s = read_json(LOGS / 'split.json')
    return tuple(torch.tensor(s[k], dtype=torch.float32).view(1, 3, 1, 1) for k in ['mean', 'std'])
