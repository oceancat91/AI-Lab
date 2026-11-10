"""Shared FP32 training, validation, checkpoint selection and final evaluation."""
import csv
import math
import os
import platform
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from config import (SEED, ORDER_SEED, AUGMENT_SEED, PROBE_SEED, BATCH_SIZE, EPOCHS,
                    EXPERIMENTS, PROTOCOL, LOGS, TABLES, CHECKPOINTS, CLASSES,
                    TRAINING_SOURCES, make_dirs)
from data import digest, save_json, read_json, tensors, normalization, transform
from models import make_model


def now():
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def source_hashes():
    return {name: digest(name) for name in TRAINING_SOURCES}


def environment():
    import matplotlib
    import torchvision
    make_dirs()
    torch.set_num_threads(2)
    probe = torch.randn(2, 3, 8, 8, requires_grad=True)
    nn.Conv2d(3, 4, 3)(probe).sum().backward()
    assert torch.isfinite(probe.grad).all()
    save_json(LOGS / 'environment.json', dict(python=platform.python_version(),
              platform=platform.system(), machine=platform.machine(), processor=platform.processor(),
              logical_cpus=os.cpu_count(), torch=torch.__version__, torchvision=torchvision.__version__,
              numpy=np.__version__, matplotlib=matplotlib.__version__, cuda_available=torch.cuda.is_available(),
              device='cpu', convolution_backward_finite=True, time_utc=now()))
    save_json(LOGS / 'protocol.json', dict(protocol=PROTOCOL, experiments=EXPERIMENTS,
              created_utc=now(), selection='validation_accuracy desc, validation_loss asc, parameters asc, id asc'))


def benchmark():
    """Synthetic timing only; neither dataset nor accuracy used to select threads."""
    times = []
    for threads in [2, 4, 6]:
        torch.set_num_threads(threads)
        seed_all(PROBE_SEED)
        model = make_model('residual')
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(BATCH_SIZE, 3, 32, 32).contiguous(memory_format=torch.channels_last)
        y = torch.arange(BATCH_SIZE) % 10
        repeats = []
        for i in range(7):
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
            if i >= 2:
                repeats.append(time.perf_counter() - start)
        times.append(dict(threads=threads, median_seconds=float(np.median(repeats)), seconds=repeats))
    selected = min(times, key=lambda r: r['median_seconds'])['threads']
    save_json(LOGS / 'runtime.json', dict(threads=selected, timings=times, source='synthetic only',
              batch_size=BATCH_SIZE, memory_format='channels_last', precision='float32'))
    print(f'CPU thread benchmark: {times}; selected={selected}', flush=True)


def lr_for_epoch(epoch):
    warm = PROTOCOL['warmup_epochs']
    if epoch <= warm:
        return PROTOCOL['lr'] * epoch / warm
    progress = (epoch - warm) / (EPOCHS - warm)
    return PROTOCOL['min_lr'] + (PROTOCOL['lr'] - PROTOCOL['min_lr']) * (1 + math.cos(math.pi * progress)) / 2


@torch.inference_mode()
def evaluate(model, x, y, mean, std, predictions=False):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    all_probs = []
    for start in range(0, len(y), BATCH_SIZE):
        target = y[start:start + BATCH_SIZE]
        output = model(transform(x[start:start + BATCH_SIZE], mean, std))
        total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
        correct += (output.argmax(1) == target).sum().item()
        count += len(target)
        if predictions:
            all_probs.append(output.softmax(1))
    result = dict(loss=total_loss / count, accuracy=correct / count, count=count)
    return (result, torch.cat(all_probs).numpy()) if predictions else result


def checkpoint_key(accuracy, loss, epoch):
    return (-accuracy, loss, epoch)


def selection_key(summary):
    return (-summary['best_val_accuracy'], summary['best_val_loss'], summary['parameters'], summary['id'])


def train_run(key):
    make_dirs()
    if (LOGS / 'selection.json').exists():
        raise RuntimeError('Model choice is frozen; do not train after final selection')
    summary_path = LOGS / f'run_{key}.json'
    if summary_path.exists():
        existing = read_json(summary_path)
        if existing['status'] == 'complete':
            assert existing['source_sha256'] == source_hashes(), 'Training source changed'
            assert digest(CHECKPOINTS / f'{key}.pt') == existing['checkpoint_sha256']
            print(f'{key}: already complete, verified and skipped', flush=True)
            return
        raise RuntimeError(f'{key}: incomplete run exists; preserve failure evidence before restarting')
    threads = read_json(LOGS / 'runtime.json')['threads']
    torch.set_num_threads(threads)
    seed_all()
    cfg = EXPERIMENTS[key]
    model = make_model(cfg['architecture'])
    opt = torch.optim.SGD(model.parameters(), lr=PROTOCOL['lr'], momentum=PROTOCOL['momentum'],
                          weight_decay=PROTOCOL['weight_decay'])
    train_x, train_y, val_x, val_y = tensors(True)
    mean, std = normalization()
    order_rng = torch.Generator().manual_seed(ORDER_SEED)
    aug_rng = torch.Generator().manual_seed(AUGMENT_SEED)
    summary = dict(id=key, config=cfg, training_protocol=PROTOCOL, source_sha256=source_hashes(),
                   split_sha256=digest(LOGS / 'split.json'), parameters=sum(p.numel() for p in model.parameters()),
                   threads=threads, status='running', started_utc=now(), command=f'python run_pipeline.py train --runs {key}')
    save_json(summary_path, summary)
    history_path = TABLES / f'history_{key}.csv'
    best = (float('inf'), float('inf'), EPOCHS + 1)
    started = time.perf_counter()
    fields = ['epoch', 'lr', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'train_count', 'val_count', 'seconds']
    with history_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            tick = time.perf_counter()
            lr = lr_for_epoch(epoch)
            for group in opt.param_groups:
                group['lr'] = lr
            model.train()
            indices = torch.randperm(len(train_y), generator=order_rng)
            total_loss, correct, count = 0.0, 0, 0
            for start in range(0, len(indices), BATCH_SIZE):
                ii = indices[start:start + BATCH_SIZE]
                x = transform(train_x[ii], mean, std, cfg['augmentation'] == 'crop_flip', aug_rng)
                y = train_y[ii]
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss')
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                count += len(y)
            val = evaluate(model, val_x, val_y, mean, std)
            row = dict(epoch=epoch, lr=lr, train_loss=total_loss / count, train_accuracy=correct / count,
                       val_loss=val['loss'], val_accuracy=val['accuracy'], train_count=count,
                       val_count=val['count'], seconds=time.perf_counter() - tick)
            writer.writerow(row)
            f.flush()
            candidate = checkpoint_key(val['accuracy'], val['loss'], epoch)
            if candidate < best:
                best = candidate
                torch.save(model.state_dict(), CHECKPOINTS / f'{key}.pt')
                summary.update(best_epoch=epoch, best_val_accuracy=val['accuracy'], best_val_loss=val['loss'])
            summary['completed_epochs'] = epoch
            save_json(summary_path, summary)
            print(f'{key} {epoch:02d}/{EPOCHS} train={row["train_accuracy"]:.4f} val={val["accuracy"]:.4f} '
                  f'loss={val["loss"]:.4f} seconds={row["seconds"]:.1f}', flush=True)
    model.load_state_dict(torch.load(CHECKPOINTS / f'{key}.pt', weights_only=True))
    clean_train = evaluate(model, train_x, train_y, mean, std)
    summary.update(status='complete', finished_utc=now(), seconds=time.perf_counter() - started,
                   checkpoint_sha256=digest(CHECKPOINTS / f'{key}.pt'), clean_train=clean_train)
    save_json(summary_path, summary)


def select():
    if (LOGS / 'test_metrics.json').exists():
        raise RuntimeError('Cannot reselect after test evaluation')
    runs = [read_json(LOGS / f'run_{k}.json') for k in EXPERIMENTS]
    assert all(r['status'] == 'complete' for r in runs)
    winner = min(runs, key=selection_key)
    out = {k: winner[k] for k in ['id', 'config', 'best_epoch', 'best_val_accuracy', 'best_val_loss', 'checkpoint_sha256']}
    out.update(test_used=False, frozen_utc=now(), selection_rule='accuracy desc, loss asc, parameters asc, id asc')
    if (LOGS / 'selection.json').exists():
        raise RuntimeError('Selection already frozen')
    save_json(LOGS / 'selection.json', out)
    print('Frozen validation winner:', out, flush=True)


def classification_metrics(y, predicted):
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (y, predicted), 1)
    per_class = []
    f1s = []
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i])
        support, called = int(cm[i].sum()), int(cm[:, i].sum())
        recall = tp / support if support else 0.0
        precision = tp / called if called else 0.0
        f1 = 2 * tp / (support + called) if support + called else 0.0
        f1s.append(f1)
        per_class.append(dict(class_id=i, name=name, support=support, correct=tp, accuracy=recall, precision=precision, f1=f1))
    n, successes = len(y), int(np.trace(cm))
    p, z = successes / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(accuracy=p, correct=successes, count=n, macro_f1=float(np.mean(f1s)),
                confusion_matrix=cm.tolist(), per_class=per_class, wilson_95=[center - radius, center + radius])


def test_final():
    if (LOGS / 'test_metrics.json').exists():
        raise RuntimeError('Test evaluation already exists; use audit, do not reevaluate for tuning')
    selection = read_json(LOGS / 'selection.json')
    key = selection['id']
    assert digest(CHECKPOINTS / f'{key}.pt') == selection['checkpoint_sha256']
    torch.set_num_threads(read_json(LOGS / 'runtime.json')['threads'])
    model = make_model(selection['config']['architecture'])
    model.load_state_dict(torch.load(CHECKPOINTS / f'{key}.pt', weights_only=True))
    x, y = tensors(False)
    evaluation, probs = evaluate(model, x, y, *normalization(), predictions=True)
    predicted, confidence = probs.argmax(1), probs.max(1)
    metrics = classification_metrics(y.numpy(), predicted)
    metrics.update(loss=evaluation['loss'], selected_id=key, time_utc=now(), selection_sha256=digest(LOGS / 'selection.json'),
                   checkpoint_sha256=selection['checkpoint_sha256'], source_sha256=source_hashes())
    with (TABLES / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'target', 'prediction', 'confidence'] + [f'p_{i}' for i in range(10)])
        for i in range(len(y)):
            writer.writerow([i, y[i].item(), int(predicted[i]), float(confidence[i])] + probs[i].astype(float).tolist())
    save_json(LOGS / 'test_metrics.json', metrics)
    with (TABLES / 'per_class.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics['per_class'][0]))
        w.writeheader()
        w.writerows(metrics['per_class'])
    print(f'Final test: {metrics["correct"]}/10000 = {metrics["accuracy"]:.4%}', flush=True)
