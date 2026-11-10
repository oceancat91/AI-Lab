"""Figures and post-test diagnostics from saved evidence; never tunes the model."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import CLASSES, EXPERIMENTS, LOGS, TABLES, FIGURES
from data import read_json, save_json, load_raw, transform, normalization, digest
from models import make_model

COLORS = ['#246D8F', '#C76B35', '#418A68', '#805799']


def save_figure(fig, name):
    fig.savefig(FIGURES / name, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def gradcam(model, image, target):
    captured = {}

    def capture(module, inputs, output):
        captured['activation'] = output
        output.retain_grad()

    handle = model.features[-1].conv2.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        logits[0, target].backward()
        activation = captured['activation']
        gradients = activation.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        raw = (weights * activation).sum(dim=1).relu()
        peak = raw.max().item()
        normalized = raw / peak if peak > 0 else torch.zeros_like(raw)
        scaled = F.interpolate(normalized[:, None], (32, 32), mode='bilinear', align_corners=False)[0, 0]
        assert torch.isfinite(gradients).all() and torch.isfinite(scaled).all()
        return scaled.detach().numpy(), dict(explained_class=target, activation_shape=list(activation.shape),
                gradient_l1=float(gradients.abs().sum().item()), raw_positive_max=peak,
                low_resolution_map=normalized[0].detach().numpy().tolist(),
                layer='features.2.conv2', method='ReLU(sum_k(mean_hw(d logit_c / d A_k) * A_k))')
    finally:
        handle.remove()


def analyze():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.constrained_layout.use': True})
    runs = {k: read_json(LOGS / f'run_{k}.json') for k in EXPERIMENTS}
    metrics = read_json(LOGS / 'test_metrics.json')
    selection = read_json(LOGS / 'selection.json')
    histories = {}
    for key in runs:
        with (TABLES / f'history_{key}.csv').open(encoding='utf-8', newline='') as f:
            histories[key] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for row, (key, run) in enumerate(runs.items()):
        history = histories[key]
        epochs = [r['epoch'] for r in history]
        for col, metric in enumerate(['loss', 'accuracy']):
            ax = axes[row, col]
            for split, color in [('train', COLORS[0]), ('val', COLORS[1])]:
                ax.plot(epochs, [r[f'{split}_{metric}'] for r in history], label='Train (online)' if split == 'train' else 'Validation', color=color)
            ax.axvline(run['best_epoch'], color='#888888', linestyle=':', linewidth=1, label='Selected epoch')
            ax.set(title=f'{key}: {run["config"]["architecture"]}, {run["config"]["augmentation"]}',
                   ylabel='Cross-entropy loss' if metric == 'loss' else 'Accuracy (fraction)', xlabel='Epoch')
            ax.grid(alpha=.18)
            ax.legend(fontsize=8, loc='best')
    fig.suptitle('CIFAR-10 learning curves | 45,000 train / 5,000 validation | seed 42', fontsize=14)
    save_figure(fig, 'learning_curves.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    keys = list(runs)
    x = np.arange(4)
    vals = [runs[k]['best_val_accuracy'] * 100 for k in keys]
    clean = [runs[k]['clean_train']['accuracy'] * 100 for k in keys]
    axes[0].bar(x - .18, clean, .36, label='Clean train at best checkpoint', color=COLORS[0])
    axes[0].bar(x + .18, vals, .36, label='Validation at best checkpoint', color=COLORS[1])
    for i, value in enumerate(vals):
        axes[0].text(i + .18, value + 1, f'{value:.2f}', ha='center', fontsize=9)
    axes[0].set(xticks=x, xticklabels=keys, ylim=(0, 104), ylabel='Accuracy (%)', xlabel='Registered configuration', title='Generalization at selected checkpoints')
    axes[0].legend(fontsize=8, loc='lower right')
    contrasts = [('B-A: augmentation, plain', 'B', 'A'), ('D-C: augmentation, residual', 'D', 'C'),
                 ('C-A: residual, no augmentation', 'C', 'A'), ('D-B: residual, augmentation', 'D', 'B')]
    effects = [(runs[a]['best_val_accuracy'] - runs[b]['best_val_accuracy']) * 100 for _, a, b in contrasts]
    axes[1].barh(np.arange(4), effects, color=COLORS[0])
    axes[1].set(yticks=np.arange(4), yticklabels=[t[0] for t in contrasts], xlabel='Validation accuracy difference (percentage points)', title='Controlled single-factor contrasts')
    axes[1].axvline(0, color='#888888', linewidth=1)
    axes[1].invert_yaxis()
    for i, value in enumerate(effects):
        axes[1].annotate(f'{value:+.2f}', (value, i), xytext=(4 if value >= 0 else -4, 0), textcoords='offset points', va='center', ha='left' if value >= 0 else 'right')
    axes[1].margins(x=.25)
    fig.suptitle('2 x 2 ablation | one split and one seed; no significance claim', fontsize=13)
    save_figure(fig, 'ablation.png')

    cm = np.asarray(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(8.4, 7))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1000)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9, color='white' if cm[i, j] > 550 else '#172B3A')
    ax.set(xticks=range(10), xticklabels=CLASSES, yticks=range(10), yticklabels=CLASSES,
           xlabel='Predicted class', ylabel='True class', title=f'Official test set | selected model {selection["id"]} | n=10,000')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    fig.colorbar(im, ax=ax, label='Images (1,000 per true class)', fraction=.04, pad=.04)
    save_figure(fig, 'confusion_matrix.png')

    with (TABLES / 'test_predictions.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    errors = []
    for c in range(10):
        candidates = [r for r in rows if int(r['target']) == c and int(r['prediction']) != c]
        if candidates:
            item = min(candidates, key=lambda r: (-float(r['confidence']), int(r['index'])))
            errors.append(dict(index=int(item['index']), target=c, prediction=int(item['prediction']), confidence=float(item['confidence'])))
    raw, labels = load_raw(False)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.3))
    for ax, example in zip(axes.flat, errors):
        i, t, p = example['index'], example['target'], example['prediction']
        ax.imshow(raw[i].transpose(1, 2, 0), interpolation='nearest')
        ax.set_title(f'#{i} | true: {CLASSES[t]}\npred: {CLASSES[p]} ({example["confidence"]:.1%})', fontsize=9)
        ax.axis('off')
    for ax in list(axes.flat)[len(errors):]:
        ax.axis('off')
    fig.suptitle('Highest-confidence error in each true class | fixed rule, no visual cherry-picking', fontsize=12)
    save_figure(fig, 'misclassified.png')

    torch.set_num_threads(read_json(LOGS / 'runtime.json')['threads'])
    model = make_model(selection['config']['architecture'])
    checkpoint = Path('checkpoints') / f'{selection["id"]}.pt'
    assert digest(checkpoint) == selection['checkpoint_sha256']
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    cam_records = []
    examples = errors[:4]
    fig, axes = plt.subplots(len(examples), 3, figsize=(7.4, 8.8))
    for row, example in enumerate(examples):
        i = example['index']
        image = raw[i].transpose(1, 2, 0) / 255.0
        tensor = transform(torch.from_numpy(raw[i:i + 1]), *normalization())
        cam, record = gradcam(model, tensor, example['prediction'])
        record.update(example)
        assert record['explained_class'] == record['prediction']
        cam_records.append(record)
        axes[row, 0].imshow(image, interpolation='nearest')
        axes[row, 0].set_title(f'#{i}: {CLASSES[example["target"]]}', fontsize=10)
        axes[row, 1].imshow(cam, cmap='inferno', vmin=0, vmax=1)
        axes[row, 1].set_title('Grad-CAM (normalized 0-1)', fontsize=9)
        axes[row, 2].imshow(image, interpolation='nearest')
        axes[row, 2].imshow(cam, cmap='inferno', alpha=.45, vmin=0, vmax=1)
        axes[row, 2].set_title(f'Target: {CLASSES[example["prediction"]]}', fontsize=10)
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle('Post-hoc explanation of predicted logits | final convolution, 8 x 8 resolution', fontsize=11)
    save_figure(fig, 'gradcam.png')
    pairs = sorted([(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j], reverse=True)[:8]
    save_json(LOGS / 'analysis.json', dict(errors=errors, gradcam=cam_records,
              contrasts=[dict(name=t[0], difference_pp=v) for t, v in zip(contrasts, effects)],
              top_confusions=[dict(count=n, true=CLASSES[i], predicted=CLASSES[j]) for n, i, j in pairs],
              figure_sha256={p.name: digest(p) for p in sorted(FIGURES.glob('*.png'))},
              selection_policy='highest-confidence error per true class, ties by smallest index; first 4 for Grad-CAM',
              limitations='single seed; selected extreme errors not representative error frequencies; Grad-CAM is not causal proof'))
    with (TABLES / 'comparison.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'architecture', 'augmentation', 'parameters', 'best_epoch', 'clean_train_accuracy', 'val_accuracy', 'val_loss', 'seconds'])
        for k, r in runs.items():
            w.writerow([k, r['config']['architecture'], r['config']['augmentation'], r['parameters'], r['best_epoch'], r['clean_train']['accuracy'], r['best_val_accuracy'], r['best_val_loss'], r['seconds']])
    print('Saved five evidence figures and post-test analysis', flush=True)


if __name__ == '__main__':
    analyze()
