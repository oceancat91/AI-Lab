"""Stage gates; exits nonzero on errors. No model training is performed here."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PDF_HASH = 'b80b838fb193a10be179c25d49bff077c10b00959e2ddc9a520baadad21f78fe'
REPOSITORY_URL = 'https://github.com/oceancat91/AI-Lab'
REPORT_MAX_PAGES = 5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_split(s):
    tr, va = s['train_indices'], s['val_indices']
    assert len(tr) == 45000 and len(va) == 5000, 'split size'
    assert len(set(tr)) == 45000 and len(set(va)) == 5000, 'duplicates'
    assert not set(tr).intersection(va), 'train/val leakage'
    assert set(tr + va) == set(range(50000)), 'split coverage'
    assert s['train_class_counts'] == [4500] * 10, 'train balance'
    assert s['val_class_counts'] == [500] * 10, 'val balance'
    assert s['statistics_fit_count'] == 45000, 'statistics source'
    assert all(0 < x < 1 for x in s['mean'] + s['std']), 'statistics range'


def validate_predictions(rows, metrics):
    assert len(rows) == 10000, 'prediction count'
    assert [int(r['index']) for r in rows] == list(range(10000)), 'prediction order'
    cm = [[0] * 10 for _ in range(10)]
    for r in rows:
        t, p = int(r['target']), int(r['prediction'])
        assert 0 <= t < 10 and 0 <= p < 10, 'label range'
        prob = float(r['confidence'])
        assert math.isfinite(prob) and 0 <= prob <= 1, 'confidence'
        probabilities = [float(r[f'p_{i}']) for i in range(10)]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities), 'probability range'
        assert abs(sum(probabilities) - 1) < 2e-6, 'probability sum'
        assert p == max(range(10), key=lambda i: probabilities[i]), 'prediction/probability mismatch'
        assert abs(prob - probabilities[p]) < 1e-7, 'confidence/probability mismatch'
        cm[t][p] += 1
    assert cm == metrics['confusion_matrix'], 'confusion matrix mismatch'
    assert all(sum(row) == 1000 for row in cm), 'test class balance'
    acc = sum(cm[i][i] for i in range(10)) / 10000
    assert abs(acc - metrics['accuracy']) < 1e-12, 'accuracy mismatch'
    for i, row in enumerate(metrics['per_class']):
        assert abs(cm[i][i] / 1000 - row['accuracy']) < 1e-12, 'class accuracy'


def validate_report_claims(text, metrics, selection, runs):
    assert f'{100 * metrics["accuracy"]:.2f}%' in text, 'report overall accuracy missing'
    assert f'{metrics["correct"]}/10,000' in text, 'report correct count missing'
    assert f'{metrics["macro_f1"]:.4f}' in text, 'report macro F1 missing'
    for row in metrics['per_class']:
        assert f'{100 * row["accuracy"]:.2f}%' in text, 'report class accuracy missing'
    for row in runs:
        assert f'{100 * row["best_val_accuracy"]:.2f}%' in text, 'report validation accuracy missing'
    assert f'配置 {selection["id"]}' in text, 'report selected model missing'


def validate_report_format(markdown, pdf):
    count = len(pdf.pages)
    if not 1 <= count <= REPORT_MAX_PAGES:
        raise AssertionError(f'report has {count} pages, expected 1 to {REPORT_MAX_PAGES}')
    lines = (line.strip() for line in markdown.splitlines() if line.strip())
    opening = next(lines, '')
    if opening.startswith('# '):
        opening = next(lines, '')
    if f']({REPOSITORY_URL})' not in opening and f'<{REPOSITORY_URL}>' not in opening:
        raise AssertionError('Markdown opening is missing the repository link')

    first = pdf.pages[0]
    # The published template places the active link in the opening area beneath the title.
    opening_bottom = float(first.cropbox.top) - float(first.cropbox.height) / 4
    for ref in first.get('/Annots', []):
        annotation = ref.get_object()
        action = annotation.get('/A', {})
        if (annotation.get('/Subtype') == '/Link' and action.get('/S') == '/URI'
                and action.get('/URI') == REPOSITORY_URL and float(annotation['/Rect'][1]) >= opening_bottom):
            return
    raise AssertionError('PDF opening is missing the clickable repository link')


def run(stage):
    checks, errors = [], []

    def check(name, fn):
        try:
            fn()
            checks.append(name)
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    check('source_pdf', lambda: require(sha256('../当代人工智能作业三.pdf') == PDF_HASH, 'source changed'))
    for p in ['../AGENTS.md', '../WORKFLOW.md', 'environment.yml', 'config.py']:
        check(p, lambda p=p: require(Path(p).is_file(), 'missing file'))
    check('environment', lambda: require(read_json('results/logs/environment.json')['torch'] == '2.5.1', 'unexpected torch'))
    if stage >= 1:
        check('split', lambda: validate_split(read_json('results/logs/split.json')))
        check('data_integrity', lambda: require(read_json('results/logs/data_integrity.json')['verified'], 'unverified data'))
    if stage >= 2:
        check('invariants', lambda: require(read_json('results/logs/tests.json')['failures'] == 0, 'failed tests'))
        check('runtime', lambda: require(read_json('results/logs/runtime.json')['threads'] in [2, 4, 6], 'thread choice'))
    if stage >= 3:
        from config import EXPERIMENTS, EPOCHS
        for key in (['A'] if stage == 3 else list(EXPERIMENTS)):
            def run_check(key=key):
                s = read_json(f'results/logs/run_{key}.json')
                with Path(f'results/tables/history_{key}.csv').open(encoding='utf-8', newline='') as f:
                    rows = list(csv.DictReader(f))
                require(s['status'] == 'complete', 'run incomplete')
                require(s['config'] == EXPERIMENTS[key], 'config changed')
                require(len(rows) == EPOCHS, 'missing epochs')
                require([int(r['epoch']) for r in rows] == list(range(1, EPOCHS + 1)), 'epoch order')
                for r in rows:
                    for k in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'seconds']:
                        require(math.isfinite(float(r[k])), 'non-finite metric')
                    require(int(r['train_count']) == 45000 and int(r['val_count']) == 5000, 'sample count')
                best = min(rows, key=lambda r: (-float(r['val_accuracy']), float(r['val_loss']), int(r['epoch'])))
                require(int(best['epoch']) == s['best_epoch'], 'best epoch mismatch')
                require(abs(float(best['val_accuracy']) - s['best_val_accuracy']) < 1e-12, 'best accuracy mismatch')
            check(f'run_{key}', run_check)
    if stage >= 4:
        def comparable():
            from config import PROTOCOL, TRAINING_SOURCES
            runs = [read_json(f'results/logs/run_{k}.json') for k in 'ABCD']
            for field in ['split_sha256', 'training_protocol', 'source_sha256', 'threads']:
                require(all(r[field] == runs[0][field] for r in runs), f'incomparable {field}')
            require(runs[0]['training_protocol'] == PROTOCOL, 'registered protocol changed')
            require(runs[0]['source_sha256'] == {n: sha256(n) for n in TRAINING_SOURCES}, 'training source changed')
        check('controlled_comparison', comparable)
    if stage >= 5:
        def selection():
            runs = [read_json(f'results/logs/run_{k}.json') for k in 'ABCD']
            best = min(runs, key=lambda r: (-r['best_val_accuracy'], r['best_val_loss'], r['parameters'], r['id']))
            sel = read_json('results/logs/selection.json')
            require(sel['id'] == best['id'], 'selection differs from validation rule')
            require(sel['test_used'] is False, 'test used in selection')
            require(sel['checkpoint_sha256'] == best['checkpoint_sha256'], 'checkpoint mismatch')
        check('validation_only_selection', selection)
    if stage >= 6:
        def test_check():
            with Path('results/tables/test_predictions.csv').open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            m = read_json('results/logs/test_metrics.json')
            validate_predictions(rows, m)
            require(m['selection_sha256'] == sha256('results/logs/selection.json'), 'selection changed')
        check('test_metrics', test_check)
    if stage >= 7:
        for p in ['REPORT.md', 'output/pdf/Lab3_report.pdf', 'results/figures/learning_curves.png',
                  'results/figures/confusion_matrix.png', 'results/figures/ablation.png',
                  'results/figures/misclassified.png', 'results/figures/gradcam.png']:
            check(p, lambda p=p: require(Path(p).is_file() and Path(p).stat().st_size > 500, 'missing/empty artifact'))
        def report_check():
            from pypdf import PdfReader
            m = read_json('results/logs/test_metrics.json')
            s = read_json('results/logs/selection.json')
            runs = [read_json(f'results/logs/run_{k}.json') for k in 'ABCD']
            markdown = Path('REPORT.md').read_text(encoding='utf-8')
            validate_report_claims(markdown, m, s, runs)
            pdf = PdfReader('output/pdf/Lab3_report.pdf')
            validate_report_format(markdown, pdf)
            validate_report_claims(''.join(p.extract_text() for p in pdf.pages), m, s, runs)
            manifest = read_json('results/logs/report_manifest.json')
            require(manifest['page_count'] == len(pdf.pages), 'report page count changed')
            require(manifest['pdf_sha256'] == sha256('output/pdf/Lab3_report.pdf'), 'PDF changed')
            require(manifest['markdown_sha256'] == sha256('REPORT.md'), 'report changed')
            for name, expected in manifest['inputs'].items():
                require(expected == sha256(Path('results/logs') / name), f'report evidence changed: {name}')
        check('report_evidence', report_check)
        def visual_check():
            review = read_json('results/logs/visual_review.json')
            require(review['passed'] is True, 'not reviewed')
            require(review['pdf_sha256'] == sha256('output/pdf/Lab3_report.pdf'), 'PDF changed since review')
            count = read_json('results/logs/report_manifest.json')['page_count']
            require(review['pages'] == list(range(1, count + 1)), 'incomplete visual page review')
            require(review['figure_sha256'] == read_json('results/logs/analysis.json')['figure_sha256'], 'figures changed since review')
        check('visual_review', visual_check)
        def explanation_check():
            a = read_json('results/logs/analysis.json')
            require(len(a['errors']) == 10 and len(a['gradcam']) == 4, 'missing explanation examples')
            for cam in a['gradcam']:
                require(cam['explained_class'] == cam['prediction'], 'wrong CAM target')
                require(cam['gradient_l1'] > 0, 'missing CAM gradient')
            audit = read_json('results/logs/image_label_audit.json')
            require(audit['all_images_equal'] and audit['all_labels_equal'] and audit['classes_equal'], 'decoder mismatch')
        check('explanation_and_decoder', explanation_check)
    if stage >= 8:
        check('reproduction', lambda: require(read_json('results/logs/reproduction.json')['passed'] is True, 'not reproduced'))
    out = {'stage': stage, 'time_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks, 'errors': errors,
           'scope': 'mechanical checks only; visual and scientific interpretation separately reviewed'}
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    Path(f'results/logs/rigor_stage_{stage}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Stage {stage}: PASS={len(checks)} ERROR={len(errors)}', flush=True)
    for err in errors:
        print(err)
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, type=int, choices=range(9))
    raise SystemExit(run(parser.parse_args().stage))
