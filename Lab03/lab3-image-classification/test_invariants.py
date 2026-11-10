"""Scientific invariants and negative controls, using unittest (no extra framework)."""
import copy
import hashlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import data
from config import LOGS, PROBE_SEED, BATCH_SIZE
from data import read_json, save_json, fit_statistics, transform
from models import make_model
from engine import seed_all, evaluate, checkpoint_key, selection_key, classification_metrics, lr_for_epoch
from check_rigor import validate_split, validate_predictions


def deterministic_smoke():
    """Four synthetic optimizer steps; used for independent-environment replay."""
    torch.set_num_threads(2)
    seed_all(PROBE_SEED)
    model = make_model('residual')
    rng = torch.Generator().manual_seed(PROBE_SEED)
    x = torch.rand(16, 3, 32, 32, generator=rng).contiguous(memory_format=torch.channels_last)
    y = torch.arange(16) % 10
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    losses = []
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().contiguous().numpy().tobytes())
    return dict(losses=losses, state_sha256=h.hexdigest(), steps=4, samples=16, seed=PROBE_SEED,
                scope='synthetic short training only, not full CIFAR-10 reproduction')


class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_split_and_negative_leakage(self):
        split = read_json(LOGS / 'split.json')
        validate_split(split)
        bad = copy.deepcopy(split)
        bad['train_indices'][0] = bad['val_indices'][0]
        with self.assertRaisesRegex(AssertionError, 'leakage'):
            validate_split(bad)
        bad = copy.deepcopy(split)
        bad['train_indices'][0] = bad['train_indices'][1]
        with self.assertRaisesRegex(AssertionError, 'duplicates'):
            validate_split(bad)

    def test_training_only_statistics(self):
        x = np.zeros((4, 3, 32, 32), dtype=np.uint8)
        x[1] = 128
        x[2:] = 255
        a = fit_statistics(x, [0, 1])
        x[2:] = 0  # Change held-out samples: training mean/std must not change.
        b = fit_statistics(x, [0, 1])
        self.assertEqual(a, b)
        self.assertAlmostEqual(a[0][0], 64 / 255)

    def test_test_set_is_sealed(self):
        sealed = Path('../tmp/sealed-test-no-selection')
        self.assertFalse((sealed / 'selection.json').exists())
        with patch.object(data, 'LOGS', sealed):
            with self.assertRaisesRegex(RuntimeError, 'sealed'):
                data.load_raw(False)

    def test_transform_determinism_and_augmentation(self):
        x = torch.arange(8 * 3 * 32 * 32).reshape(8, 3, 32, 32).remainder(256).byte()
        mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
        plain = transform(x, mean, std)
        self.assertTrue(torch.equal(plain, x.float() / 255))
        self.assertTrue(torch.equal(plain, transform(x, mean, std)))
        a = transform(x, mean, std, True, torch.Generator().manual_seed(PROBE_SEED))
        b = transform(x, mean, std, True, torch.Generator().manual_seed(PROBE_SEED))
        self.assertEqual(a.shape, (8, 3, 32, 32))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, plain))
        self.assertTrue(torch.all((a >= 0) & (a <= 1)))
        with self.assertRaises(ValueError):
            transform(x, mean, std, True)

    def test_models_gradients_and_eval_state(self):
        for architecture in ['plain', 'residual']:
            seed_all(PROBE_SEED)
            model = make_model(architecture)
            x = torch.randn(4, 3, 32, 32)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (4, 10))
            torch.nn.functional.cross_entropy(logits, torch.arange(4)).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            model.eval()
            before = {n: v.clone() for n, v in model.state_dict().items()}
            a, b = model(x), model(x)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(all(torch.equal(v, model.state_dict()[n]) for n, v in before.items()))

    def test_metrics_against_hand_computation(self):
        y = np.asarray([0, 0, 1, 1])
        p = np.asarray([0, 1, 1, 1])
        m = classification_metrics(y, p)
        self.assertEqual(m['accuracy'], 0.75)
        self.assertEqual(m['confusion_matrix'][0][:2], [1, 1])
        self.assertEqual(m['per_class'][0]['accuracy'], 0.5)
        self.assertAlmostEqual(m['per_class'][1]['f1'], 0.8)

    def test_prediction_negative_controls(self):
        y = np.repeat(np.arange(10), 1000)
        m = classification_metrics(y, y)
        rows = [dict(index=i, target=int(t), prediction=int(t), confidence=1.0) for i, t in enumerate(y)]
        for row in rows:
            row.update({f'p_{i}': float(i == row['target']) for i in range(10)})
        validate_predictions(rows, m)
        bad = copy.deepcopy(rows)
        bad[0]['prediction'] = 1
        with self.assertRaisesRegex(AssertionError, 'prediction/probability mismatch'):
            validate_predictions(bad, m)
        bad = copy.deepcopy(rows)
        bad[0]['p_0'] = 0.5
        with self.assertRaisesRegex(AssertionError, 'probability sum'):
            validate_predictions(bad, m)
        bad_metrics = copy.deepcopy(m)
        bad_metrics['confusion_matrix'][0][0] -= 1
        with self.assertRaisesRegex(AssertionError, 'matrix mismatch'):
            validate_predictions(rows, bad_metrics)
        bad = copy.deepcopy(rows)
        bad[0]['index'] = 1
        with self.assertRaisesRegex(AssertionError, 'order'):
            validate_predictions(bad, m)

    def test_selection_ties_and_schedule(self):
        self.assertLess(checkpoint_key(.8, .5, 2), checkpoint_key(.8, .6, 1))
        self.assertLess(checkpoint_key(.8, .5, 1), checkpoint_key(.8, .5, 2))
        a = dict(best_val_accuracy=.8, best_val_loss=.5, parameters=100, id='A')
        b = dict(a, id='B', parameters=101)
        self.assertLess(selection_key(a), selection_key(b))
        self.assertAlmostEqual(lr_for_epoch(1), .05)
        self.assertAlmostEqual(lr_for_epoch(2), .1)
        self.assertAlmostEqual(lr_for_epoch(20), .001)
        self.assertTrue(all(lr_for_epoch(i + 1) < lr_for_epoch(i) for i in range(2, 20)))

    def test_uneven_final_batch_weighting(self):
        model = make_model('plain')
        x = torch.zeros(BATCH_SIZE + 1, 3, 32, 32, dtype=torch.uint8)
        y = torch.arange(BATCH_SIZE + 1) % 10
        m = evaluate(model, x, y, torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1))
        with torch.no_grad():
            logits = model(x.float())
            reference = torch.nn.functional.cross_entropy(logits, y).item()
        self.assertEqual(m['count'], BATCH_SIZE + 1)
        self.assertAlmostEqual(m['loss'], reference, places=5)

    def test_repeated_short_training_is_deterministic(self):
        a, b = deterministic_smoke(), deterministic_smoke()
        self.assertEqual(a, b)
        self.assertLess(a['losses'][-1], a['losses'][0])


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Invariants)
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    log = dict(tests=result.testsRun, failures=len(result.failures) + len(result.errors), output=buffer.getvalue())
    save_json(LOGS / 'tests.json', log)
    print(buffer.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise RuntimeError('Invariant tests failed')
    save_json(LOGS / 'smoke_reference.json', deterministic_smoke())


if __name__ == '__main__':
    run_tests()
