"""Independent torchvision decoding audit; no prediction, tuning or relabeling."""
import numpy as np
from torchvision.datasets import CIFAR10

from config import DATA, LOGS, CLASSES
from data import load_raw, read_json, save_json


def audit():
    # Final selection must already exist; load_raw enforces the test seal.
    raw, labels = load_raw(False)
    official_loader = CIFAR10(root=str(DATA), train=False, download=False)
    image_match = np.array_equal(raw.transpose(0, 2, 3, 1), official_loader.data)
    label_match = np.array_equal(labels, np.asarray(official_loader.targets))
    assert image_match and label_match and official_loader.classes == CLASSES
    examples = read_json(LOGS / 'analysis.json')['errors']
    save_json(LOGS / 'image_label_audit.json', {
        'count': len(labels), 'all_images_equal': image_match, 'all_labels_equal': label_match,
        'classes_equal': official_loader.classes == CLASSES,
        'independent_reader': 'torchvision.datasets.CIFAR10 0.20.1',
        'examples': [{'index': r['index'], 'official_label': int(official_loader.targets[r['index']]),
                      'label_name': CLASSES[int(official_loader.targets[r['index']])]} for r in examples],
        'interpretation': 'Confirms image/label alignment with the official files; does not establish semantic label correctness.',
        'relabeling_performed': False, 'model_reevaluated': False,
    })
    print('All 10,000 images, labels and class names match the independent torchvision decoder.')


if __name__ == '__main__':
    audit()
