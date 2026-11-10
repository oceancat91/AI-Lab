"""Preregistered protocol: change only before formal training."""
from pathlib import Path

SEED = 42
SPLIT_SEED = SEED
ORDER_SEED = SEED + 1
AUGMENT_SEED = SEED + 2
PROBE_SEED = SEED + 3
BATCH_SIZE = 256
EPOCHS = 20
CHANNELS = (16, 32, 64)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
EXPERIMENTS = {
    'A': {'architecture': 'plain', 'augmentation': 'none'},
    'B': {'architecture': 'plain', 'augmentation': 'crop_flip'},
    'C': {'architecture': 'residual', 'augmentation': 'none'},
    'D': {'architecture': 'residual', 'augmentation': 'crop_flip'},
}
PROTOCOL = dict(seed=SEED, split_seed=SPLIT_SEED, order_seed=ORDER_SEED, augment_seed=AUGMENT_SEED,
                batch_size=BATCH_SIZE, epochs=EPOCHS, channels=list(CHANNELS), optimizer='SGD',
                lr=0.1, min_lr=0.001, momentum=0.9, weight_decay=0.0005,
                warmup_epochs=2, schedule='cosine', loss='cross_entropy', dtype='float32')
DATA = Path('data')
CHECKPOINTS = Path('checkpoints')
LOGS = Path('results/logs')
TABLES = Path('results/tables')
FIGURES = Path('results/figures')
TRAINING_SOURCES = ['config.py', 'data.py', 'models.py', 'engine.py']


def make_dirs():
    for p in [DATA, CHECKPOINTS, LOGS, TABLES, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)
