from torchvision import transforms

from typing import Optional, Dict, Any

class HyperparametersConfig:
    def __init__(self, base_dir='runs', defaults: Optional[Dict[str, Any]] = None):
        # Initialize defaults if None
        if defaults is None:
            defaults = {}

        # ============================================================================
        # TRAINING
        # ============================================================================
        self.LEARNING_RATE = defaults.get('LEARNING_RATE', 0.001)
        self.WEIGHT_DECAY = defaults.get('WEIGHT_DECAY', 0.00025)
        self.BATCH_SIZE = defaults.get('BATCH_SIZE', 128)
        self.EPOCHS = 1

        # ============================================================================
        # DATA
        # ============================================================================
        self.SHUFFLE_TRAIN = defaults.get('SHUFFLE_TRAIN', True)
        self.SHUFFLE_TEST = defaults.get('SHUFFLE_TEST', False)

        # Starting Normalization values
        self.NORMALIZE_MEAN = defaults.get('NORMALIZE_MEAN', [0.5, 0.5, 0.5])
        self.NORMALIZE_STD = defaults.get('NORMALIZE_STD', [0.5, 0.5, 0.5])

    def get_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD)
        ])