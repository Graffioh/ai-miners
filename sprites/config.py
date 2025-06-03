import os
from datetime import datetime
from pathlib import Path
from torchvision import transforms

class RunManager:
    def __init__(self, base_dir="runs"):
        self.base_dir = base_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / self.run_id
        self._create_dirs()

    def _create_dirs(self):
        """Create all required subdirectories"""
        self.models_dir = self.run_dir / "models"
        self.plots_dir = self.run_dir / "plots"
        self.logs_dir = self.run_dir / "logs"

        for d in [self.models_dir, self.plots_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, name):
        return str(self.models_dir / f"{name}.pth")

    def get_plot_path(self, name):
        return str(self.run_dir / "plots" / f"{name}.png")

    def get_log_path(self, name):
        return str(self.run_dir / "logs" / f"{name}.log")

from typing import Optional, Dict, Any

class Config:
    def __init__(self, base_dir='runs', defaults: Optional[Dict[str, Any]] = None):
        self.manager = RunManager(base_dir)

        # Initialize defaults if None
        if defaults is None:
            defaults = {}

        # ============================================================================
        # PATHS
        # ============================================================================
        self.USE_COLAB = False  # Set to True when running on Colab

        # Local paths
        self.TRAIN_PATH = defaults.get('TRAIN_PATH', "./dataset/shadowless/train")
        self.VALIDATION_PATH = defaults.get('VALIDATION_PATH', "./dataset/shadowless/validation")
        self.TEST_PATH = defaults.get('TEST_PATH', "./dataset/shadowless/test")
        self.PLOT_DIR = defaults.get('PLOT_DIR', "./plots")

        # ============================================================================
        # TRAINING
        # ============================================================================
        self.LEARNING_RATE = defaults.get('LEARNING_RATE', 0.001)
        self.BATCH_SIZE = defaults.get('BATCH_SIZE', 128)
        self.EPOCHS = 5

        # ============================================================================
        # DATA
        # ============================================================================
        self.SHUFFLE_TRAIN = defaults.get('SHUFFLE_TRAIN', True)
        self.SHUFFLE_TEST = defaults.get('SHUFFLE_TEST', False)

        # Normalization values
        self.NORMALIZE_MEAN = defaults.get('NORMALIZE_MEAN', [0.013441166840493679, 0.010885078459978104,
                        0.010833792388439178, 0.04079267755150795])
        self.NORMALIZE_STD = defaults.get('NORMALIZE_STD', [0.07362207025289536, 0.06255189329385757,
                        0.0627019852399826, 0.18982279300689697])

        # ============================================================================
        # SAVING
        # ============================================================================
        self.SAVE_MODEL = defaults.get('SAVE_MODEL', True)

        # ============================================================================
        # PLOTTING
        # ============================================================================
        self.ENABLE_PLOTTING = defaults.get('ENABLE_PLOTTING', True)

    def get_train_path(self):
        return self.TRAIN_PATH

    def get_validation_path(self):
        return self.VALIDATION_PATH

    def get_test_path(self):
        return self.TEST_PATH

    def get_plot_dir(self):
        os.makedirs(self.PLOT_DIR, exist_ok=True)
        return self.PLOT_DIR

    def get_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD)
        ])
