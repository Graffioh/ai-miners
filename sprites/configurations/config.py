import os
from datetime import datetime
from pathlib import Path

class RunConfigManager:
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
        self.manager = RunConfigManager(base_dir)

        # Initialize defaults if None
        if defaults is None:
            defaults = {}

        # ============================================================================
        # PATHS
        # ============================================================================
        self.TRAIN_PATH = defaults.get('TRAIN_PATH', "./dataset/shadowless_template/train")
        self.TEST_PATH = defaults.get('TEST_PATH', "./dataset/shadowless_template/test")
        self.PLOT_DIR = defaults.get('PLOT_DIR', "./plots")

        # ============================================================================
        # EXTRA
        # ============================================================================
        self.SAVE_MODEL = defaults.get('SAVE_MODEL', True)
        self.ENABLE_PLOTTING = defaults.get('ENABLE_PLOTTING', True)

    def get_plot_dir(self):
        os.makedirs(self.PLOT_DIR, exist_ok=True)
        return self.PLOT_DIR

    def get_train_path(self):
        return self.TRAIN_PATH

    def get_test_path(self):
        return self.TEST_PATH