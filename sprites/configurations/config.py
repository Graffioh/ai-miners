from dataclasses import dataclass, field

@dataclass(frozen=True)
class MainConfig:
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.00025
    transform_normalization_mean_3: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    transform_normalization_std_3: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    transform_normalization_mean_4: list = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    transform_normalization_std_4: list = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])

#@dataclass(frozen=True)
#class ModelConfig:

@dataclass(frozen=True)
class MiscConfig:
    dataset_train_path: str = "./dataset/shadowless_template/train"
    dataset_test_path: str = "./dataset/shadowless_template/test"
    is_alpha_enabled: bool = False

@dataclass(frozen=True)
class NeurodragonConfig:
    main: MainConfig = MainConfig()
    misc: MiscConfig = MiscConfig()
