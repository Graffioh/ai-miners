from torchvision import transforms

class Config:
    # ============================================================================
    # PATHS
    # ============================================================================
    USE_COLAB = False  # Set to True when running on Colab

    # Local paths
    LOCAL_TRAIN_PATH = "./dataset/train"
    LOCAL_TEST_PATH = "./dataset/test"

    # Colab paths
    COLAB_TRAIN_PATH = "/content/drive/MyDrive/shadowless/train"
    COLAB_TEST_PATH = "/content/drive/MyDrive/shadowless/test"

    @classmethod
    def get_train_path(cls):
        return cls.COLAB_TRAIN_PATH if cls.USE_COLAB else cls.LOCAL_TRAIN_PATH

    @classmethod
    def get_test_path(cls):
        return cls.COLAB_TEST_PATH if cls.USE_COLAB else cls.LOCAL_TEST_PATH

    # ============================================================================
    # TRAINING
    # ============================================================================
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 5

    # ============================================================================
    # DATA
    # ============================================================================
    SHUFFLE_TRAIN = True
    SHUFFLE_TEST = False

    # Normalization values
    NORMALIZE_MEAN = [0.013441166840493679, 0.010885078459978104,
                     0.010833792388439178, 0.04079267755150795]
    NORMALIZE_STD = [0.07362207025289536, 0.06255189329385757,
                    0.0627019852399826, 0.18982279300689697]

    @classmethod
    def get_transform(cls):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.NORMALIZE_MEAN, std=cls.NORMALIZE_STD)
        ])

    # ============================================================================
    # SAVING
    # ============================================================================
    SAVE_MODEL = True
    MODEL_SAVE_PATH = "./saved_model.pth"
