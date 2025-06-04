import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset.dataset import SpriteDataset
from models.simpleCNN import SimpleCNN
from models.simpleNN import SimpleNN
from training.train import train_sprites
from evaluation.test import test_model
from utils.util import print_dataset, pick_model_menu
from config import Config

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()

print(f"Using device: {device}")

def main():
    config = Config()

    model_architecture_choice = pick_model_architecture_menu(config)

    print(f"🚀 Starting new training run: {config.manager.run_id}")
    print(f"📁 Run directory: {config.manager.run_dir}")
    print("-" * 60)

    config_log_path = config.manager.get_log_path("config")
    with open(config_log_path, "w") as f:
        f.write("Training Configuration\n")
        f.write("====================\n")
        f.write(f"Run ID: {config.manager.run_id}\n")
        f.write(f"Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Epochs: {config.EPOCHS}\n")
        f.write(f"Transform normalization mean: {config.NORMALIZE_MEAN}\n")
        f.write(f"Transform normalization std: {config.NORMALIZE_STD}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Use Colab: {config.USE_COLAB}\n")
        f.write(f"Train Path: {config.get_train_path()}\n")
        f.write(f"Validation Path: {config.get_validation_path()}\n")
        f.write(f"Test Path: {config.get_test_path()}\n")

    print("Loading dataset...")
    train_dataset = SpriteDataset(config.get_train_path(), config.get_transform())
    validation_dataset = SpriteDataset(config.get_test_path(), config.get_transform())
    test_dataset = SpriteDataset(config.get_test_path(), config.get_transform())

    print("++ Testing Train dataset ++")
    print_dataset(train_dataset)

    print("++ Testing Validation dataset ++")
    print_dataset(validation_dataset)

    print("++ Testing Test dataset ++")
    print_dataset(test_dataset)

    train_data = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_TRAIN)
    validation_data = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_TEST)
    test_data = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_TEST)

    model = None
    if model_choice == config.MODEL_ARCHITECTURE_FCN:
        model = SimpleNN().to(device)
    elif model_choice == config.MODEL_ARCHITECTURE_CNN:
        model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Starting training for {config.EPOCHS} epochs...")
    train_losses = train_sprites(model, train_data, criterion, optimizer, device, config, config.EPOCHS)

    print("+---------------------------------------+")
    print("\nTesting the model...")
    validation_accuracy = test_model(model, validation_data, device, config)
    test_accuracy = test_model(model, test_data, device, config)

    if config.SAVE_MODEL:
        model_path = config.manager.get_model_path("final_model")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

        model_info_path = config.manager.get_log_path("model_info")
        with open(model_info_path, "w") as f:
            f.write("Model Information\n")
            f.write("=================\n")
            f.write(f"Model Architecture: {model_architecture_choice}")
            f.write(f"Final Validation Accuracy: {validation_accuracy:.2f}%\n")
            f.write(f"Final Test Accuracy: {test_accuracy:.2f}%\n")
            f.write(f"Training Epochs: {config.EPOCHS}\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n")
            f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    print("\nTraining completed!")
    print(f"Final Validation Accuracy: {validation_accuracy:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"All files saved to: {config.manager.run_dir}")
    print(f"   Plots: {config.manager.plots_dir}")
    print(f"   Model: {config.manager.models_dir}")
    print(f"   Logs: {config.manager.logs_dir}")

if __name__ == "__main__":
    main()
