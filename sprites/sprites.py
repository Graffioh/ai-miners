import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from dataset.dataset import SpriteDataset
from models.simpleCNN import SimpleCNN
from models.simpleNN import SimpleNN
from models.simpleNN_BN import SimpleNN_BN
from training.train import train_sprites
from evaluation.test import test_model 
from utils.util import pick_model_architecture_menu, save_training_results
from config import Config

torch.manual_seed(42)

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
        f.write("Training Configuration\n====================\n")
        f.write(f"Run ID: {config.manager.run_id}\n")
        f.write(f"Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Epochs: {config.EPOCHS}\n")
        f.write(f"Transform normalization mean: {config.NORMALIZE_MEAN}\n")
        f.write(f"Transform normalization std: {config.NORMALIZE_STD}\n")
        f.write(f"Device: {device}\n")

    print("Loading datasets...")
    full_train_dataset = SpriteDataset(config.get_train_path(), config.get_transform())
    num_total_train_samples = len(full_train_dataset)
    num_validation_samples = int(0.2 * num_total_train_samples)
    num_train_samples = num_total_train_samples - num_validation_samples

    if num_train_samples <= 0 or num_validation_samples <= 0:
        print("❌ Error: Dataset too small to split.")
        return
        
    generator = torch.Generator().manual_seed(42) 
    train_dataset, validation_dataset = random_split(full_train_dataset, 
                                                      [num_train_samples, num_validation_samples],
                                                      generator=generator)
    test_dataset = SpriteDataset(config.get_test_path(), config.get_transform())

    train_data = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_TRAIN)
    validation_data = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_TEST) 
    test_data = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE_TEST)

    model = None
    if model_architecture_choice == config.MODEL_ARCHITECTURE_FCN:
        model = SimpleNN().to(device)
    elif model_architecture_choice == config.MODEL_ARCHITECTURE_FCN_BN:
        model = SimpleNN_BN().to(device)
    elif model_architecture_choice == config.MODEL_ARCHITECTURE_CNN:
        model = SimpleCNN().to(device)
    
    if model is None:
        print(f"❌ Error: Model architecture '{model_architecture_choice}' not recognized. Exiting.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\nStarting training for {config.EPOCHS} epochs with model {model_architecture_choice}...")
    train_losses = train_sprites(model, train_data, criterion, optimizer, device, config, config.EPOCHS)

    print("+---------------------------------------+")
    print("\nEvaluating on Validation Set...")
    overall_validation_accuracy, validation_dir_accuracies, validation_char_accuracies = test_model(model, validation_data, device, config=None) 
    
    print("\nEvaluating on Test Set...")
    overall_test_accuracy, test_dir_accuracies, test_char_accuracies = test_model(model, test_data, device, config)
    
    save_training_results(
        config,
        model,
        model_architecture_choice,
        overall_validation_accuracy,
        validation_dir_accuracies,
        validation_char_accuracies,
        overall_test_accuracy,
        test_dir_accuracies,
        test_char_accuracies
    )

    print("\nTraining completed!")
    print(f"\n--- Final Validation Set Performance ---")
    print(f"Overall Validation Accuracy: {overall_validation_accuracy:.2f}%")
    if validation_char_accuracies:
        print("Validation Accuracies per Character (direction prediction):")
        for char, acc in validation_char_accuracies.items():
            print(f"  - '{char}': {acc:.2f}%")

    print(f"\n--- Final Test Set Performance ---")
    print(f"Overall Test Accuracy: {overall_test_accuracy:.2f}%")
    if test_char_accuracies:
        print("Test Accuracies per Character (direction prediction):")
        for char, acc in test_char_accuracies.items():
            print(f"  - '{char}': {acc:.2f}%")
            
    print(f"\nAll files saved to: {config.manager.run_dir}")
    print(f"   Plots: {config.manager.plots_dir}")
    print(f"   Model: {config.manager.models_dir}")
    print(f"   Logs:  {config.manager.logs_dir}")

if __name__ == "__main__":
    main()