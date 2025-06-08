import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from dataset.dataset import SpriteDataset
from models.simpleCNN import SimpleCNN
from models.simpleCNN_BN import SimpleCNN_BN
from models.simpleNN import SimpleNN
from models.simpleNN_BN import SimpleNN_BN
from training.train import train_sprites
from evaluation.evaluation_orchestrator import evaluate_model
from utils.util import pick_model_architecture_menu, log_hyperparameters_config
from config import Config
from hyperparameters import HyperparametersConfig

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
    hyperparameters_config = HyperparametersConfig()
    model_architecture_choice = pick_model_architecture_menu(hyperparameters_config)

    log_hyperparameters_config(hyperparameters_config, config.manager, device, model_architecture_choice)

    # == Train - Validation - Test
    print("Loading datasets...")
    full_train_dataset = SpriteDataset(config.get_train_path(), hyperparameters_config.get_transform())

    num_total_train_samples = len(full_train_dataset)
    num_validation_samples = int(hyperparameters_config.VALIDATION_SPLIT_RATIO * num_total_train_samples)
    num_train_samples = num_total_train_samples - num_validation_samples

    generator = torch.Generator().manual_seed(42) 
    train_dataset, validation_dataset = random_split(full_train_dataset, 
                                                      [num_train_samples, num_validation_samples],
                                                      generator=generator)
    test_dataset = SpriteDataset(config.get_test_path(), hyperparameters_config.get_transform())
    # ==

    # == Data loaders ==
    train_data = DataLoader(train_dataset, batch_size=hyperparameters_config.BATCH_SIZE, shuffle=hyperparameters_config.SHUFFLE_TRAIN)
    validation_data = DataLoader(validation_dataset, batch_size=hyperparameters_config.BATCH_SIZE, shuffle=hyperparameters_config.SHUFFLE_TEST) 
    test_data = DataLoader(test_dataset, batch_size=hyperparameters_config.BATCH_SIZE, shuffle=hyperparameters_config.SHUFFLE_TEST)
    # ==

    # == Model choice ==
    model = None
    if model_architecture_choice == hyperparameters_config.MODEL_ARCHITECTURE_FCN:
        model = SimpleNN().to(device)
    elif model_architecture_choice == hyperparameters_config.MODEL_ARCHITECTURE_FCN_BN:
        model = SimpleNN_BN().to(device)
    elif model_architecture_choice == hyperparameters_config.MODEL_ARCHITECTURE_CNN:
        model = SimpleCNN().to(device)
    elif model_architecture_choice == hyperparameters_config.MODEL_ARCHITECTURE_CNN_BN:
        model = SimpleCNN_BN().to(device)

    if model is None:
        print(f"❌ Error Model architecture ❌: '{model_architecture_choice}' not recognized. Exiting.")
        return
    # ==

    # == Core training - eval flow
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters_config.LEARNING_RATE)

    print(f"\nStarting training for {hyperparameters_config.EPOCHS} epochs with model {model_architecture_choice}...")
    train_sprites(model, train_data, criterion, optimizer, device, config, hyperparameters_config.EPOCHS)
    
    print("\nEvaluating on Validation Set...")
    overall_validation_accuracy, validation_directions_accuracies, validation_char_accuracies = evaluate_model(model, validation_data, device, config=None) 
    
    # == Save validation results ==
    validation_log_path = config.manager.get_log_path("validation")
    with open(validation_log_path, "w") as f:
        f.write("Validation Results\n==================\n")
        f.write(f"Overall Validation Accuracy: {overall_validation_accuracy:.2f}%\n\n")
        
        if validation_directions_accuracies:
            f.write("Validation Accuracies per Direction:\n")
            for direction, data in sorted(validation_directions_accuracies.items()):
                if isinstance(data, dict) and 'accuracy' in data:
                    f.write(f"  - {direction}: {data['accuracy']:.2f}%\n")
                else:
                    f.write(f"  - {direction}: {data:.2f}%\n")
            f.write("\n")
        
        if validation_char_accuracies:
            f.write("Validation Accuracies per Character (for direction prediction):\n")
            for char, data in sorted(validation_char_accuracies.items()):
                if isinstance(data, dict) and 'accuracy' in data:
                    f.write(f"  - {char}: {data['accuracy']:.2f}%\n")
                else:
                    f.write(f"  - {char}: {data:.2f}%\n")
    # ==
    
    print("\nEvaluating on Test Set...")
    overall_test_accuracy, test_directions_accuracies, test_char_accuracies = evaluate_model(model, test_data, device, config)
    # ==
    
    # == Save and print model/results ==
    if config.SAVE_MODEL:
        model_path = config.manager.get_model_path(f"final_model_{model_architecture_choice}")
        try:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
        except Exception as e:
            print(f"❌ Error saving model to {model_path}: {e}")

    print("\nTraining completed!")
    print(f"\n--- Final Validation Set Performance ---")
    print(f"Overall Validation Accuracy: {overall_validation_accuracy:.2f}%")
    if validation_directions_accuracies:
        print("Validation Accuracies per Direction:")
        for direction, data in sorted(validation_directions_accuracies.items()):
            if isinstance(data, dict) and 'accuracy' in data:
                print(f"  - {direction}: {data['accuracy']:.2f}%")
            else:
                print(f"  - {direction}: {data:.2f}%")
    if validation_char_accuracies:
        print("Validation Accuracies per Character (direction prediction):")
        for char, data in validation_char_accuracies.items():
            if isinstance(data, dict) and 'accuracy' in data:
                print(f"  - '{char}': {data['accuracy']:.2f}%")
            else:
                print(f"  - '{char}': {data:.2f}%")

    print(f"\n--- Final Test Set Performance ---")
    print(f"Overall Test Accuracy: {overall_test_accuracy:.2f}%")
    if test_directions_accuracies:
        print("Test Accuracies per Direction:")
        for direction, data in sorted(test_directions_accuracies.items()):
            if isinstance(data, dict) and 'accuracy' in data:
                print(f"  - {direction}: {data['accuracy']:.2f}%")
            else:
                print(f"  - {direction}: {data:.2f}%")
    if test_char_accuracies:
        print("Test Accuracies per Character (direction prediction):")
        for char, data in test_char_accuracies.items():
            if isinstance(data, dict) and 'accuracy' in data:
                print(f"  - '{char}': {data['accuracy']:.2f}%")
            else:
                print(f"  - '{char}': {data:.2f}%")
            
    print(f"\nAll files saved to: {str(config.manager.run_dir)}")
    print(f"   Plots: {str(config.manager.plots_dir)}")
    print(f"   Model: {str(config.manager.models_dir)}")
    print(f"   Logs:  {str(config.manager.logs_dir)}")
    # ==

if __name__ == "__main__":
    main()