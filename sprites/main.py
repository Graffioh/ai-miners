import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import SpriteDataset
from utils.util import log_hyperparameters_config, print_final_results
from utils.plotter import TrainingPlotter
from configurations.config import Config
from configurations.hyperparameters import HyperparametersConfig
from evaluation.k_fold_cv import *
from evaluation.evaluation_orchestrator import evaluate_model
from models.spritesCNN import SpritesCNN

#torch.manual_seed(42)

def get_device():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

def train_and_evaluate_model(full_train_dataset, test_dataset, model, 
                                   hyperparameters_config, config, device):
    """Train model on full training data and evaluate on test set"""
    print("\n=== Training & Testing on Full Training Data ===")
    
    # Create data loader for full training data
    full_train_loader = DataLoader(full_train_dataset, 
                                 batch_size=hyperparameters_config.BATCH_SIZE, 
                                 shuffle=hyperparameters_config.SHUFFLE_TRAIN)
    
    test_loader = DataLoader(test_dataset, 
                           batch_size=hyperparameters_config.BATCH_SIZE, 
                           shuffle=hyperparameters_config.SHUFFLE_TEST)
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters_config.LEARNING_RATE, weight_decay=hyperparameters_config.WEIGHT_DECAY)
    #optimizer = optim.Adam(model.parameters(), lr=hyperparameters_config.LEARNING_RATE)
    
    # Train model
    print("Training final model on full training data...")
    plotter = TrainingPlotter()
    train_sprites(model, full_train_loader, criterion, optimizer, device, config, hyperparameters_config.EPOCHS, plotter)
    
    # Evaluate on test set
    print("Evaluating final model on test set...")
    test_accuracy, test_directions_acc, test_char_acc = evaluate_model(model, test_loader, criterion, device, config)
    
    # Save model
    if config.SAVE_MODEL:
        model_path = config.manager.get_model_path(f"model-save")
        try:
            torch.save(model.state_dict(), model_path)
            print(f"Final model saved to: {model_path}")
        except Exception as e:
            print(f"❌ Error saving final model: {e}")
    
    return model, test_accuracy, test_directions_acc, test_char_acc

def main():
    config = Config()
    hyperparameters_config = HyperparametersConfig()
    log_hyperparameters_config(hyperparameters_config, config.manager, device)

    # Load datasets
    print("Loading datasets...")
    full_train_dataset = SpriteDataset(config.get_train_path(), hyperparameters_config.get_transform())
    test_dataset = SpriteDataset(config.get_test_path(), hyperparameters_config.get_transform())
    
    print(f"Total training samples: {len(full_train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")

    model = SpritesCNN().to(device)

    # Train final model on full training data and evaluate on test set
    _, test_accuracy, test_directions_acc, test_char_acc = train_and_evaluate_model(
        full_train_dataset=full_train_dataset,
        test_dataset=test_dataset,
        model=model,
        hyperparameters_config=hyperparameters_config,
        config=config,
        device=device
    )

    print_final_results(test_accuracy, test_directions_acc, test_char_acc)

if __name__ == "__main__":
    main()