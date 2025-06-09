import torch

from dataset.dataset import SpriteDataset
from training.train import train_final_model
from utils.util import pick_model_architecture_menu, log_hyperparameters_config
from configurations.config import Config
from configurations.hyperparameters import HyperparametersConfig
from evaluation.k_fold_cv import *

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

    # Load datasets
    print("Loading datasets...")
    full_train_dataset = SpriteDataset(config.get_train_path(), hyperparameters_config.get_transform())
    test_dataset = SpriteDataset(config.get_test_path(), hyperparameters_config.get_transform())
    
    print(f"Total training samples: {len(full_train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")

    # Set number of folds for cross validation
    k_folds = hyperparameters_config.K_FOLD
    
    # Perform K-fold cross validation
    fold_results = perform_kfold_cross_validation(
        k_folds=k_folds,
        full_train_dataset=full_train_dataset,
        test_dataset=test_dataset,
        model_architecture_choice=model_architecture_choice,
        hyperparameters_config=hyperparameters_config,
        config=config,
        device=device
    )
    
    # Print and save K-fold results
    print_kfold_summary(fold_results)
    save_kfold_results(fold_results, config)
    
    # Train final model on full training data
    final_model, test_accuracy, test_directions_acc, test_char_acc = train_final_model(
        full_train_dataset=full_train_dataset,
        test_dataset=test_dataset,
        model_architecture_choice=model_architecture_choice,
        hyperparameters_config=hyperparameters_config,
        config=config,
        device=device
    )
    
    # Print final test results
    print("\n" + "=" * 50)
    print("FINAL MODEL TEST RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    if test_directions_acc:
        print("Test Accuracies per Direction:")
        for direction, data in sorted(test_directions_acc.items()):
            if isinstance(data, dict) and 'accuracy' in data:
                print(f"  - {direction}: {data['accuracy']:.2f}%")
            else:
                print(f"  - {direction}: {data:.2f}%")
    
    if test_char_acc:
        print("Test Accuracies per Character:")
        for char, data in test_char_acc.items():
            if isinstance(data, dict) and 'accuracy' in data:
                print(f"  - '{char}': {data['accuracy']:.2f}%")
            else:
                print(f"  - '{char}': {data:.2f}%")
    
    print(f"\nAll files saved to: {str(config.manager.run_dir)}")
    print(f"   Plots: {str(config.manager.plots_dir)}")
    print(f"   Models: {str(config.manager.models_dir)}")
    print(f"   Logs:  {str(config.manager.logs_dir)}")

if __name__ == "__main__":
    main()