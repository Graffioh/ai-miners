from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.nn as nn
from training.train import create_model, train_sprites
from evaluation.evaluation_orchestrator import evaluate_model
import torch.optim as optim

def perform_kfold_cross_validation(k_folds, full_train_dataset, test_dataset, model_architecture_choice, 
                                   hyperparameters_config, config, device):
    """Perform K-fold cross validation"""
    
    # Create indices for the full training dataset
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    
    # Initialize KFold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    
    print(f"\n=== Starting {k_folds}-Fold Cross Validation ===")
    print(f"Total training samples: {dataset_size}")
    print(f"Approximate samples per fold: {dataset_size // k_folds}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Convert numpy arrays to lists for PyTorch Subset
        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()
        
        # Create data subsets for this fold
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_subset, 
                                batch_size=hyperparameters_config.BATCH_SIZE, 
                                shuffle=hyperparameters_config.SHUFFLE_TRAIN)
        val_loader = DataLoader(val_subset, 
                              batch_size=hyperparameters_config.BATCH_SIZE, 
                              shuffle=False)
        
        # Create a fresh model for this fold
        model = create_model(model_architecture_choice, hyperparameters_config, device)
        
        # Create optimizer and criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters_config.LEARNING_RATE)
        
        # Train the model for this fold
        print(f"Training fold {fold + 1}...")
        train_sprites(model, train_loader, criterion, optimizer, device, config, hyperparameters_config.EPOCHS, plotter=None)
        
        # Evaluate on validation set for this fold
        print(f"Evaluating fold {fold + 1}...")
        val_accuracy, val_directions_acc, val_char_acc = evaluate_model(model, val_loader, device, config=None)
        
        # Store results
        fold_result = {
            'fold': fold + 1,
            'validation_accuracy': val_accuracy,
            'validation_directions_accuracies': val_directions_acc,
            'validation_char_accuracies': val_char_acc,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices)
        }
        fold_results.append(fold_result)
        
        print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.2f}%")
    
    return fold_results

def save_kfold_results(fold_results, config):
    """Save K-fold cross validation results to file"""
    kfold_log_path = config.manager.get_log_path("kfold_results")
    
    with open(kfold_log_path, "w") as f:
        f.write("K-Fold Cross Validation Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Individual fold results
        for result in fold_results:
            f.write(f"Fold {result['fold']}:\n")
            f.write(f"  Training samples: {result['train_samples']}\n")
            f.write(f"  Validation samples: {result['val_samples']}\n")
            f.write(f"  Validation Accuracy: {result['validation_accuracy']:.2f}%\n")
            
            if result['validation_directions_accuracies']:
                f.write(f"  Direction Accuracies:\n")
                for direction, data in sorted(result['validation_directions_accuracies'].items()):
                    if isinstance(data, dict) and 'accuracy' in data:
                        f.write(f"    - {direction}: {data['accuracy']:.2f}%\n")
                    else:
                        f.write(f"    - {direction}: {data:.2f}%\n")
            
            if result['validation_char_accuracies']:
                f.write(f"  Character Accuracies:\n")
                for char, data in sorted(result['validation_char_accuracies'].items()):
                    if isinstance(data, dict) and 'accuracy' in data:
                        f.write(f"    - {char}: {data['accuracy']:.2f}%\n")
                    else:
                        f.write(f"    - {char}: {data:.2f}%\n")
            f.write("\n")
        
        # Summary statistics
        accuracies = [result['validation_accuracy'] for result in fold_results]
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        f.write("Summary Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Validation Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%\n")
        f.write(f"Min Validation Accuracy: {min(accuracies):.2f}%\n")
        f.write(f"Max Validation Accuracy: {max(accuracies):.2f}%\n")
        f.write(f"Individual Fold Accuracies: {[f'{acc:.2f}%' for acc in accuracies]}\n")

def print_kfold_summary(fold_results):
    """Print summary of K-fold cross validation results"""
    print("\n" + "=" * 50)
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print("=" * 50)
    
    accuracies = [result['validation_accuracy'] for result in fold_results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"Number of folds: {len(fold_results)}")
    print(f"Mean Validation Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"Min Validation Accuracy: {min(accuracies):.2f}%")
    print(f"Max Validation Accuracy: {max(accuracies):.2f}%")
    
    print("\nIndividual Fold Results:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i+1}: {result['validation_accuracy']:.2f}%")