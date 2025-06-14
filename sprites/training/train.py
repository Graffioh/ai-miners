import torch
from utils.plotter import TrainingPlotter
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from evaluation.evaluation_orchestrator import evaluate_model
from utils.util import create_model

def train_final_model(full_train_dataset, test_dataset, model_architecture_choice, 
                                   hyperparameters_config, config, device):
    """Train final model on full training data and evaluate on test set"""
    print("\n=== Training Final Model on Full Training Data ===")
    
    # Create data loader for full training data
    full_train_loader = DataLoader(full_train_dataset, 
                                 batch_size=hyperparameters_config.BATCH_SIZE, 
                                 shuffle=hyperparameters_config.SHUFFLE_TRAIN)
    
    test_loader = DataLoader(test_dataset, 
                           batch_size=hyperparameters_config.BATCH_SIZE, 
                           shuffle=hyperparameters_config.SHUFFLE_TEST)
    
    # Create final model
    final_model = create_model(model_architecture_choice, hyperparameters_config, device)
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=hyperparameters_config.LEARNING_RATE)
    #optimizer = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train final model
    print("Training final model on full training data...")
    plotter = TrainingPlotter()
    train_sprites(final_model, full_train_loader, criterion, optimizer, device, config, hyperparameters_config.EPOCHS, plotter)
    
    # Evaluate on test set
    print("Evaluating final model on test set...")
    test_accuracy, test_directions_acc, test_char_acc = evaluate_model(final_model, test_loader, device, config)
    
    # Save final model
    if config.SAVE_MODEL:
        final_model_path = config.manager.get_model_path(f"final_model_{model_architecture_choice}")
        try:
            torch.save(final_model.state_dict(), final_model_path)
            print(f"Final model saved to: {final_model_path}")
        except Exception as e:
            print(f"❌ Error saving final model: {e}")
    
    return final_model, test_accuracy, test_directions_acc, test_char_acc

def train_sprites(model, data_loader, criterion, optimizer, device, config, epochs=5, plotter=None):
    """
    Train the model with integrated plotting and saving via RunManager
    """
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (_, img_data, target, _) in enumerate(data_loader):
            # Move data to device (GPU if available)
            img_data, target = img_data.to(device), target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(img_data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Print progress
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(data_loader)}, '
                      f'Loss: {loss.item():.4f}')

        # Calculate epoch metrics
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total

        # Update plotter with epoch metrics
        if plotter:
            plotter.update(epoch_loss, epoch_acc)

        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # Save training log
        log_path = config.manager.get_log_path("training")
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n")

    # Generate and save plots at the end of training
    if plotter:
        print("\nGenerating training plots...")
        plotter.plot_loss(config.manager.get_plot_path("training_loss"))
        plotter.plot_accuracy(config.manager.get_plot_path("training_accuracy"))
        print(f"✅ Plots saved to: {config.manager.plots_dir}")