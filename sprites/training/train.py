import torch
from utils.plotter import TrainingPlotter
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from models.simpleCNN import SimpleCNN
from models.simpleCNN_BN import SimpleCNN_BN
from models.simpleNN import SimpleNN
from models.simpleNN_BN import SimpleNN_BN

from evaluation.evaluation_orchestrator import evaluate_model

def create_model(model_architecture_choice, hyperparameters_config, device):
    """Factory function to create a new model instance"""
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
        raise ValueError(f"Model architecture '{model_architecture_choice}' not recognized")
    
    return model

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
    
    # Train final model
    print("Training final model on full training data...")
    train_sprites(final_model, full_train_loader, criterion, optimizer, device, config, hyperparameters_config.EPOCHS)
    
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

def train_sprites(model, data_loader, criterion, optimizer, device, config, epochs=5):
    """
    Train the model with integrated plotting and saving via RunManager
    """
    model.train()

    # Initialize plotter if plotting is enabled
    plotter = TrainingPlotter() if config.ENABLE_PLOTTING else None

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # For tracking class losses during the epoch
        epoch_class_outputs = []
        epoch_class_targets = []

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

            # Collect data for class loss calculation
            if plotter:
                epoch_class_outputs.append(output.detach())
                epoch_class_targets.append(target.detach())

            # Print progress
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(data_loader)}, '
                      f'Loss: {loss.item():.4f}')

        # Calculate epoch metrics
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total

    # == Plot & Log ==
        if plotter:
            # Update plot for each epoch
            plotter.update(epoch_loss, epoch_acc)

            if epoch_class_outputs:
                all_outputs = torch.cat(epoch_class_outputs, dim=0)
                all_targets = torch.cat(epoch_class_targets, dim=0)
                plotter.update_class_losses(all_outputs, all_targets, criterion)

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
        plotter.plot_class_losses(config.manager.get_plot_path("class_losses"))

        print(f"✅ Plots saved to: {config.manager.plots_dir}")
    # ==