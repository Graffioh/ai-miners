import torch
from training.plotter import TrainingPlotter

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

        # Update plotter with epoch metrics
        if plotter:
            plotter.update(epoch_loss, epoch_acc)

            # Calculate class losses for this epoch
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

        # Save individual plots using RunManager paths
        plotter.plot_loss(config.manager.get_plot_path("training_loss"))
        plotter.plot_accuracy(config.manager.get_plot_path("training_accuracy"))
        plotter.plot_class_losses(config.manager.get_plot_path("class_losses"))

        print(f"✅ Plots saved to: {config.manager.plots_dir}")