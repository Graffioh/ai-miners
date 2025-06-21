import torch
from utils.plotter import plot_training_loss

def train_model(model, train_loader, criterion, optimizer, device, epochs):
    """
    Train the model 
    """
    train_loader_len = len(train_loader)
    model.train()

    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (_, img_data, target, _) in enumerate(train_loader):
            img_data, target = img_data.to(device), target.to(device)

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
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{train_loader_len}, '
                      f'Loss: {loss.item():.4f}')

        # Calculate epoch metrics
        epoch_loss = running_loss / train_loader_len
        epoch_acc = 100 * correct / total

        # Store loss for plotting
        epoch_losses.append(epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
        plot_training_loss(epochs, epoch_losses)