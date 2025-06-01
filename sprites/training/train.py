import torch

def train_sprites(model, data_loader, criterion, optimizer, device, epochs=5):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (character, img_data, target, action) in enumerate(data_loader):
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

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses
