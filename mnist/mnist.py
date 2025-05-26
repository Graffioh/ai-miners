#MNIST

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. DATA PREPARATION
# Transform to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
])

# Download and load the training and test datasets
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders (batch the data)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# 2. VISUALIZE SOME DATA
def show_sample_images():
    # Get a batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Create a grid of images
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(8):
        row, col = i // 4, i % 4
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'Label: {labels[i]}')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

# 3. DEFINE THE NEURAL NETWORK
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Input: 28x28 = 784 pixels
        self.fc1 = nn.Linear(28 * 28, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)       # Second hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes: 0-9)
        self.dropout = nn.Dropout(0.2)      # Dropout for regularization

    def forward(self, x):
        # Flatten the image (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)

        # Pass through layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation here (will be handled by loss function)

        return x

# 4. INITIALIZE MODEL, LOSS, AND OPTIMIZER
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()  # Good for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

print("Model architecture:")
print(model)

# 5. TRAINING FUNCTION
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()  # Set model to training mode
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device (GPU if available)
            data, target = data.to(device), target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
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
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses

# 6. TESTING FUNCTION
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Don't compute gradients for testing
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 7. FUNCTION TO MAKE PREDICTIONS ON INDIVIDUAL IMAGES
def predict_single_image(model, image, actual_label):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
        confidence = F.softmax(output, dim=1)
        max_confidence = torch.max(confidence).item()

        print(f"Predicted: {predicted.item()}, Actual: {actual_label}, "
              f"Confidence: {max_confidence:.2%}")
        return predicted.item()

# 8. RUN THE TRAINING
if __name__ == "__main__":
    print("Starting training...")

    # Show some sample images (optional)
    print("Sample images from dataset:")
    show_sample_images()

    # Train the model
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs=5)

    # Test the model
    print("\nTesting the model...")
    test_accuracy = test_model(model, test_loader)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Test on a few individual images
    print("\nTesting on individual images:")
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)

    for i in range(5):
        predict_single_image(model, test_images[i], test_labels[i].item())

    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("\nModel saved as 'mnist_model.pth'")
