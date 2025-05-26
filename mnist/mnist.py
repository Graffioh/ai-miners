# MNIST Handwritten Digit Recognition using PyTorch
# This script trains a neural network to classify handwritten digits (0-9)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ==============================================================================
# DEVICE SETUP
# ==============================================================================
# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
# Define the neural network architecture
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Hidden layer: 784 inputs (28x28 pixels) -> 128 neurons
        self.hidden_layer = nn.Linear(28 * 28, 128)
        # Dropout layer to prevent overfitting (randomly sets 20% of inputs to 0)
        self.dropout_layer = nn.Dropout(0.2)
        # Output layer: 128 neurons -> 10 classes (digits 0-9)
        self.output_layer = nn.Linear(128, 10)
    
    def forward(self, input_images):
        # Flatten 28x28 images into 784-dimensional vectors
        flattened_images = input_images.view(-1, 28 * 28)
        # Apply hidden layer + ReLU activation (removes negative values)
        hidden_output = F.relu(self.hidden_layer(flattened_images))
        # Apply dropout (only during training)
        dropout_output = self.dropout_layer(hidden_output)
        # Apply output layer
        raw_predictions = self.output_layer(dropout_output)
        # Apply log softmax for probability distribution over 10 classes
        return F.log_softmax(raw_predictions, dim=1)

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================
# Define transformations to apply to the images
image_transforms = transforms.Compose([
    # Convert PIL images to PyTorch tensors (0-1 range)
    transforms.ToTensor(),
    # Normalize with MNIST dataset mean and standard deviation
    # This helps with training stability
    transforms.Normalize((0.1307,), (0.3081,))
])

# ==============================================================================
# DATA LOADING
# ==============================================================================
# Download and load MNIST training dataset (60,000 images)
training_dataset = datasets.MNIST('data', train=True, download=True, transform=image_transforms)
# Download and load MNIST test dataset (10,000 images)
testing_dataset = datasets.MNIST('data', train=False, transform=image_transforms)

# Create data loaders for batching and shuffling
# Batch size of 64 means we process 64 images at once
training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
testing_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False)

# ==============================================================================
# MODEL, LOSS, AND OPTIMIZER SETUP
# ==============================================================================
# Initialize the model and move it to the selected device (CPU/GPU)
neural_network = MNISTNet().to(device)
# Negative Log Likelihood Loss - good for classification with log_softmax
loss_function = nn.NLLLoss()
# Adam optimizer - adaptive learning rate, works well for most problems
model_optimizer = optim.Adam(neural_network.parameters(), lr=0.001)

# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================
def train_model(model, data_loader, optimizer, loss_fn, current_epoch):
    """Train the model for one epoch"""
    model.train()  # Set model to training mode (enables dropout)
    
    for batch_index, (image_batch, label_batch) in enumerate(data_loader):
        # Move data to device (CPU/GPU)
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        
        # Reset gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        model_predictions = model(image_batch)
        
        # Compute loss between predictions and true labels
        current_loss = loss_fn(model_predictions, label_batch)
        
        # Backward pass: compute gradients
        current_loss.backward()
        
        # Update model parameters using gradients
        optimizer.step()
        
        # Print progress every 100 batches
        if batch_index % 100 == 0:
            print(f'Epoch {current_epoch}: [{batch_index * len(image_batch)}/{len(data_loader.dataset)} '
                  f'({100. * batch_index / len(data_loader):.0f}%)]\tLoss: {current_loss.item():.6f}')

# ==============================================================================
# TESTING FUNCTION
# ==============================================================================
def evaluate_model(model, data_loader, loss_fn):
    """Evaluate the model on test data"""
    model.eval()  # Set model to evaluation mode (disables dropout)
    total_test_loss = 0
    correct_predictions = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for image_batch, label_batch in data_loader:
            # Move data to device
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            
            # Get model predictions
            model_output = model(image_batch)
            
            # Accumulate test loss
            total_test_loss += loss_fn(model_output, label_batch).item()
            
            # Get predicted class (highest probability)
            predicted_labels = model_output.argmax(dim=1, keepdim=True)
            
            # Count correct predictions
            correct_predictions += predicted_labels.eq(label_batch.view_as(predicted_labels)).sum().item()
    
    # Calculate average loss and accuracy
    average_loss = total_test_loss / len(data_loader.dataset)
    accuracy_percentage = 100. * correct_predictions / len(data_loader.dataset)
    print(f'\nTest set: Average loss: {average_loss:.4f}, '
          f'Accuracy: {correct_predictions}/{len(data_loader.dataset)} ({accuracy_percentage:.2f}%)\n')
    return accuracy_percentage

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
print("Training the model...")
# Train for 5 epochs (full passes through the training data)
for epoch_number in range(1, 6):
    # Train for one epoch
    train_model(neural_network, training_loader, model_optimizer, loss_function, epoch_number)
    # Evaluate on test set after each epoch
    evaluate_model(neural_network, testing_loader, loss_function)

# ==============================================================================
# VISUALIZATION OF RESULTS
# ==============================================================================
# Get a batch of test data for visualization
neural_network.eval()
example_iterator = enumerate(testing_loader)
batch_index, (example_images, true_labels) = next(example_iterator)

# Make predictions on the examples
with torch.no_grad():
    model_output = neural_network(example_images.to(device))
    predicted_digits = model_output.argmax(dim=1, keepdim=True)

# Plot the first 6 images with predictions and true labels
plt.figure(figsize=(12, 4))
for image_index in range(6):
    plt.subplot(2, 3, image_index + 1)
    # Display the image in grayscale
    plt.imshow(example_images[image_index][0], cmap='gray')
    # Show prediction vs actual label
    plt.title(f'Predicted: {predicted_digits[image_index].item()}, Actual: {true_labels[image_index].item()}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# ==============================================================================
# MODEL SAVING
# ==============================================================================
# Save the trained model parameters to disk
torch.save(neural_network.state_dict(), 'mnist_trained_model.pth')
print("Model saved as 'mnist_trained_model.pth'")

