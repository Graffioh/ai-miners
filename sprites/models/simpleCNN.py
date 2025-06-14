import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple CNN 
    """
    def __init__(self, num_classes=8, input_channels=4):
        super(SimpleCNN, self).__init__()

        # Convolutional layers - detect features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Pooling and activation - ResNet style
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.flattened_size = 256 * 2 * 2  
        self.flatten = nn.Flatten()

        # Classifier layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.maxpool(self.relu(self.conv1(x)))  # 128→64
        x = self.maxpool(self.relu(self.conv2(x)))  # 64→32
        x = self.maxpool(self.relu(self.conv3(x)))  # 32→16
        x = self.maxpool(self.relu(self.conv4(x)))  # 16→8

        # Adaptive pooling to 2×2 (preserves some spatial info)
        x = self.adaptive_pool(x)  

        # Flatten for classifier
        x = self.flatten(x)  

        # Classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x