import torch.nn as nn

class SpritesCNN(nn.Module):
    """
    CNN (ResNet inspired) 
    https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
    """
    def __init__(self, num_classes=8, input_channels=3):
        super(SpritesCNN, self).__init__()

        # Convolutional layers with BatchNorm
        #self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm2d(16)

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        #self.conv1 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm2d(48)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #self.conv2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        #self.bn2 = nn.BatchNorm2d(96)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        #self.conv3 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        #self.bn3 = nn.BatchNorm2d(192)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten(start_dim=1)
        
        # Final linear layer: 256 features -> num_classes
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Feature extraction with batch normalization
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  
        x = self.relu(self.bn2(self.conv2(x)))  
        x = self.relu(self.bn3(self.conv3(x)))  
        x = self.relu(self.bn4(self.conv4(x)))  

        # Apply adaptive average pooling to get 1x1 spatial dimensions
        x = self.avg_pool(x)  # Shape: (batch_size, 256, 1, 1)
        
        # Flatten to (batch_size, 256)
        x = self.flatten(x)  
        
        # Final classification layer
        x = self.fc(x)

        return x