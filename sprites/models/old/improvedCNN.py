import torch.nn as nn

class ImprovedCNN(nn.Module):
    """
    CNN optimized for 8-directional sprite classification
    """
    def __init__(self, num_classes=8, input_channels=4):
        super(ImprovedCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)    
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)    
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)   
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)   

        # Pooling and activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Classifier
        self.flattened_size = 256 * 2 * 2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Follow ResNet's Conv-BN-ReLU pattern
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.bn3(self.conv3(x))))
        x = self.maxpool(self.relu(self.bn4(self.conv4(x))))

        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x