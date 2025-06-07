import torch.nn as nn

class SimpleCNN_BN(nn.Module):
    """
    Simple CNN with Batch Normalization
    """
    def __init__(self, num_classes=8, input_channels=4):
        super(SimpleCNN_BN, self).__init__()

        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Calculate flattened size after convolutions
        # 128x128 → 64x64 → 32x32 → 16x16 → 8x8
        self.flattened_size = 256 * 8 * 8  # 16,384 (much smaller!)

        # Classifier layers with BatchNorm
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Feature extraction (same aggressive pooling)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 128→64
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 64→32
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 32→16
        x = self.pool(self.relu(self.bn4(self.conv4(x))))  # 16→8

        # Flatten for classifier
        x = x.view(x.size(0), -1)

        # Classification with BatchNorm
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x