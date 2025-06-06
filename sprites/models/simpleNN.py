import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleNN, self).__init__()
        input_size = 128 * 128 * 4  # 65,536 for RGBA

        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)  # Output: 8 directions

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten: (batch_size, C, H, W) -> (batch_size, input_size)
        # need to flatten because the linear layer expect a tuple (batch_size, features)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x) # dropout used to prevent overfitting
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # No activation (handled by loss)

        return x
