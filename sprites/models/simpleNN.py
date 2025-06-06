import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleNN, self).__init__()
        input_size = 128 * 128 * 4  # 65,536 for RGBA

        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)  

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten: (batch_size, C, H, W) -> (batch_size, input_size)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x) 
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  

        return x
