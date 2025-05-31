# Notes

## Device

google colab T4 runtime (w mounted folder from google drive)

## Dataset structure

### shadowless, 40 characters

80% train (32 characters) / 20% test (8 characters)

train and test have different character structure/template

train shuffled

(soon™) mini batch stratified sampling

## Experiments

*thanks claudio for the code*

- Criterion (loss): CrossEntropyLoss
- Optimizer: Adam

### Fully Connected NN

12% test accuracy

#### Code

```python
class SimpleNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleNN, self).__init__()
        input_size = 128 * 128 * 4

        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x
```

### CNN

81~% test accuracy

#### Code

```python
class SpriteDirectionCNN(nn.Module):
    def __init__(self, num_classes=8, input_channels=4):
        super(SpriteDirectionCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.flattened_size = 256 * 8 * 8

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
```
