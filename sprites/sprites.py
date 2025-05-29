import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# sprite obj
class Sprite():
    def __init__(self, char, img, act, dir, frame):
        self.character = char # str
        self.image = img # np.array or tensor
        self.action = act # str
        self.direction = dir # int
        self.frame = frame # int

# prep dataset
class SpriteDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, sprite_width=128, sprite_height=128):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.sprite_width = sprite_width
        self.sprite_height = sprite_height
        self.sprites = self.split_sprite_sheet(dataset_dir) # sprite objects as features
        self.flattened_data = self._flatten_sprites()
        self.directions = [0,1,2,3,4,5,6,7] # 8 directions as label

    def __len__(self):
        return len(self.flattened_data)

    # called by DataLoader to get the items from the dataset by index
    def __getitem__(self, idx):
        sprite_obj = self.flattened_data[idx]
        sprite_img = sprite_obj.image

        if self.transform:
            sprite_img = self.transform(sprite_img)
        elif isinstance(sprite_img, np.ndarray):
            sprite_img = torch.from_numpy(sprite_img).float()

        return sprite_obj.character, sprite_img, sprite_obj.direction, sprite_obj.action

    def split_sprite_sheet(self, dataset_dir):
        """Split the sprite sheet into individual sprite objects and return them in a list"""
        sprites = []

        # Iterate through character folders
        for char_folder in os.listdir(dataset_dir):
            char_path = os.path.join(dataset_dir, char_folder)
            if not os.path.isdir(char_path):
                continue

            char_sprites = []

            # Process each sprite sheet for this character
            for action_file in os.listdir(char_path):
                if not action_file.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                action_name = os.path.splitext(action_file)[0]
                sprite_sheet_path = os.path.join(char_path, action_file)

                # Load and split the sprite sheet
                sprite_sheet = Image.open(sprite_sheet_path)
                sheet_width, sheet_height = sprite_sheet.size

                # Extract individual sprites
                rows = 8
                cols = 15
                for row in range(rows):
                    for col in range(cols):
                        left = col * self.sprite_width
                        top = row * self.sprite_height
                        right = left + self.sprite_width
                        bottom = top + self.sprite_height

                        sprite_img = sprite_sheet.crop((left, top, right, bottom))
                        sprite_array = np.array(sprite_img)

                        # Create sprite object
                        sprite = Sprite(
                            char=char_folder,
                            img=sprite_array,
                            act=action_name,
                            dir=row,
                            frame=col
                        )
                        char_sprites.append(sprite)

            sprites.append(char_sprites)
        return sprites

    def _flatten_sprites(self):
        """Flatten 2D sprite array into 1D array for easy indexing"""
        flattened = []
        for char_sprites in self.sprites:
            flattened.extend(char_sprites)
        return flattened


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def print_dataset(dataset_path):
    """Print the dataset to verify that loading works correctly"""

    print("+---------------------------------------+")
    print(f"Testing SpriteDataset {dataset_path}")

    try:
        # Create dataset
        dataset = SpriteDataset(dataset_path)

        # Basic info
        print(f"Dataset loaded: {len(dataset)} sprites")
        print(f"Directions: {dataset.directions}")

        # Test first item
        if len(dataset) > 0:
            character, image, direction, action = dataset[0]
            print(f"First sprite - Character name: {character}, Action name: {action}, Image shape: {image.shape}, Direction: {direction}")

            test_indices = [0, 15, 30, 60, 105, 121, 250, 400, 666]
            print("\n--- Testing different directions ---")
            for i in test_indices:
                if i < len(dataset):
                    character, image, direction, action = dataset[i]
                    sprite_obj = dataset.flattened_data[i]
                    print(f"Index {i}: Char={character}, Action={action}, Dir={direction}, Frame={sprite_obj.frame}")

            print("✅ Dataset working correctly!")
            print("+---------------------------------------+")
        else:
            print("❌ No sprites found!")

    except Exception as e:
        print(f"❌ Error: {e}")

class SimpleNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleNN, self).__init__()
        input_size = 128 * 128 * 4  # 65,536 for RGBA

        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)  # Output: 8 directions

        self.dropout = nn.Dropout(0.3)
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

class SpriteDirectionCNN(nn.Module):
    def __init__(self, num_classes=8, input_channels=4):
        super(SpriteDirectionCNN, self).__init__()

        # Convolutional layers - detect features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Calculate flattened size after convolutions
        # 128x128 → 64x64 → 32x32 → 16x16 → 8x8
        self.flattened_size = 256 * 8 * 8  # 16,384 (much smaller!)

        # Classifier layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.pool(self.relu(self.conv1(x)))  # 128→64
        x = self.pool(self.relu(self.conv2(x)))  # 64→32
        x = self.pool(self.relu(self.conv3(x)))  # 32→16
        x = self.pool(self.relu(self.conv4(x)))  # 16→8

        # Flatten for classifier
        x = x.view(x.size(0), -1)

        # Classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

def train_sprites(model, data_loader, criterion, optimizer, epochs=5):
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

def main():
    train_dataset_path = "./dataset/train"
    #print_dataset(train_dataset_path)

    test_dataset_path = "./dataset/test"
    #print_dataset(test_dataset_path)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # Converts to [0,1] range
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))  # Normalize to [-1,1]
    ])

    train_dataset = SpriteDataset(train_dataset_path, transform)
    test_dataset = SpriteDataset(test_dataset_path, transform)

    train_data = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SpriteDirectionCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # Good for classification
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

    train_losses = train_sprites(model, train_data, criterion, optimizer)
    print("YOOOOOOOOOOOOOOOOOOOOOOO")
    print(train_losses)



if __name__ == "__main__":
    main()
