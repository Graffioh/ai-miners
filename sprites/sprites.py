import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

# sprite obj
class Sprite():
    def __init__(self, char, img, act, dir, frame):
        self.character = char # str
        self.image = img # np.array
        self.action = act # str
        self.direction = dir # int
        self.frame = frame # int

# prep dataset
class SpriteDataset(Dataset):
    def __init__(self, dataset_dir, sprite_width=128, sprite_height=128, transform=None):
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
        else:
            print("❌ No sprites found!")

    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    train_dataset_path = "./dataset/train"
    print_dataset(train_dataset_path)

    test_dataset_path = "./dataset/test"
    print_dataset(test_dataset_path)

if __name__ == "__main__":
    main()
