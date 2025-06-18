import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

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
        
        # Apply transforms or convert to tensor
        if self.transform:
            sprite_img = self.transform(sprite_img)
        else:
            # Convert to tensor if no transforms
            sprite_img = transforms.ToTensor()(sprite_img)
        
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
                        if sprite_img.mode == 'RGBA':
                            sprite_img = sprite_img.convert('RGB')

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
