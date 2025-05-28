import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

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
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        # Features
        self.sprites = self.split_sprite_sheet(self.dataset_dir) # 2D arr of Sprite
        # Inputs
        self.directions = self.set_directions(self.sprites) # arr of int
        self.dataset_rows = len(self.sprites)
        self.dataset_cols = len(self.sprites[0]) if self.sprites else 0

    def __len__(self):
            return self.dataset_rows * self.dataset_cols

    def __getitem__(self, idx):
        # convert 1D index to 2D coordinates
        row = idx // self.dataset_cols
        col = idx % self.dataset_cols

        sprite_obj = self.sprites[row][col]
        return sprite_obj.image, sprite_obj.direction

    def split_sprite_sheet(self, dataset_dir):
        # TODO: implement this
        return []

    def set_directions(self, sprites):
        # TODO: implement this
        return []








# load datasets

# data loading



# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
