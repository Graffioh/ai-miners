import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

from dataset.dataset import SpriteDataset
from models.old.simpleCNN import SimpleCNN
from training.train import train_model
from evaluation.test import test_model

from configurations.config import NeurodragonConfig
from utils.util import print_separator, debug_show_sprites, add_white_background_and_handle_channels

torch.manual_seed(42)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

def main():
    # ++ REFACTOR ++
    # [] better plotting and dir organization
    # [] dynamic background (check color not present in the palette)

    config = NeurodragonConfig()

    print_separator()

    is_alpha_cannel_enabled = config.misc.is_alpha_enabled
    transform = transforms.Compose([
        transforms.Lambda(lambda img: add_white_background_and_handle_channels(img, is_alpha_cannel_enabled)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.main.transform_normalization_mean_3, std=config.main.transform_normalization_std_3)
    ])

    # Load datasets
    print("Loading datasets...")
    train_dataset = SpriteDataset(config.misc.dataset_train_path, transform)
    debug_show_sprites(train_dataset, num_samples=10)

    test_dataset = SpriteDataset(config.misc.dataset_test_path, transform)

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, 
                                 batch_size=config.main.batch_size, 
                                 shuffle=False)
    test_loader = DataLoader(test_dataset, 
                           batch_size=config.main.batch_size, 
                           shuffle=True)

    print("Data loaded ✔")

    print_separator()

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.main.learning_rate, weight_decay=config.main.weight_decay)

    train_model(model, train_loader, criterion, optimizer, device, config.main.epochs)

    print_separator()

    test_model(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()