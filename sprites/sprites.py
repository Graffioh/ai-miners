import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset.dataset import SpriteDataset
from models.simpleCNN import SimpleCNN
from training.train import train_sprites
from evaluation.test import test_model
from utils.util import print_dataset
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    print("Loading dataset...")
    train_dataset = SpriteDataset(Config.get_train_path(), Config.get_transform())
    test_dataset = SpriteDataset(Config.get_test_path(), Config.get_transform())

    print("++ Testing Train dataset ++")
    print_dataset(train_dataset)

    print("++ Testing Test dataset ++")
    print_dataset(test_dataset)

    train_data = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=Config.SHUFFLE_TRAIN)
    test_data = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=Config.SHUFFLE_TEST)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    train_losses = train_sprites(model, train_data, criterion, optimizer, device, Config.EPOCHS)
    print(f"Train losses: {train_losses}\n")

    print("+---------------------------------------+")
    print("\nTesting the model...")
    test_accuracy = test_model(model, test_data, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Save model if enabled
    if Config.SAVE_MODEL:
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f"Model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
