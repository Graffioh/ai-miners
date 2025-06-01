import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from dataset.dataset import SpriteDataset
from models.simpleCNN import SimpleCNN
from training.train import train_sprites
from evaluation.test import test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # google drive for colab
    train_dataset_path = "/content/drive/MyDrive/shadowless/train"
    test_dataset_path = "/content/drive/MyDrive/shadowless/test"

    # local
    #train_dataset_path = "./dataset/train"
    #test_dataset_path = "./dataset/test"

    #util.print_dataset(train_dataset_path)
    #util.print_dataset(test_dataset_path)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.013441166840493679, 0.010885078459978104, 0.010833792388439178, 0.04079267755150795], std=[0.07362207025289536, 0.06255189329385757, 0.0627019852399826, 0.18982279300689697])
    ])

    print("Loading dataset...")
    train_dataset = SpriteDataset(train_dataset_path, transform)
    test_dataset = SpriteDataset(test_dataset_path, transform)
    train_data = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = train_sprites(model, train_data, criterion, optimizer, device)
    print(f"Train losses: {train_losses}\n")

    print("+---------------------------------------+")
    print("\nTesting the model...")
    test_accuracy = test_model(model, test_data, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')


if __name__ == "__main__":
    main()
