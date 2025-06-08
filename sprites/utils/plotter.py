import matplotlib.pyplot as plt
import torch
import os
from collections import defaultdict

class TrainingPlotter:
    def __init__(self, num_classes=8):
        self.train_losses = []
        self.train_accuracies = []
        self.num_classes = num_classes
        self.class_losses = defaultdict(list)  # Store losses per class

    def update(self, train_loss, train_acc):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

    def update_class_losses(self, model_output, targets, criterion):
        """Calculate and store loss per class"""
        with torch.no_grad():
            # Calculate loss for each class
            for class_idx in range(self.num_classes):
                # Get samples for this class
                class_mask = (targets == class_idx)
                if class_mask.sum() > 0:  # If we have samples for this class
                    class_output = model_output[class_mask]
                    class_targets = targets[class_mask]
                    class_loss = criterion(class_output, class_targets).item()
                    self.class_losses[class_idx].append(class_loss)
                else:
                    # No samples for this class in this batch
                    # Use the last known loss or 0
                    last_loss = self.class_losses[class_idx][-1] if self.class_losses[class_idx] else 0
                    self.class_losses[class_idx].append(last_loss)

    def plot_loss(self, save_path=None):
        """Plot training loss over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")

        # plt.show()

    def plot_accuracy(self, save_path=None):
        """Plot training accuracy over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracies, label='Training Accuracy', color='green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)  # Accuracy is 0-100%

        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Accuracy plot saved to {save_path}")

        # plt.show()

    def plot_class_losses(self, save_path=None):
        """Plot loss per class (sprite direction)"""
        plt.figure(figsize=(12, 8))

        # Define colors for each direction
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        direction_names = ['West', "SW", "South", "SE", "East", "NE", "North", "NW"]

        for class_idx in range(self.num_classes):
            if class_idx in self.class_losses and self.class_losses[class_idx]:
                plt.plot(self.class_losses[class_idx],
                        label=f'Dir {class_idx} ({direction_names[class_idx]})',
                        color=colors[class_idx],
                        linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Sprite Direction')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Class losses plot saved to {save_path}")

        # plt.show()

    def plot_all(self, save_dir=None):
        """Plot all three charts"""
        # Create individual plots
        if save_dir:
            self._ensure_dir(save_dir)
            self.plot_loss(os.path.join(save_dir, "training_loss.png"))
            self.plot_accuracy(os.path.join(save_dir, "training_accuracy.png"))
            self.plot_class_losses(os.path.join(save_dir, "class_losses.png"))
        else:
            self.plot_loss()
            self.plot_accuracy()
            self.plot_class_losses()

    def _ensure_dir(self, path):
        """Ensure directory exists for saving plots"""
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
