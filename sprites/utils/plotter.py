import matplotlib.pyplot as plt
import os

class TrainingPlotter:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []

    def update(self, train_loss, train_acc):
        """Update training metrics for the current epoch"""
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

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

    def plot_all(self, save_dir=None):
        """Plot both loss and accuracy charts"""
        if save_dir:
            self._ensure_dir(save_dir)
            self.plot_loss(os.path.join(save_dir, "training_loss.png"))
            self.plot_accuracy(os.path.join(save_dir, "training_accuracy.png"))
        else:
            self.plot_loss()
            self.plot_accuracy()

    def _ensure_dir(self, path):
        """Ensure directory exists for saving plots"""
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)