import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
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
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None, title="Confusion Matrix"):
        """
        Plot confusion matrix for test predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            class_names: List of class names for labels
            save_path: Path to save the plot (optional)
            title: Title for the plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate accuracy
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap using matplotlib's imshow
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20)
        
        # Set ticks and labels
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        ax.set_title(f'{title}\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        # Print detailed metrics
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Print confusion matrix interpretation
        print("\nConfusion Matrix Interpretation:")
        print("- Diagonal elements: Correct predictions")
        print("- Off-diagonal elements: Misclassifications")
        print("- Each row represents the true class")
        print("- Each column represents the predicted class")
        
        return cm

    def _ensure_dir(self, path):
        """Ensure directory exists for saving plots"""
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)