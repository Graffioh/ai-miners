import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_training_loss(epochs, epoch_losses):
    """Plot the training loss vs epochs"""
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, 'b-', linewidth=2)
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.savefig(f"./plots/training_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

def create_direction_confusion_matrix(all_targets, all_predictions):
    """
    Create confusion matrix for direction classification
    """
    # Direction mapping
    direction_map = {0: 'E', 1: 'SE', 2: 'S', 3: 'SW', 4: 'W', 5: 'NW', 6: 'N', 7: 'NE'}
    
    # Create the main confusion matrix (8 directions)
    cm = confusion_matrix(all_targets, all_predictions, labels=range(8))
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Direction Classification Confusion Matrix\n(All Characters Combined)', fontsize=14, pad=20)
    plt.colorbar(im, label='Count')
    
    # Add text annotations with percentages
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Show count and percentage
            percentage = cm[i, j] / np.sum(cm[i, :]) * 100 if np.sum(cm[i, :]) > 0 else 0
            text = f"{cm[i, j]}\n({percentage:.1f}%)"
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, text, ha='center', va='center', 
                    color=color, fontsize=10, fontweight='bold')
    
    # Set compass direction labels
    direction_labels = [direction_map[i] for i in range(8)]
    plt.xticks(range(8), direction_labels)
    plt.yticks(range(8), direction_labels)
    plt.xlabel('Predicted Direction', fontsize=12)
    plt.ylabel('Actual Direction', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"./plots/direction_confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                dpi=150, bbox_inches='tight')
    plt.show()

def create_character_confusion_matrix(all_characters, all_targets, all_predictions):
    """
    Create confusion matrix for character classification
    """
    # Get unique characters and sort them
    unique_characters = sorted(set(all_characters))
    
    # Create character to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(unique_characters)}
    
    # Convert character labels to indices
    char_targets = [char_to_idx[char] for char in all_characters]
    char_predictions = [char_to_idx[char] for char in all_characters]  # This maps actual chars to predicted chars
    
    # For character confusion matrix, we need to group by actual character and see which directions were predicted
    # This is a bit different - we want to see character-wise accuracy, not character vs character prediction
    
    # Alternative approach: Create a matrix showing for each character, how well each direction was predicted
    num_chars = len(unique_characters)
    num_directions = 8
    direction_map = {0: 'E', 1: 'SE', 2: 'S', 3: 'SW', 4: 'W', 5: 'NW', 6: 'N', 7: 'NE'}
    
    # Initialize confusion matrix: rows = characters, columns = directions
    cm_char = np.zeros((num_chars, num_directions))
    
    # Fill the confusion matrix
    for i, char in enumerate(all_characters):
        char_idx = char_to_idx[char]
        target_direction = all_targets[i]
        predicted_direction = all_predictions[i]
        
        # We'll show the accuracy per character per direction
        cm_char[char_idx, target_direction] += 1 if target_direction == predicted_direction else 0
    
    # Convert to percentage for each character
    char_totals = np.zeros(num_chars)
    for i, char in enumerate(all_characters):
        char_idx = char_to_idx[char]
        char_totals[char_idx] += 1
    
    # Create accuracy matrix (percentage of correct predictions per character per direction)
    cm_char_pct = np.zeros((num_chars, num_directions))
    direction_counts = np.zeros((num_chars, num_directions))
    
    # Count total samples per character per direction
    for i, char in enumerate(all_characters):
        char_idx = char_to_idx[char]
        target_direction = all_targets[i]
        direction_counts[char_idx, target_direction] += 1
    
    # Calculate accuracy percentages
    for char_idx in range(num_chars):
        for dir_idx in range(num_directions):
            if direction_counts[char_idx, dir_idx] > 0:
                cm_char_pct[char_idx, dir_idx] = (cm_char[char_idx, dir_idx] / direction_counts[char_idx, dir_idx]) * 100
    
    # Create the plot
    plt.figure(figsize=(12, max(6, num_chars * 0.8)))
    im = plt.imshow(cm_char_pct, interpolation='nearest', cmap='Greens', vmin=0, vmax=100)
    plt.title('Character-wise Direction Classification Accuracy (%)\n(Per Character Per Direction)', fontsize=14, pad=20)
    plt.colorbar(im, label='Accuracy (%)')
    
    # Add text annotations
    for i in range(num_chars):
        for j in range(num_directions):
            if direction_counts[i, j] > 0:
                text = f"{cm_char_pct[i, j]:.1f}%\n({int(cm_char[i, j])}/{int(direction_counts[i, j])})"
                color = "white" if cm_char_pct[i, j] > 50 else "black"
                plt.text(j, i, text, ha='center', va='center', 
                        color=color, fontsize=9, fontweight='bold')
            else:
                plt.text(j, i, "N/A", ha='center', va='center', 
                        color="gray", fontsize=9)
    
    # Set labels
    direction_labels = [direction_map[i] for i in range(8)]
    plt.xticks(range(8), direction_labels)
    plt.yticks(range(num_chars), unique_characters)
    plt.xlabel('Direction', fontsize=12)
    plt.ylabel('Character', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"./plots/character_confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                dpi=150, bbox_inches='tight')
    plt.show()