import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def print_separator():
    print("==" * 30)

def add_white_background_and_handle_channels(img, is_alpha_enabled):
    """Convert to RGB add white background if alpha channel is not enabled"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    if is_alpha_enabled:
        # Keep transparent images as-is
        return img
    else:
        # Convert to RGB with white background
        if img.mode in ['RGBA', 'LA']:
            white_bg = Image.new('RGB', img.size, (255, 255, 255))
            white_bg.paste(img, mask=img.split()[-1])  
            return white_bg
        else:
            return img.convert('RGB')

def debug_show_sprites(dataset, num_samples=10):
    """Show random sprites in a grid for debugging"""
    # Get random samples
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    # Create grid
    cols = 5
    rows = (num_samples + cols - 1) // cols  # Ceiling division
    
    plt.figure(figsize=(15, 3 * rows))
    
    for i, idx in enumerate(random_indices):
        char, img_tensor, direction, action = dataset[idx]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0, 1)
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_np)
        plt.title(f'{char}\nDir: {direction}, Act: {action}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()