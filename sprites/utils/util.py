import torch 
from torch.utils.data import Subset 

def pick_model_architecture_menu(config):
    print("+---------------------------------------+")
    print("\nSelect model architecture:")
    print(f"1. {config.MODEL_ARCHITECTURE_FCN} (SimpleNN)")
    print(f"2. {config.MODEL_ARCHITECTURE_CNN} (SimpleCNN)")
    print(f"3. {config.MODEL_ARCHITECTURE_FCN_BN} (SimpleNN_BN)") # New option
    choice = input("Enter 1, 2, or 3: ").strip()
    if choice == "1":
        return config.MODEL_ARCHITECTURE_FCN
    elif choice == "2":
        return config.MODEL_ARCHITECTURE_CNN
    elif choice == "3":
        return config.MODEL_ARCHITECTURE_FCN_BN # Handle new choice
    else:
        print(f"Invalid choice, defaulting to {config.MODEL_ARCHITECTURE_FCN}")
        return config.MODEL_ARCHITECTURE_FCN # Default choice

    # This print was inside the function, usually it's outside or not present
    # print("+---------------------------------------+") 

def print_dataset(dataset):
    """Print the dataset to verify that loading works correctly"""

    print("+---------------------------------------+")
    # Identify if it's a Subset or a direct SpriteDataset
    if isinstance(dataset, Subset):
        dataset_name = "Subset of SpriteDataset"
        original_dataset = dataset.dataset 
    else:
        dataset_name = "SpriteDataset"
        original_dataset = dataset
    
    print(f"Testing {dataset_name}")

    try:
        # Basic info
        print(f"Dataset loaded: {len(dataset)} sprites")
        if hasattr(original_dataset, 'directions'): # Check if original_dataset has 'directions'
            print(f"Directions available in original dataset: {original_dataset.directions}")
        else:
            print("Original dataset does not have a 'directions' attribute.")


        # Test first item
        if len(dataset) > 0:
            # __getitem__ for Subset delegates to the original dataset with the mapped index
            character, image, direction, action = dataset[0] 
            print(f"First sprite - Character name: {character}, Action name: {action}, Image shape: {image.shape}, Direction: {direction}")

            # Test some more items using indices relative to the current dataset view (Subset or original)
            # Reduced number of test_indices for brevity in output, can be expanded
            test_indices = [0, 1, min(15, len(dataset)-1), min(30, len(dataset)-1), min(60, len(dataset)-1), min(105, len(dataset)-1)]
            # Filter out indices that are out of bounds for the current dataset
            test_indices = [i for i in test_indices if i < len(dataset)]
            
            print("\n--- Testing different items (indices relative to this dataset view) ---")
            for i in test_indices:
                # This check is redundant due to the list comprehension above but kept for safety
                if i < len(dataset): 
                    char, img, dir_val, act = dataset[i] # Fetches components

                    # To get the frame, we need the original Sprite object.
                    if isinstance(dataset, Subset):
                        original_idx = dataset.indices[i]
                        sprite_obj = original_dataset.flattened_data[original_idx]
                        print(f"Subset Index {i} (Original Index {original_idx}): Char={char}, Action={act}, Dir={dir_val}, Frame={sprite_obj.frame}")
                    else: # It's a SpriteDataset instance
                        sprite_obj = original_dataset.flattened_data[i]
                        print(f"Index {i}: Char={char}, Action={act}, Dir={dir_val}, Frame={sprite_obj.frame}")
            
            print("✅ Dataset working correctly!")
            print("+---------------------------------------+")
        else:
            print("❌ No sprites found in this dataset view!")

    except Exception as e:
        print(f"❌ Error during print_dataset: {e}")
        import traceback
        traceback.print_exc()


def save_training_results(
    config, 
    model, 
    model_architecture_choice,
    overall_validation_accuracy, 
    validation_dir_accuracies, 
    validation_char_accuracies,
    overall_test_accuracy, 
    test_dir_accuracies, 
    test_char_accuracies
):
    """
    Saves the model and a summary of training/evaluation results if config.SAVE_MODEL is True.
    """
    if not config.SAVE_MODEL:
        return

    # Save the model
    model_path = config.manager.get_model_path(f"final_model_{model_architecture_choice}")
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"❌ Error saving model to {model_path}: {e}")
        return 

    # Save model information and results log
    model_info_path = config.manager.get_log_path("model_info")
    try:
        with open(model_info_path, "w") as f:
            f.write("Model Information\n=================\n")
            f.write(f"Model Architecture: {model_architecture_choice}\n")
            
            f.write(f"\n--- Validation Set Results ---\n")
            f.write(f"Overall Validation Accuracy: {overall_validation_accuracy:.2f}%\n")
            if validation_dir_accuracies:
                f.write("Validation Accuracies per Direction:\n")
                for direction, acc in sorted(validation_dir_accuracies.items()): # Sort for consistent output
                    f.write(f"  - {direction}: {acc:.2f}%\n")
            if validation_char_accuracies:
                f.write("Validation Accuracies per Character (for direction prediction):\n")
                for char, acc in sorted(validation_char_accuracies.items()): # Sort for consistent output
                    f.write(f"  - {char}: {acc:.2f}%\n")
            
            f.write(f"\n--- Test Set Results ---\n")
            f.write(f"Overall Test Accuracy: {overall_test_accuracy:.2f}%\n")
            if test_dir_accuracies:
                f.write("Test Accuracies per Direction:\n")
                for direction, acc in sorted(test_dir_accuracies.items()): # Sort for consistent output
                    f.write(f"  - {direction}: {acc:.2f}%\n")
            if test_char_accuracies:
                f.write("Test Accuracies per Character (for direction prediction):\n")
                for char, acc in sorted(test_char_accuracies.items()): # Sort for consistent output
                    f.write(f"  - {char}: {acc:.2f}%\n")
            
            f.write(f"\nTraining Epochs: {config.EPOCHS}\n")
            if model:
                f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n")
                f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
            else:
                f.write("Model parameter information not available (model is None).\n")
        print(f"Model info and results saved to: {model_info_path}")
    except Exception as e:
        print(f"❌ Error saving model info to {model_info_path}: {e}")

