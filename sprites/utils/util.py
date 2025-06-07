from torch.utils.data import Subset 

def pick_model_architecture_menu(config):
    print("+---------------------------------------+")
    print("\nModel architectures available:")
    print(f"1. {config.MODEL_ARCHITECTURE_FCN} (SimpleNN)")
    print(f"2. {config.MODEL_ARCHITECTURE_FCN_BN} (SimpleNN_BN)") 
    print(f"3. {config.MODEL_ARCHITECTURE_CNN} (SimpleCNN)")
    print(f"4. {config.MODEL_ARCHITECTURE_CNN_BN} (SimpleCNN_BN)") 
    choice = input("Select: ").strip()
    if choice == "1":
        return config.MODEL_ARCHITECTURE_FCN
    elif choice == "2":
        return config.MODEL_ARCHITECTURE_FCN_BN 
    elif choice == "3":
        return config.MODEL_ARCHITECTURE_CNN
    elif choice == "4":
        return config.MODEL_ARCHITECTURE_CNN_BN 
    else:
        print(f"Invalid choice, defaulting to {config.MODEL_ARCHITECTURE_CNN}")
        return config.MODEL_ARCHITECTURE_CNN

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
    
def log_config(config, config_log_path, device):
    with open(config_log_path, "w") as f:
        f.write("Training Configuration\n====================\n")
        f.write(f"Run ID: {config.manager.run_id}\n")
        f.write(f"Device: {device}\n\n")
        
        # Paths
        f.write("PATHS:\n")
        f.write(f"TRAIN_PATH: {config.TRAIN_PATH}\n")
        f.write(f"TEST_PATH: {config.TEST_PATH}\n")
        f.write(f"PLOT_DIR: {config.PLOT_DIR}\n\n")
        
        # Training parameters
        f.write("TRAINING:\n")
        f.write(f"MODEL_ARCHITECTURE_FCN: {config.MODEL_ARCHITECTURE_FCN}\n")
        f.write(f"MODEL_ARCHITECTURE_FCN_BN: {config.MODEL_ARCHITECTURE_FCN_BN}\n")
        f.write(f"MODEL_ARCHITECTURE_CNN: {config.MODEL_ARCHITECTURE_CNN}\n")
        f.write(f"LEARNING_RATE: {config.LEARNING_RATE}\n")
        f.write(f"BATCH_SIZE: {config.BATCH_SIZE}\n")
        f.write(f"EPOCHS: {config.EPOCHS}\n")
        f.write(f"VALIDATION_SPLIT_RATIO: {config.VALIDATION_SPLIT_RATIO}\n\n")
        
        # Data parameters
        f.write("DATA:\n")
        f.write(f"SHUFFLE_TRAIN: {config.SHUFFLE_TRAIN}\n")
        f.write(f"SHUFFLE_TEST: {config.SHUFFLE_TEST}\n")
        f.write(f"NORMALIZE_MEAN: {config.NORMALIZE_MEAN}\n")
        f.write(f"NORMALIZE_STD: {config.NORMALIZE_STD}\n\n")
        
        # Extra parameters
        f.write("EXTRA:\n")
        f.write(f"SAVE_MODEL: {config.SAVE_MODEL}\n")
        f.write(f"ENABLE_PLOTTING: {config.ENABLE_PLOTTING}\n")