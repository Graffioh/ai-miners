def pick_model_architecture_menu(config):
    print("+---------------------------------------+")
    print("\nSelect model architecture:")
    print(f"1. {config.MODEL_ARCHITECTURE_FCN} (SimpleNN)")
    print(f"2. {config.MODEL_ARCHITECTURE_CNN} (SimpleCNN)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        return config.MODEL_ARCHITECTURE_FCN
    elif choice == "2":
        return config.MODEL_ARCHITECTURE_CNN
    else:
        print("Invalid choice, defaulting to FCN")
        return config.MODEL_ARCHITECTURE_FCN

    print("+---------------------------------------+")

def print_dataset(dataset):
    """Print the dataset to verify that loading works correctly"""

    print("+---------------------------------------+")
    print("Testing SpriteDataset")

    try:
        # Basic info
        print(f"Dataset loaded: {len(dataset)} sprites")
        print(f"Directions: {dataset.directions}")

        # Test first item
        if len(dataset) > 0:
            character, image, direction, action = dataset[0]
            print(f"First sprite - Character name: {character}, Action name: {action}, Image shape: {image.shape}, Direction: {direction}")

            test_indices = [0, 15, 30, 60, 105, 121, 250, 400, 666]
            print("\n--- Testing different directions ---")
            for i in test_indices:
                if i < len(dataset):
                    character, image, direction, action = dataset[i]
                    sprite_obj = dataset.flattened_data[i]
                    print(f"Index {i}: Char={character}, Action={action}, Dir={direction}, Frame={sprite_obj.frame}")

            print("✅ Dataset working correctly!")
            print("+---------------------------------------+")
        else:
            print("❌ No sprites found!")

    except Exception as e:
        print(f"❌ Error: {e}")


'''
def calculate_normalization_with_dataloader(dataset_path):
    """
    Calculate normalization using DataLoader - more memory efficient
    """
    print("Calculating normalization with DataLoader method...")

    # Create temporary dataset without normalization
    temp_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()  # Only convert to tensor, no normalization
    ])

    temp_dataset = SpriteDataset(dataset_path, temp_transform)
    temp_loader = DataLoader(temp_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize accumulators
    channels = 4  # RGBA
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    total_samples = 0

    print("Processing batches...")
    for batch_idx, (character, img_data, direction, action) in enumerate(temp_loader):
        # img_data shape: (batch_size, channels, height, width)
        batch_samples = img_data.size(0)
        img_data = img_data.view(batch_samples, img_data.size(1), -1)  # Flatten spatial dimensions

        mean += img_data.mean(2).sum(0)  # Sum over batch and spatial dimensions
        std += img_data.std(2).sum(0)
        total_samples += batch_samples

        if batch_idx % 50 == 0:
            print(f"Processed {batch_idx * temp_loader.batch_size} sprites...")

    mean /= total_samples
    std /= total_samples

    print("\n" + "="*50)
    print("DATALOADER NORMALIZATION VALUES")
    print("="*50)
    print(f"Mean per channel: {mean.tolist()}")
    print(f"Std per channel:  {std.tolist()}")
    print("\nUse these values in your transform:")
    print(f"transforms.Normalize(mean={mean.tolist()}, std={std.tolist()})")
    print("="*50)

    return mean.tolist(), std.tolist()
'''
