import torch
from collections import defaultdict

def test_model(model, test_loader, device, config=None):
    """
    Evaluate the model, calculate overall, per-direction, and per-character accuracies,
    and optionally save results.
    """
    model.eval()
    overall_correct = 0
    overall_total = 0

    # For per-direction accuracy
    num_directions = 8  # Assuming 8 directions based on existing code
    direction_correct = list(0. for _ in range(num_directions))
    direction_total = list(0. for _ in range(num_directions))
    direction_names = ['West', 'SW', 'South', 'SE', 'East', 'NE', 'North', 'NW']

    # For per-character accuracy (accuracy of predicting direction for that character)
    char_correct_predictions = defaultdict(int)
    char_total_samples = defaultdict(int)

    with torch.no_grad():
        # SpriteDataset yields (character, image, direction_label, action)
        # DataLoader batches these, so:
        # char_batch contains character names/IDs for the batch
        # img_data is the image tensor batch
        # target_direction_batch is the direction label batch
        # action_batch is the action label batch (currently ignored here)
        for char_batch, img_data, target_direction_batch, _ in test_loader:
            img_data = img_data.to(device)
            target_direction_batch = target_direction_batch.to(device) # True directions

            output = model(img_data) # Model predicts directions
            _, predicted_direction_batch = torch.max(output, 1)

            overall_total += target_direction_batch.size(0)
            overall_correct += (predicted_direction_batch == target_direction_batch).sum().item()

            # Per-item correctness mask for the current batch
            correct_predictions_mask = (predicted_direction_batch == target_direction_batch) # No squeeze, keep batch dim

            for i in range(target_direction_batch.size(0)):
                true_direction_label = target_direction_batch[i].item()
                is_correct = correct_predictions_mask[i].item()
                
                # Per-direction accuracy tracking
                direction_correct[true_direction_label] += is_correct
                direction_total[true_direction_label] += 1

                # Per-character accuracy tracking
                # char_batch is expected to be a list/tuple of character names from the DataLoader
                character_name = char_batch[i] 
                char_total_samples[character_name] += 1
                if is_correct:
                    char_correct_predictions[character_name] += 1

    # Calculate overall accuracy
    overall_accuracy = 100 * overall_correct / overall_total if overall_total > 0 else 0

    # Calculate and store per-direction accuracies
    per_direction_accuracies_dict = {}
    print(f"\nOverall Test/Validation Accuracy: {overall_accuracy:.2f}%") # Generic term for console
    print("\nPer-Direction Accuracy:")
    print("-" * 30)
    for i in range(num_directions):
        dir_name = direction_names[i]
        if direction_total[i] > 0:
            acc = 100 * direction_correct[i] / direction_total[i]
            per_direction_accuracies_dict[dir_name] = acc
            print(f"Direction {i} ({dir_name}): {acc:.2f}% ({int(direction_correct[i])}/{int(direction_total[i])})")
        else:
            per_direction_accuracies_dict[dir_name] = 0
            print(f"Direction {i} ({dir_name}): No samples")

    # Calculate and store per-character accuracies
    per_character_accuracies_dict = {}
    print("\nPer-Character Direction Accuracy:")
    print("-" * 30)
    sorted_characters = sorted(char_total_samples.keys())
    for char_name in sorted_characters:
        if char_total_samples[char_name] > 0:
            acc = 100 * char_correct_predictions[char_name] / char_total_samples[char_name]
            per_character_accuracies_dict[char_name] = acc
            print(f"Character '{char_name}': {acc:.2f}% ({char_correct_predictions[char_name]}/{char_total_samples[char_name]})")
        else:
            per_character_accuracies_dict[char_name] = 0
            print(f"Character '{char_name}': No samples (should not happen if data exists)")


    # Save detailed results if config provided (intended for final test run)
    if config:
        # Log file name can be made more specific if distinguishing val/test here
        results_path = config.manager.get_log_path("test_evaluation_details") 
        with open(results_path, "w") as f:
            f.write(f"Detailed Evaluation Results\n")
            f.write(f"===========================\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
            f.write(f"Total samples: {overall_total}\n")
            f.write(f"Correct predictions: {overall_correct}\n\n")

            f.write(f"Per-Direction Accuracy:\n")
            for i in range(num_directions):
                dir_name = direction_names[i]
                if direction_total[i] > 0:
                    acc = 100 * direction_correct[i] / direction_total[i]
                    f.write(f"  - Direction {i} ({dir_name}): {acc:.2f}% ({int(direction_correct[i])}/{int(direction_total[i])})\n")
                else:
                    f.write(f"  - Direction {i} ({dir_name}): No samples\n")
            f.write("\n")

            f.write(f"Per-Character Direction Accuracy:\n")
            for char_name in sorted_characters:
                if char_total_samples[char_name] > 0:
                    acc = 100 * char_correct_predictions[char_name] / char_total_samples[char_name]
                    f.write(f"  - Character '{char_name}': {acc:.2f}% ({char_correct_predictions[char_name]}/{char_total_samples[char_name]})\n")
                else:
                    f.write(f"  - Character '{char_name}': No samples\n")
        print(f"✅ Detailed evaluation results saved to: {results_path}")

    return overall_accuracy, per_direction_accuracies_dict, per_character_accuracies_dict