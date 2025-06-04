import torch

def test_model(model, test_loader, device, config=None):
    """
    Evaluate the model and optionally save results
    """
    model.eval()
    correct = 0
    total = 0

    # For detailed analysis
    class_correct = list(0. for i in range(8))  # 8 directions
    class_total = list(0. for i in range(8))

    with torch.no_grad():
        for _, img_data, target, _ in test_loader:
            img_data, target = img_data.to(device), target.to(device)
            output = model(img_data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Per-class accuracy tracking
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Calculate overall accuracy
    accuracy = 100 * correct / total

    # Calculate per-class accuracies
    class_accuracies = []
    direction_names = ['West', 'SW', 'South', 'SE', 'East', 'NE', 'North', 'NW']

    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    print("\nPer-Direction Accuracy:")
    print("-" * 30)

    for i in range(8):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracies.append(class_acc)
            print(f"Direction {i} ({direction_names[i]}): {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            class_accuracies.append(0)
            print(f"Direction {i} ({direction_names[i]}): No samples")

    # Save detailed results if config provided
    if config:
        results_path = config.manager.get_log_path("test_results")
        with open(results_path, "w") as f:
            f.write(f"Test Results\n")
            f.write(f"============\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct predictions: {correct}\n\n")
            f.write(f"Per-Direction Results:\n")
            for i in range(8):
                if class_total[i] > 0:
                    class_acc = 100 * class_correct[i] / class_total[i]
                    f.write(f"Direction {i} ({direction_names[i]}): {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})\n")
                else:
                    f.write(f"Direction {i} ({direction_names[i]}): No samples\n")

        print(f"✅ Detailed results saved to: {results_path}")

    return accuracy
