import torch

def test_model(model, test_loader, device):
    """
    Run the test loop and collect predictions and targets.
    Returns raw data for further processing.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_characters = []
    
    with torch.no_grad():
        for char_batch, img_data, target_direction_batch, _ in test_loader:
            img_data = img_data.to(device)
            target_direction_batch = target_direction_batch.to(device)
            
            output = model(img_data)
            _, predicted_direction_batch = torch.max(output, 1)
            
            # Collect results
            all_predictions.extend(predicted_direction_batch.cpu().numpy())
            all_targets.extend(target_direction_batch.cpu().numpy())
            all_characters.extend(char_batch)
    
    return all_predictions, all_targets, all_characters