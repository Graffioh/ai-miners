import torch

def test_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data for each epoch
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Store predictions and targets for confusion matrix
    all_predictions = []
    all_targets = []
    all_characters = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle different batch formats
            if len(batch_data) == 4:
                _, img_data, target, character = batch_data
            else:
                img_data, target = batch_data
                character = None
            
            # Move data to device
            img_data, target = img_data.to(device), target.to(device)
            
            # Forward pass
            output = model(img_data)
            loss = criterion(output, target)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    # Calculate test metrics
    #test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total

    print("TEST ACCURACY: ", test_acc)