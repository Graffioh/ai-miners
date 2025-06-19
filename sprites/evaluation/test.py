import torch

def test_model(model, test_loader, criterion, device, plotter=None):
    """
    Evaluate the model on test data for each epoch
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _, img_data, target, _ in test_loader:
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
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    # Update plotter with test metrics
    if plotter:
        plotter.update_test(test_loss, test_acc)
    
    return test_loss, test_acc