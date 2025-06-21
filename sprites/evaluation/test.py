import torch
from  utils.plotter import create_direction_confusion_matrix, create_character_confusion_matrix

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
            character, img_data, target, _ = batch_data  
            
            img_data, target = img_data.to(device), target.to(device)
            
            # Forward pass
            output = model(img_data)
            loss = criterion(output, target)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Store for analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_characters.extend(character)
            
    # Calculate test metrics
    test_acc = 100 * correct / total
    print("DIRECTION CLASSIFICATION ACCURACY: ", test_acc)
    
    # Generate both confusion matrices
    create_direction_confusion_matrix(all_targets, all_predictions)
    create_character_confusion_matrix(all_characters, all_targets, all_predictions)
    
    return test_acc
