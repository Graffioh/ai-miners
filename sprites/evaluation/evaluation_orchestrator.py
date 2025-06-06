from evaluation.test import test_model
from evaluation.evaluation_metrics import EvaluationMetrics
from evaluation.evaluation_reporter import EvaluationReporter

def evaluate_model(model, test_loader, device, config=None, print_results=True):
    """
    High-level function to evaluate model and optionally save results.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        config: Configuration object (optional, for saving results)
        print_results: Whether to print results to console
        
    Returns:
        tuple: (overall_accuracy, per_direction_dict, per_character_dict)
    """
    # Run the test loop
    predictions, targets, characters = test_model(model, test_loader, device)
    
    # Calculate metrics
    metrics_calculator = EvaluationMetrics()
    overall_accuracy, per_direction_dict, per_character_dict = metrics_calculator.calculate_accuracies(
        predictions, targets, characters
    )
    
    # Handle reporting
    reporter = EvaluationReporter()
    
    if print_results:
        reporter.print_results(overall_accuracy, per_direction_dict, per_character_dict)
    
    if config:
        total_samples = len(targets)
        correct_predictions = sum(p == t for p, t in zip(predictions, targets))
        reporter.save_results(
            overall_accuracy, per_direction_dict, per_character_dict,
            config, total_samples, correct_predictions
        )
    
    return overall_accuracy, per_direction_dict, per_character_dict