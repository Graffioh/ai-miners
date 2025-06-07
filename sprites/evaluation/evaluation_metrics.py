from collections import defaultdict

class EvaluationMetrics:
    """Calculate various accuracy metrics from predictions and targets."""
    
    def __init__(self):
        self.direction_names = ['West', 'SW', 'South', 'SE', 'East', 'NE', 'North', 'NW']
        self.num_directions = len(self.direction_names)
    
    def calculate_accuracies(self, predictions, targets, characters):
        """
        Calculate overall, per-direction, and per-character accuracies.
        
        Args:
            predictions: List of predicted direction indices
            targets: List of true direction indices  
            characters: List of character names corresponding to each sample
            
        Returns:
            tuple: (overall_accuracy, per_direction_dict, per_character_dict)
        """
        overall_accuracy = self._calculate_overall_accuracy(predictions, targets)
        per_direction_dict = self._calculate_per_direction_accuracy(predictions, targets)
        per_character_dict = self._calculate_per_character_accuracy(predictions, targets, characters)
        
        return overall_accuracy, per_direction_dict, per_character_dict
    
    def _calculate_overall_accuracy(self, predictions, targets):
        """Calculate overall accuracy percentage."""
        correct = sum(p == t for p, t in zip(predictions, targets))
        total = len(targets)
        return 100 * correct / total if total > 0 else 0
    
    def _calculate_per_direction_accuracy(self, predictions, targets):
        """Calculate accuracy for each direction."""
        direction_correct = [0] * self.num_directions
        direction_total = [0] * self.num_directions
        
        for pred, target in zip(predictions, targets):
            direction_total[target] += 1
            if pred == target:
                direction_correct[target] += 1
        
        per_direction_dict = {}
        for i in range(self.num_directions):
            if direction_total[i] > 0:
                accuracy = 100 * direction_correct[i] / direction_total[i]
                per_direction_dict[self.direction_names[i]] = {
                    'accuracy': accuracy,
                    'correct': direction_correct[i],
                    'total': direction_total[i]
                }
            else:
                per_direction_dict[self.direction_names[i]] = {
                    'accuracy': 0,
                    'correct': 0,
                    'total': 0
                }
        
        return per_direction_dict
    
    def _calculate_per_character_accuracy(self, predictions, targets, characters):
        """Calculate accuracy for each character."""
        char_correct = defaultdict(int)
        char_total = defaultdict(int)
        
        for pred, target, char in zip(predictions, targets, characters):
            char_total[char] += 1
            if pred == target:
                char_correct[char] += 1
        
        per_character_dict = {}
        for char in char_total:
            if char_total[char] > 0:
                accuracy = 100 * char_correct[char] / char_total[char]
                per_character_dict[char] = {
                    'accuracy': accuracy,
                    'correct': char_correct[char],
                    'total': char_total[char]
                }
            else:
                per_character_dict[char] = {
                    'accuracy': 0,
                    'correct': 0,
                    'total': 0
                }
        
        return per_character_dict