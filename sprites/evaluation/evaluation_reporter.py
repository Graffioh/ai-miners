class EvaluationReporter:
    """Handle printing and saving of evaluation results."""
    
    def __init__(self, direction_names=None):
        self.direction_names = direction_names or ['West', 'SW', 'South', 'SE', 'East', 'NE', 'North', 'NW']
    
    def print_results(self, overall_accuracy, per_direction_dict, per_character_dict):
        """Print evaluation results to console."""
        self._print_overall_accuracy(overall_accuracy)
        self._print_per_direction_accuracy(per_direction_dict)
        self._print_per_character_accuracy(per_character_dict)
    
    def save_results(self, overall_accuracy, per_direction_dict, per_character_dict, 
                    config, total_samples, correct_predictions):
        """Save detailed results to file."""
        results_path = config.manager.get_log_path("test_evaluation_details")
        
        with open(results_path, "w") as f:
            f.write("Detailed Evaluation Results\n")
            f.write("===========================\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Correct predictions: {correct_predictions}\n\n")
            
            self._write_per_direction_results(f, per_direction_dict)
            self._write_per_character_results(f, per_character_dict)
        
        print(f"✅ Detailed evaluation results saved to: {results_path}")
    
    def _print_overall_accuracy(self, overall_accuracy):
        """Print overall accuracy."""
        print(f"\nOverall Test/Validation Accuracy: {overall_accuracy:.2f}%")
    
    def _print_per_direction_accuracy(self, per_direction_dict):
        """Print per-direction accuracy results."""
        print("\nPer-Direction Accuracy:")
        print("-" * 30)
        
        for i, dir_name in enumerate(self.direction_names):
            if dir_name in per_direction_dict:
                data = per_direction_dict[dir_name]
                if data['total'] > 0:
                    print(f"Direction {i} ({dir_name}): {data['accuracy']:.2f}% "
                          f"({data['correct']}/{data['total']})")
                else:
                    print(f"Direction {i} ({dir_name}): No samples")
    
    def _print_per_character_accuracy(self, per_character_dict):
        """Print per-character accuracy results."""
        print("\nPer-Character Direction Accuracy:")
        print("-" * 30)
        
        sorted_characters = sorted(per_character_dict.keys())
        for char_name in sorted_characters:
            data = per_character_dict[char_name]
            if data['total'] > 0:
                print(f"Character '{char_name}': {data['accuracy']:.2f}% "
                      f"({data['correct']}/{data['total']})")
            else:
                print(f"Character '{char_name}': No samples")
    
    def _write_per_direction_results(self, f, per_direction_dict):
        """Write per-direction results to file."""
        f.write("Per-Direction Accuracy:\n")
        for i, dir_name in enumerate(self.direction_names):
            if dir_name in per_direction_dict:
                data = per_direction_dict[dir_name]
                if data['total'] > 0:
                    f.write(f"  - Direction {i} ({dir_name}): {data['accuracy']:.2f}% "
                           f"({data['correct']}/{data['total']})\n")
                else:
                    f.write(f"  - Direction {i} ({dir_name}): No samples\n")
        f.write("\n")
    
    def _write_per_character_results(self, f, per_character_dict):
        """Write per-character results to file."""
        f.write("Per-Character Direction Accuracy:\n")
        sorted_characters = sorted(per_character_dict.keys())
        for char_name in sorted_characters:
            data = per_character_dict[char_name]
            if data['total'] > 0:
                f.write(f"  - Character '{char_name}': {data['accuracy']:.2f}% "
                       f"({data['correct']}/{data['total']})\n")
            else:
                f.write(f"  - Character '{char_name}': No samples\n")