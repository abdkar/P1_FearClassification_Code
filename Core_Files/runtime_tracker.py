"""
Runtime tracking and summary module for Fear Classification project.
"""

import time
import numpy as np
from typing import Dict, List


class RuntimeTracker:
    """Track runtime for different phases of the model pipeline."""
    
    def __init__(self):
        """Initialize runtime tracker."""
        self.times = {}
        self.start_times = {}
        self.fold_times = []
        self.inference_times = []
        
    def start_timer(self, phase: str) -> None:
        """Start timing a phase."""
        self.start_times[phase] = time.time()
        
    def end_timer(self, phase: str) -> float:
        """End timing a phase and return duration."""
        if phase not in self.start_times:
            raise ValueError(f"Timer for phase '{phase}' was not started")
        
        duration = time.time() - self.start_times[phase]
        self.times[phase] = duration
        return duration
        
    def add_fold_time(self, fold_time: float) -> None:
        """Add training time for a fold."""
        self.fold_times.append(fold_time)
        
    def add_inference_time(self, inference_time: float) -> None:
        """Add inference time for a test set."""
        self.inference_times.append(inference_time)
        
    def get_runtime_summary(self, num_folds: int = None) -> Dict:
        """
        Generate comprehensive runtime summary.
        
        Args:
            num_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with runtime summary statistics
        """
        if not num_folds:
            num_folds = len(self.fold_times) if self.fold_times else 1
            
        # Calculate training statistics
        total_training_time = sum(self.fold_times) if self.fold_times else 0
        avg_training_time = total_training_time / num_folds if num_folds > 0 else 0
        
        # Calculate inference statistics
        total_inference_time = sum(self.inference_times) if self.inference_times else 0
        avg_inference_time = total_inference_time / len(self.inference_times) if self.inference_times else 0
        
        # Total runtime
        total_runtime = total_training_time + total_inference_time
        
        summary = {
            'metric': ['Total Runtime for {} Folds'.format(num_folds),
                      'Average Time per Fold',
                      'Average Training Time per Fold',
                      'Average Inference Time per Test Set'],
            'time_seconds': [total_runtime, 
                           total_runtime / num_folds if num_folds > 0 else 0,
                           avg_training_time,
                           avg_inference_time],
            'time_minutes': [total_runtime / 60,
                           (total_runtime / num_folds) / 60 if num_folds > 0 else 0,
                           avg_training_time / 60,
                           avg_inference_time / 60 if avg_inference_time > 0 else 0]
        }
        
        return summary
        
    def print_runtime_summary(self, num_folds: int = None) -> None:
        """Print formatted runtime summary."""
        summary = self.get_runtime_summary(num_folds)
        
        print("=" * 60)
        print("1D-CNN Cross-Validation Runtime Summary")
        print("=" * 60)
        print(f"{'Metric':<40} {'Time (seconds)':<15} {'Time (minutes)':<15}")
        print("=" * 60)
        
        for i, metric in enumerate(summary['metric']):
            time_sec = summary['time_seconds'][i]
            time_min = summary['time_minutes'][i]
            
            if time_sec >= 60:
                print(f"{metric:<40} {time_sec:<15.2f} {time_min:<15.2f}")
            else:
                print(f"{metric:<40} {time_sec:<15.4f} {'-':<15}")
                
        print("=" * 60)
        
    def save_runtime_summary(self, filepath: str, num_folds: int = None) -> None:
        """
        Save runtime summary to file.
        
        Args:
            filepath: Path to save the summary
            num_folds: Number of cross-validation folds
        """
        summary = self.get_runtime_summary(num_folds)
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("1D-CNN Cross-Validation Runtime Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Metric':<40} {'Time (seconds)':<15} {'Time (minutes)':<15}\n")
            f.write("=" * 60 + "\n")
            
            for i, metric in enumerate(summary['metric']):
                time_sec = summary['time_seconds'][i]
                time_min = summary['time_minutes'][i]
                
                if time_sec >= 60:
                    f.write(f"{metric:<40} {time_sec:<15.2f} {time_min:<15.2f}\n")
                else:
                    f.write(f"{metric:<40} {time_sec:<15.4f} {'-':<15}\n")
                    
            f.write("=" * 60 + "\n")
            
        print(f"Runtime summary saved to: {filepath}")


# Global runtime tracker instance
runtime_tracker = RuntimeTracker()
