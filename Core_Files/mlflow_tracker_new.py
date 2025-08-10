"""
Enhanced MLflow integration for Fear Classification project.
Uses direct database writing to bypass pyarrow compatibility issues.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import warnings

# Import direct MLflow writer to bypass pyarrow issues
import sys
sys.path.append('.')
from direct_mlflow_writer import DirectMLflowWriter, MLflowContextManager

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


class NoOpContextManager:
    """A no-op context manager for when MLflow is disabled."""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MLflowTracker:
    """Enhanced MLflow tracker with direct database writing to bypass pyarrow issues."""
    
    def __init__(self):
        """Initialize MLflow tracker with direct database access."""
        self.active = False
        self.direct_writer = None
        self.experiment_name = "Fear_Classification_LOPOCV"
        self.current_context = None
        
        try:
            # Use direct writer to bypass pyarrow issues
            self.direct_writer = DirectMLflowWriter("./mlruns")
            self.direct_writer.create_experiment(self.experiment_name)
            self.active = True
            
            print(f"✅ MLflow initialized with direct database writer")
            print(f"📁 Tracking URI: ./mlruns")
            print(f"🧪 Experiment: {self.experiment_name}")
            
        except Exception as e:
            print(f"⚠️ MLflow direct writer failed: {e}")
            print("📊 Disabling MLflow tracking for this session")
            self.active = False
    
    def start_run(self, run_name: str, subject_name: str, test_n: int):
        """
        Start a new MLflow run using direct database writer.
        
        Args:
            run_name: Name for this run
            subject_name: Subject identifier
            test_n: Test number
        
        Returns:
            Context manager for the run
        """
        if not self.active or not self.direct_writer:
            return NoOpContextManager()
        
        # Create detailed run name
        detailed_run_name = f"{run_name}_{subject_name}_Test{test_n}"
        
        # Store reference for parameter/metric logging
        self.current_context = {
            'run_name': detailed_run_name,
            'subject_name': subject_name,
            'test_n': test_n
        }
        
        return MLflowContextManager(
            writer=self.direct_writer,
            run_name=detailed_run_name,
            experiment_name=self.experiment_name
        )
    
    def log_params(self, params_dict: Dict[str, Any]):
        """Log parameters using direct writer."""
        if not self.active or not self.direct_writer:
            return
        
        try:
            self.direct_writer.log_params_dict(params_dict)
            print(f"📊 Logged parameters: {list(params_dict.keys())}")
        except Exception as e:
            print(f"⚠️ Failed to log parameters: {e}")
    
    def log_metrics(self, metrics_dict: Dict[str, Any]):
        """Log metrics using direct writer."""
        if not self.active or not self.direct_writer:
            return
        
        try:
            # Handle nested metrics structure
            flat_metrics = {}
            
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    flat_metrics[key] = float(value)
                elif isinstance(value, dict):
                    # Handle nested dictionaries like 'metrics': {'accuracy': 0.8, ...}
                    if key == 'metrics':
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                flat_metrics[sub_key] = float(sub_value)
                    elif key == 'test_accuracies':
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                flat_metrics[f"accuracy_threshold_{sub_key}"] = float(sub_value)
                elif key == 'test_loss' and isinstance(value, (int, float)):
                    flat_metrics['test_loss'] = float(value)
            
            # Log the flattened metrics
            for metric_key, metric_value in flat_metrics.items():
                self.direct_writer.log_metric(metric_key, metric_value)
            
            print(f"📊 Logged metrics: {list(flat_metrics.keys())}")
            
        except Exception as e:
            print(f"⚠️ Failed to log metrics: {e}")
    
    def log_artifact(self, artifact_path: str):
        """Log artifact reference (simplified for direct writer)."""
        if not self.active or not self.direct_writer:
            return
        
        try:
            # Just log the artifact path as a tag since we're using simplified approach
            artifact_name = os.path.basename(artifact_path)
            self.direct_writer.log_tag("artifact_path", artifact_path)
            self.direct_writer.log_tag("model_file", artifact_name)
            print(f"📊 Logged artifact reference: {artifact_name}")
        except Exception as e:
            print(f"⚠️ Failed to log artifact: {e}")
    
    def finalize_run(self):
        """Finalize the current run (handled by context manager)."""
        if self.current_context:
            print(f"📊 Run finalized: {self.current_context['run_name']}")
            self.current_context = None
    
    def is_active(self) -> bool:
        """Check if MLflow tracking is active."""
        return self.active
    
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get current run information."""
        if not self.active or not self.current_context:
            return None
        
        return {
            'run_name': self.current_context.get('run_name', 'Unknown'),
            'subject_name': self.current_context.get('subject_name', 'Unknown'),
            'test_n': self.current_context.get('test_n', 0),
            'tracking_uri': './mlruns',
            'experiment_name': self.experiment_name
        }
