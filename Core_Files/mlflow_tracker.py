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
            # Use direct writer to bypass pyarrow issues with absolute path
            mlruns_path = os.path.abspath("./mlruns")
            self.direct_writer = DirectMLflowWriter(mlruns_path)
            self.direct_writer.create_experiment(self.experiment_name)
            self.active = True
            
            print(f"✅ MLflow initialized with direct database writer")
            print(f"📁 Tracking URI: {mlruns_path}")
            print(f"🧪 Experiment: {self.experiment_name}")
            
        except Exception as e:
            print(f"⚠️ MLflow initialization failed: {e}")
            print(f"   Continuing without MLflow tracking...")
            self.active = False
    
    def start_run(self, run_name: str = None, subject_name: str = None, test_n: int = None):
        """Start an MLflow run with context manager support."""
        if not self.active or not self.direct_writer:
            return NoOpContextManager()
        
        # Store context for later use
        self.current_context = {
            'run_name': run_name,
            'subject_name': subject_name, 
            'test_n': test_n
        }
        
        return MLflowContextManager(
            writer=self.direct_writer,
            run_name=run_name
        )
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not self.active or not self.direct_writer:
            return
        
        try:
            # Flatten nested parameters
            flat_params = {}
            for key, value in params.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_params[f"{key}_{sub_key}"] = str(sub_value)
                else:
                    flat_params[key] = str(value)
            
            # Log parameters
            for param_key, param_value in flat_params.items():
                self.direct_writer.log_param(param_key, param_value)
            
            print(f"📊 Logged parameters: {list(flat_params.keys())}")
            
        except Exception as e:
            print(f"⚠️ Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to MLflow.""" 
        if not self.active or not self.direct_writer:
            return
        
        try:
            # Flatten nested metrics
            flat_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, np.number)):
                            flat_metrics[f"{key}_{sub_key}"] = float(sub_value)
                else:
                    if isinstance(value, (int, float, np.number)):
                        flat_metrics[key] = float(value)
            
            # Log metrics 
            for metric_key, metric_value in flat_metrics.items():
                self.direct_writer.log_metric(metric_key, metric_value)
            
            print(f"📊 Logged metrics: {list(flat_metrics.keys())}")
            
        except Exception as e:
            print(f"⚠️ Failed to log metrics: {e}")
    
    def log_artifact(self, artifact_path: str):
        """Log an artifact to MLflow."""
        if not self.active or not self.direct_writer:
            return
        
        try:
            # For direct writer, we need to copy the file to the artifacts directory
            # Get the run artifacts directory
            run_artifacts_dir = os.path.join(
                self.direct_writer.tracking_uri,
                str(self.direct_writer.current_experiment_id),
                self.direct_writer.current_run_uuid,
                "artifacts"
            )
            
            # Ensure artifacts directory exists
            os.makedirs(run_artifacts_dir, exist_ok=True)
            
            # Copy artifact to the artifacts directory
            artifact_name = os.path.basename(artifact_path)
            artifact_dest = os.path.join(run_artifacts_dir, artifact_name)
            
            if os.path.exists(artifact_path):
                import shutil
                shutil.copy2(artifact_path, artifact_dest)
                print(f"📊 Logged artifact: {artifact_name}")
            else:
                print(f"⚠️ Artifact file not found: {artifact_path}")
                
        except Exception as e:
            print(f"⚠️ Failed to log artifact: {e}")
    
    def finalize_run(self):
        """Finalize the current MLflow run."""
        if not self.active:
            return
        
        if self.current_context:
            print(f"📊 Run finalized: {self.current_context['run_name']}")
        
        # Direct writer auto-finalizes
        self.current_context = None
    
    def get_fallback_data(self) -> Dict[str, Any]:
        """Get fallback data structure for saving results when MLflow fails."""
        if not self.active or not self.current_context:
            return {}
        
        return {
            'run_name': self.current_context.get('run_name', 'Unknown'),
            'subject_name': self.current_context.get('subject_name', 'Unknown'),
            'test_n': self.current_context.get('test_n', 0),
            'mlflow_available': self.active,
            'tracking_uri': './mlruns'
        }
    
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
