"""
MLflow integration for Fear Classification project.
Provides comprehensive experiment tracking with graceful fallback.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import json
import tempfile
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
    
    def _enable_fallback_mode(self):
        """Enable local fallback tracking when MLflow is unavailable."""
        self.fallback_mode = True
        self.active = True  # Still active, but in fallback mode
        
        # Create local tracking directory
        self.fallback_dir = "./fallback_mlflow_tracking"
        os.makedirs(self.fallback_dir, exist_ok=True)
        
        print("📊 Fallback tracking mode enabled")
        print(f"📁 Local tracking directory: {self.fallback_dir}")
    
    def _save_fallback_data(self, data_type: str, data: Dict[str, Any]):
        """Save data in fallback mode."""
        if not self.fallback_mode:
            return
            
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{timestamp}.json"
        filepath = os.path.join(self.fallback_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"📊 Saved {data_type} to {filepath}")
            
    def start_run(self, run_name: str, subject_name: str, test_n: int):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            subject_name: Subject identifier
            test_n: Test number
        
        Returns:
            Context manager (actual MLflow run or no-op)
        """
        if self.fallback_mode:
            # Initialize fallback run data
            self.current_run_data = {
                'run_name': f"{run_name}_{subject_name}_Test{test_n}",
                'subject_name': subject_name,
                'test_n': test_n,
                'start_time': pd.Timestamp.now().isoformat(),
                'params': {},
                'metrics': {},
                'tags': {}
            }
            print(f"📊 Started fallback tracking for: {self.current_run_data['run_name']}")
            return NoOpContextManager()
            
        if not self.active:
            return NoOpContextManager()
            
        try:
            import mlflow
            
            # Create detailed run name
            detailed_run_name = f"{run_name}_{subject_name}_Test{test_n}"
            
            # Start MLflow run
            self.run = mlflow.start_run(run_name=detailed_run_name)
            self.run_id = self.run.info.run_id
            
            # Log basic tags
            mlflow.set_tag("subject", subject_name)
            mlflow.set_tag("test_number", test_n)
            mlflow.set_tag("model_type", "1D_CNN")
            mlflow.set_tag("task", "fear_classification")
            
            return self.run
            
        except Exception as e:
            print(f"⚠️ MLflow run start failed: {e}")
            print("📊 Using no-op context manager")
            return NoOpContextManager()
    
    def log_parameters(self, config):
        """Log configuration parameters."""
        if self.fallback_mode:
            # Store params in fallback data
            params = {
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "early_stopping_patience": config.early_stopping_patience,
                "reduce_lr_patience": config.reduce_lr_patience,
                "land_type": config.land,
                "need_norm": config.need_norm,
                "need_standardize": config.need_standardize,
                "class_weight_0": config.class_weights[0],
                "class_weight_1": config.class_weights[1]
            }
            self.current_run_data['params'].update(params)
            print(f"📊 Logged parameters to fallback: {list(params.keys())}")
            return
            
        if not self.active or self.run_id is None:
            return
            
        try:
            import mlflow
            mlflow.log_param("batch_size", config.batch_size)
            mlflow.log_param("epochs", config.epochs)
            mlflow.log_param("learning_rate", config.learning_rate)
            mlflow.log_param("early_stopping_patience", config.early_stopping_patience)
            mlflow.log_param("reduce_lr_patience", config.reduce_lr_patience)
            mlflow.log_param("land_type", config.land)
            mlflow.log_param("need_norm", config.need_norm)
            mlflow.log_param("need_standardize", config.need_standardize)
            
            # Log class weights
            mlflow.log_param("class_weight_0", config.class_weights[0])
            mlflow.log_param("class_weight_1", config.class_weights[1])
            
        except Exception as e:
            print(f"⚠️ Failed to log parameters: {e}")
    
    def log_dataset_info(self, X_train_shape, X_test_shape, y_train, y_test):
        """Log dataset information."""
        if not self.active or self.run_id is None:
            return
            
        try:
            import mlflow
            # Log data dimensions
            mlflow.log_param("train_samples", X_train_shape[0])
            mlflow.log_param("test_samples", X_test_shape[0])
            mlflow.log_param("features", X_train_shape[2])
            mlflow.log_param("time_steps", X_train_shape[1])
            
            # Log class distribution
            train_unique, train_counts = np.unique(y_train, return_counts=True)
            test_unique, test_counts = np.unique(y_test, return_counts=True)
            
            for cls, count in zip(train_unique, train_counts):
                mlflow.log_param(f"train_class_{int(cls)}_count", count)
                
            for cls, count in zip(test_unique, test_counts):
                mlflow.log_param(f"test_class_{int(cls)}_count", count)
                
        except Exception as e:
            print(f"⚠️ Failed to log dataset info: {e}")
    
    def log_metrics(self, metrics_dict: Dict[str, Any]):
        """Log evaluation metrics."""
        if self.fallback_mode:
            # Store metrics in fallback data
            self.current_run_data['metrics'].update(metrics_dict)
            print(f"📊 Logged metrics to fallback: {list(metrics_dict.keys())}")
            return
            
        if not self.active or self.run_id is None:
            return
            
        try:
            import mlflow
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float, np.number)):
                    mlflow.log_metric(key, float(value))
                    
        except Exception as e:
            print(f"⚠️ Failed to log metrics: {e}")
    
    def log_model(self, model, model_name: str = "cnn_model"):
        """Log the trained model."""
        if not self.active or self.run_id is None:
            return None
            
        try:
            import mlflow.tensorflow
            
            # Log model as TensorFlow format
            mlflow.tensorflow.log_model(
                model, 
                model_name,
                registered_model_name=f"fear_classification_{model_name}"
            )
            
            # Get model info
            model_uri = f"runs:/{self.run_id}/{model_name}"
            return model_uri
            
        except Exception as e:
            print(f"⚠️ Failed to log model: {e}")
            return None
    
    def log_artifacts(self, artifacts_dict: Dict[str, Any]):
        """Log artifacts like plots, results, etc."""
        if not self.active or self.run_id is None:
            return
            
        try:
            import mlflow
            
            for artifact_name, artifact_data in artifacts_dict.items():
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(artifact_data, f, indent=2, default=str)
                    temp_path = f.name
                
                # Log artifact
                mlflow.log_artifact(temp_path, artifact_path=f"results/{artifact_name}.json")
                
                # Clean up
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"⚠️ Failed to log artifacts: {e}")
    
    def log_training_history(self, history):
        """Log training history metrics."""
        if not self.active or self.run_id is None or history is None:
            return
            
        try:
            import mlflow
            
            # Log training metrics for each epoch
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history.get('loss', []),
                history.history.get('accuracy', []),
                history.history.get('val_loss', []),
                history.history.get('val_accuracy', [])
            )):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("train_accuracy", acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
        except Exception as e:
            print(f"⚠️ Failed to log training history: {e}")
    
    def end_run(self):
        """End the current MLflow run."""
        if not self.active or self.run_id is None:
            return
            
        try:
            import mlflow
            mlflow.end_run()
            self.run_id = None
            self.run = None
            
        except Exception as e:
            print(f"⚠️ Failed to end run: {e}")
    
    def get_experiment_url(self) -> Optional[str]:
        """Get the MLflow experiment URL."""
        if not self.active or self.experiment_id is None:
            return None
            
        try:
            import mlflow
            tracking_uri = mlflow.get_tracking_uri()
            return f"{tracking_uri}/#/experiments/{self.experiment_id}"
            
        except Exception as e:
            print(f"⚠️ Failed to get experiment URL: {e}")
            return None
    
    def log_params(self, params_dict: Dict[str, Any]):
        """Log multiple parameters at once."""
        if not self.active or self.run_id is None:
            return
            
        try:
            import mlflow
            for key, value in params_dict.items():
                mlflow.log_param(key, value)
                
        except Exception as e:
            print(f"⚠️ Failed to log params: {e}")
    
    def set_tags(self, tags_dict: Dict[str, str]):
        """Set multiple tags at once."""
        if not self.active or self.run_id is None:
            return
            
        try:
            import mlflow
            for key, value in tags_dict.items():
                mlflow.set_tag(key, str(value))
                
        except Exception as e:
            print(f"⚠️ Failed to set tags: {e}")
    
    def is_active(self) -> bool:
        """Check if MLflow tracking is active."""
        return self.active
    
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get current run information."""
        if not self.active or self.run is None:
            return None
            
        try:
            return {
                'run_id': self.run.info.run_id,
                'experiment_id': self.run.info.experiment_id,
                'status': self.run.info.status,
                'start_time': self.run.info.start_time,
                'artifact_uri': self.run.info.artifact_uri
            }
        except Exception as e:
            print(f"⚠️ Failed to get run info: {e}")
            return None
    
    def finalize_run(self):
        """Finalize the current run and save fallback data if needed."""
        if self.fallback_mode and self.current_run_data:
            # Add end time
            self.current_run_data['end_time'] = pd.Timestamp.now().isoformat()
            
            # Save complete run data
            self._save_fallback_data('run_complete', self.current_run_data)
            
            # Create a summary file for easy viewing
            self._create_fallback_summary()
            
            print(f"📊 Fallback run data saved for: {self.current_run_data['run_name']}")
    
    def _create_fallback_summary(self):
        """Create a human-readable summary of the fallback run."""
        if not self.current_run_data:
            return
            
        summary_file = os.path.join(self.fallback_dir, f"run_summary_{self.current_run_data['run_name']}.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"MLflow Fallback Tracking Summary\n")
            f.write(f"={'='*50}\n\n")
            f.write(f"Run Name: {self.current_run_data['run_name']}\n")
            f.write(f"Subject: {self.current_run_data['subject_name']}\n")
            f.write(f"Start Time: {self.current_run_data['start_time']}\n")
            f.write(f"End Time: {self.current_run_data.get('end_time', 'N/A')}\n\n")
            
            f.write("Parameters:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.current_run_data['params'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nMetrics:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.current_run_data['metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"📋 Summary saved: {summary_file}")
    
    def get_all_fallback_runs(self):
        """Get all fallback run summaries for viewing."""
        if not self.fallback_mode or not os.path.exists(self.fallback_dir):
            return []
            
        summaries = []
        for filename in os.listdir(self.fallback_dir):
            if filename.startswith('run_summary_') and filename.endswith('.txt'):
                filepath = os.path.join(self.fallback_dir, filename)
                with open(filepath, 'r') as f:
                    summaries.append({
                        'filename': filename,
                        'content': f.read()
                    })
        
        return summaries
