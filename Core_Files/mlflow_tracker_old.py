"""
MLflow integration for Fear Classification project.
Provides comprehensive experiment tracking with graceful fallback.
"""

import mlflow
import mlflow.tensorflow
import os
import pickle
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


class NoOpContextManager:
    """A no-op context manager for when MLflow is disabled."""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MLflowTracker:

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import json
import tempfile
import warnings

# Try to import MLflow with graceful fallback
try:
    import mlflow
    import mlflow.tensorflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    print("✅ MLflow imported successfully")
except ImportError as e:
    MLFLOW_AVAILABLE = False
    print(f"⚠️ MLflow not available: {e}")
    print("📊 MLflow tracking will be disabled, but experiments will continue normally")
except Exception as e:
    MLFLOW_AVAILABLE = False
    print(f"⚠️ MLflow initialization failed: {e}")
    print("📊 MLflow tracking will be disabled, but experiments will continue normally")


class MLflowTracker:
    """MLflow experiment tracking for Fear Classification experiments with graceful fallback."""
    
    def __init__(self, experiment_name: str = "Fear_Classification_Experiments"):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.run_id = None
        self.active = MLFLOW_AVAILABLE
        
        if self.active:
            self.setup_mlflow()
        else:
            print("📊 MLflow tracking disabled - continuing without experiment tracking")
        
    def setup_mlflow(self):
        """Setup MLflow experiment and tracking URI."""
        if not self.active:
            return
            
        try:
            # Set tracking URI to local directory
            mlflow.set_tracking_uri("./mlflow_tracking")
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"📊 Created new MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"📊 Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            print(f"⚠️ MLflow setup failed: {e}")
            print("📊 Disabling MLflow tracking for this session")
            self.active = False
            
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
        if not self.active:
            return NoOpContextManager()
            
        try:
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
            
            print(f"🔄 Started MLflow run: {detailed_run_name}")
            
        except Exception as e:
            print(f"⚠️ MLflow run start failed: {e}")
            self.active = False
            self.run_id = None
            
    def log_parameters(self, config):
        """Log experiment parameters."""
        if not self.active or self.run_id is None:
            return
            
        try:
            # Log hyperparameters
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
            
            print("📝 Logged parameters to MLflow")
            
        except Exception as e:
            print(f"⚠️ MLflow parameter logging failed: {e}")
            
    def log_dataset_info(self, X_train_shape, X_test_shape, y_train, y_test):
        """Log dataset information."""
        if not self.active or self.run_id is None:
            return
            
        try:
            # Log data shapes
            mlflow.log_param("train_samples", X_train_shape[0])
            mlflow.log_param("test_samples", X_test_shape[0])
            mlflow.log_param("timesteps", X_train_shape[1])
            mlflow.log_param("features", X_train_shape[2])
            
            # Log class distribution
            train_class_dist = np.bincount(y_train.astype(int))
            test_class_dist = np.bincount(y_test.astype(int))
            
            mlflow.log_param("train_class_0", int(train_class_dist[0]))
            mlflow.log_param("train_class_1", int(train_class_dist[1]))
            mlflow.log_param("test_class_0", int(test_class_dist[0]))
            mlflow.log_param("test_class_1", int(test_class_dist[1]))
            
            print("📊 Logged dataset info to MLflow")
            
        except Exception as e:
            print(f"⚠️ MLflow dataset logging failed: {e}")
            
    def log_training_metrics(self, history, training_time: float):
        """Log training metrics and history."""
        if not self.active or self.run_id is None:
            return
            
        try:
            # Log training time
            mlflow.log_metric("training_time_seconds", training_time)
            mlflow.log_metric("training_time_minutes", training_time / 60)
            
            # Log final training metrics
            if hasattr(history, 'history'):
                hist = history.history
                
                # Log final epoch metrics
                final_epoch = len(hist['loss']) - 1
                mlflow.log_metric("final_train_loss", hist['loss'][final_epoch])
                mlflow.log_metric("final_train_accuracy", hist['accuracy'][final_epoch])
                mlflow.log_metric("final_val_loss", hist['val_loss'][final_epoch])
                mlflow.log_metric("final_val_accuracy", hist['val_accuracy'][final_epoch])
                mlflow.log_metric("epochs_trained", final_epoch + 1)
                
                # Log best validation metrics
                best_val_acc_epoch = np.argmax(hist['val_accuracy'])
                best_val_loss_epoch = np.argmin(hist['val_loss'])
                
                mlflow.log_metric("best_val_accuracy", max(hist['val_accuracy']))
                mlflow.log_metric("best_val_accuracy_epoch", best_val_acc_epoch + 1)
                mlflow.log_metric("best_val_loss", min(hist['val_loss']))
                mlflow.log_metric("best_val_loss_epoch", best_val_loss_epoch + 1)
                
                # Log training curves as metrics over epochs
                for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                    hist['loss'], hist['accuracy'], hist['val_loss'], hist['val_accuracy']
                )):
                    mlflow.log_metrics({
                        "epoch_train_loss": loss,
                        "epoch_train_accuracy": acc,
                        "epoch_val_loss": val_loss,
                        "epoch_val_accuracy": val_acc
                    }, step=epoch)
            
            print("📈 Logged training metrics to MLflow")
            
        except Exception as e:
            print(f"⚠️ MLflow training metrics logging failed: {e}")
            
    def log_evaluation_metrics(self, eval_results: Dict[str, Any], inference_time: float):
        """Log evaluation metrics."""
        if not self.active or self.run_id is None:
            return
            
        try:
            # Log inference time
            mlflow.log_metric("inference_time_seconds", inference_time)
            mlflow.log_metric("inference_time_milliseconds", inference_time * 1000)
            
            # Log test accuracies for different thresholds
            if 'test_accuracies' in eval_results:
                for threshold, accuracy in eval_results['test_accuracies'].items():
                    mlflow.log_metric(f"test_accuracy_threshold_{threshold}", accuracy)
            
            # Log comprehensive metrics
            if 'metrics' in eval_results:
                metrics = eval_results['metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float, np.number)):
                        mlflow.log_metric(f"test_{metric_name}", float(metric_value))
            
            # Log test loss
            if 'test_loss' in eval_results:
                mlflow.log_metric("test_loss", eval_results['test_loss'])
                
            print("🎯 Logged evaluation metrics to MLflow")
            
        except Exception as e:
            print(f"⚠️ MLflow evaluation metrics logging failed: {e}")
            
    def log_model(self, model, model_name: str = "fear_classification_model"):
        """Log the trained model."""
        if not self.active or self.run_id is None:
            return
            
        try:
            # Log model with MLflow
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path=model_name,
                registered_model_name=f"{self.experiment_name}_{model_name}"
            )
            
            print(f"🤖 Logged model '{model_name}' to MLflow")
            
        except Exception as e:
            print(f"⚠️ MLflow model logging failed: {e}")
            
    def log_artifacts(self, results_df: pd.DataFrame, feature_importance: np.ndarray, 
                     runtime_summary: Dict[str, Any]):
        """Log experiment artifacts."""
        if not self.active or self.run_id is None:
            return
            
        try:
            # Create temporary directory for artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Save results dataframe
                results_path = os.path.join(temp_dir, "results.csv")
                results_df.to_csv(results_path, index=False)
                mlflow.log_artifact(results_path, "results")
                
                # Save feature importance
                importance_path = os.path.join(temp_dir, "feature_importance.csv")
                pd.DataFrame({
                    'feature_index': range(len(feature_importance)),
                    'importance_score': feature_importance
                }).to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, "feature_importance")
                
                # Save runtime summary
                runtime_path = os.path.join(temp_dir, "runtime_summary.json")
                with open(runtime_path, 'w') as f:
                    json.dump(runtime_summary, f, indent=2)
                mlflow.log_artifact(runtime_path, "runtime")
                
            print("📁 Logged artifacts to MLflow")
            
        except Exception as e:
            print(f"⚠️ MLflow artifacts logging failed: {e}")
            
    def log_confusion_matrices(self, eval_results: Dict[str, Any]):
        """Log confusion matrices for different thresholds."""
        if not self.active or self.run_id is None:
            return
            
        try:
            if 'confusion_matrices' in eval_results:
                for threshold, cm in eval_results['confusion_matrices'].items():
                    if hasattr(cm, 'flatten'):
                        cm_flat = cm.flatten()
                        mlflow.log_metric(f"cm_{threshold}_tn", int(cm_flat[0]))
                        mlflow.log_metric(f"cm_{threshold}_fp", int(cm_flat[1]))
                        mlflow.log_metric(f"cm_{threshold}_fn", int(cm_flat[2]))
                        mlflow.log_metric(f"cm_{threshold}_tp", int(cm_flat[3]))
                        
            print("🔢 Logged confusion matrices to MLflow")
            
        except Exception as e:
            print(f"⚠️ MLflow confusion matrix logging failed: {e}")
            
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if not self.active or self.run_id is None:
            return
            
        try:
            mlflow.end_run(status=status)
            print(f"✅ Ended MLflow run with status: {status}")
            self.run_id = None
            
        except Exception as e:
            print(f"⚠️ MLflow run end failed: {e}")
            
    def get_experiment_url(self) -> Optional[str]:
        """Get MLflow experiment UI URL."""
        if not self.active:
            return None
            
        try:
            return f"http://localhost:5000/#/experiments/{mlflow.get_experiment_by_name(self.experiment_name).experiment_id}"
        except:
            return None


# Global MLflow tracker instance
mlflow_tracker = MLflowTracker()
