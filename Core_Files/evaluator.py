"""
Evaluation module for Fear Classification project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple, Dict, Any, List
from tensorflow.keras.utils import to_categorical


from utils import (
    get_file_name, calculate_metrics, apply_threshold_predictions,
    calculate_threshold_accuracy, create_results_dataframe,
    create_pivoted_dataframe, save_results, print_results_summary,
    color_row
)

# Optional Integrated Gradients import
try:  # pragma: no cover
    from Core_Files.integrated_gradients import IntegratedGradients
except Exception:
    IntegratedGradients = None

# Attempt MLflow tracker import
try:  # pragma: no cover
    from mlflow_tracker import mlflow_tracker
except Exception:
    mlflow_tracker = None


class Evaluator:
    """Model evaluator for Fear Classification."""
    
    def __init__(self, config):
        """Initialize Evaluator with configuration."""
        self.config = config
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Tuple of (evaluation results dictionary, inference time in seconds)
        """
        # Start timing inference
        inference_start = time.time()
        
        # Get predictions
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        predictions = model.predict(X_test)[:, 1]
        
        # Calculate basic accuracy
        test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test.astype(int)), verbose=0)
        
        # End timing inference
        inference_time = time.time() - inference_start
        
        # Apply different thresholds
        thresholds = [0.43, 0.35]
        threshold_results = apply_threshold_predictions(predictions, thresholds)
        
        # Calculate accuracies for different thresholds
        test_accuracies = {'0.5': test_acc}
        confusion_matrices = {'0.5': calculate_metrics(y_test, y_pred)['confusion_matrix']}
        
        for threshold_name, threshold_data in threshold_results.items():
            threshold_val = threshold_data['threshold']
            class_predictions = threshold_data['predictions']
            
            acc = calculate_threshold_accuracy(y_test, class_predictions)
            cm = calculate_metrics(y_test, class_predictions)['confusion_matrix']
            
            test_accuracies[str(threshold_val)] = acc
            confusion_matrices[str(threshold_val)] = cm
        
        # Print results summary
        print_results_summary(y_test, y_pred, test_accuracies, confusion_matrices)
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        results = {
            'y_pred': y_pred,
            'predictions': predictions,
            'y_test': y_test,
            'test_accuracies': test_accuracies,
            'confusion_matrices': confusion_matrices,
            'threshold_results': threshold_results,
            'metrics': metrics,
            'test_loss': test_loss
        }
        
        # Log evaluation metrics to MLflow
        try:
            mlflow_tracker.log_evaluation_metrics(results, inference_time)
            mlflow_tracker.log_confusion_matrices(results)
        except Exception as e:
            print(f"⚠️ MLflow evaluation logging warning: {e}")
        
        return results, inference_time
    
    def create_results_tables(self, eval_results: Dict[str, Any], paths: Dict[str, str]) -> pd.DataFrame:
        """
        Create results tables and save them.
        
        Args:
            eval_results: Evaluation results
            paths: Dictionary containing file paths
            
        Returns:
            Merged results DataFrame
        """
        # Get file names from test folders
        path_AT = paths['HF_test']
        path_BT = paths['LF_test']
        
        file_names_AT = [get_file_name(file) for file in os.listdir(path_AT) if file.endswith('.csv')]
        file_names_BT = [get_file_name(file) for file in os.listdir(path_BT) if file.endswith('.csv')]
        
        # Create results DataFrame
        results = create_results_dataframe(
            file_names_AT, file_names_BT,
            eval_results['y_test'], eval_results['y_pred'],
            np.round(eval_results['predictions'], decimals=3).astype(str),
            eval_results['threshold_results']
        )
        
        # Create pivoted DataFrame
        pivoted_df = create_pivoted_dataframe(results, self.config.get_hop_range())
        
        return results, pivoted_df
    
    def create_comprehensive_results(self, eval_results: Dict[str, Any], 
                                   training_summary: Dict[str, Any],
                                   pivoted_df: pd.DataFrame,
                                   X_train_shape: Tuple) -> pd.DataFrame:
        """
        Create comprehensive results DataFrame.
        
        Args:
            eval_results: Evaluation results
            training_summary: Training summary
            pivoted_df: Pivoted results DataFrame
            X_train_shape: Shape of training data
            
        Returns:
            Merged comprehensive results DataFrame
        """
        # Calculate errors
        y_test = eval_results['y_test']
        y_pred = eval_results['y_pred']
        errors = np.sum(np.abs(y_pred - y_test))
        
        # Create comprehensive results
        result_data = {
            'No errors': [errors.astype(int)],
            'Test_accuracy (p=0.5)': [eval_results['test_accuracies']['0.5']],
            'Test_accuracy (p=0.43)': [eval_results['test_accuracies']['0.43']],
            'Test_accuracy (p=0.35)': [eval_results['test_accuracies']['0.35']],
            'Precision_weighted': [eval_results['metrics']['precision_weighted']],
            'Recall_weighted': [eval_results['metrics']['recall_weighted']],
            'F1_Score_weighted': [eval_results['metrics']['f1_score_weighted']],
            'Precision_macro': [eval_results['metrics']['precision_macro']],
            'Recall_macro': [eval_results['metrics']['recall_macro']],
            'F1_Score_macro': [eval_results['metrics']['f1_score_macro']],
            'Confusion_Matrix (p=0.5)': [eval_results['confusion_matrices']['0.5']],
            'Confusion_Matrix (p=0.43)': [eval_results['confusion_matrices']['0.43']],
            'Confusion_Matrix (p=0.35)': [eval_results['confusion_matrices']['0.35']],
            'batch_size': [self.config.batch_size],
            'learning_rate': [self.config.learning_rate],
            'class_weights': [self.config.class_weights],
            'X_train.shape': [X_train_shape],
            'epochs': [self.config.epochs],
            'Stopped_epoch': [training_summary['stopped_epoch']],
            'Training_loss_last': [training_summary['loss']],
            'Training_accuracy_last': [training_summary['accuracy']],
            'val_loss_last': [training_summary['val_loss']],
            'val_accuracy_last': [training_summary['val_accuracy']],
        }
        
        results_df = pd.DataFrame(result_data)
        # Store summary internally so we can save it separately later
        self._last_summary_df = results_df.copy()
        
        # Merge with pivoted DataFrame
        merged_df = pd.concat([pivoted_df, results_df], axis=1)
        
        return merged_df
    
    def save_all_results(self, results_df: pd.DataFrame, merged_df: pd.DataFrame,
                        feature_importance: np.ndarray, paths: Dict[str, str],
                        file_names: Dict[str, str]) -> None:
        """
        Save all results to files.
        
        Args:
            results_df: Results DataFrame
            merged_df: Merged comprehensive results
            feature_importance: Feature importance from integrated gradients
            paths: Dictionary containing paths
            file_names: Dictionary containing file names
        """
        base_dir = paths['base_dir']
        
        # Save styled results table
        output_folder = os.path.join(base_dir, f'Results_{self.config.save_name}')
        excel_file_path = os.path.join(output_folder, file_names['excel_file'])
        
        results_styled = results_df.style.apply(color_row, axis=1)
        save_results(results_styled, excel_file_path, 'excel')
        
        # Save feature importance (sum)
        output_folder_sum = os.path.join(base_dir, f'feature_im_{self.config.save_name}_sum/Output')
        csv_file_path_sum_local = os.path.join(output_folder_sum, get_file_name(excel_file_path) + '_sum.csv')
        save_results(feature_importance, csv_file_path_sum_local, 'numpy')
        
        # Save feature importance to global folder
        new_folder_sum = os.path.join(self.config.datadir2, f'Final_result_{self.config.save_name}_sum')
        csv_file_path_sum_global = os.path.join(new_folder_sum, get_file_name(excel_file_path) + '.csv')
        save_results(feature_importance, csv_file_path_sum_global, 'numpy')
        
        # Save comprehensive results
        new_folder = os.path.join(self.config.datadir2, f'Final_result_{self.config.save_name}')
        csv_file_path_final = os.path.join(new_folder, get_file_name(excel_file_path) + '.csv')
        save_results(merged_df, csv_file_path_final, 'csv')
        
        # NEW: save summary metrics (result_data) alone if available
        try:
            if hasattr(self, '_last_summary_df'):
                summary_only_path = os.path.join(new_folder, get_file_name(excel_file_path) + '_summary_only.csv')
                save_results(self._last_summary_df, summary_only_path, 'csv')
                if mlflow_tracker and mlflow_tracker.is_active():  # pragma: no cover
                    mlflow_tracker.log_artifact(summary_only_path)
        except Exception as e:
            print(f"⚠️ Failed to save summary-only CSV: {e}")
        
        # Log Integrated Gradients artifacts to MLflow if available
        if mlflow_tracker and mlflow_tracker.is_active():  # pragma: no cover
            try:
                ig_tmp_path = os.path.join(new_folder_sum, 'integrated_gradients_sum.csv')
                if feature_importance is not None:
                    np.savetxt(ig_tmp_path, feature_importance, fmt='%s', delimiter=',')
                    mlflow_tracker.log_artifact(ig_tmp_path)
                    if feature_importance.shape[0] > 1:
                        ig_array = feature_importance[1:].astype(float)
                        per_feature = np.sum(np.abs(ig_array), axis=0)
                        top_idx = np.argsort(per_feature)[::-1][:10]
                        top_dict = {f'ig_top_feat_{i}': str(int(idx)) for i, idx in enumerate(top_idx)}
                        mlflow_tracker.log_params(top_dict)
                        mlflow_tracker.log_metrics({f'ig_importance_sum_feat_{idx}': float(val) for idx, val in zip(top_idx, per_feature[top_idx])})
            except Exception as e:
                print(f"⚠️ IG MLflow logging failed: {e}")

    # New helper to compute Integrated Gradients inside evaluator
    def compute_integrated_gradients(self, model: Any, X_test: np.ndarray, target_class: int = 1):
        """Compute Integrated Gradients sum using existing IG module.
        Returns feature_importance array (header + values) or None if IG disabled/unavailable.
        """
        if IntegratedGradients is None:
            return None
        # Config object needs m_steps and parameters
        class _Cfg:
            def __init__(self, m_steps, parameters):
                self.m_steps = m_steps
                self.parameters = parameters
        m_steps = getattr(self.config, 'm_steps', 100)
        feature_names = getattr(self.config, 'parameters', [f'feat_{i}' for i in range(X_test.shape[-1])])
        cfg = _Cfg(m_steps, feature_names)
        ig = IntegratedGradients(model, cfg)
        try:
            return ig.compute_feature_importance(X_test, target_class_idx=target_class)
        except Exception as e:
            print(f"⚠️ IG computation failed: {e}")
            return None
    
    def create_plots(self, history: Any, paths: Dict[str, str], file_names: Dict[str, str]) -> None:
        """
        Create and save training plots.
        
        Args:
            history: Training history
            paths: Dictionary containing paths
            file_names: Dictionary containing file names
        """
        base_dir = paths['base_dir']
        
        # Create output folder for plots
        output_folder = os.path.join(base_dir, f'Plot_{self.config.save_name}')
        os.makedirs(output_folder, exist_ok=True)
        
        # Plot training history
        df_history = pd.DataFrame(history.history)
        df_history.plot(figsize=(8, 5))
        
        plot_file_path = os.path.join(output_folder, file_names['plot_file'])
        plt.savefig(plot_file_path)
        plt.show()
        
        # Plot learning rate
        plt.figure()
        plt.plot(np.log10(history.history['lr']))
        plt.title('Learning rate')
        plt.ylabel('Log_10 LR')
        plt.xlabel('epoch')
        plt.show()
