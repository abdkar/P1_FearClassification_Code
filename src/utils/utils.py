"""
Utility functions for Fear Classification project.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Any
import sklearn.metrics


def get_file_name(path: str) -> str:
    """Get the file name from the path without extension."""
    return os.path.splitext(os.path.basename(path))[0]


def setup_gpu():
    """Setup GPU configuration for TensorFlow."""
    import tensorflow as tf
    import sys
    
    # Test for GPU and determine what GPU we have
    gpu_devices = tf.config.list_physical_devices('GPU')

    if not gpu_devices:
        print("No GPU was detected. Neural nets can be very slow without a GPU.")
        if "google.colab" in sys.modules:
            print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
        if "kaggle_secrets" in sys.modules:
            print("Go to Settings > Accelerator and select GPU.")
    else:
        print("GPU devices found:")
        for device in gpu_devices:
            print(f"  {device}")

    # Enable mixed precision if possible
    if gpu_devices:
        try:
            details = tf.config.experimental.get_device_details(gpu_devices[0])
            compute_capability = details.get('compute_capability')
            print(f"Compute capability: {compute_capability}")
            if compute_capability and compute_capability[0] > 6:
                print("Turning on mixed_float16")
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")


def color_row(row: pd.Series) -> List[str]:
    """Apply color to rows based on prediction accuracy."""
    color = 'red' if row['y_test'] != row['y_pred'] else 'black'
    return [f'color: {color}'] * len(row)


def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate various classification metrics.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing various metrics
    """
    precision_w = sklearn.metrics.precision_score(y_test, y_pred, average='weighted')
    recall_w = sklearn.metrics.recall_score(y_test, y_pred, average='weighted')
    f1_score_w = sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
    precision_m = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
    recall_m = sklearn.metrics.recall_score(y_test, y_pred, average='macro')
    f1_score_m = sklearn.metrics.f1_score(y_test, y_pred, average='macro')
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    
    return {
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_score_weighted': f1_score_w,
        'precision_macro': precision_m,
        'recall_macro': recall_m,
        'f1_score_macro': f1_score_m,
        'confusion_matrix': cm
    }


def apply_threshold_predictions(predictions: np.ndarray, thresholds: List[float]) -> dict:
    """
    Apply different thresholds to predictions and calculate accuracies.
    
    Args:
        predictions: Model prediction probabilities
        thresholds: List of thresholds to apply
        
    Returns:
        Dictionary with threshold results
    """
    results = {}
    
    for threshold in thresholds:
        class_predictions = (predictions >= threshold).astype(int)
        results[f'threshold_{threshold}'] = {
            'predictions': class_predictions,
            'threshold': threshold
        }
    
    return results


def calculate_threshold_accuracy(y_test: np.ndarray, class_predictions: np.ndarray) -> float:
    """Calculate accuracy for threshold predictions."""
    cm = sklearn.metrics.confusion_matrix(y_test, class_predictions)
    
    if cm.shape == (1, 1):
        return 1.0
    else:
        return (cm[0, 0] + cm[1, 1]) / np.sum(cm)


def create_results_dataframe(file_names_AT: List[str], file_names_BT: List[str], 
                           y_test: np.ndarray, y_pred: np.ndarray, 
                           predictions_score: np.ndarray, 
                           threshold_results: dict) -> pd.DataFrame:
    """
    Create results DataFrame with predictions and threshold results.
    
    Args:
        file_names_AT: High fear test file names
        file_names_BT: Low fear test file names
        y_test: True labels
        y_pred: Predicted labels
        predictions_score: Prediction scores
        threshold_results: Results from different thresholds
        
    Returns:
        Combined results DataFrame
    """
    # Create DataFrames for HF and LF results
    results_AT = pd.DataFrame({
        'File Name': file_names_AT,
        'y_test': y_test[:len(file_names_AT)],
        'y_pred': y_pred[:len(file_names_AT)],
        'Prediction_score 0.5': predictions_score[:len(file_names_AT)]
    })
    
    results_BT = pd.DataFrame({
        'File Name': file_names_BT,
        'y_test': y_test[len(file_names_AT):],
        'y_pred': y_pred[len(file_names_AT):],
        'Prediction_score 0.5': predictions_score[len(file_names_AT):]
    })
    
    # Add threshold results
    for threshold_name, threshold_data in threshold_results.items():
        threshold_val = threshold_data['threshold']
        predictions = threshold_data['predictions']
        
        col_name = f'class_predictions {threshold_val}'
        results_AT[col_name] = predictions[:len(file_names_AT)]
        results_BT[col_name] = predictions[len(file_names_AT):]
    
    # Concatenate results
    results = pd.concat([results_AT, results_BT], ignore_index=True)
    results = results.sort_values(by='File Name')
    
    return results


def calc_lf_percent(row: pd.Series) -> float:
    """Calculate percentage of Low Fear predictions."""
    count = 0
    values = row.iloc[:-1].drop('Ground_truth', errors='ignore')
    for val in values:
        if val == 0:
            count += 1
    non_values = (row == 'non').sum()
    if len(values) - non_values == 0:
        return 0.0
    percent = count / (len(values) - non_values) * 100
    return round(percent, 2)


def create_pivoted_dataframe(results: pd.DataFrame, hop_range: List[int]) -> pd.DataFrame:
    """
    Create pivoted DataFrame for hop analysis.
    
    Args:
        results: Results DataFrame
        hop_range: Range of hop values
        
    Returns:
        Pivoted DataFrame
    """
    df = results.copy()
    
    # Extract filename and hop number
    df['Filename'] = df['File Name'].str.extract(r'(.*?)(_\d+)?$', expand=False)[0]
    df['Hop'] = df['File Name'].str.extract(r'(.*?)(_\d+)?$', expand=False)[1]
    df['Hop'] = df['Hop'].str.extract(r'_(\d+)').astype(int)
    
    # Pivot the dataframe
    pivoted_df = df.pivot(index='Filename', columns='Hop', values='y_pred')
    
    # Add missing hop columns
    for hop in hop_range:
        if hop not in pivoted_df.columns:
            pivoted_df[hop] = 'non'
    
    # Add ground truth
    pivoted_df['Ground_truth'] = df.groupby('Filename')['y_test'].first()
    
    # Fill missing values
    pivoted_df = pivoted_df.fillna('non')
    
    # Rename columns
    pivoted_df.columns = [f'hop {c}' if isinstance(c, int) else c for c in pivoted_df.columns]
    
    # Convert values to integers
    pivoted_df = pivoted_df.applymap(lambda x: 1 if x == 1.0 else 0 if x == 0.0 else x)
    
    # Add LF percentage
    pivoted_df['LF_percent'] = pivoted_df.apply(calc_lf_percent, axis=1)
    
    # Reset index
    pivoted_df.reset_index(inplace=True)
    
    return pivoted_df


def save_results(data: Any, file_path: str, data_type: str = 'csv') -> None:
    """
    Save data to file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        data_type: Type of data ('csv', 'excel', 'numpy')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if data_type == 'csv':
        data.to_csv(file_path, index=True)
    elif data_type == 'excel':
        data.to_excel(file_path, engine='openpyxl', index=False)
    elif data_type == 'numpy':
        np.savetxt(file_path, data, delimiter=',', fmt='%s')
    
    print(f"Data saved successfully to {file_path}")


def print_results_summary(y_test: np.ndarray, y_pred: np.ndarray, 
                         test_accuracies: dict, confusion_matrices: dict) -> None:
    """Print summary of results."""
    errors = np.sum(np.abs(y_pred - y_test))
    accuracy = 1 - errors / len(y_pred)
    
    print(f"Confusion matrix (p=0.5):\n{confusion_matrices['0.5']}\n")
    print(f"No errors={errors}, this is an accuracy = {accuracy}")
    print(f"Test accuracy (p=0.5): {test_accuracies['0.5']:.3f}")
    
    for threshold, acc in test_accuracies.items():
        if threshold != '0.5':
            print(f"Test accuracy (p={threshold}): {acc:.3f}")
            print(f"Confusion matrix (p={threshold}):\n{confusion_matrices[threshold]}\n")
