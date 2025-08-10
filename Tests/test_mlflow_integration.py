#!/usr/bin/env python3
"""
MLflow Integration Demo for Fear Classification Project
Demonstrates the MLflow tracking capabilities with a simple test run.
"""

import os
import sys
import numpy as np
from config import Config
from mlflow_tracker import mlflow_tracker

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mlflow_integration():
    """Test the MLflow integration with a mock experiment."""
    
    print("🔬 Testing MLflow Integration for Fear Classification")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Test MLflow tracker initialization
    print("📊 Initializing MLflow tracker...")
    try:
        # Start a test run
        mlflow_tracker.start_run("MLflow_Integration_Test", "TestSubject", 1)
        print("✅ MLflow run started successfully")
        
        # Log parameters
        mlflow_tracker.log_parameters(config)
        print("✅ Parameters logged successfully")
        
        # Log mock dataset info
        mock_X_train_shape = (100, 101, 24)
        mock_X_test_shape = (30, 101, 24)
        mock_y_train = np.random.randint(0, 2, 100)
        mock_y_test = np.random.randint(0, 2, 30)
        
        mlflow_tracker.log_dataset_info(
            mock_X_train_shape, mock_X_test_shape, mock_y_train, mock_y_test
        )
        print("✅ Dataset info logged successfully")
        
        # Log mock training metrics
        mock_history = type('MockHistory', (), {
            'history': {
                'loss': [0.6, 0.4, 0.3, 0.2],
                'accuracy': [0.6, 0.75, 0.85, 0.9],
                'val_loss': [0.65, 0.45, 0.35, 0.25],
                'val_accuracy': [0.55, 0.7, 0.8, 0.85]
            }
        })()
        
        mlflow_tracker.log_training_metrics(mock_history, 120.5)
        print("✅ Training metrics logged successfully")
        
        # Log mock evaluation metrics
        mock_eval_results = {
            'test_accuracies': {'0.5': 0.85, '0.43': 0.87, '0.35': 0.82},
            'metrics': {
                'precision': 0.88,
                'recall': 0.82,
                'f1_score': 0.85,
                'auc': 0.91
            },
            'test_loss': 0.23,
            'confusion_matrices': {
                '0.5': np.array([[12, 3], [2, 13]]),
                '0.43': np.array([[11, 4], [1, 14]]),
                '0.35': np.array([[13, 2], [3, 12]])
            }
        }
        
        mlflow_tracker.log_evaluation_metrics(mock_eval_results, 2.3)
        mlflow_tracker.log_confusion_matrices(mock_eval_results)
        print("✅ Evaluation metrics logged successfully")
        
        # End the run
        mlflow_tracker.end_run(status="FINISHED")
        print("✅ MLflow run ended successfully")
        
        # Print experiment URL
        experiment_url = mlflow_tracker.get_experiment_url()
        if experiment_url:
            print(f"🌐 MLflow UI URL: {experiment_url}")
        
        print("\n🎉 MLflow Integration Test Completed Successfully!")
        print("\nTo view the experiment:")
        print("1. Open a terminal in the project directory")
        print("2. Run: mlflow ui")
        print("3. Open your browser to: http://localhost:5000")
        print("4. Navigate to the 'Fear_Classification_Experiments' experiment")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def setup_mlflow_environment():
    """Setup MLflow environment and check dependencies."""
    print("🔧 Setting up MLflow environment...")
    
    # Check if MLflow is installed
    try:
        import mlflow
        print(f"✅ MLflow version: {mlflow.__version__}")
        return True
    except ImportError:
        print("❌ MLflow not found. Please install it with: pip install mlflow>=2.0.0")
        return False
    except Exception as e:
        print(f"⚠️ MLflow has compatibility issues: {e}")
        print("📊 MLflow will be disabled but experiments can continue normally")
        return True  # Allow the test to continue without MLflow


def main():
    """Main function to run the MLflow integration test."""
    print("🚀 Fear Classification MLflow Integration Demo")
    print("=" * 60)
    
    # Setup environment
    if not setup_mlflow_environment():
        print("❌ Environment setup failed. Exiting.")
        return 1
    
    # Test MLflow integration
    if test_mlflow_integration():
        print("\n✨ All tests passed! MLflow is ready for Fear Classification experiments.")
        return 0
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
