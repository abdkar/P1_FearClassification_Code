#!/usr/bin/env python3

"""
Test script to verify standard MLflow integration works properly
This will create a simple experiment using the official MLflow library
"""

import mlflow
import mlflow.tensorflow
import numpy as np
import time
import os

def test_standard_mlflow():
    """Test standard MLflow functionality"""
    
    # Set tracking URI
    tracking_uri = "file://./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"📁 MLflow tracking URI: {tracking_uri}")
    
    # Set experiment
    experiment_name = "Standard_MLflow_Test"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"🆕 Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"🔄 Using existing experiment: {experiment_name} (ID: {experiment_id})")
            
        mlflow.set_experiment(experiment_name)
        
        # Create and start a run
        with mlflow.start_run(run_name="Standard_Test_Run") as run:
            print(f"🏃 Started run: {run.info.run_name} (ID: {run.info.run_id})")
            
            # Log parameters
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("epochs", 10)
            mlflow.log_param("model_type", "test_model")
            print("✅ Logged parameters")
            
            # Simulate training with metrics
            for epoch in range(10):
                # Simulate training metrics
                train_loss = 1.0 - (epoch * 0.08) + np.random.normal(0, 0.05)
                train_acc = 0.5 + (epoch * 0.04) + np.random.normal(0, 0.02)
                val_loss = 1.2 - (epoch * 0.07) + np.random.normal(0, 0.06)
                val_acc = 0.45 + (epoch * 0.035) + np.random.normal(0, 0.03)
                
                # Log metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                print(f"Epoch {epoch+1}/10: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                time.sleep(0.5)  # Simulate training time
            
            # Log final metrics
            final_test_acc = 0.85 + np.random.normal(0, 0.02)
            mlflow.log_metric("test_accuracy", final_test_acc)
            
            # Log some artifacts
            with open("test_results.txt", "w") as f:
                f.write(f"Test Accuracy: {final_test_acc:.4f}\n")
                f.write("Model completed successfully\n")
            
            mlflow.log_artifact("test_results.txt")
            print("✅ Logged artifacts")
            
            # Add tags
            mlflow.set_tag("framework", "tensorflow")
            mlflow.set_tag("dataset", "test_data")
            mlflow.set_tag("experiment_type", "standard_mlflow_test")
            
            print(f"✅ Completed run: {run.info.run_name}")
            print(f"🔗 Run ID: {run.info.run_id}")
        
        print("\n🎉 Standard MLflow test completed successfully!")
        print(f"📊 View results at: http://localhost:5002")
        print(f"🧪 Experiment: {experiment_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in standard MLflow test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Standard MLflow Integration...")
    print("=" * 50)
    
    success = test_standard_mlflow()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("Check the MLflow UI to see the new experiment and run")
    else:
        print("\n❌ Test failed!")
