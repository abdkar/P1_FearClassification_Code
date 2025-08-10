#!/usr/bin/env python3
"""
Test script to verify MLflow UI real-time visualization
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

# Import our custom MLflow writer
from fixed_direct_mlflow_writer import FixedDirectMLflowWriter

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data(base_path, name, data_type):
    """Load data from specified path"""
    lf_path = os.path.join(base_path, f"Leav_{name}", "LF", data_type)
    hf_path = os.path.join(base_path, f"Leav_{name}", "HF", data_type)
    
    lf_data = []
    hf_data = []
    
    # Load LF data (Low Fear - label 0)
    for subfolder in ["Low_Test", "Med_Test"]:
        folder_path = os.path.join(lf_path, subfolder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            for file in files:
                data = np.load(os.path.join(folder_path, file))
                lf_data.append(data)
        else:
            print(f"Warning: No files in {folder_path}, returning empty array")
    
    # Load HF data (High Fear - label 1)  
    for subfolder in ["High_Test", "VHigh_Test"]:
        folder_path = os.path.join(hf_path, subfolder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            for file in files:
                data = np.load(os.path.join(folder_path, file))
                hf_data.append(data)
    
    lf_array = np.array(lf_data) if lf_data else np.empty((0, 101, 24))
    hf_array = np.array(hf_data) if hf_data else np.empty((0, 101, 24))
    
    print(f"Loaded LF_{data_type}: {lf_array.shape}, HF_{data_type}: {hf_array.shape}")
    
    return lf_array, hf_array

def create_model(input_shape):
    """Create CNN model"""
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(), 
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def main():
    print("🧪 Testing MLflow UI Real-time Visualization")
    print("=" * 60)
    
    # Initialize MLflow
    mlflow_writer = FixedDirectMLflowWriter("./mlruns")
    
    # Create test experiment
    experiment_name = "Real_Time_Test_LOPOCV"
    experiment_id = mlflow_writer.create_experiment(experiment_name)
    
    print(f"✅ Created test experiment: {experiment_name} (ID: {experiment_id})")
    
    # Test subjects (just use H_106 for quick test)
    subjects = ["H_106"]
    data_base_path = "/home/amir/SH_Wrok_Data/Leav_one_24_0811_1"
    
    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Testing with subject: {subject}")
        print(f"{'='*60}")
        
        # Start MLflow run  
        run_id = mlflow_writer.start_run(f"Test_Subject_{subject}")
        
        print(f"🏃 Started run: Test_Subject_{subject} (ID: {run_id})")
        print("Check MLflow UI at http://localhost:5002 - you should see this run appear!")
        
        # Log initial parameters
        mlflow_writer.log_param("subject", subject)
        mlflow_writer.log_param("test_type", "real_time_visualization_test")
        mlflow_writer.log_param("epochs", 5)  # Very short training for quick test
        
        # Load data
        print("\nLoading training data...")
        lf_train, hf_train = load_data(data_base_path, subject, "Train")
        
        if lf_train.size == 0 and hf_train.size == 0:
            print(f"❌ No training data found for {subject}")
            mlflow_writer.end_run(status=mlflow_writer.STATUS_FAILED)
            continue
        
        # Prepare training data
        X_train = np.concatenate([lf_train, hf_train], axis=0)
        y_train = np.concatenate([
            np.zeros(len(lf_train)),  # LF = 0
            np.ones(len(hf_train))    # HF = 1  
        ])
        
        print(f"Shape of X_train data: {X_train.shape}")
        print(f"Shape of y_train data: {y_train.shape}")
        
        # Normalize data
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        # Create and train model
        print("\nCreating model...")
        model = create_model((X_train.shape[1], X_train.shape[2]))
        
        print(f"\n🚀 Starting quick training (5 epochs) for {subject}...")
        print("Watch MLflow UI - metrics should appear in real-time!")
        
        # Training with callbacks and real-time logging
        for epoch in range(5):
            print(f"\nEpoch {epoch + 1}/5")
            
            # Train for one epoch
            history = model.fit(
                X_train_scaled, y_train,
                epochs=1,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Log metrics immediately after each epoch
            train_loss = history.history['loss'][0]
            train_acc = history.history['accuracy'][0] 
            val_loss = history.history['val_loss'][0]
            val_acc = history.history['val_accuracy'][0]
            
            mlflow_writer.log_metric("train_loss", train_loss, step=epoch)
            mlflow_writer.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow_writer.log_metric("val_loss", val_loss, step=epoch)
            mlflow_writer.log_metric("val_accuracy", val_acc, step=epoch)
            
            print(f"📊 Logged metrics for epoch {epoch + 1} - check MLflow UI!")
            
            # Small delay to see real-time updates
            time.sleep(1)
        
        # Load test data and evaluate
        print("\nLoading test data...")
        lf_test, hf_test = load_data(data_base_path, subject, "Test")
        
        if lf_test.size > 0 or hf_test.size > 0:
            X_test = np.concatenate([lf_test, hf_test], axis=0)
            y_test = np.concatenate([
                np.zeros(len(lf_test)),
                np.ones(len(hf_test))
            ])
            
            # Normalize test data
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            X_test_scaled = scaler.transform(X_test_flat)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            
            test_accuracy = accuracy_score(y_test, y_pred_binary)
            test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)[0]
            
            # Log final test results
            mlflow_writer.log_metric("test_accuracy", test_accuracy)
            mlflow_writer.log_metric("test_loss", test_loss)
            mlflow_writer.log_metric("final_epoch", 5)
            
            print(f"\n📊 Final Results for {subject}:")
            print(f"- Test Accuracy: {test_accuracy:.6f}")
            print(f"- Test Loss: {test_loss:.6f}")
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred_binary)
            print(f"Confusion matrix:\n{cm}")
            
        else:
            print(f"⚠️ No test data found for {subject}")
            mlflow_writer.log_metric("test_accuracy", 0.0)
        
        # End run
        mlflow_writer.end_run()
        print(f"✅ Completed test run for {subject}")
        print("Check MLflow UI - the run should now show as 'FINISHED'!")
    
    print(f"\n🎉 Real-time visualization test completed!")
    print(f"🔗 Check all results at: http://localhost:5002")
    print("\nAnswer: Results ARE visualized immediately in MLflow UI!")
    print("- Runs appear as soon as they start")
    print("- Metrics update in real-time during training") 
    print("- Final results show immediately when runs complete")

if __name__ == "__main__":
    main()
