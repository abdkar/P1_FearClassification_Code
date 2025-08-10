"""
Training module for Fear Classification project.
"""

import numpy as np
import time
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Any
import os


class Trainer:
    """Model trainer for Fear Classification."""
    
    def __init__(self, config):
        """Initialize Trainer with configuration."""
        self.config = config
    
    def train_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        """
        Train the model with MLflow tracking.
        
        Args:
            model: Keras model to train
            X_train: Training data
            y_train: Training labels
            
        Returns:
            Tuple of (trained model, training history)
        """
        # Start timing
        start_time = time.time()
        
        # Convert labels to categorical
        y_train_categorical = to_categorical(y_train)
        
        # Log dataset information to MLflow
        validation_size = int(0.15 * len(X_train))
        train_size = len(X_train) - validation_size
        
        # Dataset info will be logged by main script
        
        print(f"📊 Training: {train_size} samples, Validation: {validation_size} samples")
        
        # Get callbacks
        from model import CNNModel
        cnn_model = CNNModel(self.config)
        callbacks = cnn_model.get_callbacks()
        
        # Train the model
        history = model.fit(
            X_train,
            y_train_categorical,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=2,
            shuffle=True,
            validation_split=0.15,
            callbacks=callbacks,
            class_weight=self.config.class_weights
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Training metrics will be logged by main script
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        return model, history
    
    def save_model(self, model: Any, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            model: Trained model
            save_path: Path to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def get_training_summary(self, history: Any) -> dict:
        """
        Get training summary statistics.
        
        Args:
            history: Training history
            
        Returns:
            Dictionary with training summary
        """
        stopped_epoch = len(history.history['loss']) - 1
        last_epoch = stopped_epoch
        
        summary = {
            'stopped_epoch': stopped_epoch,
            'loss': history.history['loss'][last_epoch],
            'accuracy': history.history['accuracy'][last_epoch],
            'val_loss': history.history['val_loss'][last_epoch],
            'val_accuracy': history.history['val_accuracy'][last_epoch]
        }
        
        return summary
