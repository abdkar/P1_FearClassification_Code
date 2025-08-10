"""
CNN Model definition for Fear Classification project.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPool1D, Flatten, Dropout, 
    BatchNormalization, Input, Lambda, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


class CNNModel:
    """CNN Model for Fear Classification."""
    
    def __init__(self, config):
        """Initialize CNNModel with configuration."""
        self.config = config
    
    def create_model(self, input_shape):
        """
        Create the CNN model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        input_layer = Input(shape=input_shape)

        # First Conv1D block
        conv_layer1 = Conv1D(
            filters=400, 
            kernel_size=3, 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(0.01)
        )(input_layer)
        bn_layer1 = BatchNormalization()(conv_layer1)
        dropout_layer1 = Dropout(0.2)(bn_layer1)

        # Second Conv1D block
        conv_layer2 = Conv1D(
            filters=100, 
            kernel_size=7, 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(0.01)
        )(dropout_layer1)
        bn_layer2 = BatchNormalization()(conv_layer2)
        dropout_layer2 = Dropout(0.4)(bn_layer2)

        # Max pooling
        max_pool = MaxPool1D(pool_size=2)(dropout_layer2)

        # Third Conv1D block
        conv_layer3 = Conv1D(
            filters=300, 
            kernel_size=3, 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(0.01)
        )(max_pool)
        bn_layer3 = BatchNormalization()(conv_layer3)
        dropout_layer3 = Dropout(0.2)(bn_layer3)

        # Reshape for GlobalAveragePooling2D
        reshape_layer = Lambda(lambda x: K.expand_dims(x, axis=2))(dropout_layer3)

        # Global average pooling
        global_avg_pool = GlobalAveragePooling2D()(reshape_layer)

        # Dense layers
        dense_layer1 = Dense(
            units=450, 
            activation='relu', 
            kernel_regularizer=l2(0.01)
        )(global_avg_pool)
        bn_layer4 = BatchNormalization()(dense_layer1)
        dropout_layer4 = Dropout(0.3)(bn_layer4)

        # Output layer
        output_layer = Dense(units=2, activation='softmax')(dropout_layer4)

        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self):
        """
        Get training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            mode='min',
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=self.config.reduce_lr_patience,
            min_lr=0.0000001
        )
        
        return [early_stopping, reduce_lr]
