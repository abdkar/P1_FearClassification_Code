"""
Integrated Gradients implementation for model interpretability.
"""

import tensorflow as tf
import numpy as np
from typing import Any


class IntegratedGradients:
    """Integrated Gradients for CNN model interpretability."""
    
    def __init__(self, model, config):
        """
        Initialize IntegratedGradients.
        
        Args:
            model: Trained Keras model
            config: Configuration object
        """
        self.model = model
        self.config = config
    
    @tf.function
    def one_batch(self, baseline: tf.Tensor, image: tf.Tensor, 
                  alpha_batch: tf.Tensor, target_class_idx: int) -> tf.Tensor:
        """
        Compute gradients for one batch of interpolated inputs.
        
        Args:
            baseline: Baseline input
            image: Input image
            alpha_batch: Alpha values for interpolation
            target_class_idx: Target class index
            
        Returns:
            Gradient batch
        """
        # Generate interpolated inputs between baseline and input
        interpolated_path_input_batch = self.interpolate_images(
            baseline=baseline,
            image=image,
            alphas=alpha_batch
        )

        # Compute gradients between model outputs and interpolated inputs
        gradient_batch = self.compute_gradients(
            images=interpolated_path_input_batch,
            target_class_idx=target_class_idx
        )
        return gradient_batch

    def integral_approximation(self, gradients: tf.Tensor) -> tf.Tensor:
        """
        Approximate integral using trapezoidal rule.
        
        Args:
            gradients: Gradient tensor
            
        Returns:
            Integrated gradients
        """
        # Riemann trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def compute_gradients(self, images: tf.Tensor, target_class_idx: int) -> tf.Tensor:
        """
        Compute gradients with respect to inputs.
        
        Args:
            images: Input images
            target_class_idx: Target class index
            
        Returns:
            Gradients
        """
        with tf.GradientTape() as tape:
            tape.watch(images)
            images = tf.squeeze(images, axis=1)
            logits = self.model(images)
            probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
        return tape.gradient(probs, images)

    def interpolate_images(self, baseline: tf.Tensor, image: tf.Tensor, 
                          alphas: tf.Tensor) -> tf.Tensor:
        """
        Interpolate between baseline and image.
        
        Args:
            baseline: Baseline input
            image: Input image
            alphas: Alpha values for interpolation
            
        Returns:
            Interpolated images
        """
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        input_x = tf.cast(input_x, tf.float32)
        baseline_x = tf.cast(baseline_x, tf.float32)
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    def integrated_gradients(self, baseline: tf.Tensor, image: tf.Tensor,
                           target_class_idx: int, m_steps: int = 50,
                           batch_size: int = 64) -> tf.Tensor:
        """
        Compute integrated gradients.
        
        Args:
            baseline: Baseline input
            image: Input image
            target_class_idx: Target class index
            m_steps: Number of interpolation steps
            batch_size: Batch size for computation
            
        Returns:
            Integrated gradients
        """
        # Generate alphas
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

        # Collect gradients
        gradient_batches = []

        # Iterate alphas range and batch computation for speed, memory efficiency
        for alpha in tf.range(0, len(alphas), batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_:to]

            gradient_batch = self.one_batch(baseline, image, alpha_batch, target_class_idx)
            gradient_batches.append(gradient_batch)

        # Concatenate path gradients together row-wise into single tensor
        total_gradients = tf.concat(gradient_batches, axis=0)

        # Integral approximation through averaging gradients
        avg_gradients = self.integral_approximation(gradients=total_gradients)

        # Scale integrated gradients with respect to input
        integrated_gradients = (image - baseline) * avg_gradients

        return integrated_gradients
    
    def compute_feature_importance(self, X_test: np.ndarray, target_class_idx: int) -> np.ndarray:
        """
        Compute feature importance using integrated gradients.
        
        Args:
            X_test: Test data
            target_class_idx: Target class index
            
        Returns:
            Sum of integrated gradients across all test samples
        """
        baseline = np.zeros(X_test[0].shape)
        
        all_integrated_grads = []
        for x in X_test:
            ig = self.integrated_gradients(
                baseline=baseline,
                image=x,
                target_class_idx=target_class_idx,
                m_steps=self.config.m_steps,
                batch_size=64
            )
            all_integrated_grads.append(ig)

        # Sum integrated gradients across all samples
        sum_integrated_grads = np.sum(all_integrated_grads, axis=0)
        
        # Add header with parameter names
        header = np.array([self.config.parameters])
        sum_integrated_grads = np.concatenate((header, sum_integrated_grads), axis=0)
        
        return sum_integrated_grads
