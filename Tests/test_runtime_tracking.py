#!/usr/bin/env python3
"""
Test script to demonstrate runtime tracking functionality.
"""

import time
import numpy as np
from config import Config
from runtime_tracker import runtime_tracker

def test_runtime_tracking():
    """Test runtime tracking with simulated data."""
    print("🧪 Testing Runtime Tracking System")
    print("=" * 50)
    
    # Initialize config
    config = Config()
    
    # Start total timer
    runtime_tracker.start_timer('total_test')
    
    # Simulate multiple folds
    num_folds = 5
    print(f"📊 Simulating {num_folds} training folds...")
    
    for fold in range(num_folds):
        print(f"   🔄 Training fold {fold + 1}/{num_folds}")
        
        # Simulate training time (2-5 seconds per fold)
        training_time = np.random.uniform(2.0, 5.0)
        time.sleep(training_time)
        runtime_tracker.add_fold_time(training_time)
        
        # Simulate inference time (0.1-0.3 seconds per test)
        inference_time = np.random.uniform(0.1, 0.3)
        time.sleep(inference_time)
        runtime_tracker.add_inference_time(inference_time)
        
        print(f"      ⏱️  Training: {training_time:.2f}s, Inference: {inference_time:.4f}s")
    
    # End total timer
    total_time = runtime_tracker.end_timer('total_test')
    
    print(f"\n✅ Test completed in {total_time:.2f} seconds")
    
    # Print runtime summary
    print(f"\n📋 Runtime Summary:")
    runtime_tracker.print_runtime_summary(num_folds)
    
    # Save runtime summary
    runtime_tracker.save_runtime_summary('./test_runtime_summary.txt', num_folds)
    
    print(f"\n🎯 Runtime tracking test completed successfully!")

if __name__ == "__main__":
    test_runtime_tracking()
