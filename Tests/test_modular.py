"""
Test script to verify the modular Fear Classification code works correctly.
This script runs a minimal test to ensure all modules can be imported and basic functionality works.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, '/home/amir/P1_FearClassification_Code')

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from config import Config
        print("✓ Config imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Config: {e}")
        return False
    
    try:
        from data_loader import DataLoader
        print("✓ DataLoader imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DataLoader: {e}")
        return False
    
    try:
        from model import CNNModel
        print("✓ CNNModel imported successfully")
    except Exception as e:
        print(f"✗ Failed to import CNNModel: {e}")
        return False
    
    try:
        from trainer import Trainer
        print("✓ Trainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Trainer: {e}")
        return False
    
    try:
        from evaluator import Evaluator
        print("✓ Evaluator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Evaluator: {e}")
        return False
    
    try:
        from integrated_gradients import IntegratedGradients
        print("✓ IntegratedGradients imported successfully")
    except Exception as e:
        print(f"✗ Failed to import IntegratedGradients: {e}")
        return False
    
    try:
        from utils import setup_gpu, get_file_name
        print("✓ Utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Utils: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration class."""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        config = Config()
        
        # Test basic properties
        assert hasattr(config, 'batch_size'), "Config missing batch_size"
        assert hasattr(config, 'epochs'), "Config missing epochs"
        assert hasattr(config, 'learning_rate'), "Config missing learning_rate"
        assert hasattr(config, 'subfolder_names'), "Config missing subfolder_names"
        
        # Test methods
        hop_range = config.get_hop_range()
        assert isinstance(hop_range, list), "get_hop_range should return a list"
        
        # Test path generation
        paths = config.get_data_paths('H_106')
        assert isinstance(paths, dict), "get_data_paths should return a dict"
        assert 'HF_train' in paths, "paths should contain HF_train"
        
        print("✓ Configuration tests passed")
        return True
    except Exception as e:
        print(f"✗ Configuration tests failed: {e}")
        return False

def test_data_loader():
    """Test data loader functionality with mock data."""
    print("\nTesting data loader...")
    
    try:
        from config import Config
        from data_loader import DataLoader
        
        config = Config()
        data_loader = DataLoader(config)
        
        # Test normalization functions with mock data
        mock_data = np.random.rand(10, 101, 24).astype('float32')
        
        # Test normalization
        norm_params = data_loader.norminit(mock_data)
        assert 'colmin' in norm_params, "norminit should return colmin"
        assert 'colmax' in norm_params, "norminit should return colmax"
        
        normalized_data = data_loader.normapply(norm_params, mock_data)
        assert normalized_data.shape == mock_data.shape, "Normalized data should have same shape"
        
        # Test standardization
        std_params = data_loader.standardize_init(mock_data)
        assert 'colmean' in std_params, "standardize_init should return colmean"
        assert 'colstd' in std_params, "standardize_init should return colstd"
        
        standardized_data = data_loader.standardize_apply(std_params, mock_data)
        assert standardized_data.shape == mock_data.shape, "Standardized data should have same shape"
        
        print("✓ Data loader tests passed")
        return True
    except Exception as e:
        print(f"✗ Data loader tests failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from config import Config
        from model import CNNModel
        
        config = Config()
        cnn_model = CNNModel(config)
        
        # Test model creation
        input_shape = (101, 24)  # Mock input shape
        model = cnn_model.create_model(input_shape)
        
        # Basic checks
        assert model is not None, "Model should not be None"
        
        # Test callbacks
        callbacks = cnn_model.get_callbacks()
        assert isinstance(callbacks, list), "get_callbacks should return a list"
        assert len(callbacks) > 0, "Should have at least one callback"
        
        print("✓ Model creation tests passed")
        return True
    except Exception as e:
        print(f"✗ Model creation tests failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from utils import get_file_name, apply_threshold_predictions
        
        # Test get_file_name
        file_path = "/path/to/file.csv"
        file_name = get_file_name(file_path)
        assert file_name == "file", "get_file_name should return filename without extension"
        
        # Test threshold predictions
        mock_predictions = np.array([0.2, 0.4, 0.6, 0.8])
        thresholds = [0.3, 0.5]
        results = apply_threshold_predictions(mock_predictions, thresholds)
        
        assert isinstance(results, dict), "apply_threshold_predictions should return dict"
        assert len(results) == len(thresholds), "Should have results for each threshold"
        
        print("✓ Utility tests passed")
        return True
    except Exception as e:
        print(f"✗ Utility tests failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running modular Fear Classification code tests...\n")
    
    tests = [
        test_imports,
        test_configuration,
        test_data_loader,
        test_model_creation,
        test_utils
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The modular code is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
