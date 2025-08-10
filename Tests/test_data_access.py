#!/usr/bin/env python3
"""
Test script to verify simulated data access for Fear Classification project.
"""

import os
import sys
from config import Config

def test_data_access():
    """Test if the modular code can access the simulated data."""
    print("🔍 Testing Simulated Data Access")
    print("=" * 50)
    
    # Initialize config
    config = Config()
    print(f"📁 Base data directory: {config.datadir2}")
    print(f"📂 Data directory exists: {os.path.exists(config.datadir2)}")
    
    # Test a few subjects
    test_subjects = ['HF_201', 'LF_202']
    
    for subject in test_subjects:
        print(f"\n👤 Testing subject: {subject}")
        paths = config.get_data_paths(subject)
        
        # Check main paths
        for path_type, path in paths.items():
            if path_type in ['HF_train', 'LF_train', 'HF_test', 'LF_test']:
                exists = os.path.exists(path)
                file_count = len(os.listdir(path)) if exists else 0
                print(f"   📂 {path_type}: {'✅' if exists else '❌'} ({file_count} files)")
    
    # Test control groups
    print(f"\n🎯 Testing control groups:")
    control_paths = [
        f"./simulated_data/Control_KK_Healthy_24_1/C5/{config.land}_C4_D",
        f"./simulated_data/Control_KK_Athletes_24_1/C5/{config.land}_C4_D"
    ]
    
    for path in control_paths:
        exists = os.path.exists(path)
        file_count = len(os.listdir(path)) if exists else 0
        print(f"   📂 {os.path.basename(path)}: {'✅' if exists else '❌'} ({file_count} files)")
    
    print(f"\n🚀 Ready to run: python main.py")
    return True

if __name__ == "__main__":
    test_data_access()
