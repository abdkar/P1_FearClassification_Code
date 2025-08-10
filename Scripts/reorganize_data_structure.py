#!/usr/bin/env python3
"""
Reorganize simulated data structure for LOPOCV.
Clean separation between data and results.
"""

import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime


def create_clean_data_structure():
    """Create a clean, organized data structure for LOPOCV."""
    print("🧹 Reorganizing simulated data structure for LOPOCV...")
    
    # Create new clean structure
    clean_data_path = "clean_simulated_data"
    results_path = "results"
    
    # Remove old clean data if exists
    if os.path.exists(clean_data_path):
        shutil.rmtree(clean_data_path)
    
    # Create directories
    os.makedirs(clean_data_path, exist_ok=True)
    os.makedirs(os.path.join(clean_data_path, "participants"), exist_ok=True)
    os.makedirs(os.path.join(clean_data_path, "control_data"), exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "experiments"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_path, "logs"), exist_ok=True)
    
    print(f"✅ Created clean directory structure:")
    print(f"   📁 {clean_data_path}/participants/ - All participant data")
    print(f"   📁 {clean_data_path}/control_data/ - Control group data")
    print(f"   📁 {results_path}/models/ - Saved models")
    print(f"   📁 {results_path}/experiments/ - Experiment results")
    print(f"   📁 {results_path}/figures/ - Plots and visualizations")
    print(f"   📁 {results_path}/logs/ - Training logs")
    
    return clean_data_path, results_path


def reorganize_participant_data(old_data_path, clean_data_path):
    """Reorganize participant data into clean LOPOCV structure."""
    print("\n📊 Reorganizing participant data...")
    
    participants_path = os.path.join(clean_data_path, "participants")
    old_sim_path = os.path.join(old_data_path, "Simulated_Fear_Data_2024")
    
    if not os.path.exists(old_sim_path):
        print(f"❌ Source path not found: {old_sim_path}")
        return
    
    # Get all participant directories
    sim_dirs = [d for d in os.listdir(old_sim_path) if d.startswith("Sim_")]
    
    participants_copied = 0
    total_trials = 0
    
    for sim_dir in sorted(sim_dirs):
        # Extract participant ID (remove "Sim_" prefix)
        participant_id = sim_dir.replace("Sim_", "")
        
        # Determine class (HF = High Fear, LF = Low Fear)
        fear_class = "high_fear" if participant_id.startswith("HF_") else "low_fear"
        
        # Create participant directory
        participant_dir = os.path.join(participants_path, participant_id)
        os.makedirs(participant_dir, exist_ok=True)
        
        # Create class subdirectory
        class_dir = os.path.join(participant_dir, fear_class)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy all trials from this participant
        old_participant_path = os.path.join(old_sim_path, sim_dir)
        
        # Look for data in the complex old structure
        trials_copied = 0
        
        # Check for HF data
        hf_paths = [
            os.path.join(old_participant_path, "HF", "Train", "Med_Train"),
            os.path.join(old_participant_path, "HF", "Test", "Med_Test")
        ]
        
        # Check for LF data  
        lf_paths = [
            os.path.join(old_participant_path, "LF", "Train", "Med_Train"),
            os.path.join(old_participant_path, "LF", "Test", "Med_Test")
        ]
        
        # Copy all CSV files from all paths
        all_paths = hf_paths + lf_paths
        for path in all_paths:
            if os.path.exists(path):
                csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
                for csv_file in csv_files:
                    src = os.path.join(path, csv_file)
                    dst = os.path.join(class_dir, csv_file)
                    shutil.copy2(src, dst)
                    trials_copied += 1
        
        if trials_copied > 0:
            participants_copied += 1
            total_trials += trials_copied
            print(f"   ✅ {participant_id}: {trials_copied} trials -> {fear_class}")
    
    print(f"\n📈 Participant data summary:")
    print(f"   Participants processed: {participants_copied}")
    print(f"   Total trials copied: {total_trials}")
    
    return participants_copied


def reorganize_control_data(old_data_path, clean_data_path):
    """Reorganize control data."""
    print("\n🎯 Reorganizing control data...")
    
    control_path = os.path.join(clean_data_path, "control_data")
    
    # Control data sources
    control_sources = [
        ("Control_KK_Healthy_24_1", "healthy"),
        ("Control_KK_Athletes_24_1", "athletes")
    ]
    
    for old_name, clean_name in control_sources:
        old_control_path = os.path.join(old_data_path, old_name)
        if os.path.exists(old_control_path):
            new_control_path = os.path.join(control_path, clean_name)
            shutil.copytree(old_control_path, new_control_path)
            print(f"   ✅ Copied {old_name} -> control_data/{clean_name}")
        else:
            print(f"   ⚠️ Control data not found: {old_control_path}")


def create_data_info_file(clean_data_path):
    """Create information file about the clean data structure."""
    info_content = f"""# Clean Simulated Data Structure

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure

```
{os.path.basename(clean_data_path)}/
├── participants/           # Individual participant data for LOPOCV
│   ├── HF_201/            # High Fear participant 201
│   │   └── high_fear/     # All trials for this participant
│   │       ├── trial_001.csv
│   │       ├── trial_002.csv
│   │       └── ...
│   ├── LF_202/            # Low Fear participant 202
│   │   └── low_fear/      # All trials for this participant
│   │       ├── trial_001.csv
│   │       └── ...
│   └── ...
└── control_data/          # Control group data
    ├── healthy/           # Healthy control participants
    └── athletes/          # Athlete control participants
```

## LOPOCV Usage

For Leave-One-Participant-Out Cross-Validation:
1. Select one participant for testing (e.g., HF_201)
2. Use ALL other participants for training
3. Each participant's data stays together (no data leakage)

## Data Format

- Each CSV file contains one trial
- Shape: (101 timesteps, 24 features)
- Features: biomechanical measurements
- Binary classification: High Fear (1) vs Low Fear (0)

## File Naming

- Participants: HF_XXX (High Fear), LF_XXX (Low Fear)
- Trials: Sequential numbering within each participant
- No train/test split within participants (handled by LOPOCV)
"""
    
    info_file = os.path.join(clean_data_path, "DATA_STRUCTURE_INFO.md")
    with open(info_file, 'w') as f:
        f.write(info_content)
    
    print(f"📋 Created data structure documentation: {info_file}")


def create_results_info_file(results_path):
    """Create information file about results structure."""
    info_content = f"""# Results Directory Structure

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure

```
{os.path.basename(results_path)}/
├── models/               # Saved model files
│   ├── model_YYYYMMDD_HHMMSS.h5
│   └── best_model.h5
├── experiments/          # Experiment results and metrics
│   ├── lopocv_results_YYYYMMDD_HHMMSS.json
│   ├── experiment_log.csv
│   └── mlflow_runs/
├── figures/             # Plots and visualizations
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── logs/               # Training and system logs
    ├── training_YYYYMMDD_HHMMSS.log
    └── error_logs.log
```

## File Naming Convention

- Timestamps: YYYYMMDD_HHMMSS format
- Experiments: Include participant ID and cross-validation fold
- Models: Include architecture and performance metrics in filename

## MLflow Integration

- Experiment tracking in experiments/mlflow_runs/
- Automatic logging of metrics, parameters, and artifacts
- Model versioning and comparison tools
"""
    
    info_file = os.path.join(results_path, "RESULTS_STRUCTURE_INFO.md")
    with open(info_file, 'w') as f:
        f.write(info_content)
    
    print(f"📋 Created results structure documentation: {info_file}")


def main():
    """Main reorganization function."""
    print("🚀 Starting data structure reorganization for LOPOCV...")
    
    # Paths
    old_data_path = "simulated_data"
    
    # Create clean structure
    clean_data_path, results_path = create_clean_data_structure()
    
    # Reorganize data
    if os.path.exists(old_data_path):
        participants_count = reorganize_participant_data(old_data_path, clean_data_path)
        reorganize_control_data(old_data_path, clean_data_path)
        
        # Create documentation
        create_data_info_file(clean_data_path)
        create_results_info_file(results_path)
        
        print(f"\n🎉 Data reorganization complete!")
        print(f"   📁 Clean data: {clean_data_path}/")
        print(f"   📁 Results: {results_path}/")
        print(f"   👥 Participants: {participants_count}")
        print(f"\n💡 Next steps:")
        print(f"   1. Update data loader to use clean structure")
        print(f"   2. Update results saving to use results folder")
        print(f"   3. Test LOPOCV with clean data")
        
    else:
        print(f"❌ Source data path not found: {old_data_path}")


if __name__ == "__main__":
    main()
