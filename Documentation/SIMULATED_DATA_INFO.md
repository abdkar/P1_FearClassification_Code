# Simulated Data Generation - Updated

## Overview
Successfully created independent simulated data for the Fear Classification project with completely different naming scheme from the original dataset.

## Key Changes Made

### 1. **New Subject Naming Scheme**
- **Original**: `H_106`, `H_108`, `L_103`, etc.
- **New**: `HF_201`, `HF_203`, `LF_202`, `LF_204`, etc.
- All subject numbers are in the 200+ range to avoid conflicts

### 2. **New Folder Structure**
- **Original**: `Leav_H_106/`, `Leav_L_103/`
- **New**: `Sim_HF_201/`, `Sim_LF_202/`
- Main dataset folder: `Simulated_Fear_Data_2024` (instead of `Leav_one_24_0811_1`)

### 3. **Complete Independence**
- All data is stored in `./simulated_data/` within the code directory
- No dependencies on external data paths
- Ready for GitHub sharing as a self-contained project

## Dataset Structure

```
P1_FearClassification_Code/
├── simulated_data/
│   ├── Simulated_Fear_Data_2024/
│   │   ├── Sim_HF_201/
│   │   │   ├── HF/
│   │   │   │   ├── Train/Med_Train/
│   │   │   │   └── Test/Med_Test/
│   │   │   └── LF/ (empty for HF subjects)
│   │   ├── Sim_LF_202/
│   │   │   ├── HF/ (empty for LF subjects)
│   │   │   └── LF/
│   │   │       ├── Train/Med_Train/
│   │   │       └── Test/Med_Test/
│   │   └── ... (62 subjects total)
│   ├── Control_KK_Healthy_24_1/
│   └── Control_KK_Athletes_24_1/
├── [all modular code files]
└── README.md
```

## Dataset Statistics
- **Total subjects**: 62 (32 HF + 30 LF)
- **Total CSV files**: 1,120
- **Dataset size**: 48.8 MB
- **Features per file**: 24 biomechanical measurements
- **Timesteps per trial**: 101

## Subject Categories
- **High Fear (HF)**: `HF_201` to `HF_263` (odd numbers)
- **Low Fear (LF)**: `LF_202` to `LF_260` (even numbers)

## How to Use

1. **Generate Data** (if needed):
   ```bash
   python generate_simulated_data.py
   ```

2. **Run Fear Classification**:
   ```bash
   python main.py
   ```

3. **Test Data Access**:
   ```bash
   python test_data_access.py
   ```

## File Naming Convention
- **Subject files**: `{SUBJECT_ID}_{HOP_NUMBER}.csv`
- **Examples**: `HF_201_11.csv`, `LF_202_5.csv`
- **Control files**: `Control_{ID}_{HOP}.csv`

## Features
All CSV files contain 24 biomechanical features:
- Ankle moments (X, Y, Z)
- Foot positions (X, Y, Z) 
- Hip moments and positions (X, Y, Z)
- Knee moments and positions (X, Y, Z)
- Pelvis positions (X, Y, Z)
- Thorax positions (X, Y, Z)

## Ready for GitHub
✅ Self-contained project  
✅ No external dependencies  
✅ Unique naming scheme  
✅ Modular architecture  
✅ Complete documentation  
✅ Test suite included  

The project is now ready to be shared on GitHub with simulated data that won't conflict with your original dataset.
