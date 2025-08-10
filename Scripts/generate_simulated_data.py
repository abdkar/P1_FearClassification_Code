"""
Simulated Data Generator for Fear Classification Project
Creates realistic biomechanical data that matches the expected structure
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple
import random


class SimulatedDataGenerator:
    """Generate simulated biomechanical data for fear classification."""
    
    def __init__(self):
        """Initialize the data generator with realistic parameters."""
        self.timesteps = 101  # Number of time points
        self.features = 24    # Number of biomechanical features
        
        # Feature names based on your original code
        self.feature_names_med = [
            'ankle_mom_X_Med', 'ankle_mom_Y_Med', 'ankle_mom_Z_Med',
            'foot_X_Med', 'foot_Y_Med', 'foot_Z_Med',
            'hip_mom_X_Med', 'hip_mom_Y_Med', 'hip_mom_Z_Med',
            'hip_X_Med', 'hip_Y_Med', 'hip_Z_Med',
            'knee_mom_X_Med', 'knee_mom_Y_Med', 'knee_mom_Z_med',
            'knee_X_med', 'knee_Y_med', 'knee_Z_med',
            'pelvis_X_med', 'pelvis_Y_med', 'pelvis_Z_med',
            'thorax_X_Med', 'thorax_Y_Med', 'thorax_Z_Med'
        ]
        
        self.feature_names_lat = [
            'ankle_mom_X_Lat', 'ankle_mom_Y_Lat', 'ankle_mom_Z_Lat',
            'foot_X_Lat', 'foot_Y_Lat', 'foot_Z_Lat',
            'hip_mom_X_Lat', 'hip_mom_Y_Lat', 'hip_mom_Z_Lat',
            'hip_X_Lat', 'hip_Y_Lat', 'hip_Z_Lat',
            'knee_mom_X_Lat', 'knee_mom_Y_Lat', 'knee_mom_Z_med',
            'knee_X_med', 'knee_Y_med', 'knee_Z_med',
            'pelvis_X_med', 'pelvis_Y_med', 'pelvis_Z_med',
            'thorax_X_Lat', 'thorax_Y_Lat', 'thorax_Z_Lat'
        ]
        
        # Simulated subject identifiers (different from original dataset)
        self.subjects = [
            'HF_201', 'HF_203', 'HF_205', 'HF_207', 'HF_209', 'HF_211', 'HF_213', 'HF_215',
            'HF_217', 'HF_219', 'HF_221', 'HF_223', 'HF_225', 'HF_227', 'HF_229', 'HF_231',
            'HF_233', 'HF_235', 'HF_237', 'HF_239', 'HF_241', 'HF_243', 'HF_245', 'HF_247',
            'HF_249', 'HF_251', 'HF_253', 'HF_255', 'HF_257', 'HF_259', 'HF_261', 'HF_263',
            'LF_202', 'LF_204', 'LF_206', 'LF_208', 'LF_210', 'LF_212', 'LF_214', 'LF_216',
            'LF_218', 'LF_220', 'LF_222', 'LF_224', 'LF_226', 'LF_228', 'LF_230', 'LF_232',
            'LF_234', 'LF_236', 'LF_238', 'LF_240', 'LF_242', 'LF_244', 'LF_246', 'LF_248',
            'LF_250', 'LF_252', 'LF_254', 'LF_256', 'LF_258', 'LF_260'
        ]
        
        # Hop ranges for different conditions
        self.hop_ranges = {
            'Med': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'Lat': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        }
    
    def generate_biomechanical_signal(self, signal_type: str, fear_level: str = 'normal') -> np.ndarray:
        """
        Generate realistic biomechanical signal with temporal patterns.
        
        Args:
            signal_type: Type of biomechanical measurement
            fear_level: 'high', 'low', or 'normal' fear
            
        Returns:
            Time series data for one feature
        """
        # Base parameters for different signal types
        if 'mom' in signal_type:  # Moment data
            base_amplitude = np.random.uniform(0.5, 2.0)
            base_frequency = np.random.uniform(0.1, 0.3)
            noise_level = 0.1
        elif any(axis in signal_type for axis in ['_X_', '_Y_', '_Z_']):  # Position/angle data
            base_amplitude = np.random.uniform(10, 45)
            base_frequency = np.random.uniform(0.05, 0.2)
            noise_level = 2.0
        else:
            base_amplitude = np.random.uniform(1.0, 10.0)
            base_frequency = np.random.uniform(0.1, 0.4)
            noise_level = 0.5
        
        # Modify parameters based on fear level
        if fear_level == 'high':
            base_amplitude *= np.random.uniform(1.2, 1.8)  # Higher amplitude
            noise_level *= np.random.uniform(1.5, 2.0)     # More variability
        elif fear_level == 'low':
            base_amplitude *= np.random.uniform(0.7, 1.0)  # Lower amplitude
            noise_level *= np.random.uniform(0.5, 0.8)     # Less variability
        
        # Generate time vector
        t = np.linspace(0, 1, self.timesteps)
        
        # Create complex biomechanical pattern
        # Main movement pattern
        main_signal = base_amplitude * np.sin(2 * np.pi * base_frequency * t)
        
        # Add secondary harmonic
        secondary_signal = (base_amplitude * 0.3) * np.sin(4 * np.pi * base_frequency * t + np.pi/4)
        
        # Add movement initiation and termination phases
        envelope = np.exp(-((t - 0.5) ** 2) / (2 * 0.3 ** 2))  # Gaussian envelope
        
        # Combine signals
        signal = (main_signal + secondary_signal) * envelope
        
        # Add realistic noise and drift
        noise = np.random.normal(0, noise_level, self.timesteps)
        drift = np.random.uniform(-0.1, 0.1) * t
        
        final_signal = signal + noise + drift
        
        return final_signal
    
    def generate_trial_data(self, subject_id: str, hop_number: int, 
                          land_type: str, fear_category: str) -> pd.DataFrame:
        """
        Generate data for a single trial.
        
        Args:
            subject_id: Subject identifier
            hop_number: Hop number
            land_type: 'Med' or 'Lat'
            fear_category: 'HF' (high fear) or 'LF' (low fear)
            
        Returns:
            DataFrame with biomechanical data
        """
        # Determine fear level for signal generation
        if fear_category == 'HF':
            fear_level = 'high'
        elif fear_category == 'LF':
            fear_level = 'low'
        else:
            fear_level = 'normal'
        
        # Get appropriate feature names
        feature_names = self.feature_names_med if land_type == 'Med' else self.feature_names_lat
        
        # Generate data for each feature
        data = {}
        for i, feature_name in enumerate(feature_names):
            data[feature_name] = self.generate_biomechanical_signal(feature_name, fear_level)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def create_subject_data(self, subject_id: str, base_path: str, 
                          land_type: str = 'Med', num_tests: int = 5) -> None:
        """
        Create all data files for a single subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'HF_201', 'LF_202')
            base_path: Base directory path
            land_type: 'Med' or 'Lat'
            num_tests: Number of test sets to create
        """
        # Create folder name using full subject ID
        subject_path = os.path.join(base_path, f'Sim_{subject_id}')
        
        # Determine fear category from subject ID prefix
        fear_category = 'HF' if subject_id.startswith('HF_') else 'LF'
        
        # Get hop range for this land type
        hop_range = self.hop_ranges[land_type]
        
        # Create train and test data for each test number
        for test_n in range(1, num_tests + 1):
            # Randomly split hops into train and test
            # Extract numeric part from subject_id for seeding
            numeric_part = ''.join(filter(str.isdigit, subject_id))
            seed_val = 42 + int(numeric_part) + test_n if numeric_part else 42 + test_n
            np.random.seed(seed_val)  # Reproducible split
            shuffled_hops = hop_range.copy()
            np.random.shuffle(shuffled_hops)
            
            # 70% for training, 30% for testing
            split_idx = int(len(shuffled_hops) * 0.7)
            train_hops = shuffled_hops[:split_idx]
            test_hops = shuffled_hops[split_idx:]
            
            # Create training data
            train_path = os.path.join(subject_path, fear_category, 'Train', f'{land_type}_Train')
            os.makedirs(train_path, exist_ok=True)
            
            for hop in train_hops:
                filename = f'{subject_id}_{hop}.csv'
                filepath = os.path.join(train_path, filename)
                
                df = self.generate_trial_data(subject_id, hop, land_type, fear_category)
                df.to_csv(filepath, index=False)
            
            # Create testing data
            test_path = os.path.join(subject_path, fear_category, 'Test', f'{land_type}_Test')
            os.makedirs(test_path, exist_ok=True)
            
            for hop in test_hops:
                filename = f'{subject_id}_{hop}.csv'
                filepath = os.path.join(test_path, filename)
                
                df = self.generate_trial_data(subject_id, hop, land_type, fear_category)
                df.to_csv(filepath, index=False)
    
    def create_control_data(self, base_path: str, land_type: str = 'Med') -> None:
        """
        Create control group data (healthy participants and athletes).
        
        Args:
            base_path: Base directory path
            land_type: 'Med' or 'Lat'
        """
        # Create healthy control data
        healthy_paths = [
            f'Control_KK_Healthy_24_1/C5/{land_type}_C4_D',
            f'Control_KK_Healthy_24_1/C5/{land_type}_C4_ND'
        ]
        
        # Create athlete control data
        athlete_paths = [
            f'Control_KK_Athletes_24_1/C5/{land_type}_C4_D',
            f'Control_KK_Athletes_24_1/C5/{land_type}_C4_ND'
        ]
        
        all_control_paths = healthy_paths + athlete_paths
        hop_range = self.hop_ranges[land_type]
        
        for path in all_control_paths:
            full_path = os.path.join(base_path, path)
            os.makedirs(full_path, exist_ok=True)
            
            # Create 20-30 control files per condition
            num_files = np.random.randint(20, 31)
            
            for i in range(num_files):
                # Random hop selection
                hop = np.random.choice(hop_range)
                filename = f'Control_{i+1:03d}_{hop}.csv'
                filepath = os.path.join(full_path, filename)
                
                # Generate normal/control level data
                df = self.generate_trial_data(f'Control_{i+1}', hop, land_type, 'normal')
                df.to_csv(filepath, index=False)
    
    def generate_complete_dataset(self, base_path: str = './simulated_data', 
                                land_types: List[str] = ['Med', 'Lat']) -> None:
        """
        Generate the complete simulated dataset.
        
        Args:
            base_path: Base directory for data (default: current directory)
            land_types: List of land types to generate
        """
        main_path = os.path.join(base_path, 'Simulated_Fear_Data_2024')
        
        print("🚀 Starting simulated data generation...")
        print(f"📁 Base path: {main_path}")
        
        for land_type in land_types:
            print(f"\n📊 Generating {land_type} data...")
            
            # Create subject data
            for i, subject_id in enumerate(self.subjects):
                print(f"   👤 Creating data for subject {subject_id} ({i+1}/{len(self.subjects)})")
                self.create_subject_data(subject_id, main_path, land_type)
            
            # Create control data
            print(f"   🎯 Creating control group data for {land_type}")
            self.create_control_data(base_path, land_type)
        
        print("\n✅ Simulated data generation completed!")
        self.print_dataset_summary(main_path)
    
    def print_dataset_summary(self, base_path: str) -> None:
        """Print a summary of the generated dataset."""
        print("\n📋 DATASET SUMMARY:")
        print("=" * 50)
        
        total_files = 0
        total_size = 0
        
        # Count files and calculate size
        for root, dirs, files in os.walk(base_path):
            csv_files = [f for f in files if f.endswith('.csv')]
            total_files += len(csv_files)
            
            for file in csv_files:
                filepath = os.path.join(root, file)
                total_size += os.path.getsize(filepath)
        
        print(f"📊 Total CSV files created: {total_files:,}")
        print(f"💾 Total dataset size: {total_size / (1024*1024):.1f} MB")
        print(f"👥 Number of subjects: {len(self.subjects)}")
        print(f"🔢 Features per file: {self.features}")
        print(f"⏱️  Timesteps per trial: {self.timesteps}")
        
        # Show folder structure sample
        print(f"\n📁 Example folder structure:")
        example_subject = self.subjects[0]
        print(f"   📂 Sim_{example_subject}/")
        print(f"      📂 HF/")
        print(f"         📂 Train/Med_Train/")
        print(f"         📂 Test/Med_Test/")
        print(f"      📂 LF/")
        print(f"         📂 Train/Med_Train/")
        print(f"         📂 Test/Med_Test/")


def main():
    """Main function to generate simulated data."""
    print("Fear Classification - Simulated Data Generator")
    print("=" * 60)
    
    # Create data generator
    generator = SimulatedDataGenerator()
    
    # Generate complete dataset in the current directory
    # Only generate Med data for faster testing (you can add 'Lat' later)
    generator.generate_complete_dataset(base_path='./simulated_data', land_types=['Med'])
    
    print(f"\n🎯 Data generation complete!")
    print(f"📝 The simulated data mimics real biomechanical patterns:")
    print(f"   • Time-series data with {generator.timesteps} timesteps")
    print(f"   • {generator.features} biomechanical features")
    print(f"   • High fear vs Low fear patterns")
    print(f"   • Realistic noise and variability")
    print(f"   • Control group data included")
    
    print(f"\n🚀 Ready to run your Fear Classification code!")
    print(f"   Run: python main.py")


if __name__ == "__main__":
    main()
