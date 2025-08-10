"""
Configuration file for Fear Classification project.
Contains all hyperparameters and settings.
"""

import os

class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    def __init__(self):
        # Simulated subject folder names (different from original dataset)
        self.subfolder_names = [
            'HF_201', 'HF_203', 'HF_205', 'HF_207', 'HF_209', 'HF_211', 'HF_213', 'HF_215',
            'HF_217', 'HF_219', 'HF_221', 'HF_223', 'HF_225', 'HF_227', 'HF_229', 'HF_231',
            'HF_233', 'HF_235', 'HF_237', 'HF_239', 'HF_241', 'HF_243', 'HF_245', 'HF_247',
            'HF_249', 'HF_251', 'HF_253', 'HF_255', 'HF_257', 'HF_259', 'HF_261', 'HF_263',
            'LF_202', 'LF_204', 'LF_206', 'LF_208', 'LF_210', 'LF_212', 'LF_214', 'LF_216',
            'LF_218', 'LF_220', 'LF_222', 'LF_224', 'LF_226', 'LF_228', 'LF_230', 'LF_232',
            'LF_234', 'LF_236', 'LF_238', 'LF_240', 'LF_242', 'LF_244', 'LF_246', 'LF_248',
            'LF_250', 'LF_252', 'LF_254', 'LF_256', 'LF_258', 'LF_260'
        ]
        
        # Data settings
        self.land = 'Med'  # 'Med' or 'Lat'
        self.need_norm = True
        self.need_standardize = False
        
        # Model hyperparameters
        self.batch_size = 128
        self.epochs = 350
        self.learning_rate = 0.0001
        self.early_stopping_patience = 52
        self.reduce_lr_patience = 9
        self.m_steps = 100
        
        # Class weights
        self.class_weights = {0: 0.59, 1: 3.28}
        
        # Save name configuration
        self.save_name = f'HL_24N2_B128_E350_Lr0001_SP52_CA_DND2_W59328_P9_final_{self.land}'
        
        # Data paths (local to the code directory)
        self.datadir1 = './'
        self.datadir2 = './simulated_data/Simulated_Fear_Data_2024/'
        
        # Hop ranges
        self.hop_ranges = {
            'Med': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'Lat': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        }
        
        # Parameters for feature importance
        self.parameters = [
            f'ankle_mom_X_{self.land}', f'ankle_mom_Y_{self.land}', f'ankle_mom_Z_{self.land}', 
            f'foot_X_{self.land}', f'foot_Y_{self.land}', f'foot_Z_{self.land}', 
            f'hip_mom_X_{self.land}', f'hip_mom_Y_{self.land}', f'hip_mom_Z_{self.land}', 
            f'hip_X_{self.land}', f'hip_Y_{self.land}', f'hip_Z_{self.land}', 
            f'knee_mom_X_{self.land}', f'knee_mom_Y_{self.land}', f'knee_mom_Z_med', 
            f'knee_X_med', 'knee_Y_med', f'knee_Z_med', f'pelvis_X_med', 
            f'pelvis_Y_med', f'pelvis_Z_med', f'thorax_X_{self.land}', 
            f'thorax_Y_{self.land}', f'thorax_Z_{self.land}'
        ]
    
    def get_hop_range(self):
        """Get hop range based on land setting."""
        return self.hop_ranges.get(self.land, [])
    
    def get_data_paths(self, name):
        """Get data paths for a specific subject."""
        datadir = os.path.join('./simulated_data/Simulated_Fear_Data_2024', f'Sim_{name}')
        
        paths = {
            'HF_train': os.path.join(datadir, f'HF/Train/{self.land}_Train'),
            'LF_train': os.path.join(datadir, f'LF/Train/{self.land}_Train'),
            'control_healthy_D': f'./simulated_data/Control_KK_Healthy_24_1/C5/{self.land}_C4_D',
            'control_healthy_ND': f'./simulated_data/Control_KK_Healthy_24_1/C5/{self.land}_C4_ND',
            'control_athletes_D': f'./simulated_data/Control_KK_Athletes_24_1/C5/{self.land}_C4_D',
            'control_athletes_ND': f'./simulated_data/Control_KK_Athletes_24_1/C5/{self.land}_C4_ND',
            'HF_test': os.path.join(datadir, f'HF/Test/{self.land}_Test'),
            'LF_test': os.path.join(datadir, f'LF/Test/{self.land}_Test'),
            'base_dir': datadir
        }
        
        return paths
    
    def get_file_names(self, name, test_n):
        """Get file names for saving results."""
        suffix = '_N' if self.need_norm else ''
        
        return {
            'excel_file': f'results_{name}_{test_n}{suffix}.xlsx',
            'excel_file_cc': f'results_acc_{name}_{test_n}{suffix}.xlsx',
            'plot_file': f'results_plot_{name}_{test_n}{suffix}',
            'model_file': f'model_{name}_{test_n}{suffix}.h5'
        }
