# src/data/dataset_metadata_planB_4class.py
import pandas as pd
import torch
import random
from torch.utils.data import Dataset

class MetadataDatasetPlanB4Class(Dataset):
    def __init__(self, file_path, is_train=True, dropout_prob=0.3):
        self.df = pd.read_csv(file_path)
        self.is_train = is_train
        self.dropout_prob = dropout_prob
        
        self.num_A = ['age_approx', 'clin_size_long_diam_mm']
        self.cat_A = [c for c in self.df.columns if c.startswith(('sex_', 'anatom_site_general_'))]
        
        self.num_B = [
            'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 
            'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 
            'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
            'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 
            'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 
            'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 
            'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt', 
            'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle'
        ]
        self.cat_B = [c for c in self.df.columns if c.startswith(('tbp_lv_location_', 'tbp_lv_location_simple_'))]
        
        self.feature_cols = self.num_A + self.num_B + self.cat_A + self.cat_B

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        yA = row['head_A_label']
        
        # Modality Dropout
        if self.is_train and yA == 0 and random.random() < self.dropout_prob:
            for col in self.num_B: row[col] = 0.0 
            for col in self.cat_B: row[col] = 1.0 if '_unknown' in col else 0.0

        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        yA_tensor = torch.tensor(yA, dtype=torch.float32)
        yB_tensor = torch.tensor(row['head_B_label'], dtype=torch.long)
        
        return features, yA_tensor, yB_tensor