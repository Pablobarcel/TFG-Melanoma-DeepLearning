# src/data/dataset_metadata_planA_4class.py
import pandas as pd
import torch
from torch.utils.data import Dataset

class MetadataDatasetPlanA4Class(Dataset):
    def __init__(self, file_path):
        # Leemos directamente la ruta absoluta que le pasemos
        self.df = pd.read_csv(file_path)
        
        self.num_cols = ['age_approx', 'clin_size_long_diam_mm']
        self.sex_cols = [c for c in self.df.columns if c.startswith('sex_')]
        self.site_cols = [c for c in self.df.columns if c.startswith('anatom_site_general_')]
        
        self.feature_cols = self.num_cols + self.sex_cols + self.site_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        yA = torch.tensor(row['head_A_label'], dtype=torch.float32)
        yB = torch.tensor(row['head_B_label'], dtype=torch.long)
        return features, yA, yBs