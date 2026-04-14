# src/data/dataset_metadata.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.config.paths import SPLITTED_DATA_DIR

class MetadataDataset(Dataset):
    def __init__(self, csv_name):
        self.df = pd.read_csv(SPLITTED_DATA_DIR / csv_name)
        
        # --- DEFINICIÓN PLAN A ---
        # 1. Numéricas (Ya escaladas con Z-score en el CSV)
        self.num_cols = ['age_approx', 'clin_size_long_diam_mm']
        
        # 2. Categóricas (OHE ya expandido en tu CSV)
        self.sex_cols = [c for c in self.df.columns if c.startswith('sex_')]
        self.site_cols = [c for c in self.df.columns if c.startswith('anatom_site_general_')]
        
        self.feature_cols = self.num_cols + self.sex_cols + self.site_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Extraemos las características como un tensor de floats
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        
        # Etiquetas
        yA = torch.tensor(row['head_A_label'], dtype=torch.float32)
        yB = torch.tensor(row['head_B_label'], dtype=torch.long)
        
        return features, yA, yB