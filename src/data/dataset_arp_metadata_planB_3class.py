# src/data/dataset_arp_metadata_planB_3class.py

import os
import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from PIL import Image
from src.config.paths import SPLITTED_DATA_DIR, PROJECT_ROOT


class ARPMetadataDatasetPlanB3Class(Dataset):
    def __init__(self, csv_file, transform=None, is_train=True, dropout_prob=0.3):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train
        self.dropout_prob = dropout_prob
        
        # 🚨 Ruta a las imágenes transformadas
        self.img_dir = PROJECT_ROOT / "src" / "data" / "processed" / "imagenes_ARP_224" 
        
        self.num_A = ['age_approx', 'clin_size_long_diam_mm']
        self.cat_A = [c for c in self.df.columns if c.startswith(('sex_', 'anatom_site_general_'))]
        
        self.num_B = [c for c in self.df.columns if c.startswith('tbp_lv_') and not c.startswith(('tbp_lv_location_', 'tbp_lv_location_simple_'))]
        self.cat_B = [c for c in self.df.columns if c.startswith(('tbp_lv_location_', 'tbp_lv_location_simple_'))]
        
        self.feature_cols = self.num_A + self.num_B + self.cat_A + self.cat_B

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        yA = row['head_A_label']
        
        # --- Modality Dropout (Sólo Benignos en Train) ---
        if self.is_train and yA == 0 and random.random() < self.dropout_prob:
            for col in self.num_B: row[col] = 0.0 
            for col in self.cat_B: row[col] = 1.0 if '_unknown' in col else 0.0

        # --- 1. Imagen ARP (1 Canal) ---
        img_name = os.path.join(self.img_dir, f"{row['isic_id']}.jpg")
        try:
            image = Image.open(img_name).convert('L') 
        except FileNotFoundError:
            image = Image.new('L', (224, 224), 0)
            
        if self.transform:
            image = self.transform(image)
            
        # --- 2. Metadatos ---
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        
        # --- 3. Etiquetas (3 Clases) ---
        yA_tensor = torch.tensor(yA, dtype=torch.float32)
        yB_tensor = torch.tensor(row['head_B_label'], dtype=torch.long)
        
        return image, features, yA_tensor, yB_tensor