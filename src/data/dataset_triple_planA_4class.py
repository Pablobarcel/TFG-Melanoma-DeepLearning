# src/data/dataset_triple_planA_4class.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from src.config.paths import SPLITTED_DATA_DIR, IMAGES_RGB_DIR , PROJECT_ROOT

class TripleDatasetPlanA4Class(Dataset):
    def __init__(self, csv_file, transform_rgb=None, transform_arp=None):
        self.df = pd.read_csv(csv_file)
        self.transform_rgb = transform_rgb
        self.transform_arp = transform_arp
        
        self.img_dir_rgb = IMAGES_RGB_DIR
        self.img_dir_arp = PROJECT_ROOT / "src" / "data" / "processed" / "imagenes_ARP_224"
        
        # --- Features Clínicas Básicas (Plan A) ---
        self.num_cols = ['age_approx', 'clin_size_long_diam_mm']
        self.sex_cols = [c for c in self.df.columns if c.startswith('sex_')]
        self.site_cols = [c for c in self.df.columns if c.startswith('anatom_site_general_')]
        self.feature_cols = self.num_cols + self.sex_cols + self.site_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. Imagen RGB ---
        img_name_rgb = os.path.join(self.img_dir_rgb, f"{row['isic_id']}.jpg")
        try:
            image_rgb = Image.open(img_name_rgb).convert('RGB')
        except FileNotFoundError:
            image_rgb = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform_rgb:
            image_rgb = self.transform_rgb(image_rgb)

        # --- 2. Imagen ARP (Escala de Grises) ---
        img_name_arp = os.path.join(self.img_dir_arp, f"{row['isic_id']}.jpg")
        try:
            image_arp = Image.open(img_name_arp).convert('L')
        except FileNotFoundError:
            image_arp = Image.new('L', (224, 224), 0)
            
        if self.transform_arp:
            image_arp = self.transform_arp(image_arp)
            
        # --- 3. Metadatos ---
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        
        # --- 4. Etiquetas (4 Clases) ---
        yA_tensor = torch.tensor(row['head_A_label'], dtype=torch.float32)
        yB_tensor = torch.tensor(row['head_B_label'], dtype=torch.long)
        
        return image_rgb, image_arp, features, yA_tensor, yB_tensor