# src/data/dataset_rgb_metadata_planB.py

import pandas as pd
import torch
import random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from src.config.paths import SPLITTED_DATA_DIR, IMAGES_RGB_DIR

class RGBMetadataDatasetPlanB(Dataset):
    """
    Dataset Multimodal: Imagen RGB + Metadatos Plan B (Con Modality Dropout)
    """
    def __init__(self, csv_name="train.csv", transforms=None, is_train=True, dropout_prob=0.3):
        self.csv_path = SPLITTED_DATA_DIR / csv_name
        self.df = pd.read_csv(self.csv_path)
        self.transforms = transforms
        self.is_train = is_train
        self.dropout_prob = dropout_prob

        # --- FUSIÓN DE CLASES ON-THE-FLY (NMSC) ---
        if "head_B_label" in self.df.columns:
            self.df.loc[self.df["head_B_label"] == 3, "head_B_label"] = 2

        # --- DEFINICIÓN PLAN B ---
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
        
        # --- 1. IMAGEN RGB ---
        img_name = Path(row["image_path"]).name
        img_path = IMAGES_RGB_DIR / img_name
        
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        # --- 2. METADATOS (Con Modality Dropout) ---
        yA = int(row['head_A_label'])
        
        if self.is_train and yA == 0 and random.random() < self.dropout_prob:
            # Apagamos variables avanzadas TBP
            for col in self.num_B:
                row[col] = 0.0 # Valor que indica ausencia
            for col in self.cat_B:
                row[col] = 1.0 if '_unknown' in col else 0.0

        features = torch.tensor(row[self.feature_cols].values.astype('float32'))

        # --- 3. ETIQUETAS ---
        yA_tensor = torch.tensor(yA, dtype=torch.float32)
        yB_tensor = torch.tensor(row['head_B_label'], dtype=torch.long)

        return image, features, yA_tensor, yB_tensor