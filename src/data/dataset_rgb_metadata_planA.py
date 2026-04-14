# src/data/dataset_rgb_metadata_planA.py

import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from src.config.paths import SPLITTED_DATA_DIR, IMAGES_RGB_DIR

class RGBMetadataDatasetPlanA(Dataset):
    """
    Dataset Multimodal: Devuelve Imagen RGB + Metadatos Plan A
    """
    def __init__(self, csv_name="train.csv", transforms=None):
        self.csv_path = SPLITTED_DATA_DIR / csv_name
        self.df = pd.read_csv(self.csv_path)
        self.transforms = transforms

        # --- FUSIÓN DE CLASES ON-THE-FLY (NMSC) ---
        if "head_B_label" in self.df.columns:
            self.df.loc[self.df["head_B_label"] == 3, "head_B_label"] = 2

        # --- DEFINICIÓN PLAN A (Ya escaladas en CSV) ---
        self.num_cols = ['age_approx', 'clin_size_long_diam_mm']
        self.sex_cols = [c for c in self.df.columns if c.startswith('sex_')]
        self.site_cols = [c for c in self.df.columns if c.startswith('anatom_site_general_')]
        self.feature_cols = self.num_cols + self.sex_cols + self.site_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 1. IMAGEN RGB ---
        img_name = Path(row["image_path"]).name
        img_path = IMAGES_RGB_DIR / img_name
        
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        # --- 2. METADATOS ---
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))

        # --- 3. ETIQUETAS ---
        y_headA = int(row["head_A_label"])
        y_headB = int(row["head_B_label"])

        return image, features, y_headA, y_headB