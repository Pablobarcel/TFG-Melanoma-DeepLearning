# src/data/rgb/dataset_rgb.py
import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class RGBDataset6Class(Dataset):
    def __init__(self, df, images_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.malignant_classes = [1, 2, 3]

        print(f"🔍 Verificando integridad RGB en {self.images_dir.name}...")
        archivos_en_disco = set(os.listdir(self.images_dir))
        
        df = df.copy()
        df['expected_filename'] = df['isic_id'].astype(str) + '.jpg'
        filtro_existentes = df['expected_filename'].isin(archivos_en_disco)
        self.df = df[filtro_existentes].reset_index(drop=True)
        
        imgs_perdidas = len(df) - len(self.df)
        if imgs_perdidas > 0:
            print(f"⚠️ Se han descartado {imgs_perdidas} registros por falta de imagen RGB.")
        print(f"✅ Dataset RGB listo: {len(self.df)} imágenes.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row['expected_filename']
        
        # Cargar en COLOR (RGB)
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        y_headB = int(row['head_B_label']) if 'head_B_label' in row else int(row['target'])
        y_headA = float(row['head_a_label']) if 'head_a_label' in row else (1.0 if y_headB in self.malignant_classes else 0.0)

        return image, torch.tensor(y_headA, dtype=torch.float), torch.tensor(y_headB, dtype=torch.long)