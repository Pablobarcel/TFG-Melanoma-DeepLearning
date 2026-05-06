# src/data/rgb/dataset_rgb.py
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

class RGBDataset6Class(Dataset):
    def __init__(self, df, images_dir: str, transforms=None):
        # Ahora images_dir apuntará a la nueva carpeta de .npy
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.label_map = {0:0, 4:0, 5:0, 1:1, 2:2, 3:3}
        self.malignant_classes = [1, 2, 3]

        print(f"🔍 Verificando integridad NPY en {self.images_dir.name}...")
        archivos_en_disco = set(os.listdir(self.images_dir))
        
        df = df.copy()
        # Buscamos .npy en lugar de .jpg
        df['expected_filename'] = df['isic_id'].astype(str) + '.npy'
        filtro_existentes = df['expected_filename'].isin(archivos_en_disco)
        self.df = df[filtro_existentes].reset_index(drop=True)
        
        imgs_perdidas = len(df) - len(self.df)
        if imgs_perdidas > 0:
            print(f"⚠️ {imgs_perdidas} registros descartados por falta de archivo .npy.")
        print(f"✅ Dataset NPY listo: {len(self.df)} muestras.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row['expected_filename']
        
        # 1. Carga ultra rápida (Lectura directa de binario)
        # mmap_mode='r' permite leer sin cargar todo el archivo en RAM si no es necesario
        image = np.load(img_path)

        # 2. Aplicar transformaciones
        if self.transforms is not None:
            # Albumentations usa numpy arrays directamente, lo cual es perfecto aquí
            if str(type(self.transforms)).find('albumentations') == -1:
                image = Image.fromarray(image)
                image = self.transforms(image)
            else:
                image = self.transforms(image=image)["image"]

        # 3. Gestión de etiquetas
        y_headB = self.label_map[int(row['target'])]
        y_headA = float(row['head_a_label']) if 'head_a_label' in row else (1.0 if y_headB in self.malignant_classes else 0.0)

        return image, torch.tensor(y_headA, dtype=torch.float), torch.tensor(y_headB, dtype=torch.long)