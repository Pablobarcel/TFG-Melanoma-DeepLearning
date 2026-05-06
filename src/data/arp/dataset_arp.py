# src/data/arp/dataset_arp.py
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class ARPDataset6Class(Dataset):
    def __init__(self, df, images_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.label_map = {0:0, 4:0, 5:0, 1:1, 2:2, 3:3}
        self.malignant_classes = [1, 2, 3]

        print(f"🔍 Verificando integridad NPY en {self.images_dir.name}...")
        archivos_en_disco = set(os.listdir(self.images_dir))
        
        df = df.copy()
        # Buscamos archivos .npy optimizados[cite: 2]
        df['expected_filename'] = df['isic_id'].astype(str) + '.npy'
        filtro_existentes = df['expected_filename'].isin(archivos_en_disco)
        self.df = df[filtro_existentes].reset_index(drop=True)
        
        print(f"✅ Dataset ARP NPY listo: {len(self.df)} muestras.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row['expected_filename']
        
        # Carga binaria ultra rápida[cite: 2]
        image = np.load(img_path)

        if self.transforms is not None:
            # Soporte para Albumentations o Torchvision[cite: 2]
            if str(type(self.transforms)).find('albumentations') == -1:
                image = Image.fromarray(image)
                image = self.transforms(image)
            else:
                image = self.transforms(image=image)["image"]

        y_headB = self.label_map[int(row['target'])]
        y_headA = float(row['head_a_label']) if 'head_a_label' in row else (1.0 if y_headB in self.malignant_classes else 0.0)

        return image, torch.tensor(y_headA, dtype=torch.float), torch.tensor(y_headB, dtype=torch.long)