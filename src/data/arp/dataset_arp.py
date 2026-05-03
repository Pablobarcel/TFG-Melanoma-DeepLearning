# src/data/arp/dataset_arp.py

import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class ARPDataset6Class(Dataset):
    """
    Dataset para imágenes ARP (Angular Radial Partitioning) Adaptado a 4 Clases.
    Devuelve las imágenes en Escala de Grises (1 Canal).
    """
    def __init__(self, df, images_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.label_map = {0:0, 4:0, 5:0, 1:1, 2:2, 3:3} # NV, BKL, BG -> 0
        self.malignant_classes = [1, 2, 3]

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Directorio de imágenes ARP no encontrado: {self.images_dir}")

        print(f"🔍 Verificando integridad de imágenes en {self.images_dir.name}...")
        
        # 1. Obtenemos un conjunto (set) súper rápido de todo lo que hay en la carpeta
        archivos_en_disco = set(os.listdir(self.images_dir))
        
        # 2. Creamos la columna del nombre del archivo en el DataFrame
        df = df.copy()
        df['expected_filename'] = df['isic_id'].astype(str) + '.jpg'
        
        # 3. Filtramos el DataFrame: nos quedamos SOLO con las filas cuya imagen exista
        filtro_existentes = df['expected_filename'].isin(archivos_en_disco)
        self.df = df[filtro_existentes].reset_index(drop=True)
        
        # 4. Reporte de seguridad
        imgs_perdidas = len(df) - len(self.df)
        if imgs_perdidas > 0:
            print(f"⚠️ PRECAUCIÓN: Se han descartado {imgs_perdidas} registros del CSV porque la imagen no existe en el disco duro.")
        print(f"✅ Dataset final listo con {len(self.df)} imágenes verificadas.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # La ruta está garantizada porque la verificamos en el __init__
        img_name = row['expected_filename']
        img_path = self.images_dir / img_name

        # Cargar en BLANCO Y NEGRO ("L" = 1 channel)
        image = Image.open(img_path).convert("L")

        # Aplicar transformaciones de PyTorch
        if self.transforms is not None:
            image = self.transforms(image)

        # Obtener etiquetas dinámicamente
        original_label = int(row['target'])
        y_headB = self.label_map[original_label] # Mapeo a 4 clases
        
        if 'head_a_label' in row:
            y_headA = float(row['head_a_label'])
        else:
            y_headA = 1.0 if y_headB in self.malignant_classes else 0.0

        return image, torch.tensor(y_headA, dtype=torch.float), torch.tensor(y_headB, dtype=torch.long)