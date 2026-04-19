# src/data/arp/dataset_arp.py

import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class ARPDataset6Class(Dataset):
    """
    Dataset para imágenes ARP (Angular Radial Partitioning) Adaptado a 6 Clases.
    Devuelve las imágenes en Escala de Grises (1 Canal).
    """
    def __init__(self, df, images_dir: str, transforms=None):
        self.df = df.copy()
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        
        # Clases que consideramos malignas para el Head A
        self.malignant_classes = [1, 2, 3]

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Directorio de imágenes ARP no encontrado: {self.images_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Obtener la ruta de la imagen (Asumimos formato isic_id.jpg)
        img_name = f"{row['isic_id']}.jpg"
        img_path = self.images_dir / img_name

        if not img_path.exists():
            # Fallback por si en tu csv antiguo se llamaba image_path
            if 'image_path' in row:
                img_path = self.images_dir / Path(row['image_path']).name
            else:
                raise FileNotFoundError(f"Imagen ARP no encontrada: {img_path}")

        # 2. Cargar en BLANCO Y NEGRO ("L" = 1 channel)
        image = Image.open(img_path).convert("L")

        # 3. Aplicar transformaciones de PyTorch (Augmentation/Normalization)
        if self.transforms is not None:
            image = self.transforms(image)

        # 4. Obtener etiquetas dinámicamente
        target_multi = int(row['target'])
        
        # Si tienes las columnas head_a_label y head_B_label explicitas, las usamos. 
        # Si no, las derivamos del target para mayor seguridad.
        y_headB = int(row['head_B_label']) if 'head_B_label' in row else target_multi
        
        if 'head_a_label' in row:
            y_headA = float(row['head_a_label'])
        else:
            y_headA = 1.0 if y_headB in self.malignant_classes else 0.0

        return image, torch.tensor(y_headA, dtype=torch.float), torch.tensor(y_headB, dtype=torch.long)