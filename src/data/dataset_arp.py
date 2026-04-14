# src/data/dataset_arp.py

import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from src.config.paths import SPLITTED_DATA_DIR, PROJECT_ROOT

# Definimos la ruta de las imágenes ARP
IMAGES_ARP_DIR = PROJECT_ROOT / "src" / "data" / "processed" / "imagenes_ARP"

class ARPDataset(Dataset):
    """
    Dataset para imágenes ARP (Angular Radial Partitioning).
    Devuelve las imágenes en Escala de Grises (1 Canal) y las 4 clases originales.
    """
    def __init__(
        self,
        csv_name: str = "training_dev.csv",
        transforms=None,
    ):
        self.csv_path = SPLITTED_DATA_DIR / csv_name
        self.df = pd.read_csv(self.csv_path)
        
        self.transforms = transforms

        if not IMAGES_ARP_DIR.exists():
            raise FileNotFoundError(
                f"Directorio de imágenes ARP no encontrado: {IMAGES_ARP_DIR}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Obtener la ruta de la imagen
        raw_path = row["image_path"]
        img_name = Path(raw_path).name
        img_path = IMAGES_ARP_DIR / img_name

        if not img_path.exists():
            raise FileNotFoundError(f"Imagen ARP no encontrada: {img_path}")

        # 2. Cargar en BLANCO Y NEGRO ("L" = 1 channel)
        # El color no aporta en ARP, solo la geometría y la textura importan.
        image = Image.open(img_path).convert("L")

        # 3. Aplicar transformaciones
        if self.transforms is not None:
            image = self.transforms(image)

        # 4. Obtener etiquetas (sin alterar)
        y_headA = int(row["head_A_label"])
        y_headB = int(row["head_B_label"])

        return image, y_headA, y_headB