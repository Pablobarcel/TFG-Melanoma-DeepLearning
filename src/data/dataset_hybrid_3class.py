# src/data/dataset_hybrid_3class.py

import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from src.config.paths import SPLITTED_DATA_DIR, IMAGES_RGB_DIR, PROJECT_ROOT

IMAGES_ARP_DIR = PROJECT_ROOT / "src" / "data" / "processed" / "imagenes_ARP_224"

class HybridTripleDataset3Class(Dataset):
    """
    Dataset final que devuelve DOS imágenes por paciente (Adaptado a 3 Clases NMSC):
    1. Imagen RGB (Para la rama ResNet + ViT)
    2. Imagen ARP en escala de grises (Para la rama ARP)
    
    Salida en Head B: 0 (BEN), 1 (MEL), 2 (NMSC).
    """
    def __init__(
        self,
        csv_name: str = "experiment_46000_balanced/train.csv",
        transforms_rgb=None,
        transforms_arp=None,
    ):
        self.csv_path = SPLITTED_DATA_DIR / csv_name
        self.df = pd.read_csv(self.csv_path)

        # -------------------------------------------------------------------
        # 🚨 MAGIA AÑADIDA: FUSIÓN DE CLASES ON-THE-FLY (3 CLASES NMSC)
        # -------------------------------------------------------------------
        # Si la columna existe, convertimos la clase 3 (SCC+) en 2 (NMSC).
        if "head_B_label" in self.df.columns:
            self.df.loc[self.df["head_B_label"] == 3, "head_B_label"] = 2
        # -------------------------------------------------------------------

        self.transforms_rgb = transforms_rgb
        self.transforms_arp = transforms_arp

        if not IMAGES_RGB_DIR.exists() or not IMAGES_ARP_DIR.exists():
            raise FileNotFoundError("Revisa que los directorios RGB y ARP existan.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = Path(row["image_path"]).name

        # --- 1. CARGAR IMAGEN RGB ---
        path_rgb = IMAGES_RGB_DIR / img_name
        image_rgb = Image.open(path_rgb).convert("RGB")
        if self.transforms_rgb:
            image_rgb = self.transforms_rgb(image_rgb)

        # --- 2. CARGAR IMAGEN ARP ---
        path_arp = IMAGES_ARP_DIR / img_name
        image_arp = Image.open(path_arp).convert("L") # Escala de grises
        if self.transforms_arp:
            image_arp = self.transforms_arp(image_arp)

        # --- 3. ETIQUETAS ---
        y_headA = int(row["head_A_label"])
        y_headB = int(row["head_B_label"])

        # Devolvemos una tupla con las dos imágenes y las dos etiquetas
        return image_rgb, image_arp, y_headA, y_headB