# src/data/dataset_rgb_3class.py

import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from src.config.paths import SPLITTED_DATA_DIR, IMAGES_RGB_DIR


class RGBDataset3Class(Dataset):
    """
    Dataset RGB para clasificación dermatológica con doble head (Adaptado a 3 Clases NMSC):
    - Head A: binaria (0 = BEN, 1 = MAL)
    - Head B: multiclase (0 = BEN, 1 = MEL, 2 = NMSC)

    Las imágenes se cargan SIEMPRE desde IMAGES_RGB_DIR.
    El CSV solo se usa para identificar la imagen y las etiquetas.
    """

    def __init__(
        self,
        csv_name: str = "training_dev.csv",
        transforms=None,
    ):
        """
        Args:
            csv_name (str): nombre del CSV dentro de SPLITTED_DATA_DIR
            transforms: transformaciones torchvision (opcional)
        """
        self.csv_path = SPLITTED_DATA_DIR / csv_name
        self.df = pd.read_csv(self.csv_path)

        # -------------------------------------------------------------------
        # 🚨 MAGIA AÑADIDA: FUSIÓN DE CLASES ON-THE-FLY (3 CLASES NMSC)
        # -------------------------------------------------------------------
        # Convertimos todos los SCC+ (3) en la clase (2) para formar el grupo NMSC
        if "head_B_label" in self.df.columns:
            self.df.loc[self.df["head_B_label"] == 3, "head_B_label"] = 2
        # -------------------------------------------------------------------

        self.transforms = transforms

        # Comprobaciones básicas
        required_columns = ["image_path", "head_A_label", "head_B_label"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Columna obligatoria no encontrada: {col}")

        # Comprobación de directorio de imágenes RGB
        if not IMAGES_RGB_DIR.exists():
            raise FileNotFoundError(
                f"Directorio de imágenes RGB no encontrado: {IMAGES_RGB_DIR}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --------------------------------------------------
        # Imagen
        # --------------------------------------------------
        raw_path = row["image_path"]

        # Nos quedamos SOLO con el nombre del archivo
        img_name = Path(raw_path).name

        # Construimos el path REAL a la imagen RGB
        img_path = IMAGES_RGB_DIR / img_name

        if not img_path.exists():
            raise FileNotFoundError(
                f"Imagen RGB no encontrada: {img_path}"
            )

        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------
        y_headA = int(row["head_A_label"])
        y_headB = int(row["head_B_label"])

        return image, y_headA, y_headB