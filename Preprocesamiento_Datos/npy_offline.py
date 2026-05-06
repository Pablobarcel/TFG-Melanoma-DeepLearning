import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURACIÓN ---
ORIGIN_DIR = Path("C:/TFG/src/data/processed/images_RGB_ISIC")
TARGET_DIR = Path("C:/TFG/src/data/processed/images_RGB_NPY_ISIC") # Nueva carpeta
os.makedirs(TARGET_DIR, exist_ok=True)

def convert_images():
    print(f"🚀 Iniciando conversión a .npy...")
    archivos = [f for f in os.listdir(ORIGIN_DIR) if f.endswith('.jpg')]
    
    for img_name in tqdm(archivos, desc="Convirtiendo"):
        # 1. Leer imagen
        img_path = ORIGIN_DIR / img_name
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
            
        # 2. Convertir a RGB (OpenCV lee en BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Guardar como binario .npy (mismo nombre, distinta extensión)
        np_name = img_name.replace(".jpg", ".npy")
        np.save(TARGET_DIR / np_name, img)

    print(f"✨ ¡Hecho! Todas las imágenes están en: {TARGET_DIR}")

if __name__ == "__main__":
    convert_images()