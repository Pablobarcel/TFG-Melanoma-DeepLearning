import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURACIÓN PARA RGB (Comentado) ---
# ORIGIN_DIR = Path("C:/TFG/src/data/processed/images_RGB_ISIC")
# TARGET_DIR = Path("C:/TFG/src/data/processed/images_RGB_NPY_ISIC")

# --- CONFIGURACIÓN PARA ARP (Activo) ---
# Se utilizan las rutas específicas para los descriptores ARP[cite: 5]
ORIGIN_DIR = Path("C:/TFG/src/data/processed/images_ARP_ISIC")
TARGET_DIR = Path("C:/TFG/src/data/processed/images_ARP_NPY_ISIC") 
os.makedirs(TARGET_DIR, exist_ok=True)

def convert_images():
    print(f"🚀 Iniciando conversión a .npy...")
    print(f"📂 Origen: {ORIGIN_DIR}")
    print(f"📂 Destino: {TARGET_DIR}")
    
    archivos = [f for f in os.listdir(ORIGIN_DIR) if f.endswith('.jpg')]
    
    for img_name in tqdm(archivos, desc="Convirtiendo"):
        img_path = ORIGIN_DIR / img_name
        
        # 1. Leer imagen
        # Para ARP usamos cv2.IMREAD_GRAYSCALE para obtener 1 solo canal
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # --- Lógica para RGB (Comentada) ---
        # img = cv2.imread(str(img_path))
        # if img is not None:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #[cite: 2]
        
        if img is None:
            continue
            
        # 2. Guardar como binario .npy
        # El formato .npy elimina la necesidad de descompresión en la CPU durante el entrenamiento
        np_name = img_name.replace(".jpg", ".npy")
        np.save(TARGET_DIR / np_name, img)

    print(f"✨ ¡Hecho! Todas las imágenes están en: {TARGET_DIR}")

if __name__ == "__main__":
    convert_images()