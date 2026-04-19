import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
INPUT_DIR = Path("C:/TFG/images/images_ISIC_FINAL")
OUTPUT_RGB_DIR = Path("C:/TFG/src/data/processed/images_RGB_ISIC")
OUTPUT_ARP_DIR = Path("C:/TFG/src/data/processed/images_ARP_ISIC")

TARGET_SIZE = (224, 224)

# ==============================================================================
# MOTOR DE TRANSFORMACIÓN (Ejecutado por cada núcleo de la CPU)
# ==============================================================================
def procesar_imagen(img_name):
    img_path = INPUT_DIR / img_name
    rgb_save_path = OUTPUT_RGB_DIR / img_name
    arp_save_path = OUTPUT_ARP_DIR / img_name
    
    # 1. Sistema reanudable: si ambas imágenes ya existen, la saltamos
    if rgb_save_path.exists() and arp_save_path.exists():
        return "Ya existe"

    # Leer la imagen original
    img = cv2.imread(str(img_path))
    if img is None:
        return f"Error de lectura: {img_name}"

    try:
        # ---------------------------------------------------------
        # RAMA 1: Transformación RGB Offline (Resize + Lanczos)
        # ---------------------------------------------------------
        if not rgb_save_path.exists():
            # Interpolación Lanczos4 es óptima para reducir imágenes médicas sin perder nitidez
            rgb_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(rgb_save_path), rgb_resized)

        # ---------------------------------------------------------
        # RAMA 2: Transformación ARP Offline (Polar + Grises + Lanczos)
        # ---------------------------------------------------------
        if not arp_save_path.exists():
            # 1. Convertir a Escala de Grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Calcular centro y radio máximo
            h, w = gray.shape
            center = (w / 2, h / 2)
            max_radius = np.sqrt((w/2)**2 + (h/2)**2)
            
            # 3. Transformación Polar
            arp_image = cv2.linearPolar(gray, center, max_radius, cv2.WARP_FILL_OUTLIERS)
            
            # 4. Redimensionar con Lanczos
            arp_resized = cv2.resize(arp_image, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(arp_save_path), arp_resized)

        return "Procesada"
    
    except Exception as e:
        return f"Error procesando {img_name}: {e}"

# ==============================================================================
# BUCLE PRINCIPAL (MULTIPROCESAMIENTO)
# ==============================================================================
def iniciar_procesamiento():
    print("="*80)
    print(" ⚙️ INICIANDO PIPELINE DE TRANSFORMACIÓN VISUAL OFFLINE (RGB & ARP) ")
    print("="*80)

    # Crear carpetas si no existen
    OUTPUT_RGB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ARP_DIR.mkdir(parents=True, exist_ok=True)

    # Listar todas las imágenes en la carpeta de origen
    print("⏳ Escaneando imágenes descargadas...")
    imagenes = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_imagenes = len(imagenes)
    print(f"▶ Total detectadas: {total_imagenes:,}")

    # Contadores
    procesadas = 0
    existentes = 0
    errores = 0

    # Usamos ProcessPoolExecutor para saturar la CPU matemática
    # max_workers=None usa automáticamente todos los núcleos de tu PC
    with ProcessPoolExecutor() as executor:
        futuros = {executor.submit(procesar_imagen, img_name): img_name for img_name in imagenes}
        
        for futuro in tqdm(as_completed(futuros), total=total_imagenes, desc="Transformando", unit="img"):
            resultado = futuro.result()
            if resultado == "Procesada":
                procesadas += 1
            elif resultado == "Ya existe":
                existentes += 1
            else:
                errores += 1

    print("\n" + "="*80)
    print(" 🏁 INFORME DE TRANSFORMACIÓN COMPLETADO ")
    print("="*80)
    print(f" ✅ Imágenes convertidas (RGB + ARP): {procesadas:,}")
    print(f" ⏭️ Imágenes que ya estaban listas:   {existentes:,}")
    print(f" ❌ Errores (imágenes corruptas):      {errores:,}")
    print(f" 📂 Carpeta RGB: {OUTPUT_RGB_DIR}")
    print(f" 📂 Carpeta ARP: {OUTPUT_ARP_DIR}")
    print("="*80)

if __name__ == '__main__':
    # Este if __name__ == '__main__' es obligatorio en Windows para poder usar Multiprocesamiento
    iniciar_procesamiento()