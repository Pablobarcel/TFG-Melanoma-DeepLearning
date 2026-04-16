import os
from PIL import Image
from tqdm import tqdm # Barra de progreso (haz 'pip install tqdm' si no lo tienes)

# 1. Configura tus rutas (Cámbialas por las tuyas de Windows)
# Carpeta donde tienes las fotos originales ahora mismo
RUTA_ORIGEN = r"C:/TFG/src/data/processed/imagenes_RGB" 
# Carpeta nueva donde se guardarán las fotos en miniatura
RUTA_DESTINO = r"C:/TFG/src/data/imagenes_RGB_224"

# Crear la carpeta de destino si no existe
os.makedirs(RUTA_DESTINO, exist_ok=True)

# 2. Obtener la lista de imágenes
imagenes = [f for f in os.listdir(RUTA_ORIGEN) if f.endswith('.jpg') or f.endswith('.png')]

print(f"🚀 Iniciando redimensionamiento de {len(imagenes)} imágenes a 224x224...")

# 3. Bucle de procesamiento
for nombre_archivo in tqdm(imagenes, desc="Procesando"):
    ruta_input = os.path.join(RUTA_ORIGEN, nombre_archivo)
    ruta_output = os.path.join(RUTA_DESTINO, nombre_archivo)
    
    # Si la imagen ya existe en destino, la saltamos (por si se corta a la mitad)
    if os.path.exists(ruta_output):
        continue
        
    try:
        # Abrir, redimensionar en modo Bilinear (rápido) y guardar
        img = Image.open(ruta_input).convert("RGB")
        img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
        # Guardamos con buena calidad pero comprimido para que pese poco
        img_resized.save(ruta_output, format="JPEG", quality=85)
    except Exception as e:
        print(f"Error con la imagen {nombre_archivo}: {e}")

print("✅ ¡Proceso terminado! Ya puedes subir la carpeta a Kaggle.")