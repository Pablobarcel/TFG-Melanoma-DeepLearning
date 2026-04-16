import os
from PIL import Image
from tqdm import tqdm

# 1. Configura tus rutas (Cámbialas por las tuyas de Windows)
# Carpeta donde tienes las fotos ARP originales gigantes
RUTA_ORIGEN_ARP = r"C:/TFG/src/data/processed/imagenes_ARP" 
# Carpeta nueva donde se guardarán las fotos ARP en miniatura
RUTA_DESTINO_ARP = r"C:/TFG/src/data/processed/imagenes_ARP_224"

os.makedirs(RUTA_DESTINO_ARP, exist_ok=True)

imagenes = [f for f in os.listdir(RUTA_ORIGEN_ARP) if f.endswith('.jpg') or f.endswith('.png')]

print(f"🚀 Iniciando redimensionamiento de {len(imagenes)} imágenes ARP a 224x224...")

for nombre_archivo in tqdm(imagenes, desc="Procesando ARP"):
    ruta_input = os.path.join(RUTA_ORIGEN_ARP, nombre_archivo)
    ruta_output = os.path.join(RUTA_DESTINO_ARP, nombre_archivo)
    
    if os.path.exists(ruta_output):
        continue
        
    try:
        # 🎯 LA CLAVE: convert("L") transforma la imagen a Escala de Grises (1 canal)
        # Esto concuerda con tu Normalize(mean=[0.5], std=[0.5])
        img = Image.open(ruta_input).convert("L")
        
        # Como es offline, usamos LANCZOS para máxima nitidez geométrica
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Guardamos con calidad alta (90) para no perder el detalle de los bordes
        img_resized.save(ruta_output, format="JPEG", quality=90)
    except Exception as e:
        print(f"Error con la imagen {nombre_archivo}: {e}")

print("✅ ¡Proceso ARP terminado! Ya puedes subir la carpeta a Kaggle.")