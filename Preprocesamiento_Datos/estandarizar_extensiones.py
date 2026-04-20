import os
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
DIRECTORIOS = [
    Path("C:/TFG/src/data/processed/images_RGB_ISIC"),
    Path("C:/TFG/src/data/processed/images_ARP_ISIC")
]

TARGET_EXTENSION = ".jpg"

def estandarizar_directorio(directorio: Path):
    if not directorio.exists():
        print(f"⚠️ El directorio no existe: {directorio}")
        return

    print(f"\n📂 Escaneando: {directorio.name}")
    archivos = [f for f in directorio.iterdir() if f.is_file()]
    
    renombrados = 0
    ignorados = 0

    # Usamos tqdm para ver la barra de progreso (son muchísimas imágenes)
    for archivo in tqdm(archivos, desc="Estandarizando extensiones", unit="img"):
        # Comprobamos si la extensión (en minúsculas) es diferente a .jpg
        if archivo.suffix.lower() != TARGET_EXTENSION:
            # Creamos la nueva ruta manteniendo el nombre original pero forzando .jpg
            nuevo_nombre = archivo.stem + TARGET_EXTENSION
            nueva_ruta = directorio / nuevo_nombre
            
            try:
                # Renombramos el archivo físicamente en el disco
                archivo.rename(nueva_ruta)
                renombrados += 1
            except Exception as e:
                print(f"❌ Error renombrando {archivo.name}: {e}")
        else:
            # Si ya es .jpg, pero está en mayúsculas (ej. .JPG), lo forzamos a minúsculas
            if archivo.suffix != TARGET_EXTENSION:
                nuevo_nombre = archivo.stem + TARGET_EXTENSION
                nueva_ruta = directorio / nuevo_nombre
                archivo.rename(nueva_ruta)
                renombrados += 1
            else:
                ignorados += 1

    print(f"✅ Proceso completado en {directorio.name}:")
    print(f"   - Imágenes renombradas a {TARGET_EXTENSION}: {renombrados}")
    print(f"   - Imágenes que ya estaban correctas: {ignorados}")

if __name__ == '__main__':
    print("="*70)
    print(" 🔄 INICIANDO ESTANDARIZACIÓN DE EXTENSIONES A .jpg")
    print("="*70)
    
    for directorio in DIRECTORIOS:
        estandarizar_directorio(directorio)
        
    print("\n" + "="*70)
    print(" 🏁 TODAS LAS IMÁGENES ESTÁN AHORA EN FORMATO .jpg")
    print("="*70)