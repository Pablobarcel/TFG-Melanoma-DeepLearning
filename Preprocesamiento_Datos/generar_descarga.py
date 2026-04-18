import pandas as pd
import os

# CONFIGURACIÓN
CSV_FINAL = "C:/TFG/data/Original_Data/ISIC_FINAL/isic_metadata_transformado.csv"
CARPETA_IMAGENES = "C:/TFG/images/images_ISIC_FINAL"
ARCHIVO_BAT = "C:/TFG/descargar_imagenes.bat"

def generar_bat():
    print(" leyendo CSV para generar el archivo de comandos...")
    df = pd.read_csv(CSV_FINAL, low_memory=False)
    ids = df['isic_id'].unique().tolist()
    
    # Agrupamos de 50 en 50 para que sea mucho más rápido
    # (ISIC permite buscar varios IDs a la vez con 'OR')
    chunk_size = 50
    chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    
    print(f" Generando comandos para {len(ids)} imágenes en {len(chunks)} bloques...")

    with open(ARCHIVO_BAT, 'w') as f:
        f.write("@echo off\n")
        f.write("echo INICIANDO DESCARGA AUTOMATIZADA...\n")
        
        for i, bloque in enumerate(chunks, 1):
            # Creamos la consulta: isic_id:ISIC_1 OR isic_id:ISIC_2...
            query = " OR ".join([f"isic_id:{id_img}" for id_img in bloque])
            
            # Escribimos la línea de comando exacta que tú querías
            f.write(f'echo Procesando bloque {i} de {len(chunks)}...\n')
            f.write(f'isic image download --search "{query}" "{CARPETA_IMAGENES}"\n')

    print(f"✅ ¡LISTO! Se ha creado el archivo: {ARCHIVO_BAT}")
    print(" PASOS A SEGUIR:")
    print(f"1. Copia '{ARCHIVO_BAT}' a la carpeta donde te funciona el comando 'isic'.")
    print("2. Abre el CMD en esa carpeta.")
    print("3. Escribe 'descargar_imagenes.bat' y pulsa Enter.")

if __name__ == "__main__":
    generar_bat()