import pandas as pd
import numpy as np

# Configuración de ruta
CSV_PATH = "isic_metadata_full.csv"

def auditar_dataset(csv_path):
    print("="*65)
    print("🕵️‍♂️ INICIANDO AUDITORÍA FORENSE DEL METADATA DE ISIC")
    print("="*65)

    # 1. CARGA DE DATOS
    print(f"⏳ Cargando {csv_path}... (puede tardar unos segundos)")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return

    total_imagenes = len(df)
    print(f"✅ Archivo cargado con éxito.")
    print(f"📊 Dimensiones totales: {total_imagenes:,} imágenes y {df.shape[1]} variables clínicas.\n")

    # 2. ANÁLISIS DE IDENTIFICADORES (DATA LEAKAGE)
    print("-" *65)
    print("🔍 1. ANÁLISIS DE IDENTIFICADORES Y FUGAS DE DATOS")
    print("-" *65)
    
    # ISIC ID (Las imágenes físicas)
    if 'isic_id' in df.columns:
        unicos = df['isic_id'].nunique()
        print(f"🔸 ISIC IDs: {unicos:,} únicos. Duplicados exactos: {total_imagenes - unicos} (Deben ser 0)")
    
    # Patient ID (Pacientes únicos)
    if 'patient_id' in df.columns:
        nulos_pac = df['patient_id'].isna().sum()
        pacs_unicos = df['patient_id'].nunique()
        max_img_pac = df['patient_id'].value_counts().max() if pacs_unicos > 0 else 0
        print(f"🔸 Patient IDs: {pacs_unicos:,} pacientes únicos registrados.")
        print(f"   -> Imágenes SIN paciente asignado (NaN): {nulos_pac:,} ({(nulos_pac/total_imagenes)*100:.1f}%)")
        print(f"   -> Paciente con más fotos: {max_img_pac} imágenes.")
        
    # Lesion ID (Evolución temporal del mismo lunar)
    if 'lesion_id' in df.columns:
        nulos_les = df['lesion_id'].isna().sum()
        les_unicas = df['lesion_id'].nunique()
        max_img_les = df['lesion_id'].value_counts().max() if les_unicas > 0 else 0
        print(f"🔸 Lesion IDs: {les_unicas:,} lesiones únicas documentadas.")
        print(f"   -> Imágenes SIN lesión asignada (NaN): {nulos_les:,} ({(nulos_les/total_imagenes)*100:.1f}%)")
        print(f"   -> Lesión más fotografiada: {max_img_les} veces.")

    # 3. BALANCEO CLÍNICO Y GRANULARIDAD
    print("\n" + "-" *65)
    print("🔬 2. DIAGNÓSTICOS CLÍNICOS (GRANULARIDAD DE ETIQUETAS)")
    print("-" *65)
    
    if 'benign_malignant' in df.columns:
        print("📈 Distribución Maestra (Benigno vs Maligno):")
        print(df['benign_malignant'].value_counts(dropna=False).to_string())
        
        df_benign = df[df['benign_malignant'] == 'benign']
        df_malign = df[df['benign_malignant'] == 'malignant']
        
        if 'diagnosis' in df.columns:
            print("\n🟢 TOP Diagnósticos Benignos:")
            print(df_benign['diagnosis'].value_counts(dropna=False).head(10).to_string())
            
            print("\n🔴 TOP Diagnósticos Malignos (Tipos de Cáncer):")
            print(df_malign['diagnosis'].value_counts(dropna=False).head(10).to_string())
            
            # Detectar ruido: diagnósticos marcados como 'unknown'
            unknown_benignos = len(df_benign[df_benign['diagnosis'] == 'unknown'])
            print(f"\n⚠️ Hay {unknown_benignos:,} imágenes 'benignas' que no dicen QUÉ son (unknown).")
            print(f"   (Deberías eliminarlas para subir la calidad del dataset).")

    # 4. DIAGNÓSTICO DEL INGENIERO: ¿MASTER GROUP ID?
    print("\n" + "-" *65)
    print("🧠 3. VEREDICTO DE ARQUITECTURA: ¿NECESITAMOS MASTER GROUP ID?")
    print("-" *65)
    
    if ('patient_id' in df.columns) and ('lesion_id' in df.columns):
        if nulos_pac > 0 or nulos_les > 0:
            print("❌ ALARMA DE DATA LEAKAGE: Faltan muchísimos IDs de pacientes y lesiones.")
            print("   Si haces un split K-Fold usando solo 'patient_id', todas las imágenes")
            print("   que tienen NaN se mezclarán aleatoriamente entre Train y Val.")
            print("   La red se evaluará con fotos de pacientes que ya ha visto en el entreno.")
            print("\n✅ SOLUCIÓN OBLIGATORIA: Sí, tienes que fabricar el 'Master Group ID'.")
            print("   Deberás programar una lógica que junte (patient_id + lesion_id + isic_id)")
            print("   para rellenar los huecos y garantizar que los grupos sean herméticos.")
        else:
            print("✅ Tienes una base de datos impecable (0 nulos). Podrías usar patient_id directamente.")

    print("\n" + "="*65)
    print("🏁 FIN DE LA AUDITORÍA")

if __name__ == "__main__":
    auditar_dataset(CSV_PATH)