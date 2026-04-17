import pandas as pd
import numpy as np

# Rutas (ajústalas según tu entorno)
CSV_ORIGINAL = "C:/TFG/data/Original_Data/ISIC_FINAL/isic_metadata_full.csv"
CSV_LIMPIO = "C:/TFG/data/Original_Data/ISIC_FINAL/isic_metadata_limpio.csv"

def limpieza_pura_metadatos(csv_in, csv_out):
    print("="*60)
    print(" 🧹 SCRIPT DE LIMPIEZA Y AGRUPACIÓN DE METADATOS ISIC ")
    print("="*60)

    # 1. CARGA DE DATOS
    print(f"⏳ Cargando datos desde: {csv_in}...")
    try:
        df = pd.read_csv(csv_in, low_memory=False)
    except Exception as e:
        print(f"❌ Error al cargar: {e}")
        return
        
    total_inicial = len(df)
    total_columnas_inicial = len(df.columns)
    print(f"   -> Filas iniciales cargadas: {total_inicial:,}")

    # ---------------------------------------------------------
    # FILTRO 1: RUIDO DIAGNÓSTICO
    # ---------------------------------------------------------
    print("\n▶ APLICANDO FILTRO 1: Diagnosis inútil (NaN o Indeterminate/Unknown)...")
    
    mask_diag_nan = df['diagnosis_1'].isna()
    mask_diag_incierto = df['diagnosis_1'].isin(['Indeterminate', 'unknown', 'unclassified'])
    
    df = df[~(mask_diag_nan | mask_diag_incierto)].copy()
    
    total_post_filtro1 = len(df)
    print(f"   ✅ Eliminadas {total_inicial - total_post_filtro1:,} filas basura.")

    # ---------------------------------------------------------
    # FILTRO 2: EL LIMBO ABSOLUTO DE IDs
    # ---------------------------------------------------------
    print("\n▶ APLICANDO FILTRO 2: Prevención de Data Leakage (Sin patient_id NI lesion_id)...")
    
    mask_limbo = df['patient_id'].isna() & df['lesion_id'].isna()
    df = df[~mask_limbo].copy()
    
    total_post_filtro2 = len(df)
    print(f"   ✅ Eliminadas {total_post_filtro1 - total_post_filtro2:,} filas del Limbo Absoluto.")

    # ---------------------------------------------------------
    # 3. CREACIÓN DEL MASTER ID (AGRUPACIÓN HERMÉTICA)
    # ---------------------------------------------------------
    print("\n▶ CREANDO 'master_id': Agrupando imágenes para el K-Fold...")
    
    df['master_id'] = df['patient_id'].fillna(df['lesion_id'])
    
    print(f"   ✅ Creados {df['master_id'].nunique():,} 'master_id' únicos en total.")

    # ---------------------------------------------------------
    # 4. LIMPIEZA DE COLUMNAS (ELIMINAR TOXICIDAD)
    # ---------------------------------------------------------
    print("\n▶ APLICANDO FILTRO 3: Poda de columnas (Conservando IDs, Diagnósticos y el Cuarteto de Oro)...")
    
    # Mantenemos lo estrictamente necesario
    columnas_deseadas = [
        'isic_id', 'patient_id', 'lesion_id', 'master_id', # IDs de control
        'diagnosis_1', 'diagnosis_2', 'diagnosis_3',       # Diagnósticos para mapeo
        'sex', 'age_approx', 'anatom_site_general', 'image_type' # Cuarteto de metadatos (image_type es temporal)
    ]
    
    columnas_existentes = [col for col in columnas_deseadas if col in df.columns]
    df = df[columnas_existentes].copy()
    
    print(f"   ✅ Columnas reducidas de {total_columnas_inicial} a {len(df.columns)} esenciales.")

    # ---------------------------------------------------------
    # 5. SUBMUESTREO ESTRATIFICADO DE BENIGNOS (POR MASTER_ID)
    # ---------------------------------------------------------
    print("\n▶ APLICANDO FILTRO 4: Submuestreo inteligente de Benignos (Group-wise)...")

    mask_benign = df['diagnosis_1'] == 'Benign'
    df_benign = df[mask_benign]
    df_malignant = df[~mask_benign]

    print(f"   - Total Benignos disponibles: {len(df_benign):,}")
    print(f"   - Total Cánceres a conservar intactos: {len(df_malignant):,}")

    mask_tbp = df_benign['image_type'] == 'TBP tile: close-up'
    df_benign_valiosos = df_benign[~mask_tbp]
    df_benign_tbp = df_benign[mask_tbp]

    num_valiosos = len(df_benign_valiosos)
    objetivo_tbp = 200000 - num_valiosos

    print(f"   - Rescatados el 100% de Benignos Variados (Dermatoscopio/Clínico): {num_valiosos:,}")
    print(f"   - Objetivo aproximado de Benignos TBP a rellenar: {objetivo_tbp:,}")

    # --- LÓGICA DE MUESTREO POR MASTER_ID ---
    conteo_por_master = df_benign_tbp['master_id'].value_counts()
    conteo_barajado = conteo_por_master.sample(frac=1, random_state=42)
    suma_acumulativa = conteo_barajado.cumsum()
    master_ids_seleccionados = suma_acumulativa[suma_acumulativa <= objetivo_tbp].index
    df_benign_tbp_sampled = df_benign_tbp[df_benign_tbp['master_id'].isin(master_ids_seleccionados)]
    
    print(f"   - Seleccionados {len(master_ids_seleccionados):,} 'master_ids' enteros.")
    print(f"   - Esto nos da un total de {len(df_benign_tbp_sampled):,} imágenes TBP íntegras.")

    # Unir todo
    df_benign_final = pd.concat([df_benign_valiosos, df_benign_tbp_sampled])
    df_final = pd.concat([df_benign_final, df_malignant]).sample(frac=1, random_state=42).reset_index(drop=True)

    # ---------------------------------------------------------
    # FILTRO 5: ELIMINAR EL ANDAMIO (IMAGE_TYPE)
    # ---------------------------------------------------------
    print("\n▶ APLICANDO FILTRO 5: Eliminando metadato de sacrificio ('image_type')...")
    df_final = df_final.drop(columns=['image_type'])
    print("   ✅ Columna 'image_type' eliminada para evitar el Efecto Clever Hans.")

    # ---------------------------------------------------------
    # 6. CREACIÓN DE LAS 6 CLASES CLÍNICAS (TARGET MAPPING)
    # ---------------------------------------------------------
    print("\n▶ APLICANDO FILTRO 6: Mapeando diagnósticos a 6 Clases Clínicas (Target)...")

    df_final['target'] = np.nan

    # 1. BASE: Todo lo benigno va al cajón desastre (Clase 5: Benigno Genérico)
    df_final.loc[df_final['diagnosis_1'] == 'Benign', 'target'] = 5

    # 2. SOBRESCRITURA: Mapeo específico basado en diagnosis_3
    mapeo_clases = {
        # 🔴 MALIGNOS
        'Melanoma, NOS': 1, 'Melanoma in situ': 1, 'Melanoma Invasive': 1, 'Melanoma metastasis': 1,
        'Basal cell carcinoma': 2,
        'Squamous cell carcinoma, NOS': 3, 'Squamous cell carcinoma, Invasive': 3, 'Squamous cell carcinoma in situ': 3,
        
        # 🟢 BENIGNOS ESPECÍFICOS (Sobrescriben el 5 genérico)
        'Nevus': 0,
        'Seborrheic keratosis': 4, 'Pigmented benign keratosis': 4, 'Solar lentigo': 4, 'Lichen planus like keratosis': 4
    }

    # Aplicamos el mapeo. Si diagnosis_3 tiene uno de esos valores, sobrescribe el target.
    target_mapeado = df_final['diagnosis_3'].map(mapeo_clases)
    df_final['target'] = target_mapeado.fillna(df_final['target'])

    # 3. LIMPIEZA FINAL DE RAROS: Borramos lo que se quedó en NaN (ej. Sarcomas, Cánceres raros)
    nulos_target_antes = df_final['target'].isna().sum()
    df_final = df_final.dropna(subset=['target']).copy()
    df_final['target'] = df_final['target'].astype(int)

    print(f"   ✅ Se eliminaron {nulos_target_antes:,} imágenes con cánceres raros no clasificables (<1000 fotos).")

    # 4. BORRAR COLUMNAS DIAGNÓSTICAS (Ya están resumidas en 'target')
    df_final = df_final.drop(columns=['diagnosis_1', 'diagnosis_2', 'diagnosis_3'], errors='ignore')
    print("   ✅ Columnas de diagnóstico de texto eliminadas (Limpieza total).")

    # ---------------------------------------------------------
    # RESUMEN Y GUARDADO
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" 📊 RESUMEN FINAL DEL DATASET (6 CLASES) ")
    print("="*60)
    print(f"  [0] Nevus (NV):                     {len(df_final[df_final['target'] == 0]):,}")
    print(f"  [1] Melanoma (MEL):                 {len(df_final[df_final['target'] == 1]):,}")
    print(f"  [2] Carcinoma Basocelular (BCC):    {len(df_final[df_final['target'] == 2]):,}")
    print(f"  [3] Carcinoma Espinocelular (SCC):  {len(df_final[df_final['target'] == 3]):,}")
    print(f"  [4] Queratosis Benignas (BKL):      {len(df_final[df_final['target'] == 4]):,}")
    print(f"  [5] Benigno Genérico (BG):          {len(df_final[df_final['target'] == 5]):,}")
    print("-" * 60)
    print(f"Total Columnas:  {len(df_final.columns)}")
    print(f"TOTAL DATASET:   {len(df_final):,}")
    print("="*60)

    print(f"\n⏳ Guardando archivo limpio en: {csv_out}...")
    df_final.to_csv(csv_out, index=False)
    print("✅ ¡Guardado con éxito! Tu dataset respeta historiales clínicos íntegros y está listo para PyTorch.")

if __name__ == "__main__":
    limpieza_pura_metadatos(CSV_ORIGINAL, CSV_LIMPIO)