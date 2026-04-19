import pandas as pd
import numpy as np

# Configuración de ruta
CSV_PATH = "C:\TFG\data\Original_Data\ISIC_FINAL\isic_metadata_full.csv"

def auditoria_clinica_avanzada(csv_path):
    print("="*80)
    print(" 🏥 AUDITORÍA FORENSE AVANZADA: METADATOS CLÍNICOS ISIC ".center(80))
    print("="*80)

    # 1. CARGA DE DATOS
    print(f"⏳ Cargando base de datos ({csv_path})...")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return

    total_img = len(df)
    print(f"✅ Carga completada. Dimensiones: {total_img:,} imágenes | {df.shape[1]} columnas.\n")

    # ==========================================
    # 2. MAPA DE VALORES NULOS (MISSING DATA)
    # ==========================================
    print("-" * 80)
    print(" 📉 1. ANÁLISIS DE COMPLETITUD (VALORES NULOS / NaN) ")
    print("-" * 80)
    nulos = df.isnull().sum()
    nulos_pct = (nulos / total_img) * 100
    df_nulos = pd.DataFrame({'Nulos': nulos, 'Porcentaje (%)': nulos_pct})
    df_nulos = df_nulos[df_nulos['Nulos'] > 0].sort_values(by='Porcentaje (%)', ascending=False)
    
    if not df_nulos.empty:
        print(df_nulos.head(20).to_string(formatters={'Porcentaje (%)': '{:.2f}%'.format}))
        print("...")
    else:
        print("¡Increíble! No hay valores nulos en todo el dataset.")
        
    # ==========================================
    # 2.5. ANÁLISIS DE DUPLICADOS (DATA INTEGRITY)
    # ==========================================
    print("\n" + "-" * 80)
    print(" 👯 2. ANÁLISIS DE DUPLICADOS (INTEGRIDAD DE LA BASE DE DATOS) ")
    print("-" * 80)

    duplicados_absolutos = df.duplicated().sum()
    print(f"▶ Duplicados absolutos (Filas idénticas en todo): {duplicados_absolutos:,} filas.")

    if 'isic_id' in df.columns:
        duplicados_isic = df.duplicated(subset=['isic_id']).sum()
        print(f"▶ Identificadores de imagen (isic_id) repetidos: {duplicados_isic:,} casos.")
    else:
        print("▶ No se encontró la columna 'isic_id'.")

    # ==========================================
    # 3. IDENTIFICADORES Y FUGAS (DATA LEAKAGE)
    # ==========================================
    print("\n" + "-" * 80)
    print(" 🔍 3. RASTREO DE IDs ORIGINALES ")
    print("-" * 80)
    
    isic_unicos = df['isic_id'].nunique()
    print(f"🔹 Imágenes (isic_id): {isic_unicos:,} únicas.")
    
    mask_has_patient_has_lesion = df['patient_id'].notna() & df['lesion_id'].notna()
    mask_has_patient_no_lesion = df['patient_id'].notna() & df['lesion_id'].isna()
    mask_no_patient_has_lesion = df['patient_id'].isna() & df['lesion_id'].notna()
    mask_no_patient_no_lesion = df['patient_id'].isna() & df['lesion_id'].isna()

    print("📊 DISTRIBUCIÓN DE IDs ORIGINALES:")
    print(f"  🟢 Tienen 'patient_id' Y 'lesion_id': {mask_has_patient_has_lesion.sum():,} ({(mask_has_patient_has_lesion.sum()/total_img)*100:.1f}%)")
    print(f"  🟢 Tienen 'patient_id' pero NO 'lesion_id': {mask_has_patient_no_lesion.sum():,} ({(mask_has_patient_no_lesion.sum()/total_img)*100:.1f}%)")
    print(f"  🟡 NO tienen 'patient_id', pero SÍ 'lesion_id': {mask_no_patient_has_lesion.sum():,} ({(mask_no_patient_has_lesion.sum()/total_img)*100:.1f}%)")
    print(f"  🔴 NO tienen 'patient_id' NI 'lesion_id' (Limbo): {mask_no_patient_no_lesion.sum():,} ({(mask_no_patient_no_lesion.sum()/total_img)*100:.1f}%)")

    # ==========================================
    # 3.5. NUEVO: AUDITORÍA DEL MASTER_ID
    # ==========================================
    print("\n" + "-" * 80)
    print(" 🔑 3.5. AUDITORÍA DEL MASTER_ID (AGRUPACIÓN HERMÉTICA) ")
    print("-" * 80)
    
    if 'master_id' in df.columns:
        nulos_master = df['master_id'].isna().sum()
        unicos_master = df['master_id'].nunique()
        imgs_per_master = df['master_id'].value_counts()
        
        print(f"▶ Total de Master IDs únicos (Pacientes virtuales): {unicos_master:,}")
        print(f"▶ Valores nulos en Master ID: {nulos_master} ", end="")
        if nulos_master == 0:
            print("✅ (PERFECTO)")
        else:
            print("❌ (ERROR CRÍTICO)")
            
        print(f"\n▶ Estadística de agrupación por Master ID:")
        print(f"   - Máximo de imágenes en un solo grupo: {imgs_per_master.max()}")
        print(f"   - Mínimo de imágenes en un solo grupo: {imgs_per_master.min()}")
        print(f"   - Promedio de imágenes por grupo: {imgs_per_master.mean():.2f}")
    else:
        print("❌ ERROR: No se encontró la columna 'master_id'.")

    # ==========================================
    # 4. DIAGNÓSTICOS CLÍNICOS GLOBALES
    # ==========================================
    print("\n" + "-" * 80)
    print(" 🔬 4. DISTRIBUCIÓN GLOBAL DE DIAGNÓSTICOS ")
    print("-" * 80)
    
    if 'diagnosis_1' in df.columns:
        diag_counts = df['diagnosis_1'].value_counts(dropna=False)
        melanomas = len(df[df['diagnosis_1'] == 'melanoma']) if 'melanoma' in diag_counts else 0
        bcc = len(df[df['diagnosis_1'] == 'basal cell carcinoma']) if 'basal cell carcinoma' in diag_counts else 0
        scc = len(df[df['diagnosis_1'] == 'squamous cell carcinoma']) if 'squamous cell carcinoma' in diag_counts else 0
        
        print(f"🚨 Cánceres en diagnosis_1 -> Melanomas: {melanomas:,} | BCC: {bcc:,} | SCC: {scc:,}")
        
    # ==========================================
    # 4.5. TAXONOMÍA CLÍNICA EXHAUSTIVA (PARA AGRUPACIÓN DE CLASES)
    # ==========================================
    print("\n" + "=" * 80)
    print(" 🧬 3. TAXONOMÍA CLÍNICA EXHAUSTIVA (MAPEO DE ENFERMEDADES) ")
    print(" (Usa esta lista para decidir qué diagnósticos fusionar en tus clases) ")
    print("=" * 80)

    for i in range(1, 6):
        diag_col = f'diagnosis_{i}'
        if diag_col in df.columns:
            print(f"\n▶ CATÁLOGO COMPLETO DE '{diag_col.upper()}':")
            # Obtenemos TODOS los valores sin límite, excluyendo NaNs para ver solo la clínica
            counts = df[diag_col].value_counts(dropna=False)
            
            for diag, count in counts.items():
                pct = (count / total_img) * 100
                print(f"   - {str(diag):<60} : {count:>7,} ({pct:>5.2f}%)")

    # ==========================================
    # 5. DEMOGRAFÍA Y SESGOS CLÍNICOS
    # ==========================================
    print("\n" + "-" * 80)
    print(" 📊 5. SESGOS DEMOGRÁFICOS Y ANATÓMICOS ")
    print("-" * 80)
    
    if 'sex' in df.columns:
        print("Distribución por Sexo:")
        print(df['sex'].value_counts(dropna=False, normalize=True).apply(lambda x: f"{x*100:.1f}%").to_string())
        
    if 'age_approx' in df.columns:
        media_edad = df['age_approx'].mean()
        print(f"\nEdad aproximada media: {media_edad:.1f} años")

    # ==========================================
    # 5.5. NUEVO: ANÁLISIS CRUZADO DE METADATOS Y DIAGNÓSTICO
    # ==========================================
    print("\n" + "=" * 80)
    print(" 📋 5.5. ANÁLISIS CRUZADO DE METADATOS Y DIAGNÓSTICOS ")
    print(" (¿Faltan datos? ¿Hay sesgos en los cánceres?) ")
    print("=" * 80)

    # Elegimos los metadatos tabulares más importantes
    meta_cols = ['sex', 'age_approx', 'anatom_site_general', 'image_type']
    diag_col = 'diagnosis_1'
    
    if diag_col in df.columns:
        for col in meta_cols:
            if col in df.columns:
                print(f"\n▶ METADATO: {col.upper()}")
                
                # 1. Análisis de Nulos
                nulos_col = df[col].isna().sum()
                pct_nulos = (nulos_col / total_img) * 100
                print(f"   - Valores Nulos (NaN): {nulos_col:,} ({pct_nulos:.1f}%)")
                
                # 2. Análisis Cruzado (Top categorías de este metadato)
                if nulos_col < total_img:
                    print(f"   - Distribución del diagnóstico dentro de cada grupo de {col}:")
                    
                    # Agrupar edades para que sea legible, si es la columna edad
                    if col == 'age_approx':
                        bins = [0, 30, 60, 120]
                        labels = ['Joven (0-30)', 'Adulto (31-60)', 'Mayor (61+)']
                        df_temp = df.copy()
                        df_temp['age_group'] = pd.cut(df_temp['age_approx'], bins=bins, labels=labels)
                        top_vals = labels
                        col_analizar = 'age_group'
                    else:
                        top_vals = df[col].value_counts().head(4).index
                        col_analizar = col
                        df_temp = df

                    for val in top_vals:
                        subset = df_temp[df_temp[col_analizar] == val]
                        total_subset = len(subset)
                        if total_subset == 0: continue
                        
                        print(f"     * Cuando es '{val}' (Total: {total_subset:,}):")
                        diag_counts = subset[diag_col].value_counts().head(3)
                        
                        for d_idx, d_val in diag_counts.items():
                            d_pct = (d_val / total_subset) * 100
                            print(f"         > {str(d_idx):<20}: {d_val:<6,} ({d_pct:.1f}%)")
    else:
        print(f"No se encontró la columna {diag_col} para cruzar.")

    # ==========================================
    # 5.6. ANÁLISIS PROFUNDO: % DE NaNs POR TIPO DE DIAGNÓSTICO
    # ==========================================
    print("\n" + "=" * 80)
    print(" 🕵️‍♀️ 5.6. DECISIÓN DE METADATOS: % DE NaNs SEGÚN EL DIAGNÓSTICO ")
    print(" (Identifica si una columna vacía esconde oro en los cánceres) ")
    print("=" * 80)

    # Excluir columnas de ID y diagnóstico para centrarnos en los metadatos clínicos puros
    cols_excluir = ['isic_id', 'patient_id', 'lesion_id', 'master_id'] + [c for c in df.columns if 'diagnosis' in c]
    meta_cols = [c for c in df.columns if c not in cols_excluir]

    # Sacamos las clases únicas de diagnosis_1 (Deberían ser Benign y Malignant)
    clases_diag = df['diagnosis_1'].dropna().unique()

    for col in meta_cols:
        nulos_totales = df[col].isna().sum()
        if nulos_totales == 0:
            continue # Si la columna está 100% llena, no hay riesgo de borrarla, nos la saltamos
            
        print(f"\n▶ METADATO: {col.upper()} (Nulos globales: {(nulos_totales/total_img)*100:.1f}%)")
        
        for clase in clases_diag:
            df_clase = df[df['diagnosis_1'] == clase]
            total_clase = len(df_clase)
            
            if total_clase == 0: continue
            
            nulos_en_clase = df_clase[col].isna().sum()
            pct_nulos = (nulos_en_clase / total_clase) * 100
            
            # Imprimir con formato alineado
            print(f"   - En {clase:<10}: {nulos_en_clase:>7,} NaNs de {total_clase:>7,} imágenes ({pct_nulos:>5.1f}%)")

    # ==========================================
    # 7. INFORME FINAL DEL ARQUITECTO
    # ==========================================
    print("\n" + "=" * 80)
    print(" 🧠 7. INFORME FINAL Y DECISIONES ARQUITECTÓNICAS ")
    print("=" * 80)
    print("Revisa el apartado 3.5 para confirmar que no hay NaNs en master_id.")
    print("Revisa el apartado 5.5 para ver qué columnas tabulares necesitarán imputación (-1 o 'unknown').")
    print("\n🏁 ANÁLISIS COMPLETADO.")

if __name__ == "__main__":
    auditoria_clinica_avanzada(CSV_PATH)