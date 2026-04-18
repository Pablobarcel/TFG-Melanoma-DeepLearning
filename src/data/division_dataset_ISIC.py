import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# RUTAS
CSV_FINAL = "C:/TFG/data/Original_Data/ISIC_FINAL/isic_metadata_transformado.csv"
CSV_TEST = "C:/TFG/data/Original_Data/ISIC_FINAL/test.csv"
CSV_TRAIN_CV = "C:/TFG/data/Original_Data/ISIC_FINAL/train.csv"

def dividir_test_y_cv():
    print("="*60)
    print(" ✂️ DIVISIÓN HERMÉTICA DEL DATASET (TEST vs TRAIN/VAL) ")
    print("="*60)

    # 1. Cargar el dataset
    print("⏳ Cargando el dataset transformado...")
    df = pd.read_csv(CSV_FINAL, low_memory=False)
    
    # 2. Configurar GroupShuffleSplit
    # test_size=0.15 significa que el 15% de los PACIENTES irán a Test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    
    # 3. Separar los índices
    # X e y no importan mucho aquí, lo clave es el parámetro 'groups=df['master_id']'
    train_idx, test_idx = next(gss.split(df, groups=df['master_id']))
    
    # 4. Crear los DataFrames
    df_train_cv = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # --- PRUEBA DE FUEGO (SANITARY CHECK) ---
    print("\n▶ COMPROBANDO FUGAS DE DATOS (DATA LEAKAGE):")
    pacientes_train = set(df_train_cv['master_id'].unique())
    pacientes_test = set(df_test['master_id'].unique())
    
    fugas = pacientes_train.intersection(pacientes_test)
    if len(fugas) == 0:
        print("   ✅ PERFECTO: Cero fugas detectadas. Ningún paciente de Test está en Train.")
    else:
        print(f"   ❌ ERROR CRÍTICO: {len(fugas)} pacientes están en ambos conjuntos.")
        return

    # 5. Guardar y Resumir
    print("\n" + "="*60)
    print(" 📊 RESUMEN DE LA DIVISIÓN ")
    print("="*60)
    print(f"📁 TRAIN/VAL (Para K-Fold): {len(df_train_cv):,} imágenes ({len(pacientes_train):,} pacientes)")
    print(f"📁 TEST (Examen Final):     {len(df_test):,} imágenes ({len(pacientes_test):,} pacientes)")
    print("="*60)

    df_train_cv.to_csv(CSV_TRAIN_CV, index=False)
    df_test.to_csv(CSV_TEST, index=False)
    print(f"\n✅ Archivos generados con éxito. ¡Listo para la siguiente fase!")

if __name__ == "__main__":
    dividir_test_y_cv()