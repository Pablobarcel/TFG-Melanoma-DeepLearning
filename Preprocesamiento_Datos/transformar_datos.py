import pandas as pd

# Rutas de entrada y salida
CSV_ENTRADA = "C:/TFG/data/Original_Data/ISIC_FINAL/isic_metadata_limpio.csv"
CSV_SALIDA = "C:/TFG/data/Original_Data/ISIC_FINAL/metadata_TFG_Transformado.csv"

def transformar_datos(csv_in, csv_out):
    print("="*60)
    print(" 🛠️ SCRIPT DE TRANSFORMACIÓN Y CODIFICACIÓN (OHE) ")
    print("="*60)

    # 1. Cargar datos
    print(f"⏳ Cargando datos desde: {csv_in}...")
    try:
        df = pd.read_csv(csv_in, low_memory=False)
    except Exception as e:
        print(f"❌ Error al cargar: {e}")
        return

    # ---------------------------------------------------------
    # PASO 1: GESTIÓN DE VALORES FALTANTES (NaNs)
    # ---------------------------------------------------------
    print("\n▶ PASO 1: Imputación de valores faltantes...")

    # A. Variable Numérica (Edad) -> Imputar con -1 (Valor Centinela)
    # Rellenamos los NaNs con -1.0. Si ya venían como -1.0 del script anterior, se quedan igual.
    df['age_approx'] = df['age_approx'].fillna(-1.0)
    print("   ✅ Edad (age_approx): NaNs imputados con el valor centinela -1.0.")

    # B. Variables Categóricas (Sexo y Anatomía) -> Rellenar con 'unknown'
    cat_cols = ['sex', 'anatom_site_general']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    print("   ✅ Variables categóricas: NaNs rellenados con 'unknown'.")

    # ---------------------------------------------------------
    # PASO 2: ONE-HOT ENCODING (OHE)
    # ---------------------------------------------------------
    print("\n▶ PASO 2: Aplicando One-Hot Encoding a variables categóricas...")

    # Aplicamos get_dummies. dummy_na=False porque ya hemos tapado los NaNs con 'unknown'
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
    
    # Convertimos los valores booleanos (True/False) a enteros (1/0)
    ohe_cols = [col for col in df.columns if col.startswith('sex_') or col.startswith('anatom_site_general_')]
    for col in ohe_cols:
        df[col] = df[col].astype(int)

    print(f"   ✅ One-Hot Encoding completado. Se generaron {len(ohe_cols)} columnas binarias.")

    # ---------------------------------------------------------
    # RESUMEN Y GUARDADO
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" 📊 RESUMEN DE LAS COLUMNAS FINALES ")
    print("="*60)
    for col in df.columns:
        print(f"  - {col}")
    print("="*60)

    print(f"\n⏳ Guardando archivo transformado en: {csv_out}...")
    df.to_csv(csv_out, index=False)
    print("✅ ¡Operación completada! El CSV está listo para la red neuronal.")

if __name__ == "__main__":
    transformar_datos(CSV_ENTRADA, CSV_SALIDA)