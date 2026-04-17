import pandas as pd
import numpy as np

# Ruta de tu archivo final
CSV_FINAL = "C:/TFG/data/Original_Data/ISIC_FINAL/isic_metadata_transformado.csv"

def checklist_despegue(csv_path):
    print("="*80)
    print(" 🚀 CHECKLIST DE DESPEGUE: VALIDACIÓN FINAL DEL DATASET ")
    print("="*80)

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"❌ Error crítico: No se pudo cargar el archivo. {e}")
        return

    errores = 0

    # ---------------------------------------------------------
    # 1. DIMENSIONES Y VOLUMEN
    # ---------------------------------------------------------
    print("\n▶ 1. COMPROBACIÓN DE DIMENSIONES:")
    filas, columnas = df.shape
    print(f"   - Imágenes a descargar: {filas:,}")
    print(f"   - Características (Columnas): {columnas}")
    if filas < 220000 or filas > 230000:
        print("   ⚠️ AVISO: El número de filas no ronda las 225k esperadas.")
    else:
        print("   ✅ Volumen de imágenes perfecto.")

    # ---------------------------------------------------------
    # 2. PRUEBA DE FUEGO: VALORES NULOS (NaN)
    # ---------------------------------------------------------
    print("\n▶ 2. COMPROBACIÓN DE VALORES NULOS (LA PRUEBA DEL ALGODÓN):")
    # Ignoramos patient_id y lesion_id porque ES NORMAL que tengan NaNs. 
    # El master_id ya nos protege de esto.
    columnas_a_revisar = [col for col in df.columns if col not in ['patient_id', 'lesion_id']]
    
    nulos_por_columna = df[columnas_a_revisar].isna().sum()
    nulos_totales = nulos_por_columna.sum()
    
    if nulos_totales == 0:
        print("   ✅ PERFECTO: 0 valores nulos en las variables críticas. PyTorch no fallará.")
    else:
        print(f"   ❌ ERROR CRÍTICO: Se encontraron {nulos_totales} NaNs en columnas vitales.")
        # Mostrar exactamente dónde están los errores
        print("      Detalle de los NaNs encontrados:")
        print(nulos_por_columna[nulos_por_columna > 0].to_string())
        errores += 1

    # ---------------------------------------------------------
    # 3. UNICIDAD DE DESCARGA (ISIC_ID)
    # ---------------------------------------------------------
    print("\n▶ 3. COMPROBACIÓN DE UNICIDAD (Evitar sobreescrituras):")
    if 'isic_id' in df.columns:
        duplicados = df.duplicated(subset=['isic_id']).sum()
        if duplicados == 0:
            print("   ✅ Todos los 'isic_id' son únicos. La descarga será segura.")
        else:
            print(f"   ❌ ERROR CRÍTICO: Hay {duplicados} 'isic_id' repetidos.")
            errores += 1
    else:
        print("   ❌ ERROR CRÍTICO: Falta la columna 'isic_id'.")
        errores += 1

    # ---------------------------------------------------------
    # 4. PREVENCIÓN DE DATA LEAKAGE (MASTER_ID)
    # ---------------------------------------------------------
    print("\n▶ 4. COMPROBACIÓN DE AGRUPACIÓN (MASTER_ID):")
    if 'master_id' in df.columns:
        if df['master_id'].isna().sum() == 0:
            print("   ✅ 'master_id' está intacto y sin vacíos. El K-Fold será hermético.")
        else:
            print("   ❌ ERROR CRÍTICO: 'master_id' contiene NaNs.")
            errores += 1
    else:
        print("   ❌ ERROR CRÍTICO: Falta la columna 'master_id'.")
        errores += 1

    # ---------------------------------------------------------
    # 5. PREVENCIÓN DE EFECTO CLEVER HANS
    # ---------------------------------------------------------
    print("\n▶ 5. COMPROBACIÓN DE SESGOS OCULTOS:")
    columnas_prohibidas = ['image_type', 'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'sex', 'anatom_site_general']
    cols_encontradas = [c for c in columnas_prohibidas if c in df.columns]
    
    if len(cols_encontradas) == 0:
        print("   ✅ Limpio de variables tramposas. La red no podrá hacer el Efecto Clever Hans.")
    else:
        print(f"   ❌ ERROR CRÍTICO: Aún existen columnas prohibidas: {cols_encontradas}")
        errores += 1

    # ---------------------------------------------------------
    # 6. INTEGRIDAD DEL ONE-HOT ENCODING
    # ---------------------------------------------------------
    print("\n▶ 6. COMPROBACIÓN DEL ONE-HOT ENCODING:")
    ohe_cols = [c for c in df.columns if 'sex_' in c or 'anatom_site_general_' in c]
    ohe_error = False
    for col in ohe_cols:
        valores_unicos = df[col].unique()
        if not set(valores_unicos).issubset({0, 1}):
            ohe_error = True
    
    if ohe_error:
        print("   ❌ ERROR CRÍTICO: Las columnas OHE contienen números distintos de 0 y 1.")
        errores += 1
    elif len(ohe_cols) > 0:
        print(f"   ✅ Las {len(ohe_cols)} columnas OHE son binarias puras (0 y 1).")
    else:
        print("   ❌ ERROR: No se encontraron columnas OHE.")
        errores += 1

    # ---------------------------------------------------------
    # 7. INTEGRIDAD DE LA VARIABLE TARGET
    # ---------------------------------------------------------
    print("\n▶ 7. COMPROBACIÓN DE LA VARIABLE OBJETIVO (TARGET):")
    if 'target' in df.columns:
        clases_esperadas = {0, 1, 2, 3, 4, 5}
        clases_reales = set(df['target'].unique())
        
        if clases_reales.issubset(clases_esperadas):
            print("   ✅ La columna 'target' contiene exactamente las clases esperadas.")
            print("   - Distribución:")
            for clase in sorted(list(clases_reales)):
                cantidad = len(df[df['target'] == clase])
                print(f"     Clase {clase}: {cantidad:,} imágenes")
        else:
            print(f"   ❌ ERROR CRÍTICO: Hay clases no reconocidas en el target: {clases_reales}")
            errores += 1
    else:
        print("   ❌ ERROR CRÍTICO: No existe la columna 'target'.")
        errores += 1

    # ---------------------------------------------------------
    # VEREDICTO FINAL
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(" 🚦 VEREDICTO FINAL DEL SISTEMA ")
    print("="*80)
    if errores == 0:
        print(" 🟢 LUZ VERDE. El dataset es una obra de arte arquitectónica.")
        print("    Está listo para descargar las imágenes y empezar a programar PyTorch.")
    else:
        print(f" 🔴 LUZ ROJA. Se detectaron {errores} errores críticos. NO PROCEDER.")
    print("="*80)

if __name__ == "__main__":
    checklist_despegue(CSV_FINAL)