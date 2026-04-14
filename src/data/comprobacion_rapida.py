import pandas as pd
from pathlib import Path

def auditar_experiment_300k():
    print("🕵️‍♂️ INICIANDO AUDITORÍA ESTRICTA DE EXPERIMENT_300K...\n" + "="*55)
    
    # Rutas a tus archivos generados
    train_path = Path("data/Splitted_data/experiment_300k/train.csv")
    val_path = Path("data/Splitted_data/experiment_300k/val.csv")
    
    if not train_path.exists() or not val_path.exists():
        print("❌ Error: No se encuentran los archivos train.csv o val.csv. Revisa la ruta.")
        return

    # Cargar los datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # ---------------------------------------------------------
    # PRUEBA 1: DATA LEAKAGE (LA MÁS IMPORTANTE)
    # ---------------------------------------------------------
    print("🔬 PRUEBA 1: DATA LEAKAGE (Fuga de Datos)")
    
    # Comprobamos grupos
    train_groups = set(df_train['master_group_id'])
    val_groups = set(df_val['master_group_id'])
    grupos_filtrados = train_groups.intersection(val_groups)
    
    # Comprobamos imágenes exactas (por si acaso)
    train_images = set(df_train['image_path'])
    val_images = set(df_val['image_path'])
    imagenes_filtradas = train_images.intersection(val_images)
    
    if len(grupos_filtrados) == 0 and len(imagenes_filtradas) == 0:
        print("  ✅ SUPERADO: Cero fugas de datos. Train y Val son completamente independientes.")
    else:
        print(f"  ❌ ¡ALERTA DE LEAKAGE! Hay {len(grupos_filtrados)} pacientes/lesiones en ambos sets.")
        print(f"  ❌ ¡ALERTA DE LEAKAGE! Hay {len(imagenes_filtradas)} imágenes exactas duplicadas.")
        print("  ⚠️ El modelo está haciendo trampas.")
        return # Si hay leakage, paramos aquí porque no tiene sentido seguir.
        
    # ---------------------------------------------------------
    # PRUEBA 2: TAMAÑO Y RATIO
    # ---------------------------------------------------------
    print("\n📊 PRUEBA 2: VOLUMEN DE DATOS Y RATIO")
    total_train = len(df_train)
    total_val = len(df_val)
    total_imagenes = total_train + total_val
    
    ratio_train_real = total_train / total_imagenes
    
    print(f"  -> Total imágenes en Train: {total_train}")
    print(f"  -> Total imágenes en Val:   {total_val}")
    print(f"  -> Ratio real Train/Val:    {ratio_train_real:.2%} / {1 - ratio_train_real:.2%} (Objetivo era 80/20)")
    
    if 0.75 <= ratio_train_real <= 0.85:
        print("  ✅ SUPERADO: El ratio de partición es correcto.")
    else:
        print("  ⚠️ AVISO: El ratio está un poco desviado del 80% esperado.")

    # ---------------------------------------------------------
    # PRUEBA 3: DESBALANCEO (HEAD B)
    # ---------------------------------------------------------
    print("\n⚖️ PRUEBA 3: DISTRIBUCIÓN DE CLASES (HEAD B)")
    
    print("  Distribución en Train:")
    train_counts = df_train['head_B_label'].value_counts().sort_index()
    for cls, count in train_counts.items():
        print(f"    - Clase {cls}: {count} imágenes ({count/total_train:.2%})")
        
    print("  Distribución en Val:")
    val_counts = df_val['head_B_label'].value_counts().sort_index()
    for cls, count in val_counts.items():
        print(f"    - Clase {cls}: {count} imágenes ({count/total_val:.2%})")

    # ---------------------------------------------------------
    # PRUEBA 4: COLUMNAS VITALES
    # ---------------------------------------------------------
    print("\n📝 PRUEBA 4: INTEGRIDAD DE COLUMNAS")
    required_cols = {'master_group_id', 'head_A_label', 'head_B_label', 'image_path'}
    if required_cols.issubset(df_train.columns) and required_cols.issubset(df_val.columns):
        print("  ✅ SUPERADO: Todas las columnas necesarias existen en ambos CSVs.")
    else:
        print("  ❌ ERROR: Faltan columnas vitales en los CSVs generados.")

    print("\n" + "="*55)
    print("🚀 CONCLUSIÓN: Si tienes todo en ✅, ¡tienes luz verde para arrancar el entrenamiento!")

if __name__ == "__main__":
    auditar_experiment_300k()