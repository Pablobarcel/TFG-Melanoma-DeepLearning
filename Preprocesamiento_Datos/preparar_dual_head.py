import pandas as pd

# Rutas de tus archivos generados en el paso anterior
FILES = [
    "C:/TFG/data/Original_Data/ISIC_FINAL/train.csv",
    "C:/TFG/data/Original_Data/ISIC_FINAL/test.csv"
]

def agregar_etiqueta_binaria(file_path):
    print(f"⏳ Procesando: {file_path}")
    df = pd.read_csv(file_path)
    
    # Definimos la lógica: Malignos son 1 (MEL), 2 (BCC) y 3 (SCC)
    # Benignos son 0 (NV), 4 (BKL) y 5 (BG)
    malignos = [1, 2, 3]
    
    # Creamos la columna head_A_label (Binaria)
    # .isin() devuelve True/False, .astype(int) lo pasa a 1/0
    df['target_binary'] = df['target'].isin(malignos).astype(int)
    
    # Para que sea idéntico a tu Plan A, podemos renombrar o crear alias
    df['head_A_label'] = df['target_binary']
    df['head_B_label'] = df['target']
    
    df.to_csv(file_path, index=False)
    print(f"✅ Columna 'target_binary' añadida con éxito.")

if __name__ == "__main__":
    for f in FILES:
        agregar_etiqueta_binaria(f)