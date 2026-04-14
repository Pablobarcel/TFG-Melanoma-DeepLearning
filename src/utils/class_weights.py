# src/utils/class_weights.py

import torch
import pandas as pd

def compute_class_weights(df, device, label_col="head_B_label"):
    """
    Calcula automáticamente los pesos para CrossEntropyLoss basándose en el 
    desbalance de clases del DataFrame.
    
    Args:
        df (pd.DataFrame): El dataframe que contiene las etiquetas.
        device (torch.device): 'cpu' o 'cuda'.
        label_col (str): Nombre de la columna de la etiqueta (Head B).
    
    Returns:
        torch.Tensor: Tensor de pesos normalizados listo para la Loss.
    """
    print(f"\n⚖️ [Utils] Calculando pesos para columna: '{label_col}'...")
    
    # 1. Contar ocurrencias reales (asegurando orden 0, 1, 2, 3...)
    if label_col not in df.columns:
        raise ValueError(f"❌ La columna {label_col} no existe en el DataFrame.")

    counts = df[label_col].value_counts().sort_index()
    counts_tensor = torch.tensor(counts.values, dtype=torch.float)
    
    # 2. Fórmula de Balanceo Inverso: Total / (Num_Clases * Frecuencia_Clase)
    total_samples = counts_tensor.sum()
    num_classes = len(counts_tensor)
    
    # Evitamos división por cero sumando un epsilon si fuera necesario (aquí no suele serlo)
    weights = total_samples / (num_classes * counts_tensor)
    
    # 3. Reporte visual
    class_map = {0: "BEN", 1: "MEL", 2: "BCC", 3: "SCC+"} # Mapeo opcional para log
    print(f"   -> Total muestras: {int(total_samples)}")
    for i, w in enumerate(weights):
        cls_name = class_map.get(i, str(i))
        count = int(counts_tensor[i])
        print(f"   -> Clase {i} [{cls_name}]: {count:5d} imgs | Peso: {w:.4f}")
        
    return weights.to(device)