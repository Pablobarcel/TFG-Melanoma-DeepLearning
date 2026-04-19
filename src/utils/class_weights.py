# src/utils/class_weights.py
import torch
import pandas as pd

def compute_class_weights(df, device, label_col="target"):
    """
    Calcula automáticamente los pesos para CrossEntropyLoss basándose en el 
    desbalance de clases del DataFrame.
    """
    print(f"\n⚖️ [Utils] Calculando pesos para columna: '{label_col}'...")
    
    if label_col not in df.columns:
        raise ValueError(f"❌ La columna {label_col} no existe en el DataFrame.")

    counts = df[label_col].value_counts().sort_index()
    counts_tensor = torch.tensor(counts.values, dtype=torch.float)
    
    # Fórmula de Balanceo Inverso
    total_samples = counts_tensor.sum()
    num_classes = len(counts_tensor)
    
    weights = total_samples / (num_classes * counts_tensor)
    
    # Reporte visual adaptado a 6 clases
    class_map = {0: "NV", 1: "MEL", 2: "BCC", 3: "SCC", 4: "BKL", 5: "BG"}
    
    for cls_idx, w in enumerate(weights):
        cls_name = class_map.get(cls_idx, f"Clase {cls_idx}")
        print(f"   - {cls_name} (Clase {cls_idx}): {counts_tensor[cls_idx]:.0f} imgs -> Peso: {w:.4f}")
        
    return weights.to(device)