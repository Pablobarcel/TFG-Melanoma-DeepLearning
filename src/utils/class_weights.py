# src/utils/class_weights.py
import torch
import pandas as pd
import numpy as np # Añadimos numpy para la potencia

def compute_class_weights(df, device, label_col="target", smoothing=0.5):
    """
    Calcula pesos con suavizado para evitar que el modelo se vuelva 
    demasiado agresivo (reduciendo Falsos Positivos).
    smoothing: 1.0 es balanceo puro, 0.0 es sin pesos. 0.65 es un buen equilibrio.
    """
    print(f"\n⚖️ [Utils] Calculando pesos suavizados (Factor: {smoothing})...")
    
    if label_col not in df.columns:
        raise ValueError(f"❌ La columna {label_col} no existe en el DataFrame.")

    label_map = {0:0, 4:0, 5:0, 1:1, 2:2, 3:3}
    mapped_labels = df[label_col].map(label_map)
    
    counts = mapped_labels.value_counts().sort_index()
    counts_tensor = torch.tensor(counts.values, dtype=torch.float)
    
    total_samples = counts_tensor.sum()
    num_classes = len(counts_tensor)
    
    # 1. Calculamos el peso base (Inverso)
    base_weights = total_samples / (num_classes * counts_tensor)
    
    # 2. Aplicamos SUAVIZADO (Potencia)
    # Esto reduce los pesos altos y sube los bajos hacia la unidad
    weights = torch.pow(base_weights, smoothing)
    
    class_names = {0: "BEN", 1: "MEL", 2: "BCC", 3: "SCC"}
    for cls_idx, w in enumerate(weights):
        name = class_names.get(cls_idx, f"Clase {cls_idx}")
        print(f"   - {name}: Peso original: {base_weights[cls_idx]:.4f} -> Suavizado: {w:.4f}")
        
    return weights.to(device)