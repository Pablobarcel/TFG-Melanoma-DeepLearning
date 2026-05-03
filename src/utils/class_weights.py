# src/utils/class_weights.py
import torch
import pandas as pd

def compute_class_weights(df, device, label_col="target"):
    """
    Calcula pesos para CrossEntropyLoss adaptados al nuevo esquema de 4 clases:
    0: BEN (Benignos: NV+BKL+BG), 1: MEL, 2: BCC, 3: SCC.
    """
    print(f"\n⚖️ [Utils] Calculando pesos (4 Clases) para columna: '{label_col}'...")
    
    if label_col not in df.columns:
        raise ValueError(f"❌ La columna {label_col} no existe en el DataFrame.")

    # 1. Definir el mapeo de agrupación (6 -> 4 clases)
    # NV(0), BKL(4), BG(5) -> BEN(0)
    # MEL(1) -> 1, BCC(2) -> 2, SCC(3) -> 3
    label_map = {0:0, 4:0, 5:0, 1:1, 2:2, 3:3}
    
    # 2. Aplicar el mapeo a una serie temporal para el cálculo de frecuencias
    # Esto asegura que el peso se calcule sobre la distribución REAL que verá el modelo
    mapped_labels = df[label_col].map(label_map)
    
    # 3. Contar frecuencias de las 4 clases resultantes
    counts = mapped_labels.value_counts().sort_index()
    counts_tensor = torch.tensor(counts.values, dtype=torch.float)
    
    # 4. Fórmula de Balanceo Inverso: total / (n_clases * f_clase)
    total_samples = counts_tensor.sum()
    num_classes = len(counts_tensor) # Debería ser 4
    
    weights = total_samples / (num_classes * counts_tensor)
    
    # 5. Reporte visual para tu memoria del TFG
    class_names = {0: "BEN (NV+BKL+BG)", 1: "MEL", 2: "BCC", 3: "SCC"}
    
    for cls_idx, w in enumerate(weights):
        name = class_names.get(cls_idx, f"Clase {cls_idx}")
        print(f"   - {name}: {counts_tensor[cls_idx]:.0f} muestras -> Peso: {w:.4f}")
        
    return weights.to(device)