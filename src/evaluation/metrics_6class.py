# src/evaluation/metrics_6class.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score
)

# ==========================================================
# Head A — Binario (Benigno vs Maligno)
# ==========================================================
def metrics_headA(y_true, y_pred, y_prob):
    results = {}
    
    # Accuracy de referencia
    results["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Sensibilidad (Recall) - Muy importante: capacidad de detectar malignos
    results["recall_malignant"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # AUC-ROC: Mide la calidad de la probabilidad asignada
    try:
        results["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        results["auc"] = 0.0

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return results

# ==========================================================
# Head B — Multiclase (6 Clases)
# ==========================================================
def metrics_headB(y_true, y_pred):
    results = {}
    class_labels = [0, 1, 2, 3] # Antes era 6
    class_names = ["BEN", "MEL", "BCC", "SCC"] # Agrupados

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Recall específico por cada una de las 6 patologías
    recalls = recall_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)
    for i, name in enumerate(class_names):
        results[f"recall_{name}"] = recalls[i]

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    return results