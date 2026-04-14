# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    f1_score
)


# ==========================================================
# Head A — Binario (BEN vs MAL)
# ==========================================================

def metrics_headA(y_true, y_pred, y_prob):
    """
    Métricas para Head A (clasificación binaria).
    
    Args:
        y_true (array): etiquetas reales (0=BEN, 1=MAL)
        y_pred (array): predicción binaria
        y_prob (array): probabilidad de clase positiva (MAL)
    
    Returns:
        dict con métricas principales
    """
    results = {}

    # Accuracy (referencia)
    results["accuracy"] = accuracy_score(y_true, y_pred)

    # Recall / Sensibilidad de melanoma (clase 1)
    results["recall_malignant"] = recall_score(
        y_true, y_pred, pos_label=1
    )

    # AUC-ROC (si hay ambas clases presentes)
    try:
        results["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        results["auc"] = np.nan

    # Matriz de confusión
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return results


# ==========================================================
# Head B — Multiclase (BEN, MEL, BCC, SCC+)
# ==========================================================

def metrics_headB(y_true, y_pred):
    """
    Métricas para Head B (clasificación multiclase).
    
    Args:
        y_true (array): etiquetas reales (0–3)
        y_pred (array): predicciones
    
    Returns:
        dict con métricas de análisis
    """
    results = {}

    # Accuracy global
    results["accuracy"] = accuracy_score(y_true, y_pred)

    # Recall por clase
    results["recall_per_class"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro Recall
    results["macro_recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Macro F1
    results["macro_f1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Matriz de confusión
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return results
