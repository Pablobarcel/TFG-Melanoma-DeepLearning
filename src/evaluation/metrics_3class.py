# src/evaluation/metrics_3class.py

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
        y_true, y_pred, pos_label=1, zero_division=0
    )

    # AUC-ROC (si hay ambas clases presentes)
    try:
        results["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        results["auc"] = np.nan

    # Matriz de confusión
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return results


# ==========================================================
# Head B — Multiclase (BEN, MEL, NMSC) -> 3 Clases
# ==========================================================

def metrics_headB(y_true, y_pred, num_classes=3):
    """
    Métricas para Head B (clasificación multiclase 3 clases).
    
    Args:
        y_true (array): etiquetas reales (0=BEN, 1=MEL, 2=NMSC)
        y_pred (array): predicciones
        num_classes (int): número de clases, por defecto 3.
    
    Returns:
        dict con métricas de análisis
    """
    results = {}
    
    # Aseguramos que las métricas sepan exactamente cuántas clases hay
    class_labels = list(range(num_classes))

    # Accuracy global
    results["accuracy"] = accuracy_score(y_true, y_pred)

    # Recall por clase
    results["recall_per_class"] = recall_score(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )

    # Macro Recall
    results["macro_recall"] = recall_score(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )

    # Macro F1
    results["macro_f1"] = f1_score(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )

    # Matriz de confusión (forzada a 3x3)
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=class_labels)

    return results