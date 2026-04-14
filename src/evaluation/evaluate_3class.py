# src/evaluation/evaluate_3class.py

import torch
import numpy as np

# IMPORTANTE: Importamos del nuevo script de 3 clases
from src.evaluation.metrics_3class import (
    metrics_headA,
    metrics_headB
)

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evalúa el modelo en un dataloader.
    
    Args:
        model: modelo PyTorch con dos heads
        dataloader: DataLoader de validación
        device: cpu / cuda
    
    Returns:
        dict con métricas de Head A y Head B
    """
    model.eval()

    yA_true, yA_pred, yA_prob = [], [], []
    yB_true, yB_pred = [], []

    for images, yA, yB in dataloader:
        images = images.to(device)
        yA = yA.to(device)
        yB = yB.to(device)

        # Forward
        outA, outB = model(images)

        # ---------------- Head A ----------------
        probA = torch.sigmoid(outA)
        predA = (probA >= 0.3).long()

        yA_true.extend(yA.cpu().numpy())
        yA_pred.extend(predA.cpu().numpy())
        yA_prob.extend(probA.cpu().numpy())

        # ---------------- Head B ----------------
        predB = torch.argmax(outB, dim=1)

        yB_true.extend(yB.cpu().numpy())
        yB_pred.extend(predB.cpu().numpy())

    # Convertir a numpy
    yA_true = np.array(yA_true)
    yA_pred = np.array(yA_pred)
    yA_prob = np.array(yA_prob)

    yB_true = np.array(yB_true)
    yB_pred = np.array(yB_pred)

    # Calcular métricas (pasando explícitamente num_classes=3 a Head B)
    results = {
        "headA": metrics_headA(yA_true, yA_pred, yA_prob),
        "headB": metrics_headB(yB_true, yB_pred, num_classes=3)
    }

    return results