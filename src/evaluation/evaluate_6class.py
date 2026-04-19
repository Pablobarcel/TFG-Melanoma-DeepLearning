# src/evaluation/evaluate.py
import torch
import numpy as np
import torch.nn as nn
from src.evaluation.metrics_6class import metrics_headA, metrics_headB

@torch.no_grad()
def evaluate(model, dataloader, device, criterion_A, criterion_B, threshold=0.5):
    model.eval()
    
    # Listas para Head A (Binario)
    yA_true, yA_pred, yA_prob = [], [], []
    # Listas para Head B (Multiclase)
    yB_true, yB_pred = [], []
    
    running_loss = 0.0

    for feats, targets_A, targets_B in dataloader:
        feats = feats.to(device)
        targets_A = targets_A.to(device)
        targets_B = targets_B.to(device)
        
        # Forward: El modelo devuelve dos salidas
        outA, outB = model(feats)
        
        # Cálculo de Pérdida combinada (igual que en el entrenamiento)
        lossA = criterion_A(outA.view(-1), targets_A)
        lossB = criterion_B(outB, targets_B)
        loss = lossA + lossB
        running_loss += loss.item()
        
        # --- Procesamiento Head A ---
        probA = torch.sigmoid(outA).view(-1)
        predA = (probA >= threshold).long()
        
        yA_true.extend(targets_A.cpu().numpy())
        yA_pred.extend(predA.cpu().numpy())
        yA_prob.extend(probA.cpu().numpy())
        
        # --- Procesamiento Head B ---
        predB = torch.argmax(outB, dim=1)
        
        yB_true.extend(targets_B.cpu().numpy())
        yB_pred.extend(predB.cpu().numpy())

    # Cálculo de métricas finales
    metrics = {
        "headA": metrics_headA(np.array(yA_true), np.array(yA_pred), np.array(yA_prob)),
        "headB": metrics_headB(np.array(yB_true), np.array(yB_pred))
    }
    
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, metrics