# src/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multiclase.
    Combina el balanceo por pesos (class_weights) con el factor de enfoque (gamma)
    para penalizar fuertemente los ejemplos difíciles de clasificar (Melanomas sutiles, etc).
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Args:
            weight (Tensor): Pesos para cada clase (para lidiar con el desbalance).
            gamma (float): Factor de enfoque. A mayor gamma, más ignora los casos fáciles. (2.0 es el estándar).
            reduction (str): 'mean', 'sum' o 'none'.
        """
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Calculamos la Cross Entropy normal (con pesos si los hay) pero sin reducirla
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # 2. Obtenemos la probabilidad 'p_t' que el modelo asignó a la clase correcta
        # Como ce_loss = -log(p_t), entonces p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. Aplicamos la fórmula del Focal Loss: (1 - p_t)^gamma * CE_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Reducción
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_clinical_bce_loss(df, factor_seguridad=2.0, device='cuda'):
    """
    Lee el DataFrame, calcula el ratio de Falsos Negativos y devuelve 
    la función BCEWithLogitsLoss ya configurada con el pos_weight clínico.
    """
    # 1. Leemos los conteos de la columna binaria
    conteo_A = df["head_A_label"].value_counts()
    num_benignos = conteo_A.get(0, 1) # Evita divisiones por cero
    num_malignos = conteo_A.get(1, 1)
    
    # 2. Calculamos el castigo
    ratio_natural = num_benignos / num_malignos
    peso_clinico_A = torch.tensor([ratio_natural * factor_seguridad]).to(device)
    
    print(f"⚖️ [Head A Loss] Benignos: {num_benignos} | Malignos: {num_malignos}")
    print(f"⚖️ [Head A Loss] Aplicando pos_weight clínico: {peso_clinico_A.item():.2f}")
    
    # 3. Devolvemos la función de PyTorch ya lista para usar
    return nn.BCEWithLogitsLoss(pos_weight=peso_clinico_A)