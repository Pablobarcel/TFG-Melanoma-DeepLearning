# src/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multiclase.
    Penaliza fuertemente los ejemplos difíciles de clasificar.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def get_clinical_bce_loss(df, factor_seguridad=2.0, device='cuda'):
    """
    Calcula el desbalance binario y devuelve la función BCEWithLogitsLoss.
    Le da prioridad a no fallar en los cánceres.
    """
    # Usamos la nueva columna target_binary
    conteo_A = df["target_binary"].value_counts()
    num_benignos = conteo_A.get(0, 1) 
    num_malignos = conteo_A.get(1, 1)
    
    ratio_natural = num_benignos / num_malignos
    castigo_final = ratio_natural * factor_seguridad
    
    print(f"⚖️ [Losses] Ratio Benignos/Malignos: {ratio_natural:.2f}")
    print(f"⚖️ [Losses] Castigo final pos_weight aplicado: {castigo_final:.2f}")
    
    pos_weight_tensor = torch.tensor([castigo_final], dtype=torch.float).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)