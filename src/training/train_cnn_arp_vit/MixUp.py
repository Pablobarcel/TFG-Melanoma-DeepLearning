# src/training/train_cnn_arp_vit/MixUp.py

import torch
import numpy as np

def mixup_data(x1, x2, y_a, y_b, alpha=0.2, use_cuda=True):
    '''
    Devuelve datos mixtos (RGB y ARP) y pares de etiquetas.
    
    Args:
        x1: Batch de imágenes RGB
        x2: Batch de imágenes ARP
        y_a: Etiquetas Head A (Binarias)
        y_b: Etiquetas Head B (Multiclase)
        alpha: Parámetro de la distribución Beta (0.2 recomendado para ImageNet)
        use_cuda: Booleano para usar GPU en los índices
        
    Returns:
        mixed_x1, mixed_x2, y_a_orig, y_a_shuf, y_b_orig, y_b_shuf, lam
    '''
    if alpha > 0:
        # Generamos lambda de una distribución Beta (forma de U)
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # 1. Mezclamos las imágenes RGB
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    
    # 2. Mezclamos las imágenes ARP (¡Con el MISMO índice y lambda!)
    # Esto es vital: La textura RGB debe coincidir con la geometría ARP
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]

    # 3. Preparamos los pares de etiquetas
    y_a_orig, y_a_shuffled = y_a, y_a[index]
    y_b_orig, y_b_shuffled = y_b, y_b[index]

    return mixed_x1, mixed_x2, y_a_orig, y_a_shuffled, y_b_orig, y_b_shuffled, lam

def mixup_criterion(criterion, pred, y_orig, y_shuffled, lam):
    '''
    Calcula la Loss ponderada: lam * Loss(original) + (1-lam) * Loss(shuffled)
    '''
    return lam * criterion(pred, y_orig) + (1 - lam) * criterion(pred, y_shuffled)