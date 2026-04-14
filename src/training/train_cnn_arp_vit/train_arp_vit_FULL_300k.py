# src/training/train_cnn_arp_vit/train_arp_vit_FULL_300k.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # <--- AMP para velocidad
import numpy as np
import os

from src.utils.logger import ExperimentLogger
from src.evaluation.metrics import metrics_headA, metrics_headB
from src.utils.class_weights import compute_class_weights

# Importamos dataset y modelo
from src.data.dataset_hybrid import HybridTripleDataset
from src.models.cnn_rgb_arp_vit.arp_vit_model import TripleHybridModel

# Importamos transformaciones
from src.data.transforms import (
    get_train_transforms, get_eval_transforms,
    get_train_transforms_arp, get_eval_transforms_arp
)

# Importamos MixUp (Aunque esté desactivado, la función lo requiere importar)
from src.training.train_cnn_arp_vit.MixUp import mixup_data, mixup_criterion

# -------------------------------------------------------------------------
# 1. CLASE FOCAL LOSS
# -------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss: Vital ahora que no usamos MixUp. 
        Ayudará a que el modelo no ignore los BCC/SCC.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -------------------------------------------------------------------------
# 2. FUNCIÓN DE ÉPOCA
# -------------------------------------------------------------------------
def run_epoch(model, dataloader, criterion_A, criterion_B, optimizer, scaler, device, is_train=True, accumulation_steps=1, use_mixup=False):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_yA_true, all_yA_pred, all_yA_prob = [], [], []
    all_yB_true, all_yB_pred = [], []

    if is_train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for i, (img_rgb, img_arp, yA, yB) in enumerate(dataloader):
            img_rgb = img_rgb.to(device)
            img_arp = img_arp.to(device)
            yA = yA.float().to(device)
            yB = yB.long().to(device)

            # --- AMP (Mixed Precision) ---
            with autocast(enabled=is_train):
                # Lógica MixUp (Solo si use_mixup es True, que ahora será False)
                if is_train and use_mixup:
                    img_rgb, img_arp, yA_a, yA_b, yB_a, yB_b, lam = mixup_data(
                        img_rgb, img_arp, yA, yB, alpha=0.2, use_cuda=(device.type == 'cuda')
                    )
                    out_A, out_B = model(img_rgb, img_arp)
                    out_A = out_A.squeeze(1)

                    loss_A = mixup_criterion(criterion_A, out_A, yA_a, yA_b, lam)
                    loss_B = mixup_criterion(criterion_B, out_B, yB_a, yB_b, lam)
                
                else:
                    # ENTRENAMIENTO ESTÁNDAR (Lo que se ejecutará ahora)
                    out_A, out_B = model(img_rgb, img_arp)
                    out_A = out_A.squeeze(1)
                    loss_A = criterion_A(out_A, yA)
                    loss_B = criterion_B(out_B, yB)

                loss = loss_A + loss_B 

                if is_train:
                    loss = loss / accumulation_steps

            # --- BACKPROP CON SCALER ---
            if is_train:
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            running_loss += loss.item() * (accumulation_steps if is_train else 1)

            # --- MÉTRICAS ---
            probA = torch.sigmoid(out_A.float()) 
            predA = (probA >= 0.5).long()
            
            all_yA_true.extend(yA.cpu().detach().numpy())
            all_yA_pred.extend(predA.cpu().detach().numpy())
            all_yA_prob.extend(probA.cpu().detach().numpy())

            predB = torch.argmax(out_B.float(), dim=1)
            all_yB_true.extend(yB.cpu().detach().numpy())
            all_yB_pred.extend(predB.cpu().detach().numpy())

    if is_train and len(dataloader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(dataloader)
    results_A = metrics_headA(np.array(all_yA_true), np.array(all_yA_pred), np.array(all_yA_prob))
    results_B = metrics_headB(np.array(all_yB_true), np.array(all_yB_pred))

    return epoch_loss, {"headA": results_A, "headB": results_B}

# -------------------------------------------------------------------------
# 3. ENTRENAMIENTO PRINCIPAL (330k)
# -------------------------------------------------------------------------
def train_full_scale(
    learning_rate=1e-4,   
    batch_size=32,        
    accumulation_steps=4, # Batch efectivo = 128
    num_epochs=15,        
    patience=4,           # Early Stopping Patience
    experiment_path="cnn_rgb_arp_vit/FULL_RUN_300k_FocalLoss_NoMixUp", 
    use_mixup=False       # <--- DESACTIVADO POR DEFECTO
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 [CHAMPIONS LEAGUE RUN] 330k Images | FocalLoss + AMP | MixUp={use_mixup}")
    
    # CORRECCIÓN: Sintaxis moderna para evitar FutureWarning
    scaler = torch.amp.GradScaler('cuda')

    config = {
        "model": "TripleHybrid (ResNet18 + ViT + ARP)",
        "dataset": "Experiment 300k (330k Train / 5k Val)",
        "lr": learning_rate,
        "loss": "FocalLoss (Gamma=2.0)",
        "optimizer": "AdamW + AMP",
        "mixup": use_mixup
    }
    logger = ExperimentLogger(experiment_name=experiment_path, config=config)

    # 1. RUTAS CORREGIDAS
    # Dataset busca dentro de 'data/Splitted_data/', así que solo ponemos la carpeta del experimento
    csv_train_path = "experiment_300k/train.csv"
    csv_val_path = "experiment_300k/val.csv"
    
    print(f"📂 Training Set: {csv_train_path}")
    print(f"📂 Validation Set: {csv_val_path}")

    # Carga de datasets
    train_dataset = HybridTripleDataset(
        csv_train_path, 
        transforms_rgb=get_train_transforms(),
        transforms_arp=get_train_transforms_arp()
    )
    val_dataset = HybridTripleDataset(
        csv_val_path, 
        transforms_rgb=get_eval_transforms(),
        transforms_arp=get_eval_transforms_arp()
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2        
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 2. MODELO
    ruta_pesos_arp = "experiments/cnn_arp/experiment_0001_baseline/results/run_20260210_175038_LR0.0001_BS32/best_model.pth" 
    
    model = TripleHybridModel(
        num_classes_headB=4, 
        pretrained_rgb=True,
        arp_pretrained_path=ruta_pesos_arp
    ).to(device)

    # 3. LOSS & OPTIMIZER
    print("⚖️ Calculando pesos de clase para Focal Loss...")
    class_weights = compute_class_weights(train_dataset.df, device)
    
    criterion_headA = nn.BCEWithLogitsLoss()
    criterion_headB = FocalLoss(alpha=class_weights, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    # Scheduler Patience = 2 (Si en 2 épocas no mejora, baja LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_val_loss = float('inf')
    best_f1 = 0.0
    patience_counter = 0

    # 4. BUCLE
    for epoch in range(num_epochs):
        print(f"  Epoch {epoch + 1}/{num_epochs}...", end="")
        
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion_headA, criterion_headB, optimizer, scaler, device, 
            is_train=True, accumulation_steps=accumulation_steps, use_mixup=use_mixup
        )
        
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion_headA, criterion_headB, optimizer, scaler, device, 
            is_train=False, use_mixup=False
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f" Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {val_metrics['headB']['macro_f1']:.4f} | LR: {current_lr:.2e}")

        logger.log_scalar("Loss/Train", train_loss, epoch)
        logger.log_scalar("Loss/Val", val_loss, epoch)
        logger.log_full_report(train_metrics, epoch, phase="Train")
        logger.log_full_report(val_metrics, epoch, phase="Val")

        current_f1 = val_metrics['headB']['macro_f1']
        best_f1 = logger.save_checkpoint(model, current_f1, best_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No Mejora: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("    🛑 Early Stopping")
                break

    logger.close()

if __name__ == "__main__":
    # Confirmamos: NO MixUp
    train_full_scale(use_mixup=False)