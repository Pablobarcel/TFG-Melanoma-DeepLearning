# src/training/train_cnn_arp_vit/train_arp_vit_3class.py

# --- PARCHE ANTI-CUELGUES ---
import matplotlib
matplotlib.use('Agg')
# ----------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.utils.logger import ExperimentLogger
from src.evaluation.metrics_3class import metrics_headA, metrics_headB
from src.utils.class_weights import compute_class_weights

from src.data.dataset_hybrid_3class import HybridTripleDataset3Class
from src.models.cnn_rgb_arp_vit.arp_vit_model_3class import TripleHybridModel3Class

# 🚨 IMPORTANTE: Añadimos get_clinical_bce_loss a los imports
from src.utils.losses import FocalLoss, get_clinical_bce_loss

from src.data.transforms import (
    get_train_transforms, get_eval_transforms,
    get_train_transforms_arp, get_eval_transforms_arp
)

def run_epoch(model, dataloader, criterion_A, criterion_B, optimizer, device, is_train=True, accumulation_steps=1):
    if is_train: model.train()
    else: model.eval()

    running_loss = 0.0
    all_yA_true, all_yA_pred, all_yA_prob = [], [], []
    all_yB_true, all_yB_pred = [], []

    if is_train: optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for i, (img_rgb, img_arp, yA, yB) in enumerate(dataloader):
            img_rgb = img_rgb.to(device)
            img_arp = img_arp.to(device)
            yA = yA.float().to(device)
            yB = yB.long().to(device)

            out_A, out_B = model(img_rgb, img_arp)
            out_A = out_A.squeeze(1)
            
            loss_A = criterion_A(out_A, yA)
            loss_B = criterion_B(out_B, yB)
            loss = loss_A + (2.0 * loss_B)

            if is_train:
                loss = loss / accumulation_steps 
                loss.backward() 
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * (accumulation_steps if is_train else 1)

            probA = torch.sigmoid(out_A)
            # 🚨 UMBRAL CLÍNICO
            predA = (probA >= 0.3).long()
            all_yA_true.extend(yA.cpu().detach().numpy())
            all_yA_pred.extend(predA.cpu().detach().numpy())
            all_yA_prob.extend(probA.cpu().detach().numpy())

            predB = torch.argmax(out_B, dim=1)
            all_yB_true.extend(yB.cpu().detach().numpy())
            all_yB_pred.extend(predB.cpu().detach().numpy())

    if is_train and len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(dataloader)
    results_A = metrics_headA(np.array(all_yA_true), np.array(all_yA_pred), np.array(all_yA_prob))
    results_B = metrics_headB(np.array(all_yB_true), np.array(all_yB_pred))

    return epoch_loss, {"headA": results_A, "headB": results_B}

def train_triple_model(
    learning_rate=5e-5,
    batch_size=32,       
    accumulation_steps=4,
    num_epochs=20, 
    patience=4,
    experiment_path="cnn_rgb_arp_vit/experiment_0004_200k_3classes_fusionConocimientos" 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_batch = batch_size * accumulation_steps
    print(f"\n🚀 [Triple Hybrid Run - 3 CLASSES NMSC] Batch Físico: {batch_size} | Matemático: {effective_batch} | Clinical Mode")

    config = {
        "model": "TripleHybrid3Class (Full Fusion)",
        "lr": learning_rate,
        "batch_size_effective": effective_batch,
        "optimizer": "AdamW + Scheduler + GradAccum",
        "loss": "Focal Loss (Head B) + Weighted BCE (Head A)"
    }
    logger = ExperimentLogger(experiment_name=experiment_path, config=config)

    train_dataset = HybridTripleDataset3Class(
        "experiment_200k_3classes/train.csv", 
        transforms_rgb=get_train_transforms(),
        transforms_arp=get_train_transforms_arp()
    )
    val_dataset = HybridTripleDataset3Class(
        "experiment_200k_3classes/val.csv", 
        transforms_rgb=get_eval_transforms(),
        transforms_arp=get_eval_transforms_arp()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # CAMBIAR RUTA CUANDO ENTRENE LOS DOS MODELOS DE 200K
    ruta_pesos_arp = "experiments/cnn_arp/experiment_0003_200k_3classes_FocalLoss/results/run_20260313_111503_LR0.0001_BS32/best_model.pth"
    ruta_pesos_rgb = "experiments/cnn_vit_rgb/experiment_0001_200k_3classes_FocalLoss/results/run_20260320_125703/best_model.pth"
    
    model = TripleHybridModel3Class(
        num_classes_headB=3,
        pretrained_rgb=False,
        arp_pretrained_path=ruta_pesos_arp,
        rgb_pretrained_path=ruta_pesos_rgb
    ).to(device)
    
    # 🚨 CONFIGURACIÓN CLÍNICA 🚨
    class_weights = compute_class_weights(train_dataset.df, device)
    
    # Head A (Falsos Negativos) - CÁLCULO AUTOMÁTICO CENTRALIZADO
    criterion_headA = get_clinical_bce_loss(train_dataset.df, factor_seguridad=2.0, device=device)

    pesos_clinicos_B = class_weights.clone()
    pesos_clinicos_B[1] *= 2.0  
    pesos_clinicos_B[2] *= 2.0  
    criterion_headB = FocalLoss(weight=pesos_clinicos_B, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_loss = float('inf')
    patience_counter = 0
    best_auc = 0.0

    for epoch in range(num_epochs):
        print(f"  Epoch {epoch + 1}/{num_epochs}...", end="")
        
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion_headA, criterion_headB, optimizer, device, 
            is_train=True, accumulation_steps=accumulation_steps
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion_headA, criterion_headB, optimizer, device, 
            is_train=False
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f" Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {val_metrics['headB']['macro_f1']:.4f} | LR: {current_lr:.2e}")

        logger.log_scalar("Loss/Train", train_loss, epoch)
        logger.log_scalar("Loss/Val", val_loss, epoch)
        logger.log_full_report(train_metrics, epoch, phase="Train")
        logger.log_full_report(val_metrics, epoch, phase="Val")

        current_auc = val_metrics['headA']['auc'] if not np.isnan(val_metrics['headA']['auc']) else 0.0
        
        logger.update_csv({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_metrics['headB']['macro_f1'], "lr": current_lr
        })
        
        best_auc = logger.save_checkpoint(model, current_auc, best_auc)

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
    train_triple_model()