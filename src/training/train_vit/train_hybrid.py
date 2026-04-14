# src/training/train_vit/train_hybrid.py

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

from src.utils.logger import ExperimentLogger
from src.evaluation.metrics import metrics_headA, metrics_headB
from src.data.dataset_rgb import RGBDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.cnn_vit_rgb.hybrid_model import HybridViTCNN
from src.utils.class_weights import compute_class_weights
# 🚨 Cambiado para incluir Focal Loss
from src.utils.losses import FocalLoss, get_clinical_bce_loss

def run_epoch(model, dataloader, criterion_A, criterion_B, optimizer, device, is_train=True, accumulation_steps=1):
    if is_train: model.train()
    else: model.eval()

    running_loss = 0.0
    all_yA_true, all_yA_pred, all_yA_prob = [], [], []
    all_yB_true, all_yB_pred = [], []

    if is_train: optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for i, (images, yA, yB) in enumerate(dataloader):
            images = images.to(device)
            yA = yA.float().to(device)
            yB = yB.long().to(device)

            out_A, out_B = model(images)
            
            loss_A = criterion_A(out_A, yA)
            loss_B = criterion_B(out_B, yB)
            loss = loss_A + loss_B

            if is_train:
                loss = loss / accumulation_steps 
                loss.backward() 
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * (accumulation_steps if is_train else 1)

            probA = torch.sigmoid(out_A)
            # 🚨 UMBRAL NEUTRO 0.5
            predA = (probA >= 0.5).long()
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

# Falta por hacer la busqueda de hiperparametros con optuna
def train_hybrid_model(
    learning_rate=1e-4,   # LR Base de partida (se afinará si haces tune de 4 clases)
    weight_decay=1e-2,
    batch_size=16,        
    accumulation_steps=4, 
    num_epochs=20, 
    patience=5,
    experiment_path="cnn_vit_rgb/experiment_4class_final" 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_batch = batch_size * accumulation_steps
    print(f"\n🚀 [Hybrid Run 4 CLASSES] Batch Fisico: {batch_size} | Matemático: {effective_batch}")

    config = {
        "model": "HybridViTCNN (ResNet18 + ViT-Tiny) - 4 Clases",
        "lr_heads": learning_rate,
        "lr_backbones": learning_rate / 100,
        "batch_size_effective": effective_batch,
        "optimizer": "AdamW + Diff LR + GradAccum",
        "loss": "Focal Loss (Head B) + Weighted BCE (Head A)"
    }
    
    logger = ExperimentLogger(experiment_name=experiment_path, config=config)

    train_dataset = RGBDataset("Final_dataset_4class_200k/train.csv", get_train_transforms())
    val_dataset = RGBDataset("Final_dataset_4class_200k/val.csv", get_eval_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridViTCNN(num_classes_headB=4, pretrained=True).to(device)

    # 🚨 CONFIGURACIÓN DE PÉRDIDAS UNIFICADA 🚨
    class_weights = compute_class_weights(train_dataset.df, device)
    criterion_headA = get_clinical_bce_loss(train_dataset.df, factor_seguridad=2.0, device=device)
    
    # 🚨 Aplicamos FocalLoss universalmente
    criterion_headB = FocalLoss(weight=class_weights, gamma=2.0)

    # 🧠 DIFFERENTIAL LEARNING RATE
    optimizer = optim.AdamW([
        {'params': model.cnn_backbone.parameters(), 'lr': learning_rate / 100},
        {'params': model.vit_backbone.parameters(), 'lr': learning_rate / 100},
        {'params': model.head_A.parameters(), 'lr': learning_rate},
        {'params': model.head_B.parameters(), 'lr': learning_rate}
    ], weight_decay=weight_decay)
    
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
        current_lr_head = optimizer.param_groups[2]['lr']

        print(f" Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {val_metrics['headB']['macro_f1']:.4f} | LR_Head: {current_lr_head:.2e}")

        logger.log_scalar("Loss/Train", train_loss, epoch)
        logger.log_scalar("Loss/Val", val_loss, epoch)
        logger.log_scalar("Parameters/LearningRate_Head", current_lr_head, epoch)
        logger.log_full_report(train_metrics, epoch, phase="Train")
        logger.log_full_report(val_metrics, epoch, phase="Val")

        current_auc = val_metrics['headA']['auc'] if not np.isnan(val_metrics['headA']['auc']) else 0.0
        
        logger.update_csv({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_metrics['headB']['macro_f1'], "lr_head": current_lr_head
        })
        
        best_auc = logger.save_checkpoint(model, current_auc, best_auc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("    🛑 Early Stopping")
                break

    logger.close()

if __name__ == "__main__":
    train_hybrid_model(
        batch_size=16,       
        accumulation_steps=4 
    )