# src/training/train_arp/train_arp_kfold.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import os
from datetime import datetime
from torchvision import transforms # Provisional

from src.data.arp.dataset_arp import ARPDataset6Class
from src.models.cnn_arp.arp_model_6class import ARPCNN6Class
from src.utils.class_weights import compute_class_weights
from src.utils.losses import get_clinical_bce_loss 
from src.utils.logger import ExperimentLogger
from src.evaluation.evaluate_6class import evaluate
from src.evaluation.metrics_6class import metrics_headA, metrics_headB

def train_arp_kfold():
    print("="*60)
    print(" 🚀 INICIANDO ENTRENAMIENTO CNN ARP (6 Clases + Binario)")
    print("="*60)

    # --- 1. CONFIGURACIÓN ---
    CSV_PATH = "C:/TFG/data/Original_Data/ISIC_FINAL/dataset_train_cv.csv"
    IMAGES_DIR = "C:/TFG/src/data/processed/images_ARP_ISIC"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_FOLDS = 5
    EPOCHS = 40 # Las CNN tardan más, bajamos a 40
    BATCH_SIZE = 128
    LR = 1e-4 # Learning rate estándar hasta que pasemos Optuna
    WD = 1e-3

    print(f"🖥️ Dispositivo: {DEVICE}")
    df = pd.read_csv(CSV_PATH)
    
    shared_run_name = f"run_ARP_KFold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = ExperimentLogger(experiment_name="arp_cnn_kfold", run_name=shared_run_name)
    
    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS)
    splits = list(sgkf.split(df, y=df['target'], groups=df['master_id']))

    metricas_finales_f1 = []
    
    # Acumulador maestro
    avg_metrics = {
        epoch: {
            "train_loss": 0.0, "val_loss": 0.0,
            "train_acc_A": 0.0, "val_acc_A": 0.0, "train_rec_A": 0.0, "val_rec_A": 0.0, "train_auc_A": 0.0, "val_auc_A": 0.0,
            "train_acc_B": 0.0, "val_acc_B": 0.0, "train_rec_B": 0.0, "val_rec_B": 0.0, "train_f1_B": 0.0, "val_f1_B": 0.0,
            "cm_train_A": np.zeros((2, 2)), "cm_val_A": np.zeros((2, 2)),
            "cm_train_B": np.zeros((6, 6)), "cm_val_B": np.zeros((6, 6))
        } for epoch in range(EPOCHS)
    }

    # Transformaciones base (las ampliaremos en el próximo paso)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # ARP es 1 canal
    ])

    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_prefix = f"Fold_{fold+1}"
        print(f"\n" + "═"*60)
        print(f" 📂 INICIANDO {fold_prefix}/{NUM_FOLDS}")
        print("═"*60)
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        train_ds = ARPDataset6Class(df_train, IMAGES_DIR, transforms=base_transform)
        val_ds = ARPDataset6Class(df_val, IMAGES_DIR, transforms=base_transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = ARPCNN6Class(num_classes_multiclass=6).to(DEVICE)
        
        criterion_A = get_clinical_bce_loss(df_train, factor_seguridad=2.0, device=DEVICE)
        w_multi = compute_class_weights(df_train, DEVICE, label_col="target")
        criterion_B = nn.CrossEntropyLoss(weight=w_multi) # Optuna dirá si usamos FocalLoss después
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_val_f1 = 0.0

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            yA_true_train, yA_pred_train, yA_prob_train = [], [], []
            yB_true_train, yB_pred_train = [], []
            
            for images, yA, yB in train_loader:
                images, yA, yB = images.to(DEVICE), yA.to(DEVICE), yB.to(DEVICE)
                
                optimizer.zero_grad()
                outA, outB = model(images)
                
                lossA = criterion_A(outA.view(-1), yA)
                lossB = criterion_B(outB, yB)
                total_loss = lossA + lossB 
                
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
                
                with torch.no_grad():
                    probA = torch.sigmoid(outA).view(-1)
                    predA = (probA >= 0.5).long()
                    predB = torch.argmax(outB, dim=1)
                    
                    yA_true_train.extend(yA.cpu().numpy())
                    yA_pred_train.extend(predA.cpu().numpy())
                    yA_prob_train.extend(probA.cpu().numpy())
                    
                    yB_true_train.extend(yB.cpu().numpy())
                    yB_pred_train.extend(predB.cpu().numpy())
                
            train_loss = running_loss / len(train_loader)
            
            train_metrics = {
                "headA": metrics_headA(np.array(yA_true_train), np.array(yA_pred_train), np.array(yA_prob_train)),
                "headB": metrics_headB(np.array(yB_true_train), np.array(yB_pred_train))
            }
            
            val_loss, val_metrics = evaluate(model, val_loader, DEVICE, criterion_A, criterion_B, threshold=0.5)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            val_f1 = val_metrics['headB']['macro_f1']
            train_f1 = train_metrics['headB']['macro_f1']
            
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
            
            logger.log_losses(train_loss, val_loss, epoch, fold_prefix)
            logger.log_lr(current_lr, epoch, fold_prefix)
            logger.log_metrics_both_phases(train_metrics, val_metrics, epoch, fold_prefix)
            
            logger.update_csv({
                "fold": fold + 1, "epoch": epoch + 1, 
                "train_loss": train_loss, "val_loss": val_loss,
                "train_f1_headB": train_f1, "val_f1_headB": val_f1,
                "train_auc_headA": train_metrics['headA']['auc'], "val_auc_headA": val_metrics['headA']['auc']
            })
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                logger.save_checkpoint(model, val_f1, best_val_f1, fold_prefix)
                
            # ACUMULAR PARA LA MEDIA FINAL
            avg_metrics[epoch]["train_loss"] += train_loss
            avg_metrics[epoch]["val_loss"] += val_loss
            avg_metrics[epoch]["train_acc_A"] += train_metrics["headA"]["accuracy"]
            avg_metrics[epoch]["val_acc_A"] += val_metrics["headA"]["accuracy"]
            avg_metrics[epoch]["train_rec_A"] += train_metrics["headA"]["recall_malignant"]
            avg_metrics[epoch]["val_rec_A"] += val_metrics["headA"]["recall_malignant"]
            avg_metrics[epoch]["train_auc_A"] += train_metrics["headA"]["auc"]
            avg_metrics[epoch]["val_auc_A"] += val_metrics["headA"]["auc"]
            avg_metrics[epoch]["train_acc_B"] += train_metrics["headB"]["accuracy"]
            avg_metrics[epoch]["val_acc_B"] += val_metrics["headB"]["accuracy"]
            avg_metrics[epoch]["train_rec_B"] += train_metrics["headB"]["macro_recall"]
            avg_metrics[epoch]["val_rec_B"] += val_metrics["headB"]["macro_recall"]
            avg_metrics[epoch]["train_f1_B"] += train_metrics["headB"]["macro_f1"]
            avg_metrics[epoch]["val_f1_B"] += val_metrics["headB"]["macro_f1"]
            avg_metrics[epoch]["cm_train_A"] += train_metrics["headA"]["confusion_matrix"]
            avg_metrics[epoch]["cm_val_A"] += val_metrics["headA"]["confusion_matrix"]
            avg_metrics[epoch]["cm_train_B"] += train_metrics["headB"]["confusion_matrix"]
            avg_metrics[epoch]["cm_val_B"] += val_metrics["headB"]["confusion_matrix"]

        metricas_finales_f1.append(best_val_f1)
        print(f"✅ {fold_prefix} completado. Mejor F1 (Head B): {best_val_f1:.4f}")

    print("\n" + "═"*60)
    print(" 📊 GUARDANDO GRÁFICAS PROMEDIO EN TENSORBOARD")
    print("═"*60)
    
    avg_prefix = "Average"
    for epoch in range(EPOCHS):
        t_m = { "headA": {"accuracy": avg_metrics[epoch]["train_acc_A"] / NUM_FOLDS, "recall_malignant": avg_metrics[epoch]["train_rec_A"] / NUM_FOLDS, "auc": avg_metrics[epoch]["train_auc_A"] / NUM_FOLDS, "confusion_matrix": avg_metrics[epoch]["cm_train_A"] / NUM_FOLDS }, "headB": {"accuracy": avg_metrics[epoch]["train_acc_B"] / NUM_FOLDS, "macro_recall": avg_metrics[epoch]["train_rec_B"] / NUM_FOLDS, "macro_f1": avg_metrics[epoch]["train_f1_B"] / NUM_FOLDS, "confusion_matrix": avg_metrics[epoch]["cm_train_B"] / NUM_FOLDS } }
        v_m = { "headA": {"accuracy": avg_metrics[epoch]["val_acc_A"] / NUM_FOLDS, "recall_malignant": avg_metrics[epoch]["val_rec_A"] / NUM_FOLDS, "auc": avg_metrics[epoch]["val_auc_A"] / NUM_FOLDS, "confusion_matrix": avg_metrics[epoch]["cm_val_A"] / NUM_FOLDS }, "headB": {"accuracy": avg_metrics[epoch]["val_acc_B"] / NUM_FOLDS, "macro_recall": avg_metrics[epoch]["val_rec_B"] / NUM_FOLDS, "macro_f1": avg_metrics[epoch]["val_f1_B"] / NUM_FOLDS, "confusion_matrix": avg_metrics[epoch]["cm_val_B"] / NUM_FOLDS } }
        logger.log_losses(avg_metrics[epoch]["train_loss"] / NUM_FOLDS, avg_metrics[epoch]["val_loss"] / NUM_FOLDS, epoch, avg_prefix)
        logger.log_metrics_both_phases(t_m, v_m, epoch, avg_prefix)

    print(f" ⭐ RESUMEN K-FOLD: F1-Macro Promedio = {np.mean(metricas_finales_f1):.4f} (+/- {np.std(metricas_finales_f1):.4f})")

if __name__ == "__main__":
    train_arp_kfold()