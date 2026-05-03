import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# --- Importaciones del proyecto ---
from src.config.seed import set_seed
from src.data.rgb.dataset_rgb import RGBDataset6Class
from src.models.cnn_vit_rgb.hybrid_model_6class import HybridRGBModel6Class
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.utils.class_weights import compute_class_weights
from src.utils.losses import get_clinical_bce_loss, FocalLoss 
from src.utils.logger import ExperimentLogger
from src.evaluation.evaluate_6class import evaluate
from src.evaluation.metrics_6class import metrics_headA, metrics_headB

def train_hybrid_kfold():
    set_seed(42)
    
    # =================================================================
    # 🔧 CONFIGURACIÓN DEL EXPERIMENTO Y REANUDACIÓN
    # =================================================================
    RESUME_TRAINING = FALSE  # 🚩 CAMBIAR A TRUE PARA REANUDAR
    RUN_ID_A_REANUDAR = None #"" # 📂 Nombre de la carpeta en experiments/
    FOLD_A_EMPEZAR = 1 
    
    CSV_PATH = "C:/TFG/data/Original_Data/ISIC_FINAL/train.csv"
    IMAGES_DIR = "C:/TFG/src/data/processed/images_RGB_ISIC"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NUM_FOLDS = 5
    EPOCHS_LP = 20
    EPOCHS_FT = 120
    EPOCHS_TOTAL = EPOCHS_LP + EPOCHS_FT
    
    BATCH_SIZE = 32
    BASE_LR = 0.00041544058403611043
    WD = 0.0012719398900890863
    GAMMA = 2.0

    config_logger = {
        "model_type": "Hybrid_ResNet18_ViTTiny",
        "lr": BASE_LR, "batch_size": BATCH_SIZE, "epochs_total": EPOCHS_TOTAL
    }

    # Inicialización del Logger con soporte para reanudación de carpeta
    logger = ExperimentLogger(
        experiment_name="rgb_hybrid_tiny_kfold", 
        config=config_logger,
        run_name=RUN_ID_A_REANUDAR if RESUME_TRAINING else None
    )
    
    df = pd.read_csv(CSV_PATH)
    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS)
    splits = list(sgkf.split(df, y=df['target'], groups=df['master_id']))
    metricas_finales_f1 = []
    
    # --- 📊 PERSISTENCIA DE MÉTRICAS AVERAGE ---
    # Cargamos del disco si estamos reanudando, si no, inicializamos
    avg_metrics_path = os.path.join(logger.results_dir, "avg_metrics_accumulator.npy")
    if RESUME_TRAINING and os.path.exists(avg_metrics_path):
        avg_metrics = np.load(avg_metrics_path, allow_pickle=True).item()
        print(f"📈 Historial de métricas promedio cargado desde el disco.")
    else:
        avg_metrics = {
            epoch: {
                "train_loss": 0.0, "val_loss": 0.0,
                "train_acc_A": 0.0, "val_acc_A": 0.0, "train_rec_A": 0.0, "val_rec_A": 0.0, "train_auc_A": 0.0, "val_auc_A": 0.0,
                "train_acc_B": 0.0, "val_acc_B": 0.0, "train_rec_B": 0.0, "val_rec_B": 0.0, "train_f1_B": 0.0, "val_f1_B": 0.0,
                "cm_train_A": np.zeros((2, 2)), "cm_val_A": np.zeros((2, 2)),
                "cm_train_B": np.zeros((4, 4)), "cm_val_B": np.zeros((4, 4))
            } for epoch in range(EPOCHS_TOTAL)
        }

    scaler = torch.amp.GradScaler('cuda')

    for fold, (train_idx, val_idx) in enumerate(splits):
        if fold + 1 < FOLD_A_EMPEZAR: 
            print(f"⏩ Saltando Fold {fold+1} (Ya completado)...")
            continue
            
        fold_prefix = f"Fold_{fold+1}"
        print(f"\n" + "═"*80)
        print(f" 📂 INICIANDO {fold_prefix}/{NUM_FOLDS}")
        print("═"*80)
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        train_loader = DataLoader(RGBDataset6Class(df_train, IMAGES_DIR, get_train_transforms()), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(RGBDataset6Class(df_val, IMAGES_DIR, get_eval_transforms()), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = HybridRGBModel6Class(num_classes_headB=4, pretrained=True).to(DEVICE)
        criterion_A = get_clinical_bce_loss(df_train, factor_seguridad=2.0, device=DEVICE)
        w_multi = compute_class_weights(df_train, DEVICE, label_col="target")
        criterion_B = FocalLoss(weight=w_multi, gamma=GAMMA)
        
        best_val_f1 = 0.0
        start_epoch = 0

        # --- ♻️ CARGAR CHECKPOINT DEL FOLD ACTUAL ---
        checkpoint_path = os.path.join(logger.results_dir, f"checkpoint_{fold_prefix}.pth")
        if RESUME_TRAINING and os.path.exists(checkpoint_path):
            print(f"♻️ Reanudando {fold_prefix} desde checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['best_val_f1']
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        for epoch in range(start_epoch, EPOCHS_TOTAL):
            # Fase LP o FT
            if epoch < EPOCHS_LP:
                phase_name = "LP"
                for param in model.cnn_backbone.parameters(): param.requires_grad = False
                for param in model.vit_backbone.parameters(): param.requires_grad = False
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR, weight_decay=WD)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            else:
                phase_name = "FT"
                for param in model.parameters(): param.requires_grad = True
                optimizer = optim.AdamW([
                    {'params': model.vit_backbone.parameters(), 'lr': BASE_LR / 100},
                    {'params': model.cnn_backbone.parameters(), 'lr': BASE_LR / 10},
                    {'params': model.head_A.parameters(), 'lr': BASE_LR},
                    {'params': model.head_B.parameters(), 'lr': BASE_LR}
                ], weight_decay=WD)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            # Re-cargar estado del optimizador si acabamos de cargar el checkpoint
            if RESUME_TRAINING and epoch == start_epoch and os.path.exists(checkpoint_path):
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except:
                    print("⚠️ Optimizador reiniciado por cambio de arquitectura de parámetros (LP/FT).")

            model.train()
            running_loss = 0.0
            yA_true, yA_pred, yA_prob, yB_true, yB_pred = [], [], [], [], []
            
            for images, yA, yB in train_loader:
                images, yA, yB = images.to(DEVICE), yA.to(DEVICE), yB.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outA, outB = model(images)
                    total_loss = criterion_A(outA.view(-1), yA) + criterion_B(outB, yB)
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += total_loss.item()
                
                with torch.no_grad():
                    probA = torch.sigmoid(outA).view(-1)
                    yA_true.extend(yA.cpu().numpy()); yA_pred.extend((probA >= 0.5).long().cpu().numpy()); yA_prob.extend(probA.cpu().numpy())
                    yB_true.extend(yB.cpu().numpy()); yB_pred.extend(torch.argmax(outB, dim=1).cpu().numpy())
                
            train_loss = running_loss / len(train_loader)
            train_m = {"headA": metrics_headA(np.array(yA_true), np.array(yA_pred), np.array(yA_prob)), "headB": metrics_headB(np.array(yB_true), np.array(yB_pred))}
            
            val_loss, val_m = evaluate(model, val_loader, DEVICE, criterion_A, criterion_B)
            scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[-1]['lr']
            val_f1 = val_m['headB']['macro_f1']

            print(f"Epoch {epoch+1:03d}/{EPOCHS_TOTAL} | Phase: {phase_name} | Loss T/V: {train_loss:.4f}/{val_loss:.4f} | F1: {val_f1:.4f} | LR: {current_lr:.2e}")

            # --- 💾 GUARDAR ESTADO DE SEGURIDAD (RESUME) ---
            checkpoint_data = {
                'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'best_val_f1': best_val_f1
            }
            torch.save(checkpoint_data, checkpoint_path)

            # --- LOGS TENSORBOARD ---
            train_m["loss"] = train_loss; val_m["loss"] = val_loss
            logger.log_full_report(train_m, val_m, epoch, fold_prefix=fold_prefix)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                logger.save_checkpoint(model, fold_prefix, is_best=True)

            # --- ACUMULACIÓN PARA AVERAGE ---
            avg_metrics[epoch]["train_loss"] += train_loss; avg_metrics[epoch]["val_loss"] += val_loss
            avg_metrics[epoch]["train_acc_A"] += train_m["headA"]["accuracy"]; avg_metrics[epoch]["val_acc_A"] += val_m["headA"]["accuracy"]
            avg_metrics[epoch]["train_rec_A"] += train_m["headA"]["recall_malignant"]; avg_metrics[epoch]["val_rec_A"] += val_m["headA"]["recall_malignant"]
            avg_metrics[epoch]["train_auc_A"] += train_m["headA"]["auc"]; avg_metrics[epoch]["val_auc_A"] += val_m["headA"]["auc"]
            avg_metrics[epoch]["train_f1_B"] += train_m["headB"]["macro_f1"]; avg_metrics[epoch]["val_f1_B"] += val_m["headB"]["macro_f1"]
            avg_metrics[epoch]["cm_train_A"] += train_m["headA"]["confusion_matrix"]; avg_metrics[epoch]["cm_val_A"] += val_m["headA"]["confusion_matrix"]
            avg_metrics[epoch]["cm_train_B"] += train_m["headB"]["confusion_matrix"]; avg_metrics[epoch]["cm_val_B"] += val_m["headB"]["confusion_matrix"]
            
            # Guardar acumulador en disco por si acaso
            np.save(avg_metrics_path, avg_metrics)

        metricas_finales_f1.append(best_val_f1)

    # =================================================================================
    # ✅ REGISTRO DE MEDIAS FINALES (Solo si se han completado los 5 folds)
    # =================================================================================
    if FOLD_A_EMPEZAR == 1 or len(metricas_finales_f1) == NUM_FOLDS:
        print("\n📊 GENERANDO PROMEDIOS FINALES EN TENSORBOARD...")
        for epoch in range(EPOCHS_TOTAL):
            t_m_avg = {
                "loss": avg_metrics[epoch]["train_loss"] / NUM_FOLDS,
                "headA": {"accuracy": avg_metrics[epoch]["train_acc_A"] / NUM_FOLDS, "recall_malignant": avg_metrics[epoch]["train_rec_A"] / NUM_FOLDS, "auc": avg_metrics[epoch]["train_auc_A"] / NUM_FOLDS, "confusion_matrix": avg_metrics[epoch]["cm_train_A"] / NUM_FOLDS},
                "headB": {"accuracy": avg_metrics[epoch]["train_acc_B"] / NUM_FOLDS, "macro_recall": avg_metrics[epoch]["train_rec_B"] / NUM_FOLDS, "macro_f1": avg_metrics[epoch]["train_f1_B"] / NUM_FOLDS, "confusion_matrix": avg_metrics[epoch]["cm_train_B"] / NUM_FOLDS}
            }
            v_m_avg = {
                "loss": avg_metrics[epoch]["val_loss"] / NUM_FOLDS,
                "headA": {"accuracy": avg_metrics[epoch]["val_acc_A"] / NUM_FOLDS, "recall_malignant": avg_metrics[epoch]["val_rec_A"] / NUM_FOLDS, "auc": avg_metrics[epoch]["val_auc_A"] / NUM_FOLDS, "confusion_matrix": avg_metrics[epoch]["cm_val_A"] / NUM_FOLDS},
                "headB": {"accuracy": avg_metrics[epoch]["val_acc_B"] / NUM_FOLDS, "macro_recall": avg_metrics[epoch]["val_rec_B"] / NUM_FOLDS, "macro_f1": avg_metrics[epoch]["val_f1_B"] / NUM_FOLDS, "confusion_matrix": avg_metrics[epoch]["cm_val_B"] / NUM_FOLDS}
            }
            logger.log_full_report(t_m_avg, v_m_avg, epoch, fold_prefix="Average")
    
    logger.close()

if __name__ == "__main__":
    train_hybrid_kfold()