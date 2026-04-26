# src/training/train_vit/train_hybrid_6class.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Importaciones del proyecto ---
from src.config.seed import set_seed
from src.data.rgb.dataset_rgb import RGBDataset6Class
from src.models.cnn_vit_rgb.hybrid_model_6class import HybridRGBModel6Class
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.utils.class_weights import compute_class_weights
from src.utils.losses import get_clinical_bce_loss 
from src.utils.logger import ExperimentLogger
from src.evaluation.evaluate_6class import evaluate
from src.evaluation.metrics_6class import metrics_headA, metrics_headB

def train_hybrid_kfold():
    set_seed(42)
    
    print("="*80)
    print(" 🚀 INICIANDO ENTRENAMIENTO HÍBRIDO RGB (ResNet18 + ViT-Base)")
    print(" Protocolo: Linear Probing (LP) -> Fine-Tuning (FT) con DLR")
    print("="*80)

    # =================================================================
    # 🔧 CONFIGURACIÓN PARA REANUDAR EL ENTRENAMIENTO
    # =================================================================
    FOLD_A_EMPEZAR = 1  # ⚠️ Cambia este número por el Fold donde quieres continuar (ej. 3)

    # =================================================================
    # 🔧 CONFIGURACIÓN DEL EXPERIMENTO
    # =================================================================
    CSV_PATH = "C:/TFG/data/Original_Data/ISIC_FINAL/train.csv"
    IMAGES_DIR = "C:/TFG/src/data/processed/images_RGB_ISIC"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NUM_FOLDS = 5
    EPOCHS_LP = 5     # Épocas Fase 1: Solo aprenden las cabezas (Backbone Congelado)
    EPOCHS_FT = 35    # Épocas Fase 2: Aprende todo con DLR
    EPOCHS_TOTAL = EPOCHS_LP + EPOCHS_FT
    
    BATCH_SIZE = 32   # Batch bajo porque ViT + ResNet consumen mucha VRAM
    BASE_LR = 1e-4
    WD = 1e-4
    GAMMA = 2.0

    print(f"🖥️ Dispositivo: {DEVICE}")
    df = pd.read_csv(CSV_PATH)
    
    shared_run_name = f"run_Hybrid_Final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = ExperimentLogger(experiment_name="rgb_hybrid_kfold", run_name=shared_run_name)
    
    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS)
    splits = list(sgkf.split(df, y=df['target'], groups=df['master_id']))

    metricas_finales_f1 = []
    
    # Acumulador para promedios de TensorBoard
    avg_metrics = {
        epoch: {
            "train_loss": 0.0, "val_loss": 0.0,
            "train_acc_A": 0.0, "val_acc_A": 0.0, "train_rec_A": 0.0, "val_rec_A": 0.0, "train_auc_A": 0.0, "val_auc_A": 0.0,
            "train_acc_B": 0.0, "val_acc_B": 0.0, "train_rec_B": 0.0, "val_rec_B": 0.0, "train_f1_B": 0.0, "val_f1_B": 0.0,
            "cm_train_A": np.zeros((2, 2)), "cm_val_A": np.zeros((2, 2)),
            "cm_train_B": np.zeros((6, 6)), "cm_val_B": np.zeros((6, 6))
        } for epoch in range(EPOCHS_TOTAL)
    }

    # Inicializamos el Scaler para AMP (Mixed Precision) para ahorrar memoria
    scaler = torch.amp.GradScaler('cuda')

    for fold, (train_idx, val_idx) in enumerate(splits):
        # 🚀 LÓGICA PARA SALTAR FOLDS YA COMPLETADOS
        if fold + 1 < FOLD_A_EMPEZAR:
            continue
            
        fold_prefix = f"Fold_{fold+1}"
        print(f"\n" + "═"*80)
        print(f" 📂 INICIANDO {fold_prefix}/{NUM_FOLDS}")
        print("═"*80)
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        train_ds = RGBDataset6Class(df_train, IMAGES_DIR, transforms=get_train_transforms())
        val_ds = RGBDataset6Class(df_val, IMAGES_DIR, transforms=get_eval_transforms())
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = HybridRGBModel6Class(num_classes_headB=6, pretrained=True).to(DEVICE)
        
        # --- CONFIGURACIÓN DE PÉRDIDAS ---
        criterion_A = get_clinical_bce_loss(df_train, factor_seguridad=2.0, device=DEVICE)
        w_multi = compute_class_weights(df_train, DEVICE, label_col="target")
        criterion_B = FocalLoss(weight=w_multi, gamma=GAMMA)
        
        best_val_f1 = 0.0
        optimizer = None
        scheduler = None

        for epoch in range(EPOCHS_TOTAL):
            
            # =================================================================
            # CAMBIOS DE FASE LP -> FT DINÁMICOS
            # =================================================================
            if epoch == 0:
                print(f"\n 🧊 FASE 1: LINEAR PROBING (Épocas 1 a {EPOCHS_LP}) - Congelando Backbones...")
                for param in model.cnn_backbone.parameters(): param.requires_grad = False
                for param in model.vit_backbone.parameters(): param.requires_grad = False
                
                # Optimizador solo para las cabezas
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR, weight_decay=WD)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
                
            elif epoch == EPOCHS_LP:
                print(f"\n 🔥 FASE 2: FINE-TUNING (Épocas {EPOCHS_LP+1} a {EPOCHS_TOTAL}) - Descongelando y aplicando DLR...")
                for param in model.parameters(): param.requires_grad = True
                
                # Differential Learning Rates (ViT muy lento, CNN medio, Heads normal)
                optimizer = optim.AdamW([
                    {'params': model.vit_backbone.parameters(), 'lr': BASE_LR / 100},
                    {'params': model.cnn_backbone.parameters(), 'lr': BASE_LR / 10},
                    {'params': model.head_A.parameters(), 'lr': BASE_LR},
                    {'params': model.head_B.parameters(), 'lr': BASE_LR}
                ], weight_decay=WD)
                # Reseteamos el scheduler para la nueva fase
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            # =================================================================

            model.train()
            running_loss = 0.0
            
            yA_true_train, yA_pred_train, yA_prob_train = [], [], []
            yB_true_train, yB_pred_train = [], []
            
            for images, yA, yB in train_loader:
                images, yA, yB = images.to(DEVICE), yA.to(DEVICE), yB.to(DEVICE)
                
                optimizer.zero_grad()
                
                # AMP: Mixed Precision Forward Pass
                with torch.amp.autocast('cuda'):
                    outA, outB = model(images)
                    lossA = criterion_A(outA.view(-1), yA)
                    lossB = criterion_B(outB, yB)
                    total_loss = lossA + lossB 
                
                # AMP: Backward Pass
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
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
            
            # Evaluación
            val_loss, val_metrics = evaluate(model, val_loader, DEVICE, criterion_A, criterion_B, threshold=0.5)
            
            scheduler.step(val_loss)
            
            # Guardamos el LR del Head (el grupo de parámetros índice -1) para el log
            current_lr_head = optimizer.param_groups[-1]['lr']
            
            val_f1 = val_metrics['headB']['macro_f1']
            train_f1 = train_metrics['headB']['macro_f1']
            
            phase_tag = "LP" if epoch < EPOCHS_LP else "FT"
            print(f"[{phase_tag}] Epoch {epoch+1:02d}/{EPOCHS_TOTAL} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
            
            # --- LOGGING UNIFICADO TENSORBOARD ---
            logger.log_losses(train_loss, val_loss, epoch, fold_prefix)
            logger.log_lr(current_lr_head, epoch, fold_prefix)
            logger.log_metrics_both_phases(train_metrics, val_metrics, epoch, fold_prefix)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                logger.save_checkpoint(model, val_f1, best_val_f1, fold_prefix)
                
            # ACUMULAR DATOS PARA LA MEDIA FINAL
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

    # Registro de medias finales (Average)
    if FOLD_A_EMPEZAR == 1:
        print("\n" + "═"*80)
        print(" 📊 GENERANDO MÉTRICAS PROMEDIO DE LOS 5 FOLDS")
        print("═"*80)
        # (El logger ya procesa estos datos de tu dict avg_metrics si tienes el script adaptado como en ARP)

        print(f" ⭐ RESUMEN K-FOLD RGB HÍBRIDO: F1-Macro Promedio = {np.mean(metricas_finales_f1):.4f}")
    else:
        print("\n" + "═"*80)
        print(f" ⚠️ Entrenamiento finalizado. (Reanudado desde el Fold {FOLD_A_EMPEZAR})")
        print(" No se calculan las medias globales para evitar inconsistencias matemáticas.")
        print("═"*80)

if __name__ == "__main__":
    train_hybrid_kfold()