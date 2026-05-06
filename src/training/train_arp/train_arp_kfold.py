import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Importaciones del proyecto ---
from src.config.seed import set_seed
from src.data.arp.dataset_arp import ARPDataset6Class
from src.models.cnn_arp.arp_model_6class import ARPCNN6Class
from src.data.transforms import get_train_transforms_arp, get_eval_transforms_arp
from src.utils.class_weights import compute_class_weights
from src.utils.losses import get_clinical_bce_loss, FocalLoss 
from src.utils.logger import ExperimentLogger
from src.evaluation.evaluate_6class import evaluate
from src.evaluation.metrics_6class import metrics_headA, metrics_headB

def train_arp_kfold():
    set_seed(42)
    
    # =================================================================
    # 🔧 CONFIGURACIÓN DEL EXPERIMENTO
    # =================================================================
    RESUME_TRAINING = False  
    RUN_ID_A_REANUDAR = None 
    FOLD_A_EMPEZAR = 1 
    
    CSV_PATH = "C:/TFG/data/Original_Data/ISIC_FINAL/train.csv"
    # Asegúrate de que esta carpeta contenga los archivos ARP en formato .npy
    IMAGES_DIR = "C:/TFG/src/data/processed/images_ARP_NPY_ISIC" 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NUM_FOLDS = 5
    EPOCHS = 100
    
    # Hiperparámetros optimizados para ARP NPY[cite: 5, 6]
    BATCH_SIZE = 128 
    BASE_LR = 8.995e-05
    WD = 5.142e-05
    GAMMA = 3.0
    
    UMBRAL = 0.60 
    SMOOTHING = 0.65
    EARLY_STOPPING_PATIENCE = 15 
    # =================================================================

    config_logger = {
        "model_type": "ARP_CNN_NPY_CosineWarm",
        "lr": BASE_LR, "batch_size": BATCH_SIZE, 
        "epochs_total": EPOCHS, "threshold": UMBRAL
    }

    logger = ExperimentLogger(
        experiment_name="arp_cnn_npy_optimized", 
        config=config_logger,
        run_name=RUN_ID_A_REANUDAR if RESUME_TRAINING else None
    )
    
    df = pd.read_csv(CSV_PATH)
    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS)
    splits = list(sgkf.split(df, y=df['target'], groups=df['master_id']))
    
    avg_metrics_path = os.path.join(logger.results_dir, "avg_metrics_accumulator.npy")
    if RESUME_TRAINING and os.path.exists(avg_metrics_path):
        avg_metrics = np.load(avg_metrics_path, allow_pickle=True).item()
    else:
        avg_metrics = {e: {"train_loss": 0.0, "val_loss": 0.0, "train_f1_B": 0.0, "val_f1_B": 0.0} for e in range(EPOCHS)}

    # 🚩 Precisión Mixta para acelerar el entrenamiento en la RTX
    scaler = torch.amp.GradScaler('cuda')

    for fold, (train_idx, val_idx) in enumerate(splits):
        if fold + 1 < FOLD_A_EMPEZAR: continue
            
        fold_prefix = f"Fold_{fold+1}"
        print(f"\n" + "═"*80 + f"\n 📂 INICIANDO {fold_prefix} (ARP NPY) \n" + "═"*80)
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        # DataLoader optimizado para evitar saturar la RAM compartida
        train_loader = DataLoader(
            ARPDataset6Class(df_train, IMAGES_DIR, get_train_transforms_arp()), 
            batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, 
            persistent_workers=True, prefetch_factor=2
        )
        val_loader = DataLoader(
            ARPDataset6Class(df_val, IMAGES_DIR, get_eval_transforms_arp()), 
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, 
            persistent_workers=True
        )

        model = ARPCNN6Class(num_classes_multiclass=4).to(DEVICE)
        
        # Pérdidas con Label Smoothing y Focal Loss
        criterion_A = get_clinical_bce_loss(df_train, device=DEVICE)
        w_multi = compute_class_weights(df_train, DEVICE, smoothing=SMOOTHING)
        criterion_B = FocalLoss(weight=w_multi, gamma=GAMMA)
        
        # 🚩 El optimizador y scheduler se definen UNA VEZ por Fold para que persistan[cite: 6]
        optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WD)
        
        # Scheduler de Coseno con reinicios cada 15 épocas[cite: 6]
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=1, eta_min=1e-6
        )

        best_val_f1 = 0.0
        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            yA_true, yA_pred, yA_prob, yB_true, yB_pred = [], [], [], [], []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [ARP]", leave=False)
            
            for images, yA, yB in pbar:
                images, yA, yB = images.to(DEVICE), yA.to(DEVICE), yB.to(DEVICE)
                optimizer.zero_grad()
                
                # Bloque AMP para optimización de GPU[cite: 6]
                with torch.amp.autocast('cuda'):
                    outA, outB = model(images)
                    total_loss = criterion_A(outA.view(-1), yA) + criterion_B(outB, yB)
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += total_loss.item()
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
                
                with torch.no_grad():
                    probA = torch.sigmoid(outA).view(-1)
                    yA_true.extend(yA.cpu().numpy()); yA_pred.extend((probA >= UMBRAL).long().cpu().numpy())
                    yA_prob.extend(probA.cpu().numpy()); yB_true.extend(yB.cpu().numpy())
                    yB_pred.extend(torch.argmax(outB, dim=1).cpu().numpy())
            
            # --- EVALUACIÓN Y LOGS ---
            train_loss = running_loss / len(train_loader)
            val_loss, val_m = evaluate(model, val_loader, DEVICE, criterion_A, criterion_B, threshold=UMBRAL)
            
            # 🚩 Actualizar scheduler una vez por época[cite: 6]
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            val_f1 = val_m['headB']['macro_f1']
            print(f"✅ Epoch {epoch+1:03d} | Loss T/V: {train_loss:.4f}/{val_loss:.4f} | F1: {val_f1:.4f} | LR: {current_lr:.2e}")

            # Registro de reportes completos
            train_m = {"headA": metrics_headA(np.array(yA_true), np.array(yA_pred), np.array(yA_prob)), 
                       "headB": metrics_headB(np.array(yB_true), np.array(yB_pred)), "loss": train_loss}
            val_m["loss"] = val_loss
            logger.log_full_report(train_m, val_m, epoch, fold_prefix=fold_prefix)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                logger.save_checkpoint(model, fold_prefix, is_best=True)

            # Lógica de Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"🛑 Early Stopping en {fold_prefix}.")
                    break

    logger.close()
    print("\n🏁 Entrenamiento ARP por K-Fold completado.")

if __name__ == "__main__":
    train_arp_kfold()