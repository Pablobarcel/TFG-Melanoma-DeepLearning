# src/training/train_vit/tune_vit_6class.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import argparse
import optuna
import os

# --- Importaciones del proyecto ---
from src.data.rgb.dataset_rgb import RGBDataset6Class
from src.models.cnn_vit_rgb.hybrid_model_6class import HybridRGBModel6Class
from src.utils.losses import get_clinical_bce_loss, FocalLoss
from src.evaluation.evaluate_6class import evaluate
from src.data.transforms import get_train_transforms, get_eval_transforms

# Variable global para no recalcular el Fold en cada Trial (Copia del estilo ARP)
_GLOBAL_SPLIT = None

def get_data_loaders(csv_path, images_dir, batch_size):
    global _GLOBAL_SPLIT
    df = pd.read_csv(csv_path)
    
    if _GLOBAL_SPLIT is None:
        sgkf = StratifiedGroupKFold(n_splits=5)
        # Nos quedamos con el Fold 1 para la búsqueda de hiperparámetros
        splits = list(sgkf.split(df, y=df['target'], groups=df['master_id']))
        _GLOBAL_SPLIT = splits[0]
        
    train_idx, val_idx = _GLOBAL_SPLIT
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    
    train_ds = RGBDataset6Class(df_train, images_dir, transforms=get_train_transforms())
    val_ds = RGBDataset6Class(df_val, images_dir, transforms=get_eval_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, df_train

def objective(trial, csv_path, images_dir):
    # =================================================================
    # 1. ESPACIO DE BÚSQUEDA (TUS SELECCIONADOS)
    # =================================================================
    base_lr = trial.suggest_float("base_lr", 1e-5, 1e-3, log=True)
    cnn_lr_div = trial.suggest_categorical("cnn_lr_div", [5, 10, 20, 50])
    vit_lr_div = trial.suggest_categorical("vit_lr_div", [50, 100, 200, 500])
    epochs_lp = trial.suggest_int("epochs_lp", 5, 20)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)

    # Parámetros Fijos
    BATCH_SIZE = 64 
    GAMMA = 2.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS_TOTAL = 50 

    print(f"\n🚀 Iniciando Trial #{trial.number}")
    print(f"📊 Params: LR={base_lr:.2e}, LP_Epochs={epochs_lp}, Drop={dropout_rate}, CNN_Div={cnn_lr_div}, ViT_Div={vit_lr_div}")

    # =================================================================
    # 2. CARGA DE DATOS
    # =================================================================
    train_loader, val_loader, df_train = get_data_loaders(csv_path, images_dir, BATCH_SIZE)

    # =================================================================
    # 3. MODELO Y PÉRDIDAS
    # =================================================================
    model = HybridRGBModel6Class(num_classes_headB=6, pretrained=True, dropout_rate=dropout_rate).to(DEVICE)
    
    # Head A Loss (Factor seguridad 1.0 según pediste)
    criterion_A = get_clinical_bce_loss(df_train, factor_seguridad=1.0, device=DEVICE)
    
    # Head B Loss (Focal Loss con Gamma 2.0)
    counts = df_train['target'].value_counts().sort_index()
    counts_tensor = torch.tensor(counts.values, dtype=torch.float).to(DEVICE)
    weights = counts_tensor.sum() / (len(counts_tensor) * counts_tensor)
    criterion_B = FocalLoss(weight=weights, gamma=GAMMA)

    scaler = torch.amp.GradScaler('cuda')
    optimizer = None
    best_f1 = 0.0

    # =================================================================
    # 4. BUCLE DE ENTRENAMIENTO
    # =================================================================
    for epoch in range(EPOCHS_TOTAL):
        # Cambio dinámico de fases
        if epoch == 0:
            for param in model.cnn_backbone.parameters(): param.requires_grad = False
            for param in model.vit_backbone.parameters(): param.requires_grad = False
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=weight_decay)
        elif epoch == epochs_lp:
            for param in model.parameters(): param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.vit_backbone.parameters(), 'lr': base_lr / vit_lr_div},
                {'params': model.cnn_backbone.parameters(), 'lr': base_lr / cnn_lr_div},
                {'params': model.head_A.parameters(), 'lr': base_lr},
                {'params': model.head_B.parameters(), 'lr': base_lr}
            ], weight_decay=weight_decay)

        model.train()
        running_loss = 0.0
        for images, yA, yB in train_loader:
            images, yA, yB = images.to(DEVICE), yA.to(DEVICE), yB.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outA, outB = model(images)
                lossA = criterion_A(outA.view(-1), yA)
                lossB = criterion_B(outB, yB)
                loss = lossA + lossB
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        # Evaluación
        val_loss, val_metrics = evaluate(model, val_loader, DEVICE, criterion_A, criterion_B)
        current_f1 = val_metrics['headB']['macro_f1']
        
        # --- ESTE PRINT ES EL QUE TE FALTABA ---
        phase = "LP" if epoch < epochs_lp else "FT"
        print(f"  > [E{epoch+1:02d}/{EPOCHS_TOTAL} {phase}] Loss: {val_loss:.4f} | F1: {current_f1:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1

        # Reportar y Podar
        trial.report(current_f1, epoch)
        if trial.should_prune():
            print(f"  🛑 Trial {trial.number} podado en época {epoch+1}")
            raise optuna.exceptions.TrialPruned()

    print(f"✅ Trial {trial.number} finalizado. Mejor F1: {best_f1:.4f}")
    return best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna para Híbrido RGB (ViT + ResNet)")
    parser.add_argument('--csv_path', type=str, default="C:/TFG/data/Original_Data/ISIC_FINAL/train.csv")
    parser.add_argument('--images_dir', type=str, default="C:/TFG/src/data/processed/images_RGB_ISIC")
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f" 🧪 INICIANDO BÚSQUEDA OPTUNA HÍBRIDO RGB (50 TRIALS)")
    print("="*80)
    
    # Pruner para no perder el tiempo con trials malos
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="optuna_rgb_hybrid_final",
        storage="sqlite:///optuna_resultados_rgb.db", 
        load_if_exists=True,
        pruner=pruner
    )
    
    study.optimize(lambda trial: objective(trial, args.csv_path, args.images_dir), n_trials=args.trials)

    print("\n" + "="*80)
    print(" 🏆 MEJORES RESULTADOS")
    print("="*80)
    print(f"Mejor Trial: {study.best_trial.number}")
    print(f"Mejor F1-Macro: {study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f" - {k}: {v}")