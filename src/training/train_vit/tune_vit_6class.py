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
from src.config.seed import set_seed
from src.data.rgb.dataset_rgb import RGBDataset6Class
from src.models.cnn_vit_rgb.hybrid_model_6class import HybridRGBModel6Class
from src.utils.losses import get_clinical_bce_loss, FocalLoss
from src.utils.class_weights import compute_class_weights
from src.evaluation.evaluate_6class import evaluate
from src.data.transforms import get_train_transforms, get_eval_transforms

_GLOBAL_SPLIT = None

def get_data_loaders(csv_path, images_dir, batch_size):
    global _GLOBAL_SPLIT
    df = pd.read_csv(csv_path)
    
    if _GLOBAL_SPLIT is None:
        sgkf = StratifiedGroupKFold(n_splits=5)
        splits = list(sgkf.split(df, y=df['target'], groups=df['master_id']))
        _GLOBAL_SPLIT = splits[0]
    
    train_idx, val_idx = _GLOBAL_SPLIT
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    
    # 🚀 OPTIMIZACIÓN EXTREMA: Usar solo el 25% de los datos (Estratificado)
    print(f"  -> Tamaño original Train: {len(df_train)} imgs")
    
    # ✅ FORMA MODERNA DE PANDAS: Directo con .sample() (Sin warnings y sin perder columnas)
    df_train = df_train.groupby('target', group_keys=False).sample(frac=0.25, random_state=42).reset_index(drop=True)
    df_val = df_val.groupby('target', group_keys=False).sample(frac=0.25, random_state=42).reset_index(drop=True)
    
    print(f"  -> Tamaño reducido para Optuna: {len(df_train)} imgs")
    
    train_ds = RGBDataset6Class(df_train, images_dir, transforms=get_train_transforms())
    val_ds = RGBDataset6Class(df_val, images_dir, transforms=get_eval_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, df_train

def objective(trial, csv_path, images_dir):
    # 1. 🎯 ESPACIO DE BÚSQUEDA REDUCIDO (Solo parámetros de Alta Importancia)
    base_lr = trial.suggest_float("base_lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    gamma_val = trial.suggest_float("gamma", 2.0, 5.0, step=0.5)

    # 2. ⚓ PARÁMETROS FIJOS (Cero Importancia en la gráfica anterior)
    cnn_lr_div = 10
    vit_lr_div = 100
    epochs_lp = 4
    dropout_rate = 0.4

    # Parámetros Fijos Generales
    BATCH_SIZE = 32 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS_TOTAL = 15

    print(f"\n🚀 Trial #{trial.number} | LR={base_lr:.2e} | WD={weight_decay:.2e} | Gamma={gamma_val:.1f}")

    # 2. CARGA DE DATOS
    train_loader, val_loader, df_train = get_data_loaders(csv_path, images_dir, BATCH_SIZE)

    # 3. MODELO Y PÉRDIDAS
    model = HybridRGBModel6Class(num_classes_headB=4, pretrained=True, dropout_rate=dropout_rate).to(DEVICE)
    
    criterion_A = get_clinical_bce_loss(df_train, factor_seguridad=1.0, device=DEVICE)
    weights = compute_class_weights(df_train, DEVICE, label_col="target")
    # Aplicamos el gamma sugerido por Optuna
    criterion_B = FocalLoss(weight=weights, gamma=gamma_val)

    scaler = torch.amp.GradScaler('cuda')
    optimizer = None
    best_f1 = 0.0

    # 4. BUCLE DE ENTRENAMIENTO RÁPIDO
    for epoch in range(EPOCHS_TOTAL):
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

        # Evaluación
        val_loss, val_metrics = evaluate(model, val_loader, DEVICE, criterion_A, criterion_B)
        current_f1 = val_metrics['headB']['macro_f1']
        
        phase = "LP" if epoch < epochs_lp else "FT"
        print(f"  > [E{epoch+1:02d}/{EPOCHS_TOTAL} {phase}] Loss: {val_loss:.4f} | F1: {current_f1:.4f}")

        if current_f1 > best_f1: best_f1 = current_f1

        # Reportar y Podar rápido
        trial.report(current_f1, epoch)
        if trial.should_prune():
            print(f"  🛑 Trial {trial.number} podado (Pruned) por bajo rendimiento.")
            raise optuna.exceptions.TrialPruned()

    print(f"✅ Trial {trial.number} finalizado. Mejor F1: {best_f1:.4f}")
    return best_f1

if __name__ == "__main__":
    # 1. 🛡️ Fijamos la semilla global al principio
    set_seed(42)
    
    parser = argparse.ArgumentParser(description="Optuna Enfocado para Híbrido RGB")
    parser.add_argument('--csv_path', type=str, default="C:/TFG/data/Original_Data/ISIC_FINAL/train.csv")
    parser.add_argument('--images_dir', type=str, default="C:/TFG/src/data/processed/images_RGB_ISIC")
    parser.add_argument('--trials', type=int, default=25)
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f" 🚀 INICIANDO BÚSQUEDA OPTUNA ENFOCADA (Solo 25% de Datos | 15 Épocas)")
    print("="*80)
    
    # 2. 🧠 Creamos el Sampler de Optuna con semilla
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # 🚀 PRUNER AGRESIVO
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="optuna_rgb_hybrid_focused_v3", # ⚠️ NUEVO NOMBRE OBLIGATORIO
        storage="sqlite:///optuna_resultados_rgb_focused_v3.db", # ⚠️ NUEVA BBDD OBLIGATORIA
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler # 3. 🎯 Asignamos el sampler con semilla al estudio
    )
    
    study.optimize(lambda trial: objective(trial, args.csv_path, args.images_dir), n_trials=args.trials)

    print("\n" + "="*80)
    print(" 🏆 MEJORES RESULTADOS")
    print(f"Mejor Trial: {study.best_trial.number}")
    print(f"Mejor F1-Macro: {study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f" - {k}: {v}")