# src/training/train_arp/tune_arp.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import argparse
import optuna

# --- Importaciones del proyecto ---
from src.data.arp.dataset_arp import ARPDataset6Class
from src.models.cnn_arp.arp_model_6class import ARPCNN6Class
from src.utils.losses import get_clinical_bce_loss, FocalLoss
from src.utils.class_weights import compute_class_weights
from src.evaluation.evaluate_6class import evaluate
from src.data.transforms import get_train_transforms_arp, get_eval_transforms_arp

# Variable global para no recalcular el Fold en cada Trial
_GLOBAL_SPLIT = None

def get_data_loaders(csv_path, images_dir, batch_size):
    """
    Carga el CSV y crea un único Fold estático para la búsqueda de hiperparámetros.
    """
    global _GLOBAL_SPLIT
    df = pd.read_csv(csv_path)
    
    if _GLOBAL_SPLIT is None:
        sgkf = StratifiedGroupKFold(n_splits=5)
        # Nos quedamos solo con la primera partición (Fold 1) para ser rápidos
        splits = list(sgkf.split(df, y=df['target'], groups=df['master_id']))
        _GLOBAL_SPLIT = splits[0] 
        
    train_idx, val_idx = _GLOBAL_SPLIT
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    
    train_ds = ARPDataset6Class(df_train, images_dir, transforms=get_train_transforms_arp())
    val_ds = ARPDataset6Class(df_val, images_dir, transforms=get_eval_transforms_arp())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, df_train

def objective(trial, csv_path, images_dir):
    """Función objetivo que Optuna intentará maximizar"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS
    lr = trial.suggest_float("lr", 1e-5, 2e-3, log=True) # LR algo más bajos para CNNs
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # Lotes controlados para no saturar la VRAM
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) 
    
    loss_type = trial.suggest_categorical("loss_type_B", ["CrossEntropy", "FocalLoss"])
    factor_malignos = trial.suggest_categorical("factor_malignos", [1.0, 1.5, 2.0, 3.0])
    focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0]) if loss_type == "FocalLoss" else 0.0

    # 2. CARGAR DATOS Y MODELO
    train_loader, val_loader, df_train = get_data_loaders(csv_path, images_dir, batch_size)
    
    model = ARPCNN6Class(num_classes_multiclass=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 3. CONFIGURACIÓN DE PÉRDIDAS
    criterion_A = get_clinical_bce_loss(df_train, factor_seguridad=2.0, device=device)
    
    class_weights = compute_class_weights(df_train, device, label_col="target")
    pesos_B = class_weights.clone()
    
    pesos_B[1] *= factor_malignos # MEL
    pesos_B[2] *= factor_malignos # BCC
    pesos_B[3] *= factor_malignos # SCC
    
    if loss_type == "CrossEntropy":
        criterion_B = nn.CrossEntropyLoss(weight=pesos_B)
    else:
        criterion_B = FocalLoss(weight=pesos_B, gamma=focal_gamma)
    
    # Entrenamos solo 8 épocas por trial (Las imágenes tardan mucho más que los metadatos)
    num_epochs = 8 
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        
        for images, yA, yB in train_loader:
            images, yA, yB = images.to(device), yA.to(device), yB.to(device)
            
            optimizer.zero_grad()
            outA, outB = model(images)
            
            lossA = criterion_A(outA.view(-1), yA)
            lossB = criterion_B(outB, yB)
            total_loss = lossA + lossB
            
            total_loss.backward()
            optimizer.step()
            
        # 4. EVALUACIÓN Y PRUNING
        val_loss, metrics = evaluate(model, val_loader, device, criterion_A, criterion_B, threshold=0.5)
        
        current_f1 = metrics['headB']['macro_f1']
        
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            
        trial.report(current_f1, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Búsqueda de Hiperparámetros (Optuna) para CNN ARP 6 Clases")
    parser.add_argument('--csv_path', type=str, default="C:/TFG/data/Original_Data/ISIC_FINAL/dataset_train_cv.csv")
    parser.add_argument('--images_dir', type=str, default="C:/TFG/src/data/processed/images_ARP_ISIC")
    parser.add_argument('--trials', type=int, default=20, help="Número de pruebas a realizar")
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f" 🧪 INICIANDO BÚSQUEDA DE HIPERPARÁMETROS OPTUNA PARA ARP (Dual-Head)")
    print("="*80)
    
    study_name = "optuna_arp_dualhead"
    
    # Pruner para cancelar trials malos rápidamente
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", study_name=study_name, pruner=pruner)
    
    study.optimize(lambda trial: objective(trial, args.csv_path, args.images_dir), n_trials=args.trials)
    
    print("\n" + "="*80)
    print(" 🏆 ¡BÚSQUEDA OPTUNA ARP COMPLETADA! ")
    print("="*80)
    print(f"▶ Mejor Trial: {study.best_trial.number}")
    print(f"▶ Mejor Macro F1 alcanzado: {study.best_trial.value:.4f}")
    print("\n✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")
    print("="*80)