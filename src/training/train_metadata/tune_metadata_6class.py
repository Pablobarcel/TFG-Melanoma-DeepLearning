# src/training/train_metadata/tune_metadata_6class.py

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
from src.data.metadata.dataset_metadata import MetadataDataset
from src.models.cnn_metadata.metadata_model import MetadataMLP
from src.utils.losses import get_clinical_bce_loss, FocalLoss
from src.utils.class_weights import compute_class_weights
from src.evaluation.evaluate_6class import evaluate # Usamos el evaluador dual-head

# Variable global para no recalcular el Fold en cada Trial
_GLOBAL_SPLIT = None

def get_data_loaders(csv_path, batch_size):
    """
    Carga el CSV, crea un único Fold estático para la búsqueda de hiperparámetros
    y aplica la lógica del Z-Score dinámico.
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
    
    # 1. Dataset de Entrenamiento (Calcula media y std)
    train_ds = MetadataDataset(df_train)
    
    # 2. Dataset de Validación (Hereda media y std para evitar Data Leakage)
    val_ds = MetadataDataset(df_val, mean_age=train_ds.mean_age, std_age=train_ds.std_age)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, df_train

def objective(trial, csv_path):
    """Función objetivo que Optuna intentará maximizar"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =====================================================================
    # 1. ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS
    # =====================================================================
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    
    # Parámetros para la función de pérdida del Head B
    loss_type = trial.suggest_categorical("loss_type_B", ["CrossEntropy", "FocalLoss"])
    factor_malignos = trial.suggest_categorical("factor_malignos", [1.0, 1.5, 2.0, 3.0])
    focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0]) if loss_type == "FocalLoss" else 0.0

    # =====================================================================
    # 2. PREPARACIÓN DE DATOS Y MODELO
    # =====================================================================
    train_loader, val_loader, df_train = get_data_loaders(csv_path, batch_size)
    
    model = MetadataMLP(input_dim=13, num_classes_multiclass=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # =====================================================================
    # 3. CONFIGURACIÓN DE PÉRDIDAS
    # =====================================================================
    # Head A (Binario): Siempre BCE con desbalance calculado
    criterion_A = get_clinical_bce_loss(df_train, factor_seguridad=2.0, device=device)
    
    # Head B (Multiclase): Ajuste dinámico según Optuna
    class_weights = compute_class_weights(df_train, device, label_col="target")
    pesos_B = class_weights.clone()
    
    # Multiplicamos el peso de las clases malignas (1=MEL, 2=BCC, 3=SCC)
    pesos_B[1] *= factor_malignos
    pesos_B[2] *= factor_malignos
    pesos_B[3] *= factor_malignos
    
    if loss_type == "CrossEntropy":
        criterion_B = nn.CrossEntropyLoss(weight=pesos_B)
    else:
        criterion_B = FocalLoss(weight=pesos_B, gamma=focal_gamma)
    
    # =====================================================================
    # 4. BUCLE DE ENTRENAMIENTO RÁPIDO (12 Épocas por Trial)
    # =====================================================================
    num_epochs = 12 
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        
        for feats, yA, yB in train_loader:
            feats, yA, yB = feats.to(device), yA.to(device), yB.to(device)
            
            optimizer.zero_grad()
            outA, outB = model(feats)
            
            # Suma de pérdidas
            lossA = criterion_A(outA.view(-1), yA)
            lossB = criterion_B(outB, yB)
            total_loss = lossA + lossB
            
            total_loss.backward()
            optimizer.step()
            
        # =====================================================================
        # 5. EVALUACIÓN Y PRUNING
        # =====================================================================
        val_loss, metrics = evaluate(model, val_loader, device, criterion_A, criterion_B)
        
        # Métrica a maximizar: El F1-Macro del Head B (Multiclase)
        current_f1 = metrics['headB']['macro_f1']
        
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            
        # Reportar a Optuna
        trial.report(current_f1, epoch)
        
        # Poda (Pruning): Si el trial va muy mal respecto a los anteriores, se aborta temprano
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Búsqueda de Hiperparámetros (Optuna) para Metadatos 6 Clases")
    parser.add_argument('--csv_path', type=str, default="C:/TFG/data/Original_Data/ISIC_FINAL/train.csv", help="Ruta al CSV de entrenamiento")
    parser.add_argument('--trials', type=int, default=30, help="Número de pruebas a realizar")
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f" 🧪 INICIANDO BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA (Dual-Head)")
    print("="*80)
    
    study_name = "optuna_metadata_dualhead"
    
    # Usamos el median_pruner para descartar rápido las combinaciones que no prometen
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", study_name=study_name, pruner=pruner)
    
    study.optimize(lambda trial: objective(trial, args.csv_path), n_trials=args.trials)
    
    print("\n" + "="*80)
    print(" 🏆 ¡BÚSQUEDA OPTUNA COMPLETADA! ")
    print("="*80)
    print(f"▶ Mejor Trial: {study.best_trial.number}")
    print(f"▶ Mejor Macro F1 alcanzado: {study.best_trial.value:.4f}")
    print("\n✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")
    print("="*80)