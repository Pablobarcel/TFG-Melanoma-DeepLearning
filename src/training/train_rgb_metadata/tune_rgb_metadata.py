# src/training/train_cnn_rgb_metadata/tune_rgb_metadata.py

import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import optuna

from src.utils.losses import get_clinical_bce_loss, FocalLoss
from src.utils.class_weights import compute_class_weights
from src.evaluation.metrics_3class import metrics_headA, metrics_headB
from src.data.transforms import get_train_transforms, get_eval_transforms

# Importaciones Plan A
from src.data.dataset_rgb_metadata_planA import RGBMetadataDatasetPlanA
from src.models.cnn_rgb_metadata.model_rgb_metadata_planA import RGBMetadataModelPlanA

# Importaciones Plan B
from src.data.dataset_rgb_metadata_planB import RGBMetadataDatasetPlanB
from src.models.cnn_rgb_metadata.model_rgb_metadata_planB import RGBMetadataModelPlanB

# ==========================================================
# 🚨 PON AQUÍ LAS RUTAS A TUS EXPERTOS (Igual que en tus trains)
# ==========================================================
RGB_WEIGHTS_PATH = "experiments/cnn_vit_rgb/PONER_RUTA/best_model.pth"
META_PLAN_A_WEIGHTS_PATH = "experiments/metadata_mlp/PONER_RUTA_PLANA/best_model.pth"
META_PLAN_B_WEIGHTS_PATH = "experiments/metadata_mlp/PONER_RUTA_PLANB/best_model.pth"


def get_data_and_model(plan_type, trial, device):
    """Inicializa Dataset, DataLoader y Modelo dependiendo del Plan"""
    
    batch_size = 32 # Fijo para no saturar VRAM con imágenes
    
    if plan_type == 'A':
        train_ds = RGBMetadataDatasetPlanA("experiment_200k_3classes/train.csv", get_train_transforms())
        val_ds = RGBMetadataDatasetPlanA("experiment_200k_3classes/val.csv", get_eval_transforms())
        
        model = RGBMetadataModelPlanA(
            num_classes_headB=3, 
            rgb_weights_path=RGB_WEIGHTS_PATH, 
            meta_weights_path=META_PLAN_A_WEIGHTS_PATH,
            meta_input_dim=len(train_ds.feature_cols)
        ).to(device)
        
    elif plan_type == 'B':
        mod_dropout = trial.suggest_float("modality_dropout", 0.1, 0.5, step=0.1)
        train_ds = RGBMetadataDatasetPlanB("experiment_200k_3classes/train.csv", get_train_transforms(), is_train=True, dropout_prob=mod_dropout)
        val_ds = RGBMetadataDatasetPlanB("experiment_200k_3classes/val.csv", get_eval_transforms(), is_train=False)
        
        model = RGBMetadataModelPlanB(
            num_classes_headB=3, 
            rgb_weights_path=RGB_WEIGHTS_PATH, 
            meta_weights_path=META_PLAN_B_WEIGHTS_PATH,
            meta_input_dim=len(train_ds.feature_cols)
        ).to(device)

    # 🧊 CONGELAMOS LOS EXPERTOS PARA LA BÚSQUEDA RÁPIDA (Fase 1)
    for param in model.rgb_branch.parameters():
        param.requires_grad = False
    for param in model.meta_branch.parameters():
        param.requires_grad = False
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    return train_loader, val_loader, model, train_ds.df

def objective(trial, plan_type):
    """Función objetivo para Optuna"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() # Limpiar VRAM entre intentos
    
    # 1. HIPERPARÁMETROS A BUSCAR
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
    factor_clases = trial.suggest_categorical("factor_clases_minoritarias", [1.5, 2.0, 3.0])
    
    accumulation_steps = 4
    
    # 2. CARGAR DATOS Y MODELO
    train_loader, val_loader, model, train_df = get_data_and_model(plan_type, trial, device)
    
    # 3. LOSS Y OPTIMIZADOR
    criterion_A = get_clinical_bce_loss(train_df, factor_seguridad=2.0, device=device)
    class_weights = compute_class_weights(train_df, device)
    pesos_clinicos_B = class_weights.clone()
    pesos_clinicos_B[1] *= factor_clases 
    pesos_clinicos_B[2] *= factor_clases 
    criterion_B = FocalLoss(weight=pesos_clinicos_B, gamma=focal_gamma)
    
    # Solo optimizamos las cabezas de fusión
    parameters_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(parameters_to_update, lr=lr, weight_decay=weight_decay)
    
    num_epochs = 6 # Pocas épocas para buscar rápido
    best_val_f1 = 0.0
    
    # 4. BUCLE RÁPIDO
    for epoch in range(num_epochs):
        model.train()
        for i, (img_rgb, feats, yA, yB) in enumerate(train_loader):
            img_rgb, feats = img_rgb.to(device), feats.to(device)
            yA, yB = yA.float().to(device), yB.long().to(device)
            
            out_A, out_B = model(img_rgb, feats)
            loss = criterion_A(out_A.squeeze(1), yA) + (2.0 * criterion_B(out_B, yB))
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        # VALIDACIÓN LIGERA (Solo para calcular el F1 Macro)
        model.eval()
        all_yB_true, all_yB_pred = [], []
        with torch.no_grad():
            for img_rgb, feats, yA, yB in val_loader:
                img_rgb, feats, yB = img_rgb.to(device), feats.to(device), yB.long().to(device)
                _, out_B = model(img_rgb, feats)
                predB = torch.argmax(out_B, dim=1)
                all_yB_true.extend(yB.cpu().numpy())
                all_yB_pred.extend(predB.cpu().numpy())
                
        results_B = metrics_headB(np.array(all_yB_true), np.array(all_yB_pred), num_classes=3)
        val_macro_f1 = results_B['macro_f1']
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            
        # Reportar a Optuna
        trial.report(val_macro_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning para Fusión RGB + Metadatos")
    parser.add_argument('--plan', type=str, choices=['A', 'B'], required=True, help="Elige 'A' o 'B'")
    parser.add_argument('--trials', type=int, default=15, help="Número de pruebas")
    args = parser.parse_args()

    print(f"\n🧪 Iniciando Optuna HPO - Multimodal RGB+Meta PLAN {args.plan}")
    
    study_name = f"optuna_fusion_rgb_meta_plan{args.plan}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda trial: objective(trial, args.plan), n_trials=args.trials)
    
    print(f"\n🏆 ¡Búsqueda completada para Fusión Plan {args.plan}!")
    print(f"Mejor Macro F1 alcanzado: {study.best_trial.value:.4f}")
    print("✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")