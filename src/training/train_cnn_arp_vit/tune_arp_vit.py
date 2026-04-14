# src/training/train_cnn_arp_vit/tune_arp_vit.py

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
from src.data.transforms import (
    get_train_transforms, get_eval_transforms,
    get_train_transforms_arp, get_eval_transforms_arp
)

# Importaciones 4 Clases
from src.data.dataset_hybrid import HybridTripleDataset
from src.models.cnn_rgb_arp_vit.arp_vit_model import TripleHybridModel
from src.evaluation.metrics import metrics_headB as metrics_headB_4

# Importaciones 3 Clases
from src.data.dataset_hybrid_3class import HybridTripleDataset3Class
from src.models.cnn_rgb_arp_vit.arp_vit_model_3class import TripleHybridModel3Class
from src.evaluation.metrics_3class import metrics_headB as metrics_headB_3

# ==========================================================
# 🚨 RUTAS DE PESOS EXPERTOS ACTUALIZADAS
# ==========================================================
# (Se han cambiado las barras \ por / para evitar errores en Python)
RGB_WEIGHTS_4C = "experiments/cnn_vit_rgb/RUTAAQUI/best_model.pth"
ARP_WEIGHTS_4C = "experiments/cnn_arp/RUTAAQUI/best_model.pth"

# RUTAS 3 CLASES (Tus mejores modelos)
RGB_WEIGHTS_3C = "experiments/cnn_vit_rgb/experiment_0001_200k_3classes_FocalLoss/results/run_20260320_125703/best_model.pth"
ARP_WEIGHTS_3C = "experiments/cnn_arp/experiment_0003_200k_3classes_FocalLoss/results/run_20260313_111503_LR0.0001_BS32/best_model.pth"


def get_data_and_model(mode, trial, device, batch_size):
    """Inicializa Datasets Multimodales y Modelos Base"""
    
    if mode == '3class':
        train_ds = HybridTripleDataset3Class(
            "experiment_200k_3classes/train.csv", 
            transforms_rgb=get_train_transforms(),
            transforms_arp=get_train_transforms_arp()
        )
        val_ds = HybridTripleDataset3Class(
            "experiment_200k_3classes/val.csv", 
            transforms_rgb=get_eval_transforms(),
            transforms_arp=get_eval_transforms_arp()
        )
        model = TripleHybridModel3Class(
            num_classes_headB=3, 
            pretrained_rgb=False, 
            arp_pretrained_path=ARP_WEIGHTS_3C,
            rgb_pretrained_path=RGB_WEIGHTS_3C
        ).to(device)
        
        class_weights = compute_class_weights(train_ds.df, device)
        
        # ACOTACIÓN DEL ESPACIO DE BÚSQUEDA FOCAL (Basado en tus expertos)
        focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0])
        factor_clases = trial.suggest_categorical("factor_clases", [1.5, 2.0, 2.5])
        
        class_weights[1] *= factor_clases  
        class_weights[2] *= factor_clases  
        criterion_B = FocalLoss(weight=class_weights, gamma=focal_gamma)
        
    elif mode == '4class':
        train_ds = HybridTripleDataset(
            "experiment_46000_balanced/train.csv", 
            transforms_rgb=get_train_transforms(),
            transforms_arp=get_train_transforms_arp()
        )
        val_ds = HybridTripleDataset(
            "experiment_46000_balanced/val.csv", 
            transforms_rgb=get_eval_transforms(),
            transforms_arp=get_eval_transforms_arp()
        )
        model = TripleHybridModel(
            num_classes_headB=4, 
            pretrained_rgb=False, 
            arp_pretrained_path=ARP_WEIGHTS_4C,
            rgb_pretrained_path=RGB_WEIGHTS_4C
        ).to(device)
        
        class_weights = compute_class_weights(train_ds.df, device)
        ls = trial.suggest_categorical("label_smoothing", [0.0, 0.1])
        criterion_B = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)

    # 🧊 CONGELAMOS LOS EXPERTOS (Fase 1: Linear Probing)
    for param in model.rgb_branch.parameters():
        param.requires_grad = False
    for param in model.arp_branch.parameters():
        param.requires_grad = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    criterion_A = get_clinical_bce_loss(train_ds.df, factor_seguridad=2.0, device=device)
    
    return train_loader, val_loader, model, criterion_A, criterion_B

def objective(trial, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() 
    
    # 1. HIPERPARÁMETROS ACOTADOS (Cabezas de Fusión)
    # Acotamos a un rango donde las capas lineales suelen aprender bien
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    
    batch_size = 32 # Fijo
    accumulation_steps = 4 # Fijo
    
    # 2. CARGAMOS TODO
    train_loader, val_loader, model, criterion_A, criterion_B = get_data_and_model(mode, trial, device, batch_size)
    
    # Optimizamos SOLO las cabezas que NO están congeladas
    parameters_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(parameters_to_update, lr=lr, weight_decay=weight_decay)
    
    num_epochs = 4 # 4 épocas es suficiente para la fase de calentamiento
    best_val_f1 = 0.0
    
    # 3. BUCLE RÁPIDO
    for epoch in range(num_epochs):
        model.train()
        for i, (img_rgb, img_arp, yA, yB) in enumerate(train_loader):
            img_rgb, img_arp = img_rgb.to(device), img_arp.to(device)
            yA, yB = yA.float().to(device), yB.long().to(device)
            
            # ATENCIÓN: El modelo de fusión recibe AMBAS imágenes
            out_A, out_B = model(img_rgb, img_arp)
            out_A = out_A.squeeze(1) if out_A.dim() > 1 else out_A
            
            loss_A = criterion_A(out_A, yA)
            if mode == '3class':
                loss = loss_A + (2.0 * criterion_B(out_B, yB))
            else:
                loss = loss_A + criterion_B(out_B, yB)
                
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        # VALIDACIÓN
        model.eval()
        all_yB_true, all_yB_pred = [], []
        with torch.no_grad():
            for img_rgb, img_arp, yA, yB in val_loader:
                img_rgb, img_arp, yB = img_rgb.to(device), img_arp.to(device), yB.long().to(device)
                _, out_B = model(img_rgb, img_arp)
                predB = torch.argmax(out_B, dim=1)
                all_yB_true.extend(yB.cpu().numpy())
                all_yB_pred.extend(predB.cpu().numpy())
                
        # Evaluación
        if mode == '3class':
            results_B = metrics_headB_3(np.array(all_yB_true), np.array(all_yB_pred), num_classes=3)
        else:
            results_B = metrics_headB_4(np.array(all_yB_true), np.array(all_yB_pred))
            
        val_macro_f1 = results_B['macro_f1']
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            
        trial.report(val_macro_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning para Fusión RGB + ARP")
    parser.add_argument('--mode', type=str, choices=['3class', '4class'], required=True)
    parser.add_argument('--trials', type=int, default=30, help="Número de pruebas")
    args = parser.parse_args()

    print(f"\n🧪 Iniciando Optuna HPO - Fusión Doble de Imágenes: {args.mode.upper()}")
    print("⚠️ Espacio de búsqueda acotado para convergencia rápida.")
    
    study_name = f"optuna_fusion_rgb_arp_{args.mode}"
    # Añadimos SQLite para autoguardar cada trial
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name,
        storage="sqlite:///optuna_resultados_triple.db", # <--- Base de datos local
        load_if_exists=True # <--- Si se corta y lo vuelves a lanzar, retoma donde lo dejó
    )
    study.optimize(lambda trial: objective(trial, args.mode), n_trials=args.trials)
    
    print(f"\n🏆 ¡Búsqueda completada para {args.mode.upper()}!")
    print(f"Mejor Macro F1 en Fase 1 (Linear Probing): {study.best_trial.value:.4f}")
    print("✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")