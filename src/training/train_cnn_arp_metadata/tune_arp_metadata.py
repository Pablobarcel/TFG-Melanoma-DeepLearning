# src/training/train_cnn_arp_metadata/tune_arp_metadata.py

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
from src.data.transforms import get_train_transforms_arp, get_eval_transforms_arp

# Importaciones 3 Clases (Plan B)
from src.data.dataset_arp_metadata_planB_3class import ARPMetadataDatasetPlanB3Class
from src.models.cnn_arp_metadata.model_arp_metadata_planB_3class import ARPMetadataModelPlanB3Class
from src.evaluation.metrics_3class import metrics_headB as metrics_headB_3

# Importaciones 4 Clases (Plan B)
from src.data.dataset_arp_metadata_planB_4class import ARPMetadataDatasetPlanB4Class
from src.models.cnn_arp_metadata.model_arp_metadata_planB_4class import ARPMetadataModelPlanB4Class
from src.evaluation.metrics import metrics_headB as metrics_headB_4

# ==========================================================
# 🚨 RUTAS DE PESOS EXPERTOS (Actualizar antes de ejecutar)
# ==========================================================
# RUTAS 4 CLASES
ARP_WEIGHTS_4C = "experiments/cnn_arp/RUTAAQUI_4C/best_model.pth"
META_WEIGHTS_4C = "experiments/metadata_mlp/RUTAAQUI_PLAN_B_4C/best_model.pth"

# RUTAS 3 CLASES
ARP_WEIGHTS_3C = "experiments/cnn_arp/RUTAAQUI_3C/best_model.pth"
META_WEIGHTS_3C = "experiments/metadata_mlp/RUTAAQUI_PLAN_B_3C/best_model.pth"


def get_data_and_model(mode, trial, device, batch_size):
    """Inicializa Datasets y Modelos para la Fusión ARP+Metadatos"""
    
    if mode == '3class':
        # Dataset con Modality Dropout activado en Train
        train_ds = ARPMetadataDatasetPlanB3Class(
            "data/Splitted_data/experiment_200k_3classes/train.csv", 
            get_train_transforms_arp(), is_train=True, dropout_prob=0.3
        )
        val_ds = ARPMetadataDatasetPlanB3Class(
            "data/Splitted_data/experiment_200k_3classes/val.csv", 
            get_eval_transforms_arp(), is_train=False
        )
        model = ARPMetadataModelPlanB3Class(
            arp_weights_path=ARP_WEIGHTS_3C,
            meta_weights_path=META_WEIGHTS_3C,
            meta_input_dim=len(train_ds.feature_cols)
        ).to(device)
        
        class_weights = compute_class_weights(train_ds.df, device)
        
        # 🧠 BÚSQUEDA PROFUNDA: Parámetros de Focal Loss
        focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
        factor_clases = trial.suggest_categorical("factor_clases", [1.5, 2.0, 2.5, 3.0])
        
        class_weights[1] *= factor_clases  
        class_weights[2] *= factor_clases  
        criterion_B = FocalLoss(weight=class_weights, gamma=focal_gamma)
        
    elif mode == '4class':
        train_ds = ARPMetadataDatasetPlanB4Class(
            "data/Splitted_data/Final_dataset_4class_200k/train.csv", 
            get_train_transforms_arp(), is_train=True, dropout_prob=0.3
        )
        val_ds = ARPMetadataDatasetPlanB4Class(
            "data/Splitted_data/Final_dataset_4class_200k/val.csv", 
            get_eval_transforms_arp(), is_train=False
        )
        model = ARPMetadataModelPlanB4Class(
            arp_weights_path=ARP_WEIGHTS_4C,
            meta_weights_path=META_WEIGHTS_4C,
            meta_input_dim=len(train_ds.feature_cols)
        ).to(device)
        
        class_weights = compute_class_weights(train_ds.df, device)
        ls = trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1])
        criterion_B = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)

    # 🧊 CONGELAMOS LOS EXPERTOS (Fase 1: Linear Probing)
    for param in model.arp_branch.parameters():
        param.requires_grad = False
    for param in model.meta_branch.parameters():
        param.requires_grad = False

    # Como el modelo es ligero, podemos usar num_workers altos y pin_memory sin miedo
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    criterion_A = get_clinical_bce_loss(train_ds.df, factor_seguridad=2.0, device=device)
    
    return train_loader, val_loader, model, criterion_A, criterion_B

def objective(trial, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() 
    
    # 1. BÚSQUEDA PROFUNDA DE HIPERPARÁMETROS
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    
    # 🧠 Como el modelo es ligero, dejamos que Optuna busque el Batch Size ideal
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    
    # 2. CARGAMOS DATOS Y MODELO
    train_loader, val_loader, model, criterion_A, criterion_B = get_data_and_model(mode, trial, device, batch_size)
    
    # Optimizamos SOLO las cabezas de fusión
    parameters_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(parameters_to_update, lr=lr, weight_decay=weight_decay)
    
    num_epochs = 4 # Suficiente para calentar la cabeza
    best_val_f1 = 0.0
    
    # 3. BUCLE DE ENTRENAMIENTO
    for epoch in range(num_epochs):
        model.train()
        for i, (img_arp, feats, yA, yB) in enumerate(train_loader):
            img_arp, feats = img_arp.to(device), feats.to(device)
            yA, yB = yA.float().to(device), yB.long().to(device)
            
            optimizer.zero_grad()
            out_A, out_B = model(img_arp, feats)
            out_A = out_A.squeeze(1) if out_A.dim() > 1 else out_A
            
            loss_A = criterion_A(out_A, yA)
            
            # Multiplicador estructural en 3 clases
            if mode == '3class':
                loss = loss_A + (2.0 * criterion_B(out_B, yB))
            else:
                loss = loss_A + criterion_B(out_B, yB)
                
            loss.backward()
            optimizer.step()
            
        # VALIDACIÓN
        model.eval()
        all_yB_true, all_yB_pred = [], []
        with torch.no_grad():
            for img_arp, feats, yA, yB in val_loader:
                img_arp, feats, yB = img_arp.to(device), feats.to(device), yB.long().to(device)
                _, out_B = model(img_arp, feats)
                predB = torch.argmax(out_B, dim=1)
                all_yB_true.extend(yB.cpu().numpy())
                all_yB_pred.extend(predB.cpu().numpy())
                
        # Evaluación de la rama multiclase
        if mode == '3class':
            results_B = metrics_headB_3(np.array(all_yB_true), np.array(all_yB_pred), num_classes=3)
        else:
            results_B = metrics_headB_4(np.array(all_yB_true), np.array(all_yB_pred))
            
        val_macro_f1 = results_B['macro_f1']
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            
        trial.report(val_macro_f1, epoch)
        # Pruning: Optuna corta los trials malos en la época 1 o 2 para ahorrar tiempo
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning para Fusión Ligera: ARP + Metadatos")
    parser.add_argument('--mode', type=str, choices=['3class', '4class'], required=True)
    parser.add_argument('--trials', type=int, default=40, help="Número de pruebas (Trials)")
    args = parser.parse_args()

    print(f"\n🧪 Iniciando Optuna HPO - Fusión ARP + Metadatos ({args.mode.upper()})")
    print("⚡ Modelo Ligero Detectado: Búsqueda profunda activada (LR, Weight Decay, Batch Size, Focal Loss).")
    
    study_name = f"optuna_fusion_arp_meta_{args.mode}"
    
    # 🚨 AUTOGUARDADO ACTIVADO (SQLite) 🚨
    # Evita perder el progreso si el script se corta
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db", 
        load_if_exists=True 
    )
    
    study.optimize(lambda trial: objective(trial, args.mode), n_trials=args.trials)
    
    print(f"\n🏆 ¡Búsqueda completada para {args.mode.upper()}!")
    print(f"Mejor Macro F1 en Fase 1 (Linear Probing): {study.best_trial.value:.4f}")
    print("✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")