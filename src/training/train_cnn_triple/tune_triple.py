# src/training/train_cnn_triple/tune_triple.py

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
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.transforms import get_train_transforms_arp, get_eval_transforms_arp

# --- IMPORTACIONES DE 3 CLASES ---
from src.data.dataset_triple_planA_3class import TripleDatasetPlanA3Class
from src.models.cnn_triple.triple_model_planA_3class import TripleModelPlanA3Class

from src.data.dataset_triple_planB_3class import TripleDatasetPlanB3Class
from src.models.cnn_triple.triple_model_planB_3class import TripleModelPlanB3Class
from src.evaluation.metrics_3class import metrics_headB as metrics_headB_3

# --- IMPORTACIONES DE 4 CLASES ---
from src.data.dataset_triple_planA_4class import TripleDatasetPlanA4Class
from src.models.cnn_triple.triple_model_planA_4class import TripleModelPlanA4Class

from src.data.dataset_triple_planB_4class import TripleDatasetPlanB4Class
from src.models.cnn_triple.triple_model_planB_4class import TripleModelPlanB4Class
from src.evaluation.metrics import metrics_headB as metrics_headB_4

# ==========================================================
# 🚨 RUTAS DE PESOS EXPERTOS (Actualizar antes de ejecutar)
# ==========================================================
# EXPERTOS 3 CLASES
RGB_WEIGHTS_3C = "experiments/cnn_vit_rgb/PONER_RUTA_RGB_3C/best_model.pth"
ARP_WEIGHTS_3C = "experiments/cnn_arp/PONER_RUTA_ARP_3C/best_model.pth"
META_PLAN_A_3C = "experiments/metadata_mlp/PONER_RUTA_META_A_3C/best_model.pth"
META_PLAN_B_3C = "experiments/metadata_mlp/PONER_RUTA_META_B_3C/best_model.pth"

# EXPERTOS 4 CLASES
RGB_WEIGHTS_4C = "experiments/cnn_vit_rgb/PONER_RUTA_RGB_4C/best_model.pth"
ARP_WEIGHTS_4C = "experiments/cnn_arp/PONER_RUTA_ARP_4C/best_model.pth"
META_PLAN_A_4C = "experiments/metadata_mlp/PONER_RUTA_META_A_4C/best_model.pth"
META_PLAN_B_4C = "experiments/metadata_mlp/PONER_RUTA_META_B_4C/best_model.pth"


def get_data_and_model(plan, mode, trial, device, batch_size):
    """Instancia dinámicamente el Dataset y Modelo según los argumentos"""
    
    # --- RUTAS BASE ---
    csv_train_3c = "data/Splitted_data/experiment_200k_3classes/train.csv"
    csv_val_3c = "data/Splitted_data/experiment_200k_3classes/val.csv"
    
    csv_train_4c = "data/Splitted_data/Final_dataset_4class_200k/train.csv"
    csv_val_4c = "data/Splitted_data/Final_dataset_4class_200k/val.csv"

    # --- LÓGICA DE INSTANCIACIÓN ---
    if mode == '3class':
        if plan == 'A':
            train_ds = TripleDatasetPlanA3Class(csv_train_3c, get_train_transforms(), get_train_transforms_arp())
            val_ds = TripleDatasetPlanA3Class(csv_val_3c, get_eval_transforms(), get_eval_transforms_arp())
            model = TripleModelPlanA3Class(RGB_WEIGHTS_3C, ARP_WEIGHTS_3C, META_PLAN_A_3C, len(train_ds.feature_cols)).to(device)
        else: # Plan B
            train_ds = TripleDatasetPlanB3Class(csv_train_3c, get_train_transforms(), get_train_transforms_arp(), is_train=True, dropout_prob=0.3)
            val_ds = TripleDatasetPlanB3Class(csv_val_3c, get_eval_transforms(), get_eval_transforms_arp(), is_train=False)
            model = TripleModelPlanB3Class(RGB_WEIGHTS_3C, ARP_WEIGHTS_3C, META_PLAN_B_3C, len(train_ds.feature_cols)).to(device)
            
        class_weights = compute_class_weights(train_ds.df, device)
        focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
        factor_clases = trial.suggest_categorical("factor_clases", [1.5, 2.0, 2.5])
        class_weights[1] *= factor_clases  
        class_weights[2] *= factor_clases  
        criterion_B = FocalLoss(weight=class_weights, gamma=focal_gamma)

    elif mode == '4class':
        if plan == 'A':
            train_ds = TripleDatasetPlanA4Class(csv_train_4c, get_train_transforms(), get_train_transforms_arp())
            val_ds = TripleDatasetPlanA4Class(csv_val_4c, get_eval_transforms(), get_eval_transforms_arp())
            model = TripleModelPlanA4Class(RGB_WEIGHTS_4C, ARP_WEIGHTS_4C, META_PLAN_A_4C, len(train_ds.feature_cols)).to(device)
        else: # Plan B
            train_ds = TripleDatasetPlanB4Class(csv_train_4c, get_train_transforms(), get_train_transforms_arp(), is_train=True, dropout_prob=0.3)
            val_ds = TripleDatasetPlanB4Class(csv_val_4c, get_eval_transforms(), get_eval_transforms_arp(), is_train=False)
            model = TripleModelPlanB4Class(RGB_WEIGHTS_4C, ARP_WEIGHTS_4C, META_PLAN_B_4C, len(train_ds.feature_cols)).to(device)
            
        class_weights = compute_class_weights(train_ds.df, device)
        focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
        criterion_B = FocalLoss(weight=class_weights, gamma=focal_gamma)

    # 🧊 CONGELAMOS LOS EXPERTOS (Fase 1: Linear Probing)
    for param in model.rgb_branch.parameters(): param.requires_grad = False
    for param in model.arp_branch.parameters(): param.requires_grad = False
    for param in model.meta_branch.parameters(): param.requires_grad = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    criterion_A = get_clinical_bce_loss(train_ds.df, factor_seguridad=2.0, device=device)
    
    return train_loader, val_loader, model, criterion_A, criterion_B

def objective(trial, plan, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() 
    
    # 1. BÚSQUEDA DE HIPERPARÁMETROS
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    
    # 🧠 Batch Size conservador para evitar OOM (Out Of Memory) en el modelo triple
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    
    # 2. CARGAMOS DATOS Y MODELO
    train_loader, val_loader, model, criterion_A, criterion_B = get_data_and_model(plan, mode, trial, device, batch_size)
    
    # Optimizamos SOLO las cabezas de fusión
    parameters_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(parameters_to_update, lr=lr, weight_decay=weight_decay)
    
    num_epochs = 4 # Épocas de "warmup" rápido para evaluar la combinación
    best_val_f1 = 0.0
    accumulation_steps = 2 if batch_size == 16 else 1 # Para igualar gradients
    
    # 3. BUCLE DE ENTRENAMIENTO
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, (img_rgb, img_arp, feats, yA, yB) in enumerate(train_loader):
            img_rgb, img_arp, feats = img_rgb.to(device), img_arp.to(device), feats.to(device)
            yA, yB = yA.float().to(device), yB.long().to(device)
            
            out_A, out_B = model(img_rgb, img_arp, feats)
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
            
        # VALIDACIÓN
        model.eval()
        all_yB_true, all_yB_pred = [], []
        with torch.no_grad():
            for img_rgb, img_arp, feats, yA, yB in val_loader:
                img_rgb, img_arp, feats, yB = img_rgb.to(device), img_arp.to(device), feats.to(device), yB.long().to(device)
                _, out_B = model(img_rgb, img_arp, feats)
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
        
        # Pruning para cortar trials malos
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning: JEFE FINAL (RGB + ARP + Metadatos)")
    parser.add_argument('--plan', type=str, choices=['A', 'B'], required=True, help="Plan de metadatos")
    parser.add_argument('--mode', type=str, choices=['3class', '4class'], required=True, help="Régimen de clases")
    parser.add_argument('--trials', type=int, default=25, help="Número de pruebas (Trials)")
    args = parser.parse_args()

    print(f"\n🧪 Iniciando Optuna HPO - Fusión TRIPLE (Plan {args.plan} | {args.mode.upper()})")
    print("⚡ Modo Pesado: Optimizando solo las cabezas de fusión (Fase 1).")
    
    study_name = f"optuna_triple_plan{args.plan}_{args.mode}"
    
    # 🚨 Autoguardado en SQLite
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db", 
        load_if_exists=True 
    )
    
    study.optimize(lambda trial: objective(trial, args.plan, args.mode), n_trials=args.trials)
    
    print(f"\n🏆 ¡Búsqueda completada para el Jefe Final (Plan {args.plan} - {args.mode})!")
    print(f"Mejor Macro F1 en Fase 1 (Linear Probing): {study.best_trial.value:.4f}")
    print("✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")