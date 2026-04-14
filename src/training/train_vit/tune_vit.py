# src/training/train_vit/tune_vit.py

import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import optuna

# Importaciones generales
from src.utils.losses import get_clinical_bce_loss, FocalLoss
from src.utils.class_weights import compute_class_weights
from src.data.transforms import get_train_transforms, get_eval_transforms

# Importaciones 4 Clases
from src.data.dataset_rgb import RGBDataset
from src.models.cnn_vit_rgb.hybrid_model import HybridViTCNN as HybridViTCNN_4

# Importaciones 3 Clases (Clínico)
from src.evaluation.metrics_3class import metrics_headB as metrics_headB_3
from src.data.dataset_rgb_3class import RGBDataset3Class
from src.models.cnn_vit_rgb.hybrid_model_3class import HybridViTCNN as HybridViTCNN_3

# Importación de métricas 4 clases (Ajusta la ruta si es diferente en tu proyecto)
from src.evaluation.metrics import metrics_headB as metrics_headB_4

def get_data_and_model(mode, trial, device, batch_size):
    """Carga los datos, el modelo y las funciones de pérdida según si es 3 o 4 clases."""
    
    if mode == '3class':
        train_ds = RGBDataset3Class("experiment_200k_3classes/train.csv", get_train_transforms())
        val_ds = RGBDataset3Class("experiment_200k_3classes/val.csv", get_eval_transforms())
        model = HybridViTCNN_3(num_classes_headB=3, pretrained=True).to(device)
        
        # Loss Head B (Focal Loss con Gamma tuneable)
        class_weights = compute_class_weights(train_ds.df, device)
        focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
        factor_clases = trial.suggest_categorical("factor_clases", [1.5, 2.0, 2.5])
        class_weights[1] *= factor_clases  
        class_weights[2] *= factor_clases  
        criterion_B = FocalLoss(weight=class_weights, gamma=focal_gamma)
        
    elif mode == '4class':
        train_ds = RGBDataset("experiment_46000_balanced/train.csv", get_train_transforms())
        val_ds = RGBDataset("experiment_46000_balanced/val.csv", get_eval_transforms())
        model = HybridViTCNN_4(num_classes_headB=4, pretrained=True).to(device)
        
        # Loss Head B (Cross Entropy básica con pesos)
        class_weights = compute_class_weights(train_ds.df, device)
        criterion_B = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Loss Head A (Común a ambos)
    criterion_A = get_clinical_bce_loss(train_ds.df, factor_seguridad=2.0, device=device)
    
    return train_loader, val_loader, model, criterion_A, criterion_B

def objective(trial, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() # Evitar acumulación de basura en la VRAM
    
    # 1. HIPERPARÁMETROS A BUSCAR (Los que impactan al modelo de visión)
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    
    # Batch size físico fijo por VRAM, tuneamos la acumulación matemática
    batch_size_fisico = 32 if mode == '3class' else 16 
    accumulation_steps = trial.suggest_categorical("accumulation_steps", [2, 4, 8])
    
    # 2. CARGAR COMPONENTES
    train_loader, val_loader, model, criterion_A, criterion_B = get_data_and_model(mode, trial, device, batch_size_fisico)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 🚨 Puesto a 4 épocas para que no tardes semanas. Es suficiente para marcar tendencia.
    num_epochs = 4 
    best_val_f1 = 0.0
    
    # 3. BUCLE DE ENTRENAMIENTO RÁPIDO
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, (images, yA, yB) in enumerate(train_loader):
            images, yA, yB = images.to(device), yA.float().to(device), yB.long().to(device)
            
            out_A, out_B = model(images)
            loss_A = criterion_A(out_A.squeeze(1) if out_A.dim() > 1 else out_A, yA)
            
            # El Head A penaliza más en el de 3 clases según tus scripts originales
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
            
        # VALIDACIÓN LIGERA (Para extraer el Macro F1 del Head B)
        model.eval()
        all_yB_true, all_yB_pred = [], []
        with torch.no_grad():
            for images, yA, yB in val_loader:
                images, yB = images.to(device), yB.long().to(device)
                _, out_B = model(images)
                predB = torch.argmax(out_B, dim=1)
                all_yB_true.extend(yB.cpu().numpy())
                all_yB_pred.extend(predB.cpu().numpy())
                
        # Calcular métricas según el modo
        if mode == '3class':
            results_B = metrics_headB_3(np.array(all_yB_true), np.array(all_yB_pred), num_classes=3)
        else:
            results_B = metrics_headB_4(np.array(all_yB_true), np.array(all_yB_pred))
            
        val_macro_f1 = results_B['macro_f1']
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            
        # Optuna Pruning (Corta el intento si el F1 es muy bajo comparado con otros)
        trial.report(val_macro_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning para CNN+ViT (RGB)")
    parser.add_argument('--mode', type=str, choices=['3class', '4class'], required=True, help="Elige el experimento: 3class o 4class")
    parser.add_argument('--trials', type=int, default=10, help="Número de pruebas (¡Cuidado con el tiempo!)")
    args = parser.parse_args()

    print(f"\n🧪 Iniciando Optuna HPO - Experimento Visual: {args.mode.upper()}")
    print("⚠️ ATENCIÓN: Las pruebas de visión consumen mucho tiempo. Mantén el número de trials bajo (10-15).")
    
    study_name = f"optuna_vit_{args.mode}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda trial: objective(trial, args.mode), n_trials=args.trials)
    
    print(f"\n🏆 ¡Búsqueda completada para {args.mode.upper()}!")
    print(f"Mejor Macro F1 alcanzado en {4} épocas: {study.best_trial.value:.4f}")
    print("✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")