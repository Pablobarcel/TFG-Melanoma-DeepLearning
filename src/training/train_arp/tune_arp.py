# src/training/train_arp/tune_arp.py

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
from src.data.transforms import get_train_transforms_arp, get_eval_transforms_arp

# Importaciones 4 Clases (Balanceado)
from src.data.dataset_arp import ARPDataset
from src.models.cnn_arp.arp_model import ARPCNN
from src.evaluation.metrics import metrics_headB as metrics_headB_4

# Importaciones 3 Clases (Clínico Desbalanceado)
from src.data.dataset_arp_3class import ARPDataset3Class
from src.models.cnn_arp.arp_model_3class import ARPCNN3Class
from src.evaluation.metrics_3class import metrics_headB as metrics_headB_3

def get_data_and_model(mode, trial, device, batch_size):
    """Carga los datos, el modelo y las funciones de pérdida según si es 3 o 4 clases."""
    
    if mode == '3class':
        train_ds = ARPDataset3Class("experiment_200k_3classes/train.csv", get_train_transforms_arp())
        val_ds = ARPDataset3Class("experiment_200k_3classes/val.csv", get_eval_transforms_arp())
        model = ARPCNN3Class(num_classes_headB=3).to(device)
        
        # Loss Head B (Focal Loss con Gamma y factor tuneables)
        class_weights = compute_class_weights(train_ds.df, device)
        focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
        factor_clases = trial.suggest_categorical("factor_clases", [1.5, 2.0, 2.5, 3.0])
        class_weights[1] *= factor_clases  
        class_weights[2] *= factor_clases  
        criterion_B = FocalLoss(weight=class_weights, gamma=focal_gamma)
        
    elif mode == '4class':
        train_ds = ARPDataset("experiment_46000_balanced/train.csv", get_train_transforms_arp())
        val_ds = ARPDataset("experiment_46000_balanced/val.csv", get_eval_transforms_arp())
        model = ARPCNN(num_classes_headB=4).to(device)
        
        # Loss Head B (Cross Entropy básica con pesos)
        class_weights = compute_class_weights(train_ds.df, device)
        criterion_B = nn.CrossEntropyLoss(weight=class_weights)

    # 🚨 Fijamos workers a 4 y pin_memory para velocidad, Batch Fijo por VRAM
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Loss Head A (Común a ambos)
    criterion_A = get_clinical_bce_loss(train_ds.df, factor_seguridad=2.0, device=device)
    
    return train_loader, val_loader, model, criterion_A, criterion_B

def objective(trial, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() # Evitamos OOM entre trials
    
    # 1. HIPERPARÁMETROS GENÉRICOS DE APRENDIZAJE
    # Como la CNN ARP no está pre-entrenada, probamos LRs un poco más altos
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    
    # Batch size físico fijo por VRAM
    batch_size = 32 
    
    # 2. CARGAR COMPONENTES
    train_loader, val_loader, model, criterion_A, criterion_B = get_data_and_model(mode, trial, device, batch_size)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Entrenaremos 5 épocas rápido para ver el potencial de la configuración
    num_epochs = 5 
    best_val_f1 = 0.0
    
    # 3. BUCLE DE ENTRENAMIENTO RÁPIDO
    for epoch in range(num_epochs):
        model.train()
        
        for images, yA, yB in train_loader:
            images, yA, yB = images.to(device), yA.float().to(device), yB.long().to(device)
            
            optimizer.zero_grad()
            out_A, out_B = model(images)
            out_A = out_A.squeeze(1)
            
            loss_A = criterion_A(out_A, yA)
            
            # El factor x2.0 en HeadB viene de tu script train_arp_3classes
            if mode == '3class':
                loss = loss_A + (2.0 * criterion_B(out_B, yB))
            else:
                loss = loss_A + criterion_B(out_B, yB)
                
            loss.backward()
            optimizer.step()
            
        # VALIDACIÓN LIGERA (Solo para el Head B)
        model.eval()
        all_yB_true, all_yB_pred = [], []
        with torch.no_grad():
            for images, yA, yB in val_loader:
                images, yB = images.to(device), yB.long().to(device)
                _, out_B = model(images)
                predB = torch.argmax(out_B, dim=1)
                all_yB_true.extend(yB.cpu().numpy())
                all_yB_pred.extend(predB.cpu().numpy())
                
        # Calculamos F1 Macro dependiendo del modo
        if mode == '3class':
            results_B = metrics_headB_3(np.array(all_yB_true), np.array(all_yB_pred), num_classes=3)
        else:
            results_B = metrics_headB_4(np.array(all_yB_true), np.array(all_yB_pred))
            
        val_macro_f1 = results_B['macro_f1']
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            
        # Reportar a Optuna y Podar (Pruning) si va mal
        trial.report(val_macro_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning para CNN ARP (Imágenes Polares)")
    parser.add_argument('--mode', type=str, choices=['3class', '4class'], required=True, help="Elige el experimento: 3class o 4class")
    parser.add_argument('--trials', type=int, default=15, help="Número de pruebas a realizar")
    args = parser.parse_args()

    print(f"\n🧪 Iniciando Optuna HPO - Experimento ARP (Vision Polar): {args.mode.upper()}")
    
    study_name = f"optuna_arp_{args.mode}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda trial: objective(trial, args.mode), n_trials=args.trials)
    
    print(f"\n🏆 ¡Búsqueda completada para {args.mode.upper()}!")
    print(f"Mejor Macro F1 alcanzado: {study.best_trial.value:.4f}")
    print("✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")