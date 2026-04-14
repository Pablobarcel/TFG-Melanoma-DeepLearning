# src/training/train_metadata/tune_metadata.py

import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import optuna

# Importaciones de tu proyecto
from src.utils.losses import get_clinical_bce_loss, FocalLoss
from src.utils.class_weights import compute_class_weights
from src.evaluation.metrics_3class import metrics_headA, metrics_headB

# Importaciones Plan A
from src.data.dataset_metadata_planA import MetadataDataset
from src.models.cnn_metadata.metadata_model_planA import MetadataMLP

# Importaciones Plan B
from src.data.dataset_metadata_planB import MetadataDatasetPlanB
from src.models.cnn_metadata.metadata_model_planB import MetadataMLPPlanB

def get_data_and_model(plan_type, trial, device):
    """Inicializa Dataset y Modelo dependiendo del Plan y los hiperparámetros del trial"""
    
    if plan_type == 'A':
        train_ds = MetadataDataset("experiment_200k_3classes/train.csv")
        val_ds = MetadataDataset("experiment_200k_3classes/val.csv")
        input_dim = len(train_ds.feature_cols)
        model = MetadataMLP(input_dim=input_dim, num_classes_headB=3).to(device)
        
    elif plan_type == 'B':
        # En el plan B, tuneamos el Modality Dropout
        mod_dropout = trial.suggest_float("modality_dropout", 0.1, 0.5, step=0.1)
        train_ds = MetadataDatasetPlanB("experiment_200k_3classes/train.csv", is_train=True, dropout_prob=mod_dropout)
        val_ds = MetadataDatasetPlanB("experiment_200k_3classes/val.csv", is_train=False)
        input_dim = len(train_ds.feature_cols)
        model = MetadataMLPPlanB(input_dim=input_dim, num_classes_headB=3).to(device)
        
    return train_ds, val_ds, model

def objective(trial, plan_type):
    """Función objetivo que Optuna intentará maximizar"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. SUGERIR HIPERPARÁMETROS PARA ESTA PRUEBA
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    focal_gamma = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
    factor_clases = trial.suggest_categorical("factor_clases_minoritarias", [1.5, 2.0, 3.0])
    
    # 2. CARGAR DATOS Y MODELO
    train_ds, val_ds, model = get_data_and_model(plan_type, trial, device)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 3. CONFIGURAR FUNCIONES DE PÉRDIDA CON LOS HIPERPARÁMETROS
    criterion_A = get_clinical_bce_loss(train_ds.df, factor_seguridad=2.0, device=device)
    
    class_weights = compute_class_weights(train_ds.df, device)
    pesos_clinicos_B = class_weights.clone()
    pesos_clinicos_B[1] *= factor_clases  # Multiplicador para Melanoma
    pesos_clinicos_B[2] *= factor_clases  # Multiplicador para NMSC
    criterion_B = FocalLoss(weight=pesos_clinicos_B, gamma=focal_gamma)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Entrenaremos rápido (ej. 10 épocas) para ver qué parámetros prometen más
    num_epochs = 10 
    best_val_f1 = 0.0
    
    # 4. BUCLE DE ENTRENAMIENTO RÁPIDO
    for epoch in range(num_epochs):
        model.train()
        for feats, yA, yB in train_loader:
            feats, yA, yB = feats.to(device), yA.float().to(device), yB.long().to(device)
            optimizer.zero_grad()
            out_A, out_B = model(feats)
            loss = criterion_A(out_A.squeeze(1), yA) + (2.0 * criterion_B(out_B, yB))
            loss.backward()
            optimizer.step()
            
        # VALIDACIÓN
        model.eval()
        all_yB_true, all_yB_pred = [], []
        with torch.no_grad():
            for feats, yA, yB in val_loader:
                feats, yB = feats.to(device), yB.long().to(device)
                _, out_B = model(feats)
                predB = torch.argmax(out_B, dim=1)
                all_yB_true.extend(yB.cpu().numpy())
                all_yB_pred.extend(predB.cpu().numpy())
                
        # Calculamos solo la métrica que nos importa maximizar para el problema "difícil"
        results_B = metrics_headB(np.array(all_yB_true), np.array(all_yB_pred), num_classes=3)
        val_macro_f1 = results_B['macro_f1']
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            
        # Reportar a Optuna (Para que corte el entrenamiento si va muy mal)
        trial.report(val_macro_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Optuna intentará maximizar este valor final
    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Búsqueda de Hiperparámetros para Metadatos")
    parser.add_argument('--plan', type=str, choices=['A', 'B'], required=True, help="Elige 'A' o 'B'")
    parser.add_argument('--trials', type=int, default=30, help="Número de pruebas a realizar")
    args = parser.parse_args()

    print(f"\n🧪 Iniciando Búsqueda de Hiperparámetros con OPTUNA - PLAN {args.plan}")
    
    # Crear el "Estudio" de Optuna
    study_name = f"optuna_metadata_plan{args.plan}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    # Ejecutar la búsqueda
    study.optimize(lambda trial: objective(trial, args.plan), n_trials=args.trials)
    
    # --- IMPRIMIR RESULTADOS FINALES ---
    print(f"\n🏆 ¡Búsqueda completada para el Plan {args.plan}!")
    print("El mejor Trial fue el número:", study.best_trial.number)
    print(f"Mejor Macro F1 alcanzado: {study.best_trial.value:.4f}")
    print("✨ LOS MEJORES HIPERPARÁMETROS SON:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")