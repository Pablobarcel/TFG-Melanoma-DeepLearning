# src/training/train_metadata/train_metadata_planB.py

# --- PARCHE ANTI-CUELGUES ---
import matplotlib
matplotlib.use('Agg')
# ----------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.utils.logger import ExperimentLogger
from src.utils.losses import get_clinical_bce_loss, FocalLoss
from src.utils.class_weights import compute_class_weights
from src.evaluation.metrics_3class import metrics_headA, metrics_headB
from src.data.dataset_metadata_planB import MetadataDatasetPlanB
from src.models.cnn_metadata.metadata_model_planB import MetadataMLPPlanB

def run_epoch(model, dataloader, criterion_A, criterion_B, optimizer, device, is_train=True):
    if is_train: 
        model.train()
    else: 
        model.eval()

    running_loss = 0.0
    all_yA_true, all_yA_pred, all_yA_prob = [], [], []
    all_yB_true, all_yB_pred = [], []

    with torch.set_grad_enabled(is_train):
        for feats, yA, yB in dataloader:
            feats = feats.to(device)
            yA = yA.float().to(device)
            yB = yB.long().to(device)

            if is_train: 
                optimizer.zero_grad()

            out_A, out_B = model(feats)
            out_A = out_A.squeeze(1)
            
            loss_A = criterion_A(out_A, yA)
            loss_B = criterion_B(out_B, yB)
            
            # 🚨 Multiplicador estructural para equilibrar gradientes en 3 clases
            loss = loss_A + (2.0 * loss_B)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            probA = torch.sigmoid(out_A)
            # 🚨 UMBRAL NEUTRO 0.5
            predA = (probA >= 0.5).long()
            
            all_yA_true.extend(yA.cpu().detach().numpy())
            all_yA_pred.extend(predA.cpu().detach().numpy())
            all_yA_prob.extend(probA.cpu().detach().numpy())

            predB = torch.argmax(out_B, dim=1)
            all_yB_true.extend(yB.cpu().detach().numpy())
            all_yB_pred.extend(predB.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    results_A = metrics_headA(np.array(all_yA_true), np.array(all_yA_pred), np.array(all_yA_prob))
    results_B = metrics_headB(np.array(all_yB_true), np.array(all_yB_pred), num_classes=3)

    return epoch_loss, {"headA": results_A, "headB": results_B}

def train_plan_b(
    # 🎯 PARÁMETROS OPTUNA PENDIENTES
    # TODO: Ejecutar tune_metadata.py con 30 trials y reemplazar estos valores
    learning_rate=1e-3, 
    weight_decay=1e-2,
    batch_size=512,      
    num_epochs=15, 
    patience=4,
    experiment_path="metadata_mlp/experiment_0001_planB_3classes"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 [Metadata Plan B - 3 CLASES] Modelo: MLP Tabular Completo | Clinical Mode")

    config = {
        "model": "MetadataMLP (Plan B)", 
        "lr": learning_rate, 
        "weight_decay": weight_decay,
        "batch_size": batch_size, 
        "optimizer": "AdamW + Scheduler",
        "loss": "Focal Loss + Weighted BCE"
    }
    logger = ExperimentLogger(experiment_name=experiment_path, config=config)

    # 🚨 NUEVAS RUTAS DE DATASET ABSOLUTAS (3 Clases) 🚨
    train_ds = MetadataDatasetPlanB("data/Splitted_data/experiment_200k_3classes/train.csv", is_train=True, dropout_prob=0.3)
    val_ds = MetadataDatasetPlanB("data/Splitted_data/experiment_200k_3classes/val.csv", is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Obtenemos dinámicamente el tamaño de entrada basado en las columnas calculadas
    input_dim = len(train_ds.feature_cols)
    model = MetadataMLPPlanB(input_dim=input_dim, num_classes_headB=3).to(device)
    
    class_weights = compute_class_weights(train_ds.df, device)
    criterion_headA = get_clinical_bce_loss(train_ds.df, factor_seguridad=2.0, device=device)
    
    pesos_clinicos_B = class_weights.clone()
    pesos_clinicos_B[1] *= 2.0  # Asumimos valores de partida seguros hasta Optuna
    pesos_clinicos_B[2] *= 2.0  
    criterion_headB = FocalLoss(weight=pesos_clinicos_B, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience_counter = 0
    best_auc = 0.0

    for epoch in range(num_epochs):
        print(f"  Epoch {epoch + 1}/{num_epochs}...", end="")
        train_loss, train_metrics = run_epoch(model, train_loader, criterion_headA, criterion_headB, optimizer, device, is_train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion_headA, criterion_headB, optimizer, device, is_train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f" Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {val_metrics['headB']['macro_f1']:.4f}")

        # Guardar logs en TensorBoard
        logger.log_scalar("Loss/Train", train_loss, epoch)
        logger.log_scalar("Loss/Val", val_loss, epoch)
        logger.log_scalar("Parameters/LearningRate", current_lr, epoch)
        
        logger.log_full_report(train_metrics, epoch, phase="Train")
        logger.log_full_report(val_metrics, epoch, phase="Val")

        current_auc = val_metrics['headA']['auc'] if not np.isnan(val_metrics['headA']['auc']) else 0.0
        
        # Guardar en CSV
        logger.update_csv({
            "epoch": epoch + 1, 
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "val_f1": val_metrics['headB']['macro_f1'], 
            "lr": current_lr
        })
        
        # Guardar el mejor modelo
        best_auc = logger.save_checkpoint(model, current_auc, best_auc)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("    🛑 Early Stopping")
                break

    logger.close()

if __name__ == "__main__":
    train_plan_b()