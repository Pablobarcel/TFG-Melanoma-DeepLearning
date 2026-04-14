# src/training/train_rgb/train_rgb.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# ### LOGGER & METRICS ###
from src.utils.logger import ExperimentLogger
from src.evaluation.metrics import metrics_headA, metrics_headB
# ########################

from src.data.dataset_rgb import RGBDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.cnn_rgb import SimpleCNNRGB
from src.utils.class_weights import compute_class_weights
from src.utils.losses import get_clinical_bce_loss

# -------------------------------------------------------------------------
# FUNCIÓN DE UNA ÉPOCA
# -------------------------------------------------------------------------
def run_epoch(model, dataloader, criterion_A, criterion_B, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_yA_true, all_yA_pred, all_yA_prob = [], [], []
    all_yB_true, all_yB_pred = [], []

    with torch.set_grad_enabled(is_train):
        for images, yA, yB in dataloader:
            images = images.to(device)
            yA = yA.float().to(device)
            yB = yB.long().to(device)

            if is_train:
                optimizer.zero_grad()

            out_A, out_B = model(images)
            
            loss_A = criterion_A(out_A, yA)
            loss_B = criterion_B(out_B, yB)
            
            # Puedes dar más peso al Head B si sigue fallando mucho
            loss = loss_A + loss_B 

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            # Recolectar datos
            probA = torch.sigmoid(out_A)
            predA = (probA >= 0.3).long()
            all_yA_true.extend(yA.cpu().detach().numpy())
            all_yA_pred.extend(predA.cpu().detach().numpy())
            all_yA_prob.extend(probA.cpu().detach().numpy())

            predB = torch.argmax(out_B, dim=1)
            all_yB_true.extend(yB.cpu().detach().numpy())
            all_yB_pred.extend(predB.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    
    results_A = metrics_headA(np.array(all_yA_true), np.array(all_yA_pred), np.array(all_yA_prob))
    results_B = metrics_headB(np.array(all_yB_true), np.array(all_yB_pred))

    return epoch_loss, {"headA": results_A, "headB": results_B}


# -------------------------------------------------------------------------
# TRAIN MODEL (Con Weighted Loss)
# -------------------------------------------------------------------------
def train_model(
    learning_rate=1e-4, 
    batch_size=32, 
    momentum=0.9,       
    patience=5,         
    num_epochs=10, 
    experiment_name="tuning_rgb_weighted"
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 [Run Weighted] LR={learning_rate} | Momentum={momentum} | Patience={patience}")

    config = {
        "model": "SimpleCNNRGB",
        "lr": learning_rate,
        "momentum": momentum,
        "batch_size": batch_size,
        "patience": patience,
        "optimizer": "SGD",
        "loss_strategy": "Weighted CrossEntropy"
    }
    
    logger = ExperimentLogger(experiment_name=experiment_name, config=config)

    # 1. Datos
    train_dataset = RGBDataset("experiment_10000/train.csv", get_train_transforms())
    val_dataset = RGBDataset("experiment_10000/val.csv", get_eval_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Modelo
    model = SimpleCNNRGB(num_classes_headB=4).to(device)

    # 3. Definición de Losses con PESOS (Weighted Loss)
    # Pesos calculados: BEN=0.31, MEL=2.5, BCC=4.17, SCC=6.25
    # Esto fuerza al modelo a prestar atención a las clases minoritarias.
    class_weights = compute_class_weights(train_dataset.df, device, label_col="head_B_label")
    
    criterion_headA = get_clinical_bce_loss(train_dataset.df, factor_seguridad=2.0, device=device)
    criterion_headB = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Optimizador
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    best_auc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    # 5. Bucle
    for epoch in range(num_epochs):
        print(f"  Epoch {epoch + 1}/{num_epochs}...", end="")
        
        train_loss, train_metrics = run_epoch(model, train_loader, criterion_headA, criterion_headB, optimizer, device, is_train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion_headA, criterion_headB, optimizer, device, is_train=False)

        print(f" Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {val_metrics['headA']['auc']:.4f}")

        logger.log_scalar("Loss/Train", train_loss, epoch)
        logger.log_scalar("Loss/Val", val_loss, epoch)
        logger.log_full_report(train_metrics, epoch, phase="Train")
        logger.log_full_report(val_metrics, epoch, phase="Val")

        current_auc = val_metrics['headA']['auc'] if not np.isnan(val_metrics['headA']['auc']) else 0.0
        
        logger.update_csv({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": current_auc,
            "val_f1_macro": val_metrics['headB']['macro_f1']
        })
        
        best_auc = logger.save_checkpoint(model, current_auc, best_auc)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"    ⚠️ No mejora ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"    🛑 Early Stopping")
                break

    logger.close()

if __name__ == "__main__":
    train_model(learning_rate=1e-4, momentum=0.9, batch_size=32)