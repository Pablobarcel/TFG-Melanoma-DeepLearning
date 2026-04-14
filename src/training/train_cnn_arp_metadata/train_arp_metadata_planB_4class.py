import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.utils.logger import ExperimentLogger
from src.evaluation.metrics import metrics_headA, metrics_headB
from src.utils.class_weights import compute_class_weights
from src.utils.losses import FocalLoss, get_clinical_bce_loss
from src.data.transforms import get_train_transforms_arp, get_eval_transforms_arp
from src.data.dataset_arp_metadata_planB_4class import ARPMetadataDatasetPlanB4Class
from src.models.cnn_arp_metadata.model_arp_metadata_planB_4class import ARPMetadataModelPlanB4Class

def run_epoch(model, dataloader, criterion_A, criterion_B, optimizer, device, is_train=True, accumulation_steps=4):
    if is_train: model.train()
    else: model.eval()

    running_loss = 0.0
    all_yA_true, all_yA_pred, all_yA_prob, all_yB_true, all_yB_pred = [], [], [], [], []

    if is_train: optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for i, (img_arp, feats, yA, yB) in enumerate(dataloader):
            img_arp, feats = img_arp.to(device), feats.to(device)
            yA, yB = yA.float().to(device), yB.long().to(device)

            out_A, out_B = model(img_arp, feats)
            
            loss_A = criterion_A(out_A, yA)
            loss_B = criterion_B(out_B, yB)
            loss = loss_A + loss_B

            if is_train:
                loss = loss / accumulation_steps 
                loss.backward() 
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step(); optimizer.zero_grad()

            running_loss += loss.item() * (accumulation_steps if is_train else 1)

            probA = torch.sigmoid(out_A)
            predA = (probA >= 0.5).long()
            all_yA_true.extend(yA.cpu().detach().numpy())
            all_yA_pred.extend(predA.cpu().detach().numpy())
            all_yA_prob.extend(probA.cpu().detach().numpy())
            all_yB_true.extend(yB.cpu().detach().numpy())
            all_yB_pred.extend(torch.argmax(out_B, dim=1).cpu().detach().numpy())

    if is_train and len(dataloader) % accumulation_steps != 0:
        optimizer.step(); optimizer.zero_grad()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, {
        "headA": metrics_headA(np.array(all_yA_true), np.array(all_yA_pred), np.array(all_yA_prob)),
        "headB": metrics_headB(np.array(all_yB_true), np.array(all_yB_pred))
    }

def train_arp_meta_planB_4class(batch_size=32, accumulation_steps=4, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 [ARP + Meta Plan B - 4 CLASES] Fase 1: Linear Probing | Modality Dropout")

    train_ds = ARPMetadataDatasetPlanB4Class("data/Splitted_data/Final_dataset_4class_200k/train.csv", get_train_transforms_arp(), is_train=True, dropout_prob=0.3)
    val_ds = ARPMetadataDatasetPlanB4Class("data/Splitted_data/Final_dataset_4class_200k/val.csv", get_eval_transforms_arp(), is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    arp_pth = "experiments/cnn_arp/PONER_RUTA_ARP_4CLASSES/best_model.pth"
    meta_pth = "experiments/metadata_mlp/PONER_RUTA_METADATA_B_4CLASSES/best_model.pth"

    model = ARPMetadataModelPlanB4Class(
        arp_weights_path=arp_pth, meta_weights_path=meta_pth, meta_input_dim=len(train_ds.feature_cols)
    ).to(device)

    # --- FASE 1 ---
    for param in model.arp_branch.parameters(): param.requires_grad = False
    for param in model.meta_branch.parameters(): param.requires_grad = False

    parameters_to_update = filter(lambda p: p.requires_grad, model.parameters())
    lr_head_fase1 = 5e-4 
    optimizer = optim.AdamW(parameters_to_update, lr=lr_head_fase1, weight_decay=1e-3)
    
    criterion_headA = get_clinical_bce_loss(train_ds.df, factor_seguridad=2.0, device=device)
    criterion_headB = FocalLoss(weight=compute_class_weights(train_ds.df, device), gamma=2.0)
    
    logger = ExperimentLogger(experiment_name="fusion/arp_meta_planB_4class_fase1", config={"lr": lr_head_fase1})
    best_auc = 0.0

    for epoch in range(num_epochs):
       print(f"  Epoch {epoch + 1}/{num_epochs}...", end="")
       train_loss, train_metrics = run_epoch(model, train_loader, criterion_headA, criterion_headB, optimizer, device, True, accumulation_steps)
       val_loss, val_metrics = run_epoch(model, val_loader, criterion_headA, criterion_headB, optimizer, device, False)
    
       print(f" Val Loss: {val_loss:.4f} | F1: {val_metrics['headB']['macro_f1']:.4f}")
       logger.log_full_report(train_metrics, epoch, phase="Train")
       logger.log_full_report(val_metrics, epoch, phase="Val")
       best_auc = logger.save_checkpoint(model, val_metrics['headA']['auc'] if not np.isnan(val_metrics['headA']['auc']) else 0.0, best_auc)
    logger.close()
    
    # --- FASE 2 ---
    print("✅ Fase 1 Completada. Iniciando Fase 2...")
    model.load_state_dict(torch.load(f"{logger.results_dir}/best_model.pth"))
    for param in model.parameters(): param.requires_grad = True

    base_lr = lr_head_fase1 / 5 
    optimizer_fase2 = optim.AdamW([
        {'params': model.arp_branch.parameters(), 'lr': base_lr / 10},
        {'params': model.meta_branch.parameters(), 'lr': base_lr / 10},
        {'params': list(model.head_A.parameters()) + list(model.head_B.parameters()), 'lr': base_lr}
    ], weight_decay=1e-3)
    
    scheduler_fase2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fase2, mode='min', factor=0.5, patience=2)
    logger_fase2 = ExperimentLogger(experiment_name="fusion/arp_meta_planB_4class_fase2", config={"lr_heads": base_lr})
    
    best_val_loss_fase2 = float('inf')
    patience_counter = 0

    for epoch in range(15):
        print(f"  [Fase 2] Epoch {epoch + 1}/15...", end="")
        train_loss, train_metrics = run_epoch(model, train_loader, criterion_headA, criterion_headB, optimizer_fase2, device, True, accumulation_steps)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion_headA, criterion_headB, optimizer_fase2, device, False)

        scheduler_fase2.step(val_loss)
        print(f" Val Loss: {val_loss:.4f} | F1: {val_metrics['headB']['macro_f1']:.4f}")
        logger_fase2.log_full_report(train_metrics, epoch, phase="Train")
        logger_fase2.log_full_report(val_metrics, epoch, phase="Val")
        
        logger_fase2.save_checkpoint(model, val_metrics['headA']['auc'] if not np.isnan(val_metrics['headA']['auc']) else 0.0, best_auc)

        if val_loss < best_val_loss_fase2:
            best_val_loss_fase2 = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 4: break
    logger_fase2.close()

if __name__ == "__main__":
    train_arp_meta_planB_4class()