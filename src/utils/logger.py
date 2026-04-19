# src/utils/logger.py
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception as e:
    print(f"⚠️ TensorBoard no disponible: {e}")
    TENSORBOARD_AVAILABLE = False

class ExperimentLogger:
    def __init__(self, experiment_name="metadata_kfold", run_name=None, base_dir="experiments"):
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"run_KFold_{timestamp}"
        else:
            self.run_name = run_name
            
        # Estructura LIMPIA: experiments/metadata_kfold/logs/run_XXXX/
        self.log_dir = os.path.join(base_dir, experiment_name, "logs", self.run_name)
        self.results_dir = os.path.join(base_dir, experiment_name, "results", self.run_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        if TENSORBOARD_AVAILABLE:
            # Se inicializa UNA sola vez, un solo archivo events.out
            self.writer = SummaryWriter(log_dir=self.log_dir)
            
        self.history = []

    def log_losses(self, train_loss, val_loss, epoch, fold_prefix):
        """Dibuja Train y Val en la misma gráfica usando add_scalar singular para evitar subcarpetas"""
        if not TENSORBOARD_AVAILABLE: return
        self.writer.add_scalar(f"Loss_{fold_prefix}/Train", train_loss, epoch)
        self.writer.add_scalar(f"Loss_{fold_prefix}/Val", val_loss, epoch)

    def log_lr(self, lr, epoch, fold_prefix):
        if not TENSORBOARD_AVAILABLE: return
        self.writer.add_scalar(f"Learning_Rate/{fold_prefix}", lr, epoch)

    def log_confusion_matrix(self, matrix, class_names, tag, epoch):
        if not TENSORBOARD_AVAILABLE: return
        
        fig, ax = plt.subplots(figsize=(6, 6))
        # fmt='g' para que acepte decimales en las medias
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.title(f'Confusion Matrix - {tag}')
        
        self.writer.add_figure(tag, fig, epoch)
        plt.close(fig)

    def log_metrics_both_phases(self, train_metrics, val_metrics, epoch, fold_prefix):
        """Registra Train y Val solapados sin ensuciar el disco"""
        if not TENSORBOARD_AVAILABLE: return
        
        # 1. Head A (Binario)
        self.writer.add_scalar(f"HeadA_Metrics_{fold_prefix}/Train/Accuracy", train_metrics["headA"]["accuracy"], epoch)
        self.writer.add_scalar(f"HeadA_Metrics_{fold_prefix}/Val/Accuracy", val_metrics["headA"]["accuracy"], epoch)
        
        self.writer.add_scalar(f"HeadA_Metrics_{fold_prefix}/Train/Recall_Malignant", train_metrics["headA"]["recall_malignant"], epoch)
        self.writer.add_scalar(f"HeadA_Metrics_{fold_prefix}/Val/Recall_Malignant", val_metrics["headA"]["recall_malignant"], epoch)
        
        self.writer.add_scalar(f"HeadA_Metrics_{fold_prefix}/Train/AUC", train_metrics["headA"]["auc"], epoch)
        self.writer.add_scalar(f"HeadA_Metrics_{fold_prefix}/Val/AUC", val_metrics["headA"]["auc"], epoch)
        
        # Matrices de Confusión Head A
        self.log_confusion_matrix(train_metrics["headA"]["confusion_matrix"], ["Benigno", "Maligno"], f"Confusion_Matrix{fold_prefix}/Train/HeadA", epoch)
        self.log_confusion_matrix(val_metrics["headA"]["confusion_matrix"], ["Benigno", "Maligno"], f"Confusion_Matrix{fold_prefix}/Val/HeadA", epoch)
        
        # 2. Head B (Multiclase)
        self.writer.add_scalar(f"HeadB_Metrics_{fold_prefix}/Train/Accuracy", train_metrics["headB"]["accuracy"], epoch)
        self.writer.add_scalar(f"HeadB_Metrics_{fold_prefix}/Val/Accuracy", val_metrics["headB"]["accuracy"], epoch)
        
        self.writer.add_scalar(f"HeadB_Metrics_{fold_prefix}/Train/Macro_recall", train_metrics["headB"]["macro_recall"], epoch)
        self.writer.add_scalar(f"HeadB_Metrics_{fold_prefix}/Val/Macro_recall", val_metrics["headB"]["macro_recall"], epoch)
        
        self.writer.add_scalar(f"HeadB_Metrics_{fold_prefix}/Train/Macro_F1", train_metrics["headB"]["macro_f1"], epoch)
        self.writer.add_scalar(f"HeadB_Metrics_{fold_prefix}/Val/Macro_F1", val_metrics["headB"]["macro_f1"], epoch)

        # Matrices de Confusión Head B
        self.log_confusion_matrix(train_metrics["headB"]["confusion_matrix"], ["NV", "MEL", "BCC", "SCC", "BKL", "BG"], f"Confusion_Matrix{fold_prefix}/Train/HeadB", epoch)
        self.log_confusion_matrix(val_metrics["headB"]["confusion_matrix"], ["NV", "MEL", "BCC", "SCC", "BKL", "BG"], f"Confusion_Matrix{fold_prefix}/Val/HeadB", epoch)

    def update_csv(self, epoch_data):
        self.history.append(epoch_data)
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.results_dir, "history.csv"), index=False)

    def save_checkpoint(self, model, current_metric, best_metric, fold_prefix):
        torch.save(model.state_dict(), os.path.join(self.results_dir, f"last_model_{fold_prefix}.pth"))
        if current_metric > best_metric:
            torch.save(model.state_dict(), os.path.join(self.results_dir, f"best_model_{fold_prefix}.pth"))