# src/utils/logger.py

import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- BLOQUE DE SEGURIDAD (Para tu entorno global) ---
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception as e:
    print(f"⚠️ TensorBoard no disponible: {e}")
    TENSORBOARD_AVAILABLE = False
# ----------------------------------------------------

class ExperimentLogger:
    def __init__(self, experiment_name, config=None, base_dir="experiments"):
        # Timestamp único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Si es un experimento de tuning, añadimos LR y Batch al nombre de carpeta
        extra_info = ""
        if config and "lr" in config and "batch_size" in config:
            extra_info = f"_LR{config['lr']}_BS{config['batch_size']}"
            
        self.run_name = f"run_{timestamp}{extra_info}"
        
        self.log_dir = os.path.join(base_dir, experiment_name, "logs", self.run_name)
        self.results_dir = os.path.join(base_dir, experiment_name, "results", self.run_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Inicializar TensorBoard de forma segura
        self.use_tensorboard = False
        if TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(log_dir=self.log_dir)
                self.use_tensorboard = True
            except Exception:
                pass

        # Guardar Config
        if config:
            with open(os.path.join(self.results_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
        
        self.history = []

    def log_scalar(self, tag, value, epoch):
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, epoch)

    def log_confusion_matrix(self, cm, class_names, tag, epoch):
        if not self.use_tensorboard:
            return

        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{tag} - Epoch {epoch}")
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        
        self.writer.add_figure(f"ConfusionMatrix/{tag}", fig, epoch)
        plt.close(fig)

    def log_full_report(self, metrics, epoch, phase="Train"):
        """
        Registra TODAS las métricas de golpe para Train o Val.
        """
        if not self.use_tensorboard:
            return

        # 1. Head A (Binaria)
        self.writer.add_scalar(f"HeadA/{phase}/AUC", metrics["headA"]["auc"], epoch)
        self.writer.add_scalar(f"HeadA/{phase}/Recall_Malignant", metrics["headA"]["recall_malignant"], epoch)
        self.writer.add_scalar(f"HeadA/{phase}/Accuracy", metrics["headA"]["accuracy"], epoch)
        
        self.log_confusion_matrix(
            metrics["headA"]["confusion_matrix"], 
            ["Benigno", "Maligno"], 
            f"HeadA/{phase}", 
            epoch
        )
        
        # 2. Head B (Multiclase)
        self.writer.add_scalar(f"HeadB/{phase}/Accuracy", metrics["headB"]["accuracy"], epoch)
        self.writer.add_scalar(f"HeadB/{phase}/Macro_Recall", metrics["headB"]["macro_recall"], epoch)
        self.writer.add_scalar(f"HeadB/{phase}/Macro_F1", metrics["headB"]["macro_f1"], epoch)

        self.log_confusion_matrix(
            metrics["headB"]["confusion_matrix"], 
            ["BEN", "MEL", "BCC", "SCC+"], 
            f"HeadB/{phase}", 
            epoch
        )

    def update_csv(self, epoch_data):
        self.history.append(epoch_data)
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.results_dir, "history.csv"), index=False)

    def save_checkpoint(self, model, current_metric, best_metric):
        # Guardar siempre last
        torch.save(model.state_dict(), os.path.join(self.results_dir, "last_model.pth"))
        
        metric_clean = current_metric if current_metric == current_metric else 0.0
        
        if metric_clean > best_metric:
            torch.save(model.state_dict(), os.path.join(self.results_dir, "best_model.pth"))
            return metric_clean
        return best_metric

    def close(self):
        if self.use_tensorboard:
            self.writer.close()