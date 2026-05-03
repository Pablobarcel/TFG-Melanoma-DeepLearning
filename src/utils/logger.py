# src/utils/logger.py

import os
import json
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
    def __init__(self, experiment_name, config=None, run_name=None, base_dir="experiments"):
        # Si pasamos un run_name (para reanudar), lo usamos. Si no, creamos uno nuevo.
        if run_name:
            self.run_name = run_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extra_info = ""
            if config:
                if "model_type" in config: extra_info += f"_{config['model_type']}"
                if "lr" in config: extra_info += f"_LR{config['lr']}"
            self.run_name = f"run_{timestamp}{extra_info}"
                
        # Rutas siguiendo tu estándar antiguo
        self.log_dir = os.path.join(base_dir, experiment_name, "logs", self.run_name)
        self.results_dir = os.path.join(base_dir, experiment_name, "results", self.run_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.use_tensorboard = False
        if TENSORBOARD_AVAILABLE:
            try:
                # Al inicializar aquí, cada ejecución es un item independiente en TensorBoard
                self.writer = SummaryWriter(log_dir=self.log_dir)
                self.use_tensorboard = True
            except Exception as e:
                print(f"❌ Error al iniciar SummaryWriter: {e}")

        # Guardar configuración del experimento para reproducibilidad académica
        if config:
            with open(os.path.join(self.results_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
        
        self.history = []

    def log_scalar(self, tag, value, epoch):
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, epoch)

    def log_confusion_matrix(self, cm, class_names, tag, epoch):
        if not self.use_tensorboard: return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"{tag} - Epoch {epoch}")
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicción')
        plt.tight_layout()
        
        # Organizamos las matrices en una pestaña propia en TensorBoard
        self.writer.add_figure(f"ConfusionMatrices/{tag}", fig, epoch)
        plt.close(fig)

    def log_full_report(self, train_metrics, val_metrics, epoch, fold_prefix=""):
        """
        Registra Head A, Head B y Losses para ambas fases.
        fold_prefix: ejemplo "Fold_1", "Average", etc.
        """
        if not self.use_tensorboard: return
        
        p = f"{fold_prefix}/" if fold_prefix else ""

        # --- LOSSES ---
        self.writer.add_scalar(f"Loss_{p}Phase/Train", train_metrics["loss"], epoch)
        self.writer.add_scalar(f"Loss_{p}Phase/Val", val_metrics["loss"], epoch)

        # --- HEAD A (Binario - Melanoma) ---
        for phase, m in [("Train", train_metrics), ("Val", val_metrics)]:
            self.writer.add_scalar(f"Metrics_HeadA_{fold_prefix}/{phase}/AUC", m["headA"]["auc"], epoch)
            self.writer.add_scalar(f"Metrics_HeadA_{fold_prefix}/{phase}/Recall_Malignant", m["headA"]["recall_malignant"], epoch)
            self.writer.add_scalar(f"Metrics_HeadA_{fold_prefix}/{phase}/Accuracy", m["headA"]["accuracy"], epoch)
            
            self.log_confusion_matrix(
                m["headA"]["confusion_matrix"], 
                ["Benigno", "Maligno"], 
                f"{fold_prefix}_{phase}_HeadA", 
                epoch
            )

        # --- HEAD B (Multiclase - 6 clases) ---
        class_names_b = ["BEN", "MEL", "BCC", "SCC"]
        for phase, m in [("Train", train_metrics), ("Val", val_metrics)]:
            self.writer.add_scalar(f"Metrics_HeadB_{fold_prefix}/{phase}/Accuracy", m["headB"]["accuracy"], epoch)
            self.writer.add_scalar(f"Metrics_HeadB_{fold_prefix}/{phase}/Macro_F1", m["headB"]["macro_f1"], epoch)
            self.writer.add_scalar(f"Metrics_HeadB_{fold_prefix}/{phase}/Macro_Recall", m["headB"]["macro_recall"], epoch)
            
            self.log_confusion_matrix(
                m["headB"]["confusion_matrix"], 
                class_names_b, 
                f"{fold_prefix}_{phase}_HeadB", 
                epoch
            )

    def update_csv(self, epoch_data, filename="history.csv"):
        self.history.append(epoch_data)
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.results_dir, filename), index=False)

    def save_checkpoint(self, model, fold_prefix, is_best=False):
        name = f"best_model_{fold_prefix}.pth" if is_best else f"last_model_{fold_prefix}.pth"
        save_path = os.path.join(self.results_dir, name)
        torch.save(model.state_dict(), save_path)

    def close(self):
        if self.use_tensorboard:
            self.writer.close()