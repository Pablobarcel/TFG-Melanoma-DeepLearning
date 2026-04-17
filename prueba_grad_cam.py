import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
import timm

# Librería especializada para XAI
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =========================================================
# 1. CONFIGURACIÓN DE RUTAS LOCALES
# =========================================================
# 🚨 CAMBIA ESTAS RUTAS A TUS CARPETAS REALES EN WINDOWS
IMAGES_DIR = "C:/TFG/src/data/processed/imagenes_RGB_224" 
VAL_CSV = "C:/TFG/data/Splitted_data/experiment_200k_3classes/val.csv"
MODEL_WEIGHTS = "C:/Users/pbarc/Downloads/best_model.pth" # El que bajaste de Kaggle
OUTPUT_DIR = r"C:/TFG/gradcam_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 2. DEFINICIÓN EXACTA DE LA ARQUITECTURA
# =========================================================
class HybridViTCNN(nn.Module):
    def __init__(self, num_classes_headB=3, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # 🚨 GRAD-CAM NECESITA ACCESO A LA ÚLTIMA CAPA CONVOLUCIONAL
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_dim = 512

        self.vit_backbone = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=0)
        self.vit_dim = 192

        fusion_dim = self.cnn_dim + self.vit_dim
        self.dropout = nn.Dropout(p=0.3)

        self.head_A = nn.Linear(fusion_dim, 1)
        self.head_B = nn.Linear(fusion_dim, num_classes_headB)

    def forward(self, x):
        feat_cnn = self.cnn_backbone(x)
        feat_cnn_pooled = self.cnn_pool(feat_cnn).view(feat_cnn.size(0), -1)
        feat_vit = self.vit_backbone(x)
        
        combined = torch.cat((feat_cnn_pooled, feat_vit), dim=1)
        combined = self.dropout(combined)
        
        return self.head_A(combined), self.head_B(combined)

# =========================================================
# 3. ENVOLTORIO PARA GRAD-CAM (Apuntando al Head B)
# =========================================================
# Como el modelo devuelve dos cosas (Head A y Head B), Grad-CAM se confunde.
# Creamos un envoltorio que solo devuelve la salida del Diagnóstico (Head B).
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        _, out_B = self.model(x)
        return out_B

# =========================================================
# 4. FUNCIONES AUXILIARES
# =========================================================
def get_transform():
    # Solo evaluación. Grad-CAM necesita la imagen sin deformar.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def inverse_normalize(tensor):
    # Deshace la normalización para poder dibujar el mapa encima
    # 🎯 CORRECCIÓN: Le decimos que use el mismo 'device' (CUDA o CPU) que tenga la imagen
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

# =========================================================
# 5. SCRIPT PRINCIPAL
# =========================================================
def run_gradcam_analysis():
    print(f"🚀 Iniciando análisis XAI en {device}...")

    # 1. Cargar el modelo
    base_model = HybridViTCNN(pretrained=False).to(device)
    
    # Manejar el DataParallel si el modelo se guardó así en Kaggle
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    base_model.load_state_dict(state_dict)
    base_model.eval()
    
    # 2. Configurar Grad-CAM apuntando a la última capa del ResNet18
    wrapped_model = ModelWrapper(base_model)
    target_layers = [base_model.cnn_backbone[-1]] 
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    # 3. Seleccionar imágenes sospechosas (Ej: 2 Benignas, 2 Malignas)
    df = pd.read_csv(VAL_CSV)
    df.loc[df["head_B_label"] == 3, "head_B_label"] = 2 # Ajuste de clase 3->2
    
    img_list = []
    # Selecciona un par de fotos de cada clase para investigar
    for c in [0, 1, 2]:
        muestras = df[df["head_B_label"] == c].sample(2, random_state=42)
        img_list.extend(muestras.to_dict('records'))

    transform = get_transform()
    clases_nombres = {0: "Benigno", 1: "Melanoma", 2: "Carcinoma"}

    for row in img_list:
        raw_path = str(row["image_path"])
        img_name = raw_path.split("\\")[-1].split("/")[-1] 
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        true_label = int(row["head_B_label"])
        
        if not os.path.exists(img_path):
            print(f"⚠️ Imagen no encontrada: {img_path}")
            continue
            
        # Cargar y preparar imagen
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Predicción del modelo
        with torch.no_grad():
            _, out_B = base_model(input_tensor)
            pred_class = torch.argmax(out_B, dim=1).item()
            confianza = torch.softmax(out_B, dim=1)[0][pred_class].item()
            
        # Generar Grad-CAM (Apuntando a la clase que el modelo predijo)
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Preparar la imagen visual
        rgb_img = inverse_normalize(input_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Guardar el resultado
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Real: {clases_nombres[true_label]}")
        axes[0].axis('off')
        
        axes[1].imshow(cam_image)
        axes[1].set_title(f"Pred: {clases_nombres[pred_class]} ({confianza*100:.1f}%)")
        axes[1].axis('off')
        
        save_path = os.path.join(OUTPUT_DIR, f"cam_{true_label}_{pred_class}_{img_name}")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Grad-CAM guardado: {save_path}")

    print("🏁 Análisis completado. Revisa la carpeta 'gradcam_results'.")

if __name__ == "__main__":
    run_gradcam_analysis()