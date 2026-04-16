# test_gradcam_hybrid.py

import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# 🚨 Importamos tu modelo real
from src.models.cnn_vit_rgb.hybrid_model import HybridViTCNN

# --- ENVOLTORIO DE SEGURIDAD ---
# Grad-CAM espera un modelo que devuelva UN SOLO tensor con las clases.
# Como tus modelos a veces devuelven (Head A, Head B), hacemos este puente.
class ModelWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.model = original_model

    def forward(self, x):
        outputs = self.model(x)
        # Si devuelve una tupla (Head A, Head B), nos quedamos con Head B (multiclase)
        if isinstance(outputs, tuple):
            return outputs[1] 
        return outputs

def run_hybrid_gradcam(image_path, weights_path, num_classes=4, save_path="gradcam_hybrid.jpg"):
    print("⏳ Cargando modelo...")
    
    # 1. Instanciar tu modelo (asegúrate de poner 3 o 4 clases según los pesos que cargues)
    base_model = HybridViTCNN(num_classes_headB=num_classes, pretrained=False)
    
    # 2. Cargar los pesos entrenados
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        base_model.load_state_dict(state_dict)
        print("✅ Pesos cargados correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar los pesos: {e}")
        return

    base_model.eval()
    
    # 3. Envolver el modelo
    model_wrapped = ModelWrapper(base_model)

    # 4. Definir la Capa Objetivo (Target Layer)
    # Al estar en un Sequential, apuntamos al último bloque de la ResNet (-1) 
    # y a la última subcapa convolucional de ese bloque (-1)
    target_layers = [base_model.cnn_backbone[-1][-1]]

    # 5. Preparar la imagen (Igual que en tu dataset)
    img_pil = Image.open(image_path).convert('RGB')
    img_pil = img_pil.resize((224, 224))
    rgb_img = np.float32(img_pil) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0)

    # 6. Inicializar Grad-CAM
    cam = GradCAM(model=model_wrapped, target_layers=target_layers)

    # 7. Generar el mapa de calor (usando la clase predecida con mayor probabilidad)
    print("🔍 Analizando dónde mira la IA...")
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    # 8. Superponer y guardar
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    print(f"🎉 ¡Éxito! Grad-CAM guardado en: {save_path}")

if __name__ == "__main__":
    # =========================================================
    # 🚨 CONFIGURA ESTAS TRES VARIABLES ANTES DE EJECUTAR 🚨
    # =========================================================
    
    # 1. Ruta a una imagen de tu dataset (ej: una que sepas que es Melanoma)
    RUTA_IMAGEN = "images/imagenes_Malignant/ISIC_0011428.jpg"
    
    # 2. Ruta al archivo .pth del mejor modelo RGB que hayas entrenado
    RUTA_PESOS = "experiments/cnn_vit_rgb/experiment_0001_200k_3classes_FocalLoss/results/run_20260320_125703/best_model.pth"
    
    # 3. Número de clases con el que entrenaste ese modelo (3 o 4)
    NUM_CLASES = 3
    
    run_hybrid_gradcam(RUTA_IMAGEN, RUTA_PESOS, NUM_CLASES)