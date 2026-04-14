# src/models/cnn_RGB/cnn_rgb.py

import torch
import torch.nn as nn
from torchvision import models

class SimpleCNNRGB(nn.Module):
    """
    Wrapper de ResNet-18 para tu TFG.
    Sustituye la CNN manual por un backbone profesional pre-entrenado.
    """
    def __init__(self, num_classes_headB=4, pretrained=True, **kwargs):
        """
        Args:
            pretrained (bool): Si True, carga pesos de ImageNet (Transfer Learning).
            **kwargs: Captura argumentos del tuning (hidden_layers, neurons...) 
                      para que no de error, aunque ResNet no los usa.
        """
        super().__init__()

        # 1. Cargar el Backbone (ResNet-18)
        # Usamos pesos pre-entrenados por defecto (Mejor convergencia)
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = models.resnet18(weights=weights)

        # 2. Preparar el extractor de características
        # ResNet termina con una capa 'fc' (Fully Connected) que no queremos.
        # La sustituimos por nuestras cabezas.
        # El vector de características antes de la 'fc' en ResNet18 tiene tamaño 512.
        self.num_features = self.backbone.fc.in_features # 512
        
        # Nos quedamos con todo menos la última capa
        # (Conv1 -> Bn1 -> Relu -> MaxPool -> Layers 1-4 -> AvgPool)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])

        # 3. Heads (Cabezales propios)
        # Head A: Binaria (1 salida)
        self.head_A = nn.Linear(self.num_features, 1)
        
        # Head B: Multiclase (4 salidas)
        self.head_B = nn.Linear(self.num_features, num_classes_headB)

    def forward(self, x):
        # 1. Extracción de características (Backbone)
        x = self.features(x)        # Salida: (Batch, 512, 1, 1)
        x = torch.flatten(x, 1)     # Aplanar: (Batch, 512)

        # 2. Predicciones
        out_A = self.head_A(x).squeeze(1)
        out_B = self.head_B(x)

        return out_A, out_B