# src/models/cnn_vit_rgb/hybrid_model_6class.py
import torch
import torch.nn as nn
import timm
from torchvision import models

class HybridRGBModel6Class(nn.Module):
    """
    Arquitectura Híbrida LIGERA: ResNet18 (512) + ViT-Tiny (192).
    Concatenación final: 512 + 192 = 704 dimensiones.
    """
    def __init__(self, num_classes_headB=4, pretrained=True, dropout_rate = 0.4):
        super().__init__()

        # --- Rama CNN: ResNet18 (Feature Dim: 512) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Rama ViT LIGERA: ViT-Tiny (Feature Dim: 192) ---
        self.vit_backbone = timm.create_model(
            'vit_tiny_patch16_224', # <--- Cambio a Tiny
            pretrained=pretrained, 
            num_classes=0 
        )

        # --- Fusión (ResNet 512 + ViT-Tiny 192 = 704) ---
        self.fusion_dim = 512 + 192 # <--- Nueva dimensión de fusión
        self.dropout = nn.Dropout(p=dropout_rate)

        # Dual-Head (4 clases en Head B)
        # self.head_A = nn.Linear(self.fusion_dim, 1)
        # self.head_B = nn.Linear(self.fusion_dim, num_classes_headB)
        
        self.head_A = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 1)
        )

        # Head B: Multi-clase (Tipos de lesión)
        self.head_B = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes_headB)
        )

    def forward(self, x):
        # 1. Rama CNN
        feat_cnn = self.cnn_backbone(x)
        feat_cnn = self.cnn_pool(feat_cnn).view(x.size(0), -1) 

        # 2. Rama ViT
        feat_vit = self.vit_backbone(x) 

        # 3. Concatenación
        combined = torch.cat((feat_cnn, feat_vit), dim=1) 
        combined = self.dropout(combined)

        return self.head_A(combined), self.head_B(combined)