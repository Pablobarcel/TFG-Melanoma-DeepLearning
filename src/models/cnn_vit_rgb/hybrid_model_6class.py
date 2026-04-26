# src/models/cnn_vit_rgb/hybrid_model_6class.py
import torch
import torch.nn as nn
import timm
from torchvision import models

class HybridRGBModel6Class(nn.Module):
    """
    Arquitectura Híbrida: ResNet18 (Textura/Local) + ViT-Base (Contexto Global).
    Concatenación final: 512 + 768 = 1280 dimensiones.
    """
    def __init__(self, num_classes_headB=6, pretrained=True, dropout_rate = 0.4):
        super().__init__()

        # --- Rama CNN: ResNet18 (Feature Dim: 512) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Rama ViT: Vision Transformer Base (Feature Dim: 768) ---
        # Usamos patch16_224 para compatibilidad total con tus imágenes
        self.vit_backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=pretrained, 
            num_classes=0 # Extractor de rasgos puro
        )

        # --- Fusión (Late Fusion Interna) ---
        # 512 (CNN) + 768 (ViT) = 1280
        self.fusion_dim = 1280
        self.dropout = nn.Dropout(p=dropout_rate)

        # Dual-Head
        self.head_A = nn.Linear(self.fusion_dim, 1)
        self.head_B = nn.Linear(self.fusion_dim, num_classes_headB)

    def forward(self, x):
        # 1. Rama CNN
        feat_cnn = self.cnn_backbone(x)
        feat_cnn = self.cnn_pool(feat_cnn).view(x.size(0), -1) # [B, 512]

        # 2. Rama ViT
        feat_vit = self.vit_backbone(x) # [B, 768]

        # 3. Concatenación
        combined = torch.cat((feat_cnn, feat_vit), dim=1) # [B, 1280]
        combined = self.dropout(combined)

        return self.head_A(combined), self.head_B(combined)