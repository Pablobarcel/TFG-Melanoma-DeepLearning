# src/models/cnn_vit_rgb/hybrid_model_3class.py

import torch
import torch.nn as nn
import timm
from torchvision import models

class HybridViTCNN(nn.Module):
    """
    Modelo Híbrido: ResNet18 (Textura) + ViT-Tiny (Contexto Global).
    Basado en el concepto de fusión de características del paper ARP-ViT.
    Configurado para 3 clases (NMSC).
    """
    def __init__(self, num_classes_headB=3, pretrained=True):
        super().__init__()

        # --- RAMA 1: CNN (ResNet18) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_dim = 512

        # --- RAMA 2: Vision Transformer (ViT) ---
        self.vit_backbone = timm.create_model(
            'vit_tiny_patch16_224', 
            pretrained=pretrained, 
            num_classes=0
        )
        self.vit_dim = 192

        # --- FUSIÓN Y CLASIFICACIÓN ---
        fusion_dim = self.cnn_dim + self.vit_dim
        self.dropout = nn.Dropout(p=0.3)

        self.head_A = nn.Linear(fusion_dim, 1)
        self.head_B = nn.Linear(fusion_dim, num_classes_headB)

    def forward(self, x):
        # 1. Rama CNN
        x_cnn = self.cnn_backbone(x)
        x_cnn = self.cnn_pool(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)

        # 2. Rama ViT
        x_vit = self.vit_backbone(x)

        # 3. Fusión
        x_fusion = torch.cat((x_cnn, x_vit), dim=1)
        x_fusion = self.dropout(x_fusion)

        # 4. Heads
        out_A = self.head_A(x_fusion).squeeze(1)
        out_B = self.head_B(x_fusion)

        return out_A, out_B