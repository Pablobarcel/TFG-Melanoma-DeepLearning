# src/models/cnn_vit_rgb/hybrid_model.py

import torch
import torch.nn as nn
import timm
from torchvision import models

class HybridViTCNN(nn.Module):
    """
    Modelo Híbrido: ResNet18 (Textura) + ViT-Tiny (Contexto Global).
    Basado en el concepto de fusión de características del paper ARP-ViT.
    """
    def __init__(self, num_classes_headB=4, pretrained=True):
        super().__init__()

        # --- RAMA 1: CNN (ResNet18) ---
        # Usamos ResNet18 por ser ligera y efectiva
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Eliminamos la capa 'fc' final y el 'avgpool' para tener control total
        # Salida de layer4 es (Batch, 512, 7, 7) para imagenes de 224x224
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Pooling global para aplanar a (Batch, 512)
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_dim = 512

        # --- RAMA 2: Vision Transformer (ViT) ---
        # Usamos 'vit_tiny_patch16_224' para que quepa en tu GPU junto con la CNN.
        # Es pequeño pero potente para capturar contexto global.
        self.vit_backbone = timm.create_model(
            'vit_tiny_patch16_224', 
            pretrained=pretrained, 
            num_classes=0 # num_classes=0 devuelve el vector de características (pooler output)
        )
        self.vit_dim = 192 # Dimensión de salida del ViT-Tiny

        # --- FUSIÓN Y CLASIFICACIÓN ---
        # Dimensión combinada = 512 (CNN) + 192 (ViT) = 704
        fusion_dim = self.cnn_dim + self.vit_dim
        
        # Dropout para regularizar la fusión
        self.dropout = nn.Dropout(p=0.3)

        # Head A (Binario)
        self.head_A = nn.Linear(fusion_dim, 1)
        
        # Head B (Multiclase)
        self.head_B = nn.Linear(fusion_dim, num_classes_headB)

    def forward(self, x):
        # 1. Rama CNN
        x_cnn = self.cnn_backbone(x)      # (B, 512, 7, 7)
        x_cnn = self.cnn_pool(x_cnn)      # (B, 512, 1, 1)
        x_cnn = torch.flatten(x_cnn, 1)   # (B, 512)

        # 2. Rama ViT
        # ViT espera (B, 3, 224, 224) directamente
        x_vit = self.vit_backbone(x)      # (B, 192)

        # 3. Fusión (Concatenación)
        x_fusion = torch.cat((x_cnn, x_vit), dim=1) # (B, 704)
        x_fusion = self.dropout(x_fusion)

        # 4. Heads
        out_A = self.head_A(x_fusion).squeeze(1)
        out_B = self.head_B(x_fusion)

        return out_A, out_B