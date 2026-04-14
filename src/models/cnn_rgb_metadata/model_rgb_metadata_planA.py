# src/models/cnn_rgb_metadata/model_rgb_metadata_planA.py

import torch
import torch.nn as nn
from src.models.cnn_vit_rgb.hybrid_model_3class import HybridViTCNN
from src.models.cnn_metadata.metadata_model_planA import MetadataMLP

class RGBMetadataModelPlanA(nn.Module):
    """
    Fusión de HybridViTCNN (Imagen) + MetadataMLP (Plan A)
    """
    def __init__(self, num_classes_headB=3, rgb_weights_path=None, meta_weights_path=None, meta_input_dim=11):
        super(RGBMetadataModelPlanA, self).__init__()
        
        # --- RAMA 1: RGB (ResNet + ViT) ---
        self.rgb_branch = HybridViTCNN(num_classes_headB=num_classes_headB, pretrained=False)
        self.rgb_feature_dim = 512 + 192 # (CNN + ViT) = 704
        
        # --- RAMA 2: Metadatos (Plan A) ---
        self.meta_branch = MetadataMLP(input_dim=meta_input_dim, num_classes_headB=num_classes_headB)
        self.meta_feature_dim = 64 # Salida del penúltimo Linear del MLP
        
        # --- 💉 INYECCIÓN DE CONOCIMIENTO (RGB) ---
        if rgb_weights_path:
            print(f"🔄 Cargando pesos RGB desde: {rgb_weights_path}")
            checkpoint = torch.load(rgb_weights_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Filtramos las cabezas de clasificación antiguas
            state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            self.rgb_branch.load_state_dict(state_dict, strict=False)

        # --- 💉 INYECCIÓN DE CONOCIMIENTO (Metadatos) ---
        if meta_weights_path:
            print(f"🔄 Cargando pesos Metadatos desde: {meta_weights_path}")
            checkpoint = torch.load(meta_weights_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            self.meta_branch.load_state_dict(state_dict, strict=False)

        # --- FUSIÓN (LATE FUSION) ---
        self.total_features = self.rgb_feature_dim + self.meta_feature_dim
        
        self.fusion_dropout = nn.Dropout(0.5)
        
        # Nuevas cabezas unificadas
        self.head_A = nn.Sequential(
            nn.Linear(self.total_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.head_B = nn.Sequential(
            nn.Linear(self.total_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_headB)
        )

    def forward(self, img_rgb, features_meta):
        # 1. Extraer características visuales (Mismo código que en tu HybridViTCNN)
        x_cnn = self.rgb_branch.cnn_backbone(img_rgb)
        x_cnn = self.rgb_branch.cnn_pool(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)
        x_vit = self.rgb_branch.vit_backbone(img_rgb)
        rgb_emb = torch.cat((x_cnn, x_vit), dim=1)
        
        # 2. Extraer características tabulares
        meta_emb = self.meta_branch.mlp(features_meta)
        
        # 3. Concatenar y clasificar
        fusion = torch.cat((rgb_emb, meta_emb), dim=1)
        fusion = self.fusion_dropout(fusion)
        
        out_A = self.head_A(fusion).squeeze(1)
        out_B = self.head_B(fusion)
        
        return out_A, out_B