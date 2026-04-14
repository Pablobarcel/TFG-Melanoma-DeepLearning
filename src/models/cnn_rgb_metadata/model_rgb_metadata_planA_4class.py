# src/models/cnn_rgb_metadata/model_rgb_metadata_planA_4class.py

import torch
import torch.nn as nn
from src.models.cnn_vit_rgb.hybrid_model import HybridViTCNN
from src.models.cnn_metadata.metadata_model_planA_4class import MetadataMLPPlanA4Class

class RGBMetadataModelPlanA4Class(nn.Module):
    def __init__(self, rgb_weights_path=None, meta_weights_path=None, meta_input_dim=11):
        super(RGBMetadataModelPlanA4Class, self).__init__()
        
        self.rgb_branch = HybridViTCNN(num_classes_headB=4, pretrained=False)
        self.rgb_feature_dim = 512 + 192 # = 704
        
        self.meta_branch = MetadataMLPPlanA4Class(input_dim=meta_input_dim)
        self.meta_feature_dim = 64 
        
        # 💉 INYECCIÓN DE CONOCIMIENTO
        if rgb_weights_path:
            state_dict = torch.load(rgb_weights_path, map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            self.rgb_branch.load_state_dict(state_dict, strict=False)
            
        if meta_weights_path:
            state_dict = torch.load(meta_weights_path, map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            self.meta_branch.load_state_dict(state_dict, strict=False)

        # FUSIÓN
        self.total_features = self.rgb_feature_dim + self.meta_feature_dim
        self.fusion_dropout = nn.Dropout(0.5)
        
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
            nn.Linear(256, 4) # 🚨 FIJADO A 4 CLASES
        )

    def forward(self, img_rgb, features_meta):
        x_cnn = self.rgb_branch.cnn_backbone(img_rgb)
        x_cnn = self.rgb_branch.cnn_pool(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)
        x_vit = self.rgb_branch.vit_backbone(img_rgb)
        x_rgb = torch.cat((x_cnn, x_vit), dim=1)
        
        x_meta = self.meta_branch.mlp(features_meta)
        x_fused = torch.cat((x_rgb, x_meta), dim=1)
        x_fused = self.fusion_dropout(x_fused)
        
        return self.head_A(x_fused), self.head_B(x_fused)