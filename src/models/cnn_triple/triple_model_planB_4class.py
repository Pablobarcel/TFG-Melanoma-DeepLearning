# src/models/cnn_triple/triple_model_planB_4class.py

import torch
import torch.nn as nn

# 🚨 Importamos las 3 ramas expertas en su versión de 4 clases
from src.models.cnn_vit_rgb.hybrid_model import HybridViTCNN
from src.models.cnn_arp.arp_model import ARPCNN
from src.models.cnn_metadata.metadata_model_planB_4class import MetadataMLPPlanB4Class

class TripleModelPlanB4Class(nn.Module):
    def __init__(self, rgb_weights_path=None, arp_weights_path=None, meta_weights_path=None, meta_input_dim=50):
        super(TripleModelPlanB4Class, self).__init__()
        
        # --- RAMA 1: RGB (ResNet + ViT) ---
        self.rgb_branch = HybridViTCNN(num_classes_headB=4, pretrained=False)
        self.rgb_feature_dim = 512 + 192 # = 704
        
        # --- RAMA 2: ARP (CNN Custom) ---
        self.arp_branch = ARPCNN(num_classes_headB=4)
        self.arp_feature_dim = 512 
        
        # --- RAMA 3: Metadatos Avanzados (Plan B) ---
        self.meta_branch = MetadataMLPPlanB4Class(input_dim=meta_input_dim)
        self.meta_feature_dim = 64 
        
        # --- 💉 INYECCIÓN DE CONOCIMIENTO (Filtra las cabezas) ---
        if rgb_weights_path:
            state_dict = torch.load(rgb_weights_path, map_location="cpu")
            self.rgb_branch.load_state_dict({k: v for k, v in state_dict.items() if 'head' not in k}, strict=False)
            
        if arp_weights_path:
            state_dict = torch.load(arp_weights_path, map_location="cpu")
            self.arp_branch.load_state_dict({k: v for k, v in state_dict.items() if 'head' not in k}, strict=False)
            
        if meta_weights_path:
            state_dict = torch.load(meta_weights_path, map_location="cpu")
            self.meta_branch.load_state_dict({k: v for k, v in state_dict.items() if 'head' not in k}, strict=False)

        # --- FUSIÓN TOTAL ---
        self.total_features = self.rgb_feature_dim + self.arp_feature_dim + self.meta_feature_dim # 1280
        self.fusion_dropout = nn.Dropout(0.5)
        
        self.head_A = nn.Sequential(
            nn.Linear(self.total_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        self.head_B = nn.Sequential(
            nn.Linear(self.total_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4) # 🚨 FIJADO A 4 CLASES
        )

    def extract_arp_features(self, x):
        x = self.arp_branch.pool1(self.arp_branch.bn1(self.arp_branch.conv1(x)))
        x = self.arp_branch.pool2(self.arp_branch.bn2(self.arp_branch.conv2(x)))
        x = self.arp_branch.pool3(self.arp_branch.bn3(self.arp_branch.conv3(x)))
        x = self.arp_branch.pool4(self.arp_branch.bn4(self.arp_branch.conv4(x)))
        x = self.arp_branch.dropout(self.arp_branch.fc_shared(self.arp_branch.relu(self.arp_branch.flatten(x))))
        return x

    def forward(self, img_rgb, img_arp, features_meta):
        x_cnn = self.rgb_branch.cnn_pool(self.rgb_branch.cnn_backbone(img_rgb))
        x_vit = self.rgb_branch.vit_backbone(img_rgb)
        x_rgb = torch.cat((torch.flatten(x_cnn, 1), x_vit), dim=1)
        
        x_arp = self.extract_arp_features(img_arp)
        x_meta = self.meta_branch.mlp(features_meta)
        
        x_fused = torch.cat((x_rgb, x_arp, x_meta), dim=1)
        x_fused = self.fusion_dropout(x_fused)
        
        return self.head_A(x_fused), self.head_B(x_fused)