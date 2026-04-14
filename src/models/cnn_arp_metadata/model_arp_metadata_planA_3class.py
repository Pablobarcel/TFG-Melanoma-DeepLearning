# src/models/cnn_arp_metadata/model_arp_metadata_planA_3class.py

import torch
import torch.nn as nn
from src.models.cnn_arp.arp_model_3class import ARPCNN3Class
from src.models.cnn_metadata.metadata_model_planA import MetadataMLP

class ARPMetadataModelPlanA3Class(nn.Module):
    def __init__(self, arp_weights_path=None, meta_weights_path=None, meta_input_dim=11):
        super(ARPMetadataModelPlanA3Class, self).__init__()
        
        # --- RAMA 1: ARP (CNN Custom) ---
        self.arp_branch = ARPCNN3Class(num_classes_headB=3)
        self.arp_feature_dim = 512 # Salida de self.fc_shared en tu modelo ARP
        
        # --- RAMA 2: Metadatos (Plan A) ---
        self.meta_branch = MetadataMLP(input_dim=meta_input_dim, num_classes_headB=3)
        self.meta_feature_dim = 64 # Salida del último Linear antes de las cabezas
        
        # --- 💉 INYECCIÓN DE CONOCIMIENTO ---
        if arp_weights_path:
            state_dict = torch.load(arp_weights_path, map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            self.arp_branch.load_state_dict(state_dict, strict=False)
            
        if meta_weights_path:
            state_dict = torch.load(meta_weights_path, map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            self.meta_branch.load_state_dict(state_dict, strict=False)

        # --- FUSIÓN ---
        self.total_features = self.arp_feature_dim + self.meta_feature_dim
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
            nn.Linear(256, 3) # 🚨 FIJADO A 3 CLASES
        )

    def extract_arp_features(self, x):
        """Extrae el vector 512D simulando el forward de la ARPCNN original hasta el fc_shared"""
        x = self.arp_branch.pool1(self.arp_branch.bn1(self.arp_branch.conv1(x)))
        x = self.arp_branch.pool2(self.arp_branch.bn2(self.arp_branch.conv2(x)))
        x = self.arp_branch.pool3(self.arp_branch.bn3(self.arp_branch.conv3(x)))
        x = self.arp_branch.pool4(self.arp_branch.bn4(self.arp_branch.conv4(x)))
        x = self.arp_branch.flatten(x)
        x = self.arp_branch.relu(x) # Activación antes del FC
        x = self.arp_branch.fc_shared(x)
        x = self.arp_branch.dropout(x)
        return x

    def forward(self, img_arp, features_meta):
        # 1. ARP Features
        x_arp = self.extract_arp_features(img_arp)
        
        # 2. Meta Features
        x_meta = self.meta_branch.mlp(features_meta)
        
        # 3. Fusión
        x_fused = torch.cat((x_arp, x_meta), dim=1)
        x_fused = self.fusion_dropout(x_fused)
        
        return self.head_A(x_fused), self.head_B(x_fused)