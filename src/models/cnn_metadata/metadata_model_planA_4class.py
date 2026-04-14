# src/models/cnn_metadata/metadata_model_planA_4class.py
import torch
import torch.nn as nn

class MetadataMLPPlanA4Class(nn.Module):
    def __init__(self, input_dim):
        super(MetadataMLPPlanA4Class, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.head_A = nn.Linear(64, 1)
        self.head_B = nn.Linear(64, 4) # 🚨 4 CLASES FIJAS

    def forward(self, x):
        features = self.mlp(x)
        return self.head_A(features), self.head_B(features)