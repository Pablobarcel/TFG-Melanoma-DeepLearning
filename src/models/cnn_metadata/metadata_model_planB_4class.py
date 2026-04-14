# src/models/cnn_metadata/metadata_model_planB_4class.py
import torch
import torch.nn as nn

class MetadataMLPPlanB4Class(nn.Module):
    def __init__(self, input_dim):
        super(MetadataMLPPlanB4Class, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        self.head_A = nn.Linear(64, 1)
        self.head_B = nn.Linear(64, 4) # 🚨 4 CLASES FIJAS

    def forward(self, x):
        emb = self.mlp(x)
        return self.head_A(emb), self.head_B(emb)