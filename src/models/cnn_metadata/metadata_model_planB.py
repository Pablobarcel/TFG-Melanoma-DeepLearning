# src/models/cnn_metadata/metadata_model_planB.py
import torch
import torch.nn as nn

class MetadataMLPPlanB(nn.Module):
    def __init__(self, input_dim, num_classes_headB=3):
        super(MetadataMLPPlanB, self).__init__()
        
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
        self.head_B = nn.Linear(64, num_classes_headB)

    def forward(self, x):
        emb = self.mlp(x)
        return self.head_A(emb), self.head_B(emb)