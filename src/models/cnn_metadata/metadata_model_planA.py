# src/models/cnn_metadata/metadata_model_planA.py
import torch
import torch.nn as nn

class MetadataMLP(nn.Module):
    """
    Rama MLP para procesar metadatos del Plan A.
    Recibe un vector de características y devuelve logits para Head A y B.
    """
    def __init__(self, input_dim, num_classes_headB=3):
        super(MetadataMLP, self).__init__()
        
        # Bloque extractor de características tabulares
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
        
        # Salidas consistentes con tu arquitectura
        self.head_A = nn.Linear(64, 1)
        self.head_B = nn.Linear(64, num_classes_headB)

    def forward(self, x):
        features = self.mlp(x)
        outA = self.head_A(features)
        outB = self.head_B(features)
        return outA, outB