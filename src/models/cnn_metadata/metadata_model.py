import torch
import torch.nn as nn

class MetadataMLP(nn.Module):
    def __init__(self, input_dim=13, num_classes_multiclass=4):
        super(MetadataMLP, self).__init__()
        
        # Extractor de características compartido
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Head A: Binario (Benigno vs Maligno) -> 1 salida (Logit para Sigmoid)
        self.head_A = nn.Linear(64, 1)
        
        # Head B: Multiclase (4 enfermedades) -> 4 salidas (Logits para CrossEntropy)
        self.head_B = nn.Linear(64, num_classes_multiclass)

    def forward(self, x):
        features = self.shared_mlp(x)
        outA = self.head_A(features) # Logit binario
        outB = self.head_B(features) # Logits multiclase
        return outA, outB