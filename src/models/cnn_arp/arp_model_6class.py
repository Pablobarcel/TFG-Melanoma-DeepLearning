# src/models/cnn_arp/arp_model_6class.py
import torch
import torch.nn as nn

class ARPCNN6Class(nn.Module):
    def __init__(self, num_classes_multiclass=4):
        super(ARPCNN6Class, self).__init__()
        
        # Backbones convolucionales (se mantienen igual para procesar 1 canal)[cite: 3]
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Clasificador compartido
        self.fc_shared = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        
        # Cabezas Duales Robustas (Siguiendo la lógica del modelo híbrido)
        self.head_A = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 1)
        )

        self.head_B = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes_multiclass)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.relu(self.fc_shared(x))
        return self.head_A(x), self.head_B(x)