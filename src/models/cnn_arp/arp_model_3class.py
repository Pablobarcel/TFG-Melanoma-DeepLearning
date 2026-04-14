# src/models/cnn_arp/arp_model_3class.py

import torch
import torch.nn as nn

class ARPCNN3Class(nn.Module):
    """
    Red Convolucional Ligera diseñada para procesar imágenes ARP.
    Entrada: Tensor de (Batch, 1, 224, 224)
    Configurado para 3 clases (NMSC).
    """
    def __init__(self, num_classes_headB=3): # <--- 3 por defecto
        super(ARPCNN3Class, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.fc_shared = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.4)
        
        self.head_A = nn.Linear(512, 1) 
        self.head_B = nn.Linear(512, num_classes_headB) 

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        x = self.flatten(x)
        x = self.relu(self.fc_shared(x))
        x = self.dropout(x)
        
        out_A = self.head_A(x)
        out_B = self.head_B(x)
        
        return out_A, out_B