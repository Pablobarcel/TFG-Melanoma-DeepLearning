# src/models/cnn_rgb_arp_vit/arp_vit_model.py

import torch
import torch.nn as nn
import timm

# Importamos nuestras dos arquitecturas previas
from src.models.cnn_vit_rgb.hybrid_model import HybridViTCNN
from src.models.cnn_arp.arp_model import ARPCNN

class TripleHybridModel(nn.Module):
    """
    El modelo definitivo del TFG: ResNet18 + ViT-Tiny (RGB) + CNN Custom (ARP).
    """
    def __init__(self, num_classes_headB=4, pretrained_rgb=True, arp_pretrained_path=None, rgb_pretrained_path=None):
        super(TripleHybridModel, self).__init__()
        
        # -----------------------------------------------------------
        # RAMA 1: RGB (ResNet18 + ViT)
        # -----------------------------------------------------------
        self.rgb_branch = HybridViTCNN(num_classes_headB=num_classes_headB, pretrained=pretrained_rgb)
        self.rgb_feature_dim = 512 + 192 
        
        # ---> CARGA DE PESOS EXPERTOS RGB <---
        if rgb_pretrained_path is not None:
            print(f"🔄 [Modelo Triple] Inyectando conocimiento previo en rama RGB desde: {rgb_pretrained_path}")
            try:
                checkpoint = torch.load(rgb_pretrained_path, map_location="cpu")
                if "model_state_dict" in checkpoint:
                    self.rgb_branch.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.rgb_branch.load_state_dict(checkpoint)
                print("✅ Pesos de la rama RGB cargados correctamente.")
            except Exception as e:
                print(f"⚠️ ¡Atención! No se pudieron cargar los pesos RGB. Error: {e}")

        # -----------------------------------------------------------
        # RAMA 2: ARP (CNN Custom)
        # -----------------------------------------------------------
        self.arp_branch = ARPCNN(num_classes_headB=num_classes_headB)
        self.arp_feature_dim = 512 

        # ---> CARGA DE PESOS EXPERTOS ARP <---
        if arp_pretrained_path is not None:
            print(f"🔄 [Modelo Triple] Inyectando conocimiento previo en rama ARP desde: {arp_pretrained_path}")
            try:
                # Cargamos el archivo a la CPU primero (para evitar problemas de memoria)
                checkpoint = torch.load(arp_pretrained_path, map_location="cpu")
                
                # Dependiendo de cómo guarde tu logger, puede ser directo o estar dentro de una key
                if "model_state_dict" in checkpoint:
                    self.arp_branch.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.arp_branch.load_state_dict(checkpoint)
                print("✅ Pesos de la rama ARP cargados correctamente.")
            except Exception as e:
                print(f"⚠️ ¡Atención! No se pudieron cargar los pesos ARP. Error: {e}")
                print("El entrenamiento continuará desde cero para esta rama.")

        # -----------------------------------------------------------
        # FUSIÓN (CONCATENACIÓN)
        # -----------------------------------------------------------
        self.total_features = self.rgb_feature_dim + self.arp_feature_dim # 1216
        
        self.classifier_dropout = nn.Dropout(0.5)
        
        self.head_A = nn.Sequential(
            nn.Linear(self.total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Binario
        )
        
        self.head_B = nn.Sequential(
            nn.Linear(self.total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_headB) # Multiclase
        )
        
    def forward(self, img_rgb, img_arp):
        # 1. Pasar imagen color por la rama RGB para extraer features
        # Llamamos directamente a las capas internas de tu modelo híbrido
        cnn_features = self.rgb_branch.cnn_backbone(img_rgb)
        cnn_pooled = self.rgb_branch.cnn_pool(cnn_features)
        cnn_flattened = torch.flatten(cnn_pooled, 1)
        
        # OJO AQUÍ: Tu ViT original recibe la imagen original 'img_rgb', no las 'cnn_features'
        vit_features = self.rgb_branch.vit_backbone(img_rgb)
        
        rgb_combined_features = torch.cat((cnn_flattened, vit_features), dim=1) # [Batch, 704]
        
        # 2. Pasar imagen polar por la rama ARP para extraer features
        # Hacemos las convoluciones y pasamos por la capa fc_shared
        x_arp = self.arp_branch.pool1(self.arp_branch.relu(self.arp_branch.bn1(self.arp_branch.conv1(img_arp))))
        x_arp = self.arp_branch.pool2(self.arp_branch.relu(self.arp_branch.bn2(self.arp_branch.conv2(x_arp))))
        x_arp = self.arp_branch.pool3(self.arp_branch.relu(self.arp_branch.bn3(self.arp_branch.conv3(x_arp))))
        x_arp = self.arp_branch.pool4(self.arp_branch.relu(self.arp_branch.bn4(self.arp_branch.conv4(x_arp))))
        
        x_arp = self.arp_branch.flatten(x_arp)
        arp_combined_features = self.arp_branch.relu(self.arp_branch.fc_shared(x_arp)) # [Batch, 512]
        
        # 3. FUSIÓN DE LOS DOS MUNDOS (Color + Geometría)
        final_features = torch.cat((rgb_combined_features, arp_combined_features), dim=1) # [Batch, 1216]
        final_features = self.classifier_dropout(final_features)
        
        # 4. Clasificación Final
        out_A = self.head_A(final_features)
        out_B = self.head_B(final_features)
        
        return out_A, out_B