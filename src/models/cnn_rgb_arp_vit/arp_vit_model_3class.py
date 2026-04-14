# src/models/cnn_rgb_arp_vit/arp_vit_model_3class.py

import torch
import torch.nn as nn
import timm

from src.models.cnn_vit_rgb.hybrid_model_3class import HybridViTCNN
from src.models.cnn_arp.arp_model_3class import ARPCNN3Class

class TripleHybridModel3Class(nn.Module):
    """
    El modelo definitivo del TFG: ResNet18 + ViT-Tiny (RGB) + CNN Custom (ARP).
    Configurado para 3 clases (NMSC).
    """
    def __init__(self, num_classes_headB=3, pretrained_rgb=True, arp_pretrained_path=None, rgb_pretrained_path=None):
        super(TripleHybridModel3Class, self).__init__()
        
        # --- RAMA 1: RGB (ResNet18 + ViT) ---
        self.rgb_branch = HybridViTCNN(num_classes_headB=num_classes_headB, pretrained=pretrained_rgb)
        self.rgb_feature_dim = 512 + 192 
        
        # --- RAMA 2: ARP (CNN Custom) ---
        self.arp_branch = ARPCNN3Class(num_classes_headB=num_classes_headB)
        self.arp_feature_dim = 512 

        # --- 💉 INYECCIÓN 1: CARGA DE PESOS ARP ---
        if arp_pretrained_path is not None:
            print(f"🔄 [Modelo Triple] Inyectando conocimiento previo en rama ARP desde: {arp_pretrained_path}")
            try:
                checkpoint = torch.load(arp_pretrained_path, map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.arp_branch.load_state_dict(state_dict)
                print("✅ Pesos de la rama ARP cargados correctamente.")
            except Exception as e:
                print(f"⚠️ ¡Atención! No se pudieron cargar los pesos ARP. Error: {e}")

        # --- 💉 INYECCIÓN 2: CARGA DE PESOS RGB (ResNet + ViT) ---
        if rgb_pretrained_path is not None:
            print(f"🔄 [Modelo Triple] Inyectando conocimiento previo en rama RGB desde: {rgb_pretrained_path}")
            try:
                checkpoint = torch.load(rgb_pretrained_path, map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                
                # Filtramos las cabezas de clasificación originales del modelo doble
                # para quedarnos solo con el conocimiento visual (Backbones)
                state_dict_filtrado = {k: v for k, v in state_dict.items() if 'head' not in k}
                
                self.rgb_branch.load_state_dict(state_dict_filtrado, strict=False)
                print("✅ Pesos de la rama RGB (ResNet + ViT) cargados correctamente.")
            except Exception as e:
                print(f"⚠️ ¡Atención! No se pudieron cargar los pesos RGB. Error: {e}")

        # --- FUSIÓN (CONCATENACIÓN) ---
        self.total_features = self.rgb_feature_dim + self.arp_feature_dim # 1216
        self.classifier_dropout = nn.Dropout(0.5)
        
        # Head A: Tarea Binaria (Más sencilla, mantenemos 256)
        self.head_A = nn.Sequential(
            nn.Linear(self.total_features, 256),
            nn.BatchNorm1d(256),    # <-- MEJORA: Estabiliza la fusión
            nn.GELU(),              # <-- MEJORA: Activación moderna (mejor que ReLU)
            nn.Dropout(0.3),
            nn.Linear(256, 1) 
        )
        
        # Head B: Tarea Multiclase (Más compleja, subimos a 512)
        self.head_B = nn.Sequential(
            nn.Linear(self.total_features, 512), # <-- MEJORA: Más capacidad cognitiva (mas neuronas para evitar perder mucha informacion)
            nn.BatchNorm1d(512),                 # <-- MEJORA: Estabiliza la fusión
            nn.GELU(),                           # <-- MEJORA: Activación moderna
            nn.Dropout(0.3),
            nn.Linear(512, num_classes_headB) 
        )
        
    def forward(self, img_rgb, img_arp):
        cnn_features = self.rgb_branch.cnn_backbone(img_rgb)
        cnn_pooled = self.rgb_branch.cnn_pool(cnn_features)
        cnn_flattened = torch.flatten(cnn_pooled, 1)
        
        vit_features = self.rgb_branch.vit_backbone(img_rgb)
        rgb_combined_features = torch.cat((cnn_flattened, vit_features), dim=1) 
        
        x_arp = self.arp_branch.pool1(self.arp_branch.relu(self.arp_branch.bn1(self.arp_branch.conv1(img_arp))))
        x_arp = self.arp_branch.pool2(self.arp_branch.relu(self.arp_branch.bn2(self.arp_branch.conv2(x_arp))))
        x_arp = self.arp_branch.pool3(self.arp_branch.relu(self.arp_branch.bn3(self.arp_branch.conv3(x_arp))))
        x_arp = self.arp_branch.pool4(self.arp_branch.relu(self.arp_branch.bn4(self.arp_branch.conv4(x_arp))))
        
        x_arp = self.arp_branch.flatten(x_arp)
        arp_combined_features = self.arp_branch.relu(self.arp_branch.fc_shared(x_arp)) 
        
        final_features = torch.cat((rgb_combined_features, arp_combined_features), dim=1) 
        final_features = self.classifier_dropout(final_features)
        
        out_A = self.head_A(final_features)
        out_B = self.head_B(final_features)
        
        return out_A, out_B