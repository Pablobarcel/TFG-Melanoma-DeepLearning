# src/models/cnn_rgb_arp_vit/arp_vit_model_3class.py

import torch
import torch.nn as nn
import timm
from src.models.cnn_vit_rgb.hybrid_model_3class import HybridViTCNN
from src.models.cnn_arp.arp_model_3class import ARPCNN3Class

class TripleHybridModel3Class(nn.Module):
    def __init__(self, num_classes_headB=3, pretrained_rgb=True, arp_pretrained_path=None, rgb_pretrained_path=None):
        super(TripleHybridModel3Class, self).__init__()
        
        self.rgb_branch = HybridViTCNN(num_classes_headB=num_classes_headB, pretrained=pretrained_rgb)
        self.rgb_feature_dim = 512 + 192 
        
        self.arp_branch = ARPCNN3Class(num_classes_headB=num_classes_headB)
        self.arp_feature_dim = 512 

        if arp_pretrained_path:
            checkpoint = torch.load(arp_pretrained_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.arp_branch.load_state_dict({k: v for k, v in state_dict.items() if 'head' not in k}, strict=False)

        if rgb_pretrained_path:
            checkpoint = torch.load(rgb_pretrained_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.rgb_branch.load_state_dict({k: v for k, v in state_dict.items() if 'head' not in k}, strict=False)

        self.total_features = self.rgb_feature_dim + self.arp_feature_dim 
        self.classifier_dropout = nn.Dropout(0.5)
        
        self.head_A = nn.Sequential(
            nn.Linear(self.total_features, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1) 
        )
        self.head_B = nn.Sequential(
            nn.Linear(self.total_features, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, num_classes_headB) 
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
        
        return self.head_A(final_features), self.head_B(final_features)