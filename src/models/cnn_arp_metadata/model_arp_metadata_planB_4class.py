import torch
import torch.nn as nn
from src.models.cnn_arp.arp_model import ARPCNN 
from src.models.cnn_metadata.metadata_model_planB_4class import MetadataMLPPlanB4Class

class ARPMetadataModelPlanB4Class(nn.Module):
    def __init__(self, arp_weights_path=None, meta_weights_path=None, meta_input_dim=50):
        super(ARPMetadataModelPlanB4Class, self).__init__()
        
        self.arp_branch = ARPCNN(num_classes_headB=4)
        self.arp_feature_dim = 512 
        self.meta_branch = MetadataMLPPlanB4Class(input_dim=meta_input_dim)
        self.meta_feature_dim = 64 
        
        if arp_weights_path:
            state_dict = torch.load(arp_weights_path, map_location="cpu")
            self.arp_branch.load_state_dict({k: v for k, v in state_dict.items() if 'head' not in k}, strict=False)
            
        if meta_weights_path:
            state_dict = torch.load(meta_weights_path, map_location="cpu")
            self.meta_branch.load_state_dict({k: v for k, v in state_dict.items() if 'head' not in k}, strict=False)

        self.total_features = self.arp_feature_dim + self.meta_feature_dim
        self.fusion_dropout = nn.Dropout(0.5)
        
        self.head_A = nn.Sequential(
            nn.Linear(self.total_features, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
        self.head_B = nn.Sequential(
            nn.Linear(self.total_features, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 4)
        )

    def extract_arp_features(self, x):
        x = self.arp_branch.pool1(self.arp_branch.bn1(self.arp_branch.conv1(x)))
        x = self.arp_branch.pool2(self.arp_branch.bn2(self.arp_branch.conv2(x)))
        x = self.arp_branch.pool3(self.arp_branch.bn3(self.arp_branch.conv3(x)))
        x = self.arp_branch.pool4(self.arp_branch.bn4(self.arp_branch.conv4(x)))
        x = self.arp_branch.dropout(self.arp_branch.fc_shared(self.arp_branch.relu(self.arp_branch.flatten(x))))
        return x

    def forward(self, img_arp, features_meta):
        x_arp = self.extract_arp_features(img_arp)
        x_meta = self.meta_branch.mlp(features_meta)
        x_fused = self.fusion_dropout(torch.cat((x_arp, x_meta), dim=1))
        return self.head_A(x_fused), self.head_B(x_fused)