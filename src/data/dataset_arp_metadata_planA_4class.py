import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from src.config.paths import SPLITTED_DATA_DIR, PROJECT_ROOT

class ARPMetadataDatasetPlanA4Class(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.img_dir = PROJECT_ROOT / "src" / "data" / "processed" / "imagenes_ARP_224" 
        
        self.num_cols = ['age_approx', 'clin_size_long_diam_mm']
        self.sex_cols = [c for c in self.df.columns if c.startswith('sex_')]
        self.site_cols = [c for c in self.df.columns if c.startswith('anatom_site_general_')]
        self.feature_cols = self.num_cols + self.sex_cols + self.site_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_name = os.path.join(self.img_dir, f"{row['isic_id']}.jpg")
        try:
            image = Image.open(img_name).convert('L') # 1 Canal (Grises/Polar)
        except FileNotFoundError:
            image = Image.new('L', (224, 224), 0)
            
        if self.transform:
            image = self.transform(image)
            
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        yA = torch.tensor(row['head_A_label'], dtype=torch.float32)
        yB = torch.tensor(row['head_B_label'], dtype=torch.long)
        
        return image, features, yA, yB