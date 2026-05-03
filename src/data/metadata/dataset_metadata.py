# src/data/metadata/dataset_metadata.py
import torch
import numpy as np
from torch.utils.data import Dataset

class MetadataDataset(Dataset):
    def __init__(self, df, mean_age=None, std_age=None):
        self.df = df.copy() 
        
        self.ohe_cols = [c for c in df.columns if 'sex_' in c or 'anatom_site_general_' in c]
        self.feature_cols = ['age_approx'] + self.ohe_cols
        
        self.malignant_classes = [1, 2, 3]

        # --- LÓGICA DE Z-SCORE DINÁMICO (VECTORIZADA ULTRA-RÁPIDA) ---
        
        # 1. Creamos una máscara booleana rápida para saber qué filas NO son -1.0
        valid_mask = self.df['age_approx'] != -1.0
        valid_ages = self.df.loc[valid_mask, 'age_approx']
        
        # 2. Cálculo de medias (Train) o herencia de medias (Val)
        if mean_age is None or std_age is None:
            self.mean_age = valid_ages.mean()
            self.std_age = valid_ages.std()
        else:
            self.mean_age = mean_age
            self.std_age = std_age
            
        # 3. Aplicamos el Z-Score DE GOLPE (Vectorizado) a las filas válidas
        safe_std = self.std_age if self.std_age > 0 else 1e-6
        self.df.loc[valid_mask, 'age_approx'] = (valid_ages - self.mean_age) / safe_std
        
        # 4. A los valores que ERAN -1.0, les ponemos el nuevo centinela -5.0 DE GOLPE
        self.df.loc[~valid_mask, 'age_approx'] = -5.0
        
        # DEFINIR EL NUEVO MAPEO
        # Original: 0:NV, 1:MEL, 2:BCC, 3:SCC, 4:BKL, 5:BG
        # Nuevo: 0:BEN (NV, BKL, BG), 1:MEL, 2:BCC, 3:SCC
        self.label_map = {0:0, 4:0, 5:0, 1:1, 2:2, 3:3}
        self.malignant_classes = [1, 2, 3]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        
        # APLICAR MAPEO DINÁMICO
        original_label = int(row['target'])
        y_headB = self.label_map[original_label]
        
        # Head A sigue siendo binario basado en si la nueva etiqueta es 1, 2 o 3
        y_headA = 1.0 if y_headB in self.malignant_classes else 0.0
            
        return features, torch.tensor(y_headA, dtype=torch.float), torch.tensor(y_headB, dtype=torch.long)