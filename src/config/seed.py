# src/config/seed.py
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Fija todas las semillas globales de Python, Numpy, PyTorch y CUDA 
    para garantizar una reproducibilidad del 100% en los experimentos.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Si usas múltiples GPUs
    
    # Fuerzan a la tarjeta gráfica a ser determinista
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ [Config] Semillas globales fijadas a nivel de sistema: {seed}")