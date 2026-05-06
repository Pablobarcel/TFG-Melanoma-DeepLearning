# src/data/transforms.py

import torch
from torchvision import transforms

# ==========================================
# TRANSFORMACIONES RAMA RGB
# ==========================================
def get_train_transforms():
    """
    Transformaciones robustas para la rama RGB.
    Optimizadas para preservar la semántica del color y simular la elasticidad.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # Simulación de la elasticidad de la piel (presión del dermatoscopio)
        # transforms.ElasticTransform(alpha=50.0, sigma=5.0), 
        
        # Brillo/contraste para simular diferentes iluminaciones (SIN Hue)
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.0),
        
        transforms.ToTensor(),
        
        # Random Erasing va SIEMPRE después de ToTensor. Simula pelos o reflejos.
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_eval_transforms():
    """ Validación limpia RGB """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ==========================================
# TRANSFORMACIONES RAMA ARP (Grises/Polares)
# ==========================================
def get_train_transforms_arp():
    """
    Transformaciones para rama ARP.
    ATENCIÓN: En coordenadas polares, el Eje Y es el Radio (Centro -> Borde).
    NUNCA aplicar VerticalFlip, o invertiremos el núcleo de la lesión hacia afuera.
    """
    return transforms.Compose([        
        # Horizontal Flip = Invertir el sentido del ángulo (espejo). Totalmente válido.
        transforms.RandomHorizontalFlip(p=0.5),
        
        transforms.ToTensor(),
        
        # Random Erasing ayuda a que la red no memorice un borde específico
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_eval_transforms_arp():
    """ Validación limpia ARP """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])