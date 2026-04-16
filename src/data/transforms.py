# src/data/transforms.py

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_train_transforms(img_size=224):
    """
    Transformaciones robustas para la rama RGB.
    Optimizadas para preservar la semántica del color (sin Hue) 
    y simular la elasticidad de la piel.
    """
    return transforms.Compose([
        # Rotación completa con relleno 'reflect' para no crear bordes negros artificiales
        transforms.RandomRotation(degrees=180),
        
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # Simulación de la elasticidad de la piel (presión del dermatoscopio)
        # alpha controla la intensidad de la deformación
        transforms.ElasticTransform(alpha=50.0, sigma=5.0), 
        
        # --- 2. Color Seguro ---
        # ELIMINADO: Hue (Peligroso en medicina)
        # Mantenemos brillo/contraste para simular diferentes iluminaciones
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        
        # --- 3. Conversión ---
        transforms.ToTensor(),
        
        # --- 4. Oclusión (Robustez) ---
        # Random Erasing va SIEMPRE después de ToTensor.
        # Simula pelos, reflejos o artefactos tapando parte de la lesión.
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        
        # --- 5. Normalización ---
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_eval_transforms(img_size=224):
    """
    Validación limpia: Solo Lanczos y Normalización.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_train_transforms_arp():
    """
    Transformaciones para rama ARP (Polar / Grises).
    NOTA: En coordenadas polares:
    - Eje X = Ángulo
    - Eje Y = Radio
    """
    return transforms.Compose([        
        # Horizontal Flip en ARP = Invertir el sentido del ángulo (reloj vs contra-reloj).
        # ES VÁLIDO y aporta variedad.
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Vertical Flip en ARP = Invertir el radio (de dentro a fuera).
        # A veces es raro biológicamente, pero matemáticamente aporta robustez.
        transforms.RandomVerticalFlip(p=0.5),
        
        # NO APLICAMOS ROTACIÓN GEOMÉTRICA AQUI.
        # Rotar una imagen polar rectangular la deformaría sin sentido físico.
        
        transforms.ToTensor(),
        
        # Random Erasing también es útil aquí para que la red no dependa
        # de un solo tramo del borde.
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_eval_transforms_arp():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])