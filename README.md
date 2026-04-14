# TFG – Modelo multimodal para clasificación de lesiones cutáneas

Este repositorio contiene el código y los datos organizados para el Trabajo
Fin de Grado sobre una arquitectura híbrida **imagen + metadatos** para la
clasificación de lesiones cutáneas con aprendizaje profundo multitarea.

## Estructura del proyecto

```text
TFG/
├── data/              # todos los datos (originales, intermedios y procesados)
├── images_ISIC/       # imágenes dermatoscópicas de la base ISIC (~400k)
├── imagenes_Malignant/# imágenes de la base maligna (~27k)
├── notebooks/         # cuadernos Jupyter de análisis y preprocesamiento
├── src/               # (código fuente del modelo y entrenamiento)
└── docs/              # material para la memoria (figuras, borradores, etc.)
