# Clasificación de Residuos mediante Deep Learning

**Asignatura:** Aprendizaje Profundo 

**Autores:** Carlos Gómez Sáez y Ada Nuévalos Gadea

---

## 1. Descripción del Problema

La gestión eficiente de residuos es un desafío global crítico para la sostenibilidad ambiental. La clasificación manual en plantas de reciclaje es costosa, lenta y propensa a errores humanos. 

Este proyecto busca automatizar la clasificación de basura en **6 categorías** (cartón, vidrio, metal, papel, plástico y basura general) utilizando técnicas de Visión por Computador y Deep Learning. Para ello, se utiliza el dataset público **TrashNet** disponible en Kaggle.

## 2. Estado del Arte

La clasificación de residuos basada en imágenes ha evolucionado desde métodos clásicos de Machine Learning hasta arquitecturas complejas de Deep Learning.

### Evolución de las técnicas:

1.  **Enfoques Clásicos (Machine Learning):** Los trabajos pioneros utilizaron extracción de características manuales (SIFT, HOG) junto con clasificadores como *Support Vector Machines* (SVM). Estos métodos demostraron ser robustos pero limitados en su capacidad de generalización frente a variaciones de fondo e iluminación, alcanzando precisiones en torno al 63%.

2.  **Redes Convolucionales (CNNs) desde cero:** La implementación de CNNs simples (tipo AlexNet) entrenadas desde cero sobre este dataset pequeño (~2500 imágenes) suele sufrir de *overfitting*, obteniendo resultados pobres si no se ajustan correctamente los hiperparámetros y no se usan técnicas de regularización.

3.  **Transfer Learning y Modelos Pre-entrenados:** La literatura actual coincide en que el uso de *Transfer Learning* es la estrategia dominante. Modelos pre-entrenados en ImageNet como ResNet50, VGG16, MobileNet y DenseNet han elevado la precisión por encima del 90%.
    *   **Mao et al.** demostraron que optimizar las capas densas (*fully-connected*) mediante Algoritmos Genéticos (GA) junto con *Data Augmentation* agresivo puede llevar la precisión al 99.6% con DenseNet121.
    *   **White et al.** propusieron *WasteNet*, optimizando DenseNet para dispositivos *edge* (como Jetson Nano), logrando un 97%.
    *   **Alkılınç et al.** exploraron técnicas de *Ensemble Learning*, combinando las predicciones de ConvNeXt, ResNet y DenseNet para mejorar la robustez, alcanzando un 96% con medias ponderadas.

4.  **Tendencias actuales:** Los trabajos más recientes (2024-2025) se centran en mecanismos de atención (como CE-EfficientNetV2 de *Qiu et al.*) y arquitecturas ligeras para implementación en tiempo real.

### Tabla Comparativa de Modelos (Benchmark en TrashNet)

La siguiente tabla resume los resultados reportados en la literatura científica analizada para este problema sobre el dataset TrashNet (o variantes aumentadas del mismo):

| Modelo | Dataset | Accuracy | Precision | Recall | F1-Score | Referencia |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *WasteNet* (DenseNet modificado) | TrashNet | *97.0%* | 97.0% | 97.0% | 97.0% | White et al. [Source 3] |
| *GoogleNet + SVM* | TrashNet | *97.86%* | - | - | - | Özkaya et al. [Source 2] |
| *Ensemble (Weighted Avg)* | TrashNet | 96.0% | 94.0% | 97.0% | 95.0% | Alkılınç et al. [Source 9] |
| *CE-EfficientNetV2* | TrashNet | 96.5% | - | - | - | Qiu et al. [Source 4] |
| *MobileNetV2* | TrashNet | 95.17% | - | - | - | Buchade et al. [Source 5] |
| *DenseNet169* | TrashNet | 95.3% | 95.4% | 95.3% | 95.3% | White et al. [Source 3] |
| *ResNet50* | TrashNet | 93.7% | 93.7% | 93.7% | 93.7% | White et al. [Source 3] |
| *VGG16* | TrashNet | 92.8% | 92.7% | 92.8% | 92.7% | White et al. [Source 3] |
| *AlexNet* | TrashNet | 78.4% | 78.6% | 78.4% | 78.2% | White et al. [Source 3] |
| *SVM + SIFT* (Baseline) | TrashNet | 63.0% | 59.0% | 60.0% | - | Thung & Yang [Source 8] |

> **Nota:** Las precisiones superiores al 95% en la literatura generalmente involucran técnicas intensivas de *Data Augmentation* para multiplicar artificialmente el tamaño del dataset original. Los campos marcados con "-" indican que la métrica no fue reportada explícitamente en el resumen del estudio.

---

## 3. Estructura del Proyecto

El repositorio está organizado siguiendo una estructura estándar de ciencia de datos:

```text
garbage-classification-project/
│
│
├── models/                   # Definición de arquitecturas de modelos (Entregas futuras)
│   └── __init__.py
│
├── notebooks/                # Notebooks de Jupyter
│   ├── 1_EDA_y_Datos.ipynb   # Carga de datos, Análisis Exploratorio y Visualización
│   └── ...
│
├── src/                      # Scripts de utilidad y funciones auxiliares
│   └── __init__.py
│
├── README.md                 # Documentación del proyecto
└── requirements.txt          # Dependencias del entorno


