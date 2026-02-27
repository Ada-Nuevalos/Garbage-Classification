# Clasificación Automática de Residuos mediante Deep Learning

**Asignatura:** Aprendizaje Profundo

**Autores:** Carlos Gómez Sáez y Ada Nuévalos Gadea

---

Este repositorio contiene el proyecto final de la asignatura de Aprendizaje Profundo. El objetivo principal es desarrollar un sistema automatizado capaz de clasificar residuos en diferentes categorías (vidrio, papel, cartón, plástico, metal, etc.) utilizando técnicas de Visión por Computador y Deep Learning. Este proyecto busca mejorar la eficiencia en las plantas de reciclaje y promover la sostenibilidad ambiental.

![Infografía del Proyecto](reports/figures/infografia_proyecto.png)

## 1. Definición del Problema y Datos

### El Problema

La correcta separación de residuos es un cuello de botella crítico en el proceso de reciclaje. La clasificación manual es lenta, costosa y propensa a errores. Este proyecto aborda este problema como una tarea de **Clasificación de Imágenes Multiclase (Supervisada)**.

### El Dataset

Para este proyecto, utilizamos el dataset estándar de referencia en la literatura académica: **TrashNet** (y sus variantes extendidas).

- **Origen:** Recopilado por Gary Thung y Mindy Yang (Stanford University).
- **Contenido:** 2.527 imágenes en total.
- **Clases (6 categorías):**
  1.  Vidrio (Glass) - 501 imágenes
  2.  Papel (Paper) - 594 imágenes
  3.  Cartón (Cardboard) - 403 imágenes
  4.  Plástico (Plastic) - 482 imágenes
  5.  Metal (Metal) - 410 imágenes
  6.  Basura general (Trash) - 137 imágenes

### Dimensiones de los Datos

| Propiedad               | Valor                                  |
| :---------------------- | :------------------------------------- |
| Imágenes totales        | 2.527                                  |
| Resolución original     | 512 x 384 px                           |
| Resolución preprocesada | 224 x 224 px                           |
| Canales de color        | 3 (RGB)                                |
| Tipo de dato (X)        | `float32`, normalizado [0, 1]          |
| Shape de X              | `(2527, 224, 224, 3)`                  |
| Shape de y              | `(2527,)` — enteros codificados [0..5] |
| Ratio de desbalanceo    | 4.34:1 (paper: 594 vs trash: 137)      |

### Muestras del Dataset

A continuación se muestran ejemplos representativos de cada una de las 6 categorías:

[TrashNet Dataset en Kaggle](https://www.kaggle.com/datasets/feyzazkefe/trashnet)

<table align="center">
  <tr>
    <td align="center">
      <img src="data/raw/metal/metal1.jpg" width="120"><br>
      <b>Metal</b>
    </td>
    <td align="center">
      <img src="data/raw/cardboard/cardboard1.jpg" width="120"><br>
      <b>Cardboard</b>
    </td>
    <td align="center">
      <img src="data/raw/paper/paper1.jpg" width="120"><br>
      <b>Paper</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="data/raw/plastic/plastic1.jpg" width="120"><br>
      <b>Plastic</b>
    </td>
    <td align="center">
      <img src="data/raw/trash/trash1.jpg" width="120"><br>
      <b>Trash</b>
    </td>
    <td align="center">
      <img src="data/raw/glass/glass1.jpg" width="120"><br>
      <b>Glass</b>
    </td>
  </tr>
</table>

## 2. Estado del Arte (SOTA)

La clasificación de residuos basada en imágenes ha evolucionado desde métodos clásicos de Machine Learning hasta arquitecturas complejas de Deep Learning.

### Evolución de las técnicas

1.  **Enfoques Clásicos (Machine Learning):** Los trabajos pioneros utilizaron extracción de características manuales (SIFT, HOG) junto con clasificadores como _Support Vector Machines_ (SVM). Estos métodos demostraron ser robustos pero limitados en su capacidad de generalización frente a variaciones de fondo e iluminación, alcanzando precisiones en torno al 63% [1].

2.  **Redes Convolucionales (CNNs) desde cero:** La implementación de CNNs simples (tipo AlexNet) entrenadas desde cero sobre este dataset pequeño (~2.500 imágenes) suele sufrir de _overfitting_, obteniendo resultados modestos si no se ajustan correctamente los hiperparámetros y no se aplican técnicas de regularización [3].

3.  **Transfer Learning y Modelos Pre-entrenados:** La literatura actual coincide en que el uso de _Transfer Learning_ es la estrategia dominante. Modelos pre-entrenados en ImageNet como ResNet50, VGG16, MobileNet y DenseNet han elevado la precisión por encima del 90%.
    - **Mao et al.** [6] demostraron que optimizar las capas densas mediante Algoritmos Genéticos (GA) junto con _Data Augmentation_ agresivo puede llevar la precisión al 99.6% con DenseNet121.
    - **White et al.** [3] propusieron _WasteNet_, optimizando DenseNet para dispositivos _edge_ (como Jetson Nano), logrando un 97%.
    - **Alkılınç et al.** [9] exploraron técnicas de _Ensemble Learning_, combinando las predicciones de ConvNeXt, ResNet y DenseNet para mejorar la robustez, alcanzando un 96% con medias ponderadas.

4.  **Tendencias actuales (2024-2025):** Los trabajos más recientes se centran en mecanismos de atención (como CE-EfficientNetV2 de Qiu et al. [4]) y arquitecturas ligeras para implementación en tiempo real.

### Tabla Comparativa de Modelos (Benchmark en TrashNet)

La siguiente tabla resume los resultados reportados en la literatura científica analizada para este problema sobre el dataset TrashNet (o variantes aumentadas del mismo):

| Modelo                           | Dataset  | Accuracy | Precision | Recall | F1-Score | Referencia           |
| :------------------------------- | :------- | :------- | :-------- | :----- | :------- | :------------------- |
| _WasteNet_ (DenseNet modificado) | TrashNet | _97.0%_  | 97.0%     | 97.0%  | 97.0%    | White et al. [3]     |
| _GoogleNet + SVM_                | TrashNet | _97.86%_ | -         | -      | -        | Özkaya & Seyfi [2]   |
| _Ensemble (Weighted Avg)_        | TrashNet | 96.0%    | 94.0%     | 97.0%  | 95.0%    | Alkılınç et al. [9]  |
| _CE-EfficientNetV2_              | TrashNet | 96.5%    | -         | -      | -        | Qiu et al. [4]       |
| _MobileNetV2_                    | TrashNet | 95.17%   | -         | -      | -        | Buchade & Bhoite [5] |
| _DenseNet169_                    | TrashNet | 95.3%    | 95.4%     | 95.3%  | 95.3%    | White et al. [3]     |
| _ResNet50_                       | TrashNet | 93.7%    | 93.7%     | 93.7%  | 93.7%    | White et al. [3]     |
| _VGG16_                          | TrashNet | 92.8%    | 92.7%     | 92.8%  | 92.7%    | White et al. [3]     |
| _AlexNet_                        | TrashNet | 78.4%    | 78.6%     | 78.4%  | 78.2%    | White et al. [3]     |
| _SVM + SIFT_ (Baseline)          | TrashNet | 63.0%    | 59.0%     | 60.0%  | -        | Thung & Yang [1]     |

> **Nota:** Las precisiones superiores al 95% en la literatura generalmente involucran técnicas intensivas de _Data Augmentation_ para multiplicar artificialmente el tamaño del dataset original. Los campos marcados con "-" indican que la métrica no fue reportada explícitamente en el estudio correspondiente.

## 3. Métricas de Evaluación

### Objetivo Matemático

Este proyecto se formula como un problema de **clasificación multiclase supervisada**: dada una imagen de entrada, el modelo debe asignarla a una de las 6 categorías de residuos. Para resolver este problema de forma óptima, se distinguen tres niveles:

1.  **Objetivo de clasificación:** Aprender una función $f: \mathbb{R}^{224 \times 224 \times 3} \rightarrow \{0, 1, ..., 5\}$ que minimice el error de generalización.

2.  **Función de coste para optimizar:** Se emplea la **Entropía Cruzada Categórica** (_Categorical Cross-Entropy_) como función de pérdida durante el entrenamiento:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

donde $N$ es el número de muestras, $C = 6$ es el número de clases, $y_{i,c}$ es la etiqueta real (one-hot) y $\hat{y}_{i,c}$ es la probabilidad predicha por el modelo para la clase $c$. Esta función penaliza más las predicciones que se alejan de la clase correcta y es diferenciable, permitiendo la optimización por descenso de gradiente.

3.  **Métricas de progreso para evaluar la calidad del modelo:**

- **Accuracy (Exactitud):** Porcentaje global de predicciones correctas. Útil como indicador general pero insuficiente cuando hay desbalanceo entre clases.

$$\text{Accuracy} = \frac{\text{Predicciones correctas}}{\text{Total de muestras}}$$

- **Precision:** Proporción de predicciones positivas que son realmente correctas. Responde a la pregunta: _de todo lo que el modelo predijo como clase $c$, cuánto acertó realmente._

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

- **Recall (Sensibilidad):** Proporción de muestras reales de una clase que el modelo detecta correctamente. Responde a: _de todas las muestras que realmente son clase $c$, cuántas detectó._

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

- **F1-Score (Macro):** Media armónica de Precision y Recall, promediada sobre todas las clases sin ponderar por frecuencia. Crucial en este proyecto dado el desbalanceo (ratio 4.34:1), ya que pondera por igual cada clase.

$$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} \qquad \Rightarrow \qquad F1_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} F1_c$$

- **Confusion Matrix:** Representación visual que muestra dónde se concentran los errores del modelo (ej. confundir plástico con vidrio), permitiendo diagnosticar debilidades por clase.

## 4. Resultados Experimentales

### Tabla Comparativa de Modelos

| Modelo                  | Nº Parámetros | Split | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| :---------------------- | ------------: | :---- | -------: | ----------------: | -------------: | ---------: |
| **Regresión Logística** |         7,782 | Train |   0.9955 |            0.9946 |         0.9965 |     0.9955 |
|                         |               | Val   |   0.5013 |            0.5000 |         0.5065 |     0.5023 |
|                         |               | Test  |   0.6000 |            0.5864 |         0.5802 |     0.5815 |
| **SVM**                 |     2,182,851 | Train |   0.9972 |            0.9975 |         0.9975 |     0.9975 |
|                         |               | Val   |   0.6491 |            0.6639 |         0.6209 |     0.6347 |
|                         |               | Test  |   0.7184 |            0.7455 |         0.6856 |     0.7035 |
| **CNN Simple**          |           210 | Train |   0.3535 |            0.3688 |         0.2863 |     0.2467 |
|                         |               | Val   |   0.3615 |            0.3435 |         0.2927 |     0.2508 |
|                         |               | Test  |   0.3553 |            0.3651 |         0.2896 |     0.2579 |

## 5. Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```text
├── data/                   # Carpeta para almacenar el dataset (raw y processed)
│   └── raw/
│   └── processed
│       
├── notebooks/              # Notebooks de Jupyter para experimentación
│   └── 1_EDA_Garbage_Classification.ipynb  
│   └── 2.1.Modelo_Lineal_Garbage_Classification.ipynb  
│   └── 2.2.Modelo_ML_Garbage_Classification.ipynb
│   └── 2.3.Modelo_Red_Neuronal_Simple.ipynb 
│ 
├── models/                 # Definición de arquitecturas de modelos (.py)
│   └── simple_cnn.py      
│ 
├── src/                    # Código fuente modular
│   └── utils/              # Funciones auxiliares de carga y visualización
│
├── sota_references/        # Papers de referencia del estado del arte
├── reports/                # Informes y figuras generadas
├── requirements.txt        # Dependencias del proyecto
├── code_exmaple            # Carpeta con código de ejemplo de otros proyectos (SOTA)
└── README.md               # Información general del proyecto
```

## 6. Referencias

Se presentan las referencias utilizadas, cada una acompañada de una frase que sintetiza su contribución al campo.

[1] G. Thung y M. Yang, _"Classification of Trash for Recyclability Status,"_ CS 229 Project Report, Stanford University, 2016.

- Trabajo fundacional que creó el dataset TrashNet (2.527 imágenes, 6 clases) y estableció la primera línea base con SVM + SIFT, alcanzando un 63% de accuracy.

[2] U. Özkaya y L. Seyfi, _"Fine-Tuning Models Comparisons on Garbage Classification for Recyclability,"_ arXiv:1908.04393, 2019.

- Comparación de arquitecturas preentrenadas (AlexNet, VGG16, GoogleNet, ResNet) combinadas con clasificadores Softmax y SVM, logrando el mejor resultado con GoogleNet + SVM (97.86%).

[3] G. White, C. Cabrera, A. Palade, F. Li y S. Clarke, _"WasteNet: Waste Classification at the Edge for Smart Bins,"_ arXiv:2006.05873, 2020.

- Propone WasteNet, una variante de DenseNet optimizada para despliegue en dispositivos edge (Jetson Nano), alcanzando 97% con DenseNet modificado y evaluando múltiples arquitecturas sobre TrashNet.

[4] W. Qiu, C. Xie y J. Huang, _"An Improved EfficientNetV2 for Garbage Classification,"_ arXiv:2503.21208, 2025.

- Introduce un módulo de atención Channel-Efficient (CE-Attention) sobre EfficientNetV2 para mejorar la extracción de características sin escalar dimensiones, logrando 96.5% de accuracy.

[5] S. J. Buchade y S. Bhoite, _"Comparative Study of ML Algorithms for Garbage Classification,"_ Research Square (Preprint), DOI: 10.21203/rs.3.rs-3903806/v1, 2024.

- Estudio comparativo de MobileNetV2, InceptionV3 y ResNet aplicados a la clasificación de residuos, con MobileNetV2 como modelo más eficiente (95.17%).

[6] W.-L. Mao, W.-C. Chen, C.-T. Wang y Y.-H. Lin, _"Recycling Waste Classification Using Optimized Convolutional Neural Network,"_ Resources, Conservation and Recycling, vol. 164, 105132, 2021.

- Propone una CNN optimizada mediante algoritmos evolutivos para clasificación de residuos reciclables, abordando la optimización mediante técnicas de hiperparámetros automáticos.

[7] R. Shukurov, _"Garbage Classification Based on Fine-Tuned State-of-the-Art Models,"_ 9th International Conference on Control, Decision and Information Technologies (CoDIT), IEEE, pp. 1-6, 2023.

- Evalúa modelos fine-tuned del estado del arte sobre múltiples datasets de residuos, analizando la transferibilidad de características entre dominios.

[8] K. Rathod, C. Vyas, K. Makvana, K. Kandoriya y A. Nimavat, _"Garbage Classification Based on Dense Network (GCDN) Using Transfer Learning and Modified Hyper Parameter,"_ International Journal of Intelligent Systems and Applications in Engineering, vol. 12, no. 4, 2024.

- Propone un enfoque basado en DenseNet201 con transfer learning y ajuste fino de hiperparámetros para mejorar la clasificación de basura, focalizándose en la reutilización de características densas.

[9] A. Alkılınç, F. Yıldırım Okay, İ. Kök y S. Özdemir, _"Deep Ensemble Learning Model for Waste Classification Systems,"_ Sustainability, vol. 18, no. 24, 2026.

- Diseña un sistema de ensemble profundo combinando múltiples CNNs mediante promediado ponderado, logrando 96% de accuracy con mayor robustez que los modelos individuales.

[10] M. Nahiduzzaman, M. F. Ahamed, M. Naznine, M. J. Karim et al., _"An Automated Waste Classification System Using Deep Learning Techniques,"_ Knowledge-Based Systems, vol. 313, 112840, 2025.

- Sistema automatizado que aplica técnicas profundas para clasificación de residuos a gran escala, enfocado en la sostenibilidad ambiental y la eficiencia del reciclaje con aplicaciones en entornos industriales.

