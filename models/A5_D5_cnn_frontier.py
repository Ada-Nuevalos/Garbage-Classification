"""
Modelo A5 = D5 — CNN Frontier (PUNTO DE ENCUENTRO de la tijera)
===============================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundo 

Este modelo es simultáneamente:
  - El más complejo de la serie ascendente  (A5)
  - El más simple  de la serie descendente  (D5)

Es el punto donde las dos series se encuentran

Cambio respecto a A4:
    Cuarto bloque conv (256 filtros) + capa Dense intermedia (128).
    Siguiendo la jerarquía: añadimos capa densa antes
    de complicar más la arquitectura con Transfer Learning.

Arquitectura:
    Rescaling
    → Conv2D(32,  3x3, relu, same) → BN → MaxPool(2x2)
    → Conv2D(64,  3x3, relu, same) → BN → MaxPool(2x2)
    → Conv2D(128, 3x3, relu, same) → BN → MaxPool(2x2)
    → Conv2D(256, 3x3, relu, same) → BN
    → GAP
    → Dense(128, relu) → Dropout(0.4)
    → Dense(6, softmax)

Parámetros aprox: ~500K

"""

import keras
from keras import layers

NUM_CLASSES = 6
INPUT_SHAPE = (224, 224, 3)
SEED        = 42


def build_cnn_frontier(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    CNN frontera: máxima complejidad razonable desde cero con TrashNet.
    Punto de encuentro entre la serie ascendente y la descendente.

    Parámetros
    ----------
    input_shape : tuple, default (224, 224, 3)
    num_classes : int,   default 6

    Returns
    -------
    model : keras.Sequential
    """
    keras.utils.set_random_seed(SEED)

    model = keras.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=input_shape),

        # Bloque 1
        layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Bloque 2
        layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Bloque 3
        layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Bloque 4
        layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),

        # GAP + cabeza densa
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation='softmax'),
    ], name="A5_D5_CNN_Frontier")

    return model


if __name__ == "__main__":
    model = build_cnn_frontier()
    model.summary()
