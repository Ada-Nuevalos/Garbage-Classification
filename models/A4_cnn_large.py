"""
Modelo A4 — CNN Large (Serie ascendente, paso 4)
================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundo

Cambios respecto a A3:
    1. Tercer bloque convolucional (128 filtros) con su MaxPooling.
    2. BatchNormalization en todos los bloques para estabilizar
       el entrenamiento al aumentar la profundidad.

Arquitectura:
    Rescaling
    → Conv2D(32,  3×3, relu, same) → BN → MaxPool(2×2)
    → Conv2D(64,  3×3, relu, same) → BN → MaxPool(2×2)
    → Conv2D(128, 3×3, relu, same) → BN → MaxPool(2×2)
    → GAP
    → Dense(6, softmax)

Parámetros aprox: ~95K
"""

import keras
from keras import layers

NUM_CLASSES = 6
INPUT_SHAPE = (224, 224, 3)
SEED        = 42


def build_cnn_large(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    CNN grande: tres bloques conv+BN+MaxPool.

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
        keras.layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),

        # Bloque 1 — igual que A3 + BN
        layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Bloque 2 — igual que A3 + BN + MaxPooling (A3 no tenía pool aquí)
        layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Bloque 3 — nuevo en A4
        layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax'),
    ], name="A4_CNN_Large")

    return model


if __name__ == "__main__":
    model = build_cnn_large()
    model.summary()