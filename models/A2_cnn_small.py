"""
Modelo A2 — CNN Small (Serie ascendente, paso 2)
================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundo 

Cambio respecto a A1:
    Aumentamos los filtros de la conv de 6 → 32.
    Más capacidad de extracción de características,
    mismo esquema sin pooling intermedio.

Arquitectura:
    Rescaling → Conv2D(32, 3x3, relu, same) → GAP → Dense(6, softmax)

Parámetros aprox: (3×3×3×32)+32 + (32×6)+6 ≈ 1.094
"""

import keras
from keras import layers

NUM_CLASSES = 6
INPUT_SHAPE = (224, 224, 3)
SEED        = 42


def build_cnn_small(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    CNN pequeña: una sola conv con más filtros que A1, sin pooling.

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

        # Más filtros → más capacidad de representación
        layers.Conv2D(32, kernel_size=3,
                      activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(num_classes, activation='softmax'),
    ], name="A2_CNN_Small")

    return model


if __name__ == "__main__":
    model = build_cnn_small()
    model.summary()
