"""
Modelo A3 — CNN Medium (Serie ascendente, paso 3)
=================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundoa

Cambio respecto a A2:
    Añadimos MaxPooling tras la primera conv y una segunda
    capa convolucional con más filtros (64).
    Siguiendo la jerarquía: primero añadimos pooling,
    luego más capas.

Arquitectura:
    Rescaling
    → Conv2D(32, 3x3, relu, same) → MaxPool(2x2)
    → Conv2D(64, 3x3, relu, same)
    → GAP
    → Dense(6, softmax)

Parámetros aprox: ~20K
"""

import keras
from keras import layers

NUM_CLASSES = 6
INPUT_SHAPE = (224, 224, 3)
SEED        = 42


def build_cnn_medium(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    CNN media: dos bloques conv con pooling intermedio.

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
        layers.MaxPooling2D(pool_size=2),

        # Bloque 2
        layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),

        layers.GlobalAveragePooling2D(),

        layers.Dense(num_classes, activation='softmax'),
    ], name="A3_CNN_Medium")

    return model


if __name__ == "__main__":
    model = build_cnn_medium()
    model.summary()
