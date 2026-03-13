"""
Modelo A1 — Simple CNN (Punto de partida de la serie ascendente)
================================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundo

Arquitectura:
    Conv2D(6, 3x3, relu, same) → GAP → Dense(6, softmax)

Parámetros: (3×3×3×6) + 6 + (6×6) + 6 = 162 + 6 + 36 + 6 = 210
Este es el modelo más tonto posible con algo de sentido:
un único filtro por clase, GAP que colapsa cada mapa a un escalar,
y una capa densa de salida.
"""

import keras
from keras import layers


# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
INPUT_SHAPE = (224, 224, 3)
SEED        = 42


def build_simple_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    CNN minimalista ultra-simple (modelo más tonto).

    Parámetros
    ----------
    input_shape : tuple, default (224, 224, 3)
    num_classes : int,  default 6

    Returns
    -------
    model : keras.Sequential
    """
    keras.utils.set_random_seed(SEED)

    model = keras.Sequential([
        # Normalización de entrada [0,255] → [0,1]
        layers.Rescaling(1.0 / 255, input_shape=input_shape),

        # Única capa convolucional: tantos filtros como clases
        layers.Conv2D(num_classes, kernel_size=3,
                      activation='relu', padding='same'),

        # GAP: colapsa cada mapa de características a un único escalar
        layers.GlobalAveragePooling2D(),

        # Capa de salida
        layers.Dense(num_classes, activation='softmax'),
    ], name="A1_SimpleCNN")

    return model


# ---------------------------------------------------------------------------
# Test rápido
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_simple_cnn()
    model.summary()
    # Parámetros esperados: 210
