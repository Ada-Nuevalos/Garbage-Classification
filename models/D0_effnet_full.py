"""
Modelo D0 — EfficientNetV2B0 Full (Techo de la serie descendente)
=================================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundo

El modelo más complejo de la serie descendente.
Backbone EfficientNetV2B0 completamente descongelado desde el inicio.

Modelo de referencia (punto de partida de la serie D):
    Cabeza completa con BatchNormalization + Dense(256) + Dropout(0.5).

Arquitectura:
    EfficientNetV2B0 (ImageNet, descongelado)
    → GAP
    → BatchNormalization
    → Dense(256, relu) → Dropout(0.5)
    → Dense(6, softmax)

Técnica de la tijera — Transfer Learning:
    Sin freeze: backbone completamente descongelado desde el inicio.
    LR bajo (1e-5) para no dañar los pesos preentrenados.

Parámetros entrenables: ~7M (backbone + cabeza)
Parámetros totales:     ~7M
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 6
INPUT_SHAPE = (224, 224, 3)
SEED        = 42


def build_effnet_full(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    tf.random.set_seed(SEED)

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        include_preprocessing=True,
    )
    backbone.trainable = True

    inputs  = keras.Input(shape=input_shape, name="input_image")
    x       = backbone(inputs, training=True)
    x       = layers.GlobalAveragePooling2D(name="GAP")(x)
    x       = layers.BatchNormalization(name="BN")(x)
    x       = layers.Dense(256, activation="relu", name="dense_256")(x)
    x       = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

    model = keras.Model(inputs, outputs, name="D0_EffNet_Full")
    return model


def unfreeze_all(model):
    backbone = model.layers[1]
    backbone.trainable = True
    print(f"Backbone completo descongelado: {len(backbone.layers)} capas")
    return model


if __name__ == "__main__":
    model = build_effnet_full()
    model.summary()
