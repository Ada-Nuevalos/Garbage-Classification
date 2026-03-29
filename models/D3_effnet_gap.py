"""
Modelo D3 — EfficientNetV2B0 GAP (Serie descendente, paso 3)
============================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundo 

Cambio respecto a D2:
    Reducimos la capa densa intermedia de 128 a 6 neuronas.
    Mantenemos Dropout(0.3), generando un cuello de botella agresivo.

Arquitectura:
    EfficientNetV2B0 (ImageNet, descongelado)
    → GAP
    → Dense(6, relu)
    → Dropout(0.3)
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


def build_effnet_gap(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
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
    x       = layers.Dense(num_classes, activation="relu",
                            name="dense_hidden")(x)
    x       = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

    model = keras.Model(inputs, outputs, name="D3_EffNet_GAP")
    return model


def unfreeze_top(model, num_layers=20):
    backbone = model.layers[1]
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    return model


if __name__ == "__main__":
    model = build_effnet_gap()
    model.summary()
