"""
Modelo D1 — EfficientNetV2B0 Top (Serie descendente, paso 1)
============================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundo — Valero Laparra

Cambio respecto a D2:
    Ampliamos la capa Dense a 256 neuronas.

Arquitectura:
    EfficientNetV2B0 (ImageNet, descongelado)
    → GAP
    → Dense(256, relu) → Dropout(0.4)
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


def build_effnet_top(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
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
    x       = layers.Dense(256, activation="relu", name="dense_256")(x)
    x       = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

    model = keras.Model(inputs, outputs, name="D1_EffNet_Top")
    return model


def unfreeze_top(model, num_layers=30):
    backbone = model.layers[1]
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    print(f"Capas descongeladas: {num_layers} de {len(backbone.layers)}")
    return model


if __name__ == "__main__":
    model = build_effnet_top()
    model.summary()