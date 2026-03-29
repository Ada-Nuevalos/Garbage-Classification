"""
Modelo D4 — EfficientNetV2B0 Minimal (Serie descendente, paso 4)
================================================================
Proyecto : Clasificación Automática de Residuos - TrashNet (6 clases)
Asignatura: Aprendizaje Profundo

Serie descendente con Transfer Learning y cabeza mínima.
Backbone EfficientNetV2B0 pre-entrenado en ImageNet, DESCONGELADO.
Cabeza mínima: GAP → Dropout(0.3) → Softmax.

Siguiendo la jerarquía para reducir complejidad:
empezamos quitando todas las capas densas intermedias.

Arquitectura:
    EfficientNetV2B0 (ImageNet, descongelado)
    → GAP
    → Dropout(0.3)
    → Dense(6, softmax)

Técnica de la tijera — Transfer Learning:
    Sin freeze: backbone completamente descongelado desde el inicio.
    LR bajo (1e-5) para no dañar los pesos preentrenados.

Parámetros entrenables: ~7M (backbone + cabeza)
Parámetros totales:     ~7M

IMPORTANTE: EfficientNetV2B0 incluye su propio preprocesado interno
(include_preprocessing=True), por lo que NO se debe normalizar
la entrada externamente. Pasar imágenes en rango [0, 255].
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 6
INPUT_SHAPE = (224, 224, 3)
SEED        = 42


def build_effnet_minimal(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Construye el modelo D4 con EfficientNetV2B0 descongelado.

    Arquitectura:
        EfficientNetV2B0 (include_top=False, include_preprocessing=True)
        → GlobalAveragePooling2D
        → Dropout(0.3)
        → Dense(num_classes, softmax)
    """ 
    tf.random.set_seed(SEED)

    # Backbone pre-entrenado en ImageNet, SIN cabeza de clasificación
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        include_preprocessing=True,   # normalización interna incorporada
    )
    backbone.trainable = True        

    inputs  = keras.Input(shape=input_shape, name="input_image")
    x       = backbone(inputs, training=True)
    x       = layers.GlobalAveragePooling2D(name="GAP")(x)
    x       = layers.Dropout(0.3, name="dropout")(x)         
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

    model = keras.Model(inputs, outputs, name="D4_EffNet_Minimal")
    return model


def unfreeze_top(model, num_layers=20):
    """
    Abre la tijera para fine-tuning: descongela las últimas
    `num_layers` capas del backbone.

    Uso:
        model = unfreeze_top(model, num_layers=20)
        model.compile(optimizer=keras.optimizers.Adam(3e-5), ...)
        model.fit(...)
    """
    backbone = model.layers[1]          # EfficientNetV2B0
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    return model


if __name__ == "__main__":
    model = build_effnet_minimal()
    model.summary()
