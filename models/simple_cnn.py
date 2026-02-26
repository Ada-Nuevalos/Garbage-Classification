import keras
from keras import layers


def build_simple_cnn(input_shape, num_classes):
    """
    CNN minimalista ultra-simple:
    - 1 capa convolucional con tantos filtros como clases
    - Global Average Pooling
    - Dense con tantas neuronas como clases
    
    Para 6 clases: 6 filtros -> Global pooling -> 6 neuronas
    Parámetros: (3*3*3*6) + 6 + (6*6) + 6 = 162 + 6 + 36 + 6 = 210 parámetros
    """
    model = keras.Sequential([
        layers.Conv2D(num_classes, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
