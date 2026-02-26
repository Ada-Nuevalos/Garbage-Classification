"""
Utilidades para el Proyecto de Clasificación de Residuos

Contiene funciones para:
- Configuración del entorno (Colab/local)
- Carga de datos preprocesados
- División estratificada de datos (train/val/test)
- Configuración de semillas para reproducibilidad
"""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Semilla global para reproducibilidad
SEED = 44

# ============================================
# 1. CONFIGURACIÓN DE SEMILLAS
# ============================================

def set_seeds(seed=SEED):
    """
    Configura las semillas para reproducibilidad en numpy, tensorflow, etc.
    
    Args:
        seed (int): Valor de la semilla. Default: SEED (44)
    """
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    


# ============================================
# 2. CONFIGURACIÓN DEL ENTORNO
# ============================================

def setup_environment():
    """
    Detecta el entorno de ejecución (Colab o local) y retorna las rutas necesarias.
    
    Returns:
        dict: Diccionario con claves:
            - 'in_colab' (bool): True si está en Google Colab
            - 'data_path' (Path): Ruta al archivo trashnet_224x224.npz
            - 'project_root' (Path): Ruta raíz del proyecto
    """
    
    # Detectar si estamos en Colab
    try:
        from google.colab import drive
        in_colab = True
    except ImportError:
        in_colab = False
    
    if in_colab:
        # Montar Google Drive
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Definir rutas para Colab
        data_path = Path('/content/drive/MyDrive/Proyecto-AP/data/processed/trashnet_224x224.npz')
        project_root = Path('/content/drive/MyDrive/Proyecto-AP')
        
        print("Ejecutando en Google Colab")
    else:
        # Definir rutas para ejecución local
        # utils.py está en src/, proyecto_root es un nivel arriba
        project_root = Path(__file__).parent.parent.absolute()
        data_path = project_root / 'data' / 'processed' / 'trashnet_224x224.npz'
        
        print("Ejecutando en local")
    
    return {
        'in_colab': in_colab,
        'data_path': data_path,
        'project_root': project_root
    }


# ============================================
# 3. CARGA DE DATOS
# ============================================

def load_data(data_path):
    """
    Carga los datos preprocesados desde el archivo .npz.
    
    Args:
        data_path (str or Path): Ruta al archivo trashnet_224x224.npz
        
    Returns:
        tuple: (X, y, class_names) donde:
            - X (np.ndarray): Array de imágenes (N, 224, 224, 3)
            - y (np.ndarray): Array de etiquetas (N,)
            - class_names (np.ndarray): Nombres de las clases
            
    Raises:
        FileNotFoundError: Si el archivo no existe
    """
    
    data_path = Path(data_path)
    
    if not data_path.is_file():
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    class_names = data['class_names']
    
    print(f"Cargando datos desde: {data_path}")
    print(f"  X.shape: {X.shape}")
    print(f"  y.shape: {y.shape}")
    print(f"  Clases: {', '.join(class_names)}")
    print()
    
    return X, y, class_names


# ============================================
# 4. DIVISIÓN ESTRATIFICADA DE DATOS
# ============================================

def split_data(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=SEED):
    """
    Divide los datos en conjuntos train/validación/test de forma estratificada.
    
    Args:
        X (np.ndarray): Array de características (imágenes)
        y (np.ndarray): Array de etiquetas
        train_ratio (float): Proporción para entrenamiento (default: 0.70)
        val_ratio (float): Proporción para validación (default: 0.15)
        test_ratio (float): Proporción para test (default: 0.15)
        random_state (int): Semilla para reproducibilidad (default: SEED=44)
        
    Returns:
        dict: Diccionario con las splits:
            - 'X_train', 'y_train'
            - 'X_val', 'y_val'
            - 'X_test', 'y_test'
            
    Notes:
        - La divisió es estratificada para mantener la distribución de clases
        - Se asume que train_ratio + val_ratio + test_ratio = 1.0
    """
    
    # Verificar que los ratios sumen 1
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Los ratios deben sumar 1.0, pero suman {total}")
    
    # Primer split: separar train del resto (temp)
    test_val_ratio = 1 - train_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_val_ratio,
        stratify=y,
        random_state=random_state
    )
    
    # Segundo split: separar val y test del temp
    val_test_ratio = test_ratio / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_test_ratio,
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"División estratificada (seed={random_state}):")
    print(f"  Train: {X_train.shape[0]} imágenes ({train_ratio*100:.0f}%)")
    print(f"  Val:   {X_val.shape[0]} imágenes ({val_ratio*100:.0f}%)")
    print(f"  Test:  {X_test.shape[0]} imágenes ({test_ratio*100:.0f}%)")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


# ============================================
# 5. FUNCIÓN PRINCIPAL DE INICIALIZACIÓN
# ============================================

def initialize_project():
    """
    Realiza la inicialización completa del proyecto:
    1. Configura las semillas
    2. Detecta el entorno (Colab/local)
    3. Carga los datos
    4. Divide estratificadamente
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, class_names)
    """
    
    # Configurar semillas (silencioso)
    set_seeds(SEED)
    
    # Configurar entorno
    env_config = setup_environment()
    print()
    
    # Cargar datos
    X, y, class_names = load_data(env_config['data_path'])
    
    # Dividir datos
    splits = split_data(X, y)
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names

