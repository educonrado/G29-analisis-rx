import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler 

#imagenes
import tensorflow as tf
import cv2
import numpy as np
import time 

# --- IMPORTACIONES ADICIONALES PARA EL MODELO ---
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

RUTA_DATASET = "D:\Dev\inn_311\datasets"

IMG_WIDTH = 128 
IMG_HEIGHT = 128
BATCH_SIZE = 8 

LABEL_NORMAL = 0
LABEL_COVID = 1
LABEL_NEUMONIA = 2
NUM_CLASSES = 3 


def obtener_dataframe_etiquetado():
    """
    Crea un DataFrame con las rutas de las imágenes y sus etiquetas
    para las tres clases: Normal (0), COVID (1), Neumonía (2).
    """
    image_paths = []
    labels = []

    # Procesar imágenes de COVID
    covid_images_dir = Path(RUTA_DATASET) / "covid" / "images"
    if covid_images_dir.exists():
        for nombre_imagen in os.listdir(covid_images_dir):
            if nombre_imagen.lower().endswith(".png"):
                image_paths.append(str(covid_images_dir / nombre_imagen))
                labels.append(LABEL_COVID)

    # Procesar imágenes de Neumonía
    neumonia_images_dir = Path(RUTA_DATASET) / "neumonia" / "images"
    if neumonia_images_dir.exists():
        for nombre_imagen in os.listdir(neumonia_images_dir):
            if nombre_imagen.lower().endswith(".png"):
                image_paths.append(str(neumonia_images_dir / nombre_imagen))
                labels.append(LABEL_NEUMONIA)

    # Procesar imágenes Normales
    normal_images_dir = Path(RUTA_DATASET) / "normal" / "images"
    if normal_images_dir.exists():
        for nombre_imagen in os.listdir(normal_images_dir):
            if nombre_imagen.lower().endswith(".png"):
                image_paths.append(str(normal_images_dir / nombre_imagen))
                labels.append(LABEL_NORMAL)

    print(f"Total images: {len(image_paths)}")
    print(f"Total labels: {len(labels)}")

    df = pd.DataFrame({
        'paths': image_paths,
        'label': labels
    })

    print(df.head())
    print("\nDistribución de clases en el DataFrame completo (NUEVO ETIQUETADO):")
    print(df['label'].value_counts()) 
    
    expected_labels = {LABEL_NORMAL, LABEL_COVID, LABEL_NEUMONIA}
    found_labels = set(df['label'].unique())
    if not expected_labels.issubset(found_labels):
        print(f"Advertencia: No se encontraron todas las clases esperadas. Esperadas: {expected_labels}, Encontradas: {found_labels}")

    return df

def dividir_dataframe_entrenamiento_validacion_prueba(dataframe, test_size=0.15, val_size=0.15, random_state=42):
    train_val_df, test_df = train_test_split(
        dataframe,  
        test_size=test_size,
        stratify=dataframe['label'],
        random_state=random_state
    )
    
    relative_val_size = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    print(f"Tamanio set Entrenamiento: {len(train_df)}")
    print(f"Tamanio set Validacion: {len(val_df)}")
    print(f"Tamanio set Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def balancear_dataframe_entrenamiento(train_df, random_state=42):
    """
    Aplica sobremuestreo (oversampling) a la clase minoritaria en el DataFrame de entrenamiento
    para balancear las clases.
    """
    print("\n--- Balanceo del conjunto de entrenamiento ---")

    X_train = train_df['paths'].values.reshape(-1, 1)
    y_train = train_df['label'].values 

    oversampler = RandomOverSampler(random_state=random_state)

    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    train_df_balanced = pd.DataFrame({
        'paths': X_train_resampled.flatten(),
        'label': y_train_resampled
    })

    print("\nDistribución de clases en train_df (antes del balanceo):")
    print(train_df['label'].value_counts())
    print("\nDistribución de clases en train_df_balanced (después del balanceo):")
    print(train_df_balanced['label'].value_counts())
    print(f"\nNuevo tamaño del set de Entrenamiento balanceado: {len(train_df_balanced)}")

    return train_df_balanced


def _load_and_preprocess_single_image(image_path_tensor, label_tensor):
    """
    Función auxiliar para cargar, decodificar, redimensionar y normalizar una sola imagen.
    VERSIÓN CORREGIDA sin tf.py_function.
    """
    # Leer el archivo como bytes
    image_raw = tf.io.read_file(image_path_tensor)
    
    # Decodificar la imagen
    image = tf.image.decode_image(image_raw, channels=3, expand_animations=False)
    
    # Asegurar que la imagen tenga 3 canales
    image = tf.ensure_shape(image, [None, None, 3])
    
    # Convertir a float32 y normalizar
    image = tf.cast(image, tf.float32) / 255.0
    
    # Redimensionar la imagen
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    
    # Asegurar la forma final
    image = tf.ensure_shape(image, [IMG_HEIGHT, IMG_WIDTH, 3])
    
    # Asegurar que la etiqueta sea int32
    label_tensor = tf.cast(label_tensor, tf.int32)
    
    return image, label_tensor

def crear_generador_dataset(dataframe, img_width, img_height, batch_size, shuffle=True, augment=False):
    """
    Crea un objeto tf.data.Dataset para cargar y preprocesar imágenes por lotes con concurrencia.
    VERSIÓN CORREGIDA sin tf.py_function.
    """
    # Crear datasets de rutas y etiquetas
    path_ds = tf.data.Dataset.from_tensor_slices(dataframe['paths'].values)
    label_ds = tf.data.Dataset.from_tensor_slices(dataframe['label'].values)
    
    # Combinar rutas y etiquetas
    dataset = tf.data.Dataset.zip((path_ds, label_ds))
    
    # Mapear la función de preprocesamiento (SIN tf.py_function)
    dataset = dataset.map(
        _load_and_preprocess_single_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Filtrar imágenes que no se pudieron cargar (opcional)
    dataset = dataset.filter(lambda img, label: tf.reduce_all(tf.shape(img) > 0))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(dataframe), 10000))  # Limitar buffer size
    
    # Aplicar data augmentation si se solicita
    if augment:
        dataset = dataset.map(
            lambda img, label: (aplicar_augmentacion(img), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Crear batches y prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def aplicar_augmentacion(image):
    """
    Aplica aumentación de datos a las imágenes de entrenamiento.
    """
    # Volteo horizontal aleatorio
    image = tf.image.random_flip_left_right(image)
    
    # Brillo aleatorio
    image = tf.image.random_brightness(image, 0.1)
    
    # Contraste aleatorio
    image = tf.image.random_contrast(image, 0.9, 1.1)
    
    # Asegurar que los valores estén en [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

# --- FUNCIÓN PARA CREAR EL MODELO ---
def crear_modelo_clasificacion_movilenetv2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), num_classes=NUM_CLASSES, learning_rate=0.0001):
    """
    Define y compila un modelo de clasificación de imágenes basado en MobileNetV2
    para entornos con recursos limitados.
    """
    inputs = Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False
    x = base_model(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    print("--- Inicio de la preparación de datos ---")

    df = obtener_dataframe_etiquetado()
    print("\nDistribución de clases en el DataFrame completo:")
    print(df['label'].value_counts())

    train_df, val_df, test_df = dividir_dataframe_entrenamiento_validacion_prueba(df)

    train_df_balanced = balancear_dataframe_entrenamiento(train_df)

    train_dataset = crear_generador_dataset(train_df_balanced, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, shuffle=True, augment=False)
    val_dataset = crear_generador_dataset(val_df, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, shuffle=False, augment=False)
    test_dataset = crear_generador_dataset(test_df, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, shuffle=False, augment=False)

    print("\n--- ¡Preparación de generadores de datos completada! ---")
    print("Ahora tienes los objetos `tf.data.Dataset` listos para entrenar tu modelo.")
    print(f"Ejemplo de un batch de entrenamiento (forma de la imagen, forma de la etiqueta):")
    try:
        for images_batch, labels_batch in train_dataset.take(1):
            print(f"  Forma del batch de imágenes: {images_batch.shape}")
            print(f"  Forma del batch de etiquetas: {labels_batch.shape}")
            print(f"  Tipo de datos de imágenes: {images_batch.dtype}")
            print(f"  Rango de píxeles (min/max): {tf.reduce_min(images_batch).numpy()}/{tf.reduce_max(images_batch).numpy()}")
        print("\nPrimer batch cargado y procesado correctamente. ¡El pipeline de datos funciona!")
    except Exception as e:
        print(f"\n¡ERROR al cargar el primer batch del dataset! Mensaje: {e}")
    
    print("\n--- Inicio de la definición y entrenamiento del modelo ---")

    model = crear_modelo_clasificacion_movilenetv2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
        num_classes=NUM_CLASSES,
        learning_rate=0.0001
    )

    print("\nResumen del modelo:")
    model.summary()

    checkpoint_filepath = 'best_model.keras'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_on_plateau_callback = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=5,
        mode='max',
        min_lr=1e-7,
        verbose=1
    )

    callbacks_list = [
        model_checkpoint_callback,
        early_stopping_callback,
        reduce_lr_on_plateau_callback
    ]

    EPOCHS = 50 
    print(f"\nEntrenando el modelo por hasta {EPOCHS} épocas...")
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks_list
    )
    
    end_time = time.time()
    training_duration_seconds = end_time - start_time
    training_duration_hours = training_duration_seconds / 3600

    print("\n--- Entrenamiento del modelo completado ---")
    print(f"Tiempo total de entrenamiento: {training_duration_seconds:.2f} segundos ({training_duration_hours:.2f} horas)")

    print("\n--- Evaluación del modelo en el conjunto de prueba ---")
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
    print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

    print("\n--- Pasos Completados: Definición, Entrenamiento y Evaluación Inicial del Modelo ---")
    print(f"El mejor modelo ha sido guardado en: {checkpoint_filepath}")