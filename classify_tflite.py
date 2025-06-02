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

IMG_WIDTH = 64 
IMG_HEIGHT = 64
BATCH_SIZE = 8 

LABEL_NORMAL = 0
LABEL_COVID = 1
LABEL_NEUMONIA = 2
NUM_CLASSES = 3 


def verificar_archivos_existentes(df):
 
    archivos_originales = len(df)

    existe_mask = df['paths'].apply(os.path.exists)
    archivos_existentes = existe_mask.sum()
    archivos_faltantes = archivos_originales - archivos_existentes
    
    if archivos_faltantes > 0:
        print(f"¡ADVERTENCIA! {archivos_faltantes} archivos no encontrados de {archivos_originales} totales")
        
        archivos_perdidos = df[~existe_mask]['paths'].head(5).tolist()
        print("Ejemplos de archivos faltantes:")
        for archivo in archivos_perdidos:
            print(f"  - {archivo}")

        df_limpio = df[existe_mask].copy().reset_index(drop=True)
        
        print(f"\nDataset limpiado:")
        print(f"  Archivos originales: {archivos_originales}")
        print(f"  Archivos existentes: {archivos_existentes}")
        print(f"  Archivos eliminados: {archivos_faltantes}")
        
        print("\nDistribución de clases después de la limpieza:")
        print(df_limpio['label'].value_counts())
        
        return df_limpio
    else:
        print(f"✓ Todos los {archivos_originales} archivos existen")
        return df


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
        print(f"Procesando directorio COVID: {covid_images_dir}")
        covid_count = 0
        for nombre_imagen in os.listdir(covid_images_dir):
            if nombre_imagen.lower().endswith((".png", ".jpg", ".jpeg")):
                ruta_completa = str(covid_images_dir / nombre_imagen)
                if os.path.exists(ruta_completa):
                    image_paths.append(ruta_completa)
                    labels.append(LABEL_COVID)
                    covid_count += 1
        print(f"  Encontradas {covid_count} imágenes de COVID")
    else:
        print(f"¡ADVERTENCIA! Directorio COVID no encontrado: {covid_images_dir}")

    # Procesar imágenes de Neumonía
    neumonia_images_dir = Path(RUTA_DATASET) / "neumonia" / "images"
    if neumonia_images_dir.exists():
        print(f"Procesando directorio Neumonía: {neumonia_images_dir}")
        neumonia_count = 0
        for nombre_imagen in os.listdir(neumonia_images_dir):
            if nombre_imagen.lower().endswith((".png")):
                ruta_completa = str(neumonia_images_dir / nombre_imagen)
                if os.path.exists(ruta_completa):
                    image_paths.append(ruta_completa)
                    labels.append(LABEL_NEUMONIA)
                    neumonia_count += 1
        print(f"  Encontradas {neumonia_count} imágenes de Neumonía")
    else:
        print(f"¡ADVERTENCIA! Directorio Neumonía no encontrado: {neumonia_images_dir}")

    # Procesar imágenes Normales
    normal_images_dir = Path(RUTA_DATASET) / "normal" / "images"
    if normal_images_dir.exists():
        print(f"Procesando directorio Normal: {normal_images_dir}")
        normal_count = 0
        for nombre_imagen in os.listdir(normal_images_dir):
            if nombre_imagen.lower().endswith((".png", ".jpg", ".jpeg")):
                ruta_completa = str(normal_images_dir / nombre_imagen)
                if os.path.exists(ruta_completa):
                    image_paths.append(ruta_completa)
                    labels.append(LABEL_NORMAL)
                    normal_count += 1
        print(f"  Encontradas {normal_count} imágenes Normales")
    else:
        print(f"¡ADVERTENCIA! Directorio Normal no encontrado: {normal_images_dir}")

    print(f"\nTotal images encontradas: {len(image_paths)}")
    print(f"Total labels: {len(labels)}")

    if len(image_paths) == 0:
        raise ValueError("No se encontraron imágenes. Verifica la estructura de directorios.")

    df = pd.DataFrame({
        'paths': image_paths,
        'label': labels
    })

    print("\nPrimeras 5 filas del DataFrame:")
    print(df.head())
    print("\nDistribución de clases inicial:")
    print(df['label'].value_counts())
    
    # Verificar que tenemos todas las clases esperadas
    expected_labels = {LABEL_NORMAL, LABEL_COVID, LABEL_NEUMONIA}
    found_labels = set(df['label'].unique())
    if not expected_labels.issubset(found_labels):
        print(f"¡ADVERTENCIA! No se encontraron todas las clases esperadas.")
        print(f"Esperadas: {expected_labels}, Encontradas: {found_labels}")

    return df

def dividir_dataframe_entrenamiento_validacion_prueba(dataframe, test_size=0.15, val_size=0.15, random_state=42):
    # Verificar que tenemos suficientes muestras de cada clase
    class_counts = dataframe['label'].value_counts()
    min_samples = class_counts.min()
    
    if min_samples < 3:
        raise ValueError(f"Clase con muy pocas muestras ({min_samples}). Se necesitan al menos 3 muestras por clase.")
    
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
    
    print(f"\nDivisión del dataset:")
    print(f"  Entrenamiento: {len(train_df)} muestras")
    print(f"  Validación: {len(val_df)} muestras")
    print(f"  Prueba: {len(test_df)} muestras")
    
    return train_df, val_df, test_df

def balancear_dataframe_entrenamiento(train_df, random_state=42):
    """
    Aplica sobremuestreo (oversampling) a la clase minoritaria en el DataFrame de entrenamiento
    para balancear las clases.
    """
    print("\n--- Balanceo del conjunto de entrenamiento ---")

    X_train = train_df['paths'].values.reshape(-1, 1)
    y_train = train_df['label'].values 

    print("Distribución ANTES del balanceo:")
    print(train_df['label'].value_counts())

    oversampler = RandomOverSampler(random_state=random_state)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    train_df_balanced = pd.DataFrame({
        'paths': X_train_resampled.flatten(),
        'label': y_train_resampled
    })

    print("Distribución DESPUÉS del balanceo:")
    print(train_df_balanced['label'].value_counts())
    print(f"Nuevo tamaño del conjunto de entrenamiento: {len(train_df_balanced)}")

    return train_df_balanced


def _load_and_preprocess_single_image(image_path_tensor, label_tensor):
    """
    Función auxiliar para cargar, decodificar, redimensionar y normalizar una sola imagen.
    Incluye manejo de errores mejorado.
    """
    try:
        # Leer el archivo como bytes
        image_raw = tf.io.read_file(image_path_tensor)
        
        # Decodificar la imagen
        image = tf.image.decode_image(image_raw, channels=3, expand_animations=False)
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
        
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        black_image = tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
        label_tensor = tf.cast(label_tensor, tf.int32)
        return black_image, label_tensor

def crear_generador_dataset(dataframe, img_width, img_height, batch_size, shuffle=True, augment=False):
    """
    Crea un objeto tf.data.Dataset para cargar y preprocesar imágenes por lotes con concurrencia.
    Versión con mejor manejo de errores.
    """
    if len(dataframe) == 0:
        raise ValueError("DataFrame vacío. No se pueden crear datasets.")
    
    print(f"Creando dataset con {len(dataframe)} muestras")
    
    # Crear datasets de rutas y etiquetas
    path_ds = tf.data.Dataset.from_tensor_slices(dataframe['paths'].values)
    label_ds = tf.data.Dataset.from_tensor_slices(dataframe['label'].values)
    
    # Combinar rutas y etiquetas
    dataset = tf.data.Dataset.zip((path_ds, label_ds))
    
    # Mapear la función de preprocesamiento
    dataset = dataset.map(
        _load_and_preprocess_single_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Filtrar imágenes que no se pudieron cargar correctamente
    dataset = dataset.filter(
        lambda img, label: tf.reduce_all(tf.shape(img) > 0) and tf.reduce_max(img) > 0
    )
    
    if shuffle:
        buffer_size = min(len(dataframe), 1000)  # Limitar buffer size
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
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
    
    # Saturación aleatoria
    image = tf.image.random_saturation(image, 0.9, 1.1)
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

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
    base_model.trainable = False  # Congelar capas base inicialmente
    x = base_model(x)

    # Agregar dropout para regularización
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    print("=== INICIO DEL PIPELINE DE CLASIFICACIÓN ===")
    print("--- Paso 1: Preparación de datos ---")

    try:
        # Obtener DataFrame inicial
        df = obtener_dataframe_etiquetado()
        
        # Verificar archivos existentes y limpiar dataset
        df = verificar_archivos_existentes(df)
        
        # Verificar que tenemos suficientes datos
        if len(df) < 10:
            raise ValueError(f"Dataset muy pequeño ({len(df)} muestras). Se necesitan al menos 10 muestras.")
        
        print("\nDistribución final de clases:")
        print(df['label'].value_counts())

        # Dividir en conjuntos de entrenamiento, validación y prueba
        train_df, val_df, test_df = dividir_dataframe_entrenamiento_validacion_prueba(df)

        # Balancear conjunto de entrenamiento
        train_df_balanced = balancear_dataframe_entrenamiento(train_df)

        print("\n--- Paso 2: Creación de datasets ---")
        
        # Crear datasets
        train_dataset = crear_generador_dataset(
            train_df_balanced, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, 
            shuffle=True, augment=True  # Activar augmentación para entrenamiento
        )
        val_dataset = crear_generador_dataset(
            val_df, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, 
            shuffle=False, augment=False
        )
        test_dataset = crear_generador_dataset(
            test_df, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, 
            shuffle=False, augment=False
        )

        print("--- Datasets creados exitosamente ---")
        
        # Validar que los datasets funcionan
        print("\n--- Paso 3: Validación de datasets ---")
        try:
            for images_batch, labels_batch in train_dataset.take(1):
                print(f"✓ Batch de entrenamiento cargado:")
                print(f"  Forma de imágenes: {images_batch.shape}")
                print(f"  Forma de etiquetas: {labels_batch.shape}")
                print(f"  Tipo de datos: {images_batch.dtype}")
                print(f"  Rango de valores: [{tf.reduce_min(images_batch).numpy():.3f}, {tf.reduce_max(images_batch).numpy():.3f}]")
        except Exception as e:
            print(f"¡ERROR al validar dataset! {e}")
            raise

        print("\n--- Paso 4: Definición del modelo ---")
        
        model = crear_modelo_clasificacion_movilenetv2(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
            num_classes=NUM_CLASSES,
            learning_rate=0.0001
        )

        print("\nResumen del modelo:")
        model.summary()

        checkpoint_filepath = 'best_model.keras'
        callbacks_list = [
            ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=5,
                mode='max',
                min_lr=1e-7,
                verbose=1
            )
        ]

        print("\n--- Paso 5: Entrenamiento del modelo ---")
        
        EPOCHS = 50 
        print(f"Iniciando entrenamiento por hasta {EPOCHS} épocas...")
        
        start_time = time.time()
        
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks_list
        )
        
        end_time = time.time()
        training_duration = end_time - start_time

        print(f"\n=== ENTRENAMIENTO COMPLETADO ===")
        print(f"Tiempo total: {training_duration:.2f} segundos ({training_duration/3600:.2f} horas)")

        print("\n--- Paso 6: Evaluación final ---")
        loss, accuracy = model.evaluate(test_dataset)
        print(f"Pérdida en conjunto de prueba: {loss:.4f}")
        print(f"Precisión en conjunto de prueba: {accuracy:.4f} ({accuracy*100:.1f}%)")

        print(f"\n✓ Modelo guardado en: {checkpoint_filepath}")
        print("=== PIPELINE COMPLETADO EXITOSAMENTE ===")

    except Exception as e:
        print(f"\n❌ ERROR EN EL PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        raise