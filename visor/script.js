let model;
// Asegúrate de que IMG_SIZE coincida con el IMG_WIDTH/HEIGHT de tu modelo Python
const IMG_SIZE = 128; 
const CLASS_NAMES = ['Normal', 'COVID', 'Neumonía']; // ¡Asegúrate que el orden sea el mismo que tus etiquetas numéricas!

// Obtener referencias a los elementos del DOM
const fileInput = document.getElementById('fileInput');
const imageCanvas = document.getElementById('imageCanvas');
const predictionText = document.getElementById('prediction');
const ctx = imageCanvas.getContext('2d');

// Función para cargar el modelo de TensorFlow.js
async function loadModel() {
    try {
        // Carga el modelo desde la ruta relativa al archivo HTML
        // Asegúrate de que './web_model/model.json' sea la ruta correcta
        model = await tf.loadLayersModel('./web_model/model.json');
        console.log("Modelo cargado exitosamente.");
        predictionText.innerText = "Modelo listo. Sube una imagen para clasificar.";
        predictionText.classList.remove('loading', 'error');
        predictionText.classList.add('success');
    } catch (error) {
        console.error("Error al cargar el modelo:", error);
        predictionText.innerText = "Error al cargar el modelo. Verifica la consola para más detalles.";
        predictionText.classList.remove('loading', 'success');
        predictionText.classList.add('error');
    }
}

// Función para preprocesar la imagen y hacer la predicción
async function predict() {
    if (!model) {
        predictionText.innerText = "El modelo aún no está cargado. Por favor, espera.";
        predictionText.classList.add('loading');
        return;
    }

    if (fileInput.files.length === 0) {
        predictionText.innerText = "Por favor, selecciona una imagen para clasificar.";
        predictionText.classList.remove('success', 'loading');
        predictionText.classList.add('error');
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = async (e) => {
        const img = new Image();
        img.onload = async () => {
            // Limpiar el canvas antes de dibujar la nueva imagen
            ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
            // Dibujar la imagen original en el canvas, redimensionándola al tamaño del modelo
            ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

            predictionText.innerText = "Clasificando imagen...";
            predictionText.classList.remove('error', 'success');
            predictionText.classList.add('loading');

            // --- Preprocesamiento de la imagen (DEBE COINCIDIR CON PYTHON) ---
            // 1. Convertir la imagen del canvas a un tensor de TensorFlow.js
            //    tf.browser.fromPixels escala a [0, 255]
            let tensor = tf.browser.fromPixels(imageCanvas)
                .resizeNearestNeighbor([IMG_SIZE, IMG_SIZE]) // Asegura el tamaño exacto del modelo
                .toFloat(); // Cambia el tipo de dato a float32

            // 2. Normalización a [0, 1] (similar a image / 255.0 en Python)
            tensor = tensor.div(tf.scalar(255.0));

            // 3. Normalización específica de MobileNetV2 a [-1, 1]
            //    (similar a tf.keras.applications.mobilenet_v2.preprocess_input)
            const processedTensor = tensor.mul(tf.scalar(2.0)).sub(tf.scalar(1.0));

            // 4. Añadir una dimensión para el batch (el modelo espera [batch_size, height, width, channels])
            const inputTensor = processedTensor.expandDims(0); // Ahora tiene forma [1, IMG_SIZE, IMG_SIZE, 3]

            try {
                // Realiza la predicción
                const predictions = await model.predict(inputTensor);
                // Obtiene los datos de las probabilidades como un array de JavaScript
                const probabilities = predictions.dataSync();

                // Encuentra el índice de la clase con la mayor probabilidad
                const predictedClassIndex = probabilities.indexOf(Math.max(...probabilities));
                const predictedClassName = CLASS_NAMES[predictedClassIndex];

                let predictionResult = `Predicción: <strong>${predictedClassName}</strong>`;
                predictionResult += `<br><small>Probabilidades:</small>`;
                CLASS_NAMES.forEach((name, i) => {
                    predictionResult += `<br><small>${name}: ${(probabilities[i] * 100).toFixed(2)}%</small>`;
                });

                predictionText.innerHTML = predictionResult;
                predictionText.classList.remove('loading', 'error');
                predictionText.classList.add('success');

            } catch (error) {
                console.error("Error durante la predicción:", error);
                predictionText.innerText = "Error al predecir la imagen. Consulta la consola.";
                predictionText.classList.remove('loading', 'success');
                predictionText.classList.add('error');
            } finally {
                // Libera la memoria de los tensores de TensorFlow.js después de usarlos
                tf.dispose([tensor, processedTensor, inputTensor, predictions]);
            }
        };
        img.src = e.target.result; // Carga la imagen al objeto Image
    };
    reader.readAsDataURL(file); // Lee el archivo como URL de datos
}

// --- Event Listeners ---
// Cuando el DOM esté completamente cargado, carga el modelo
document.addEventListener('DOMContentLoaded', loadModel);
// Cuando se selecciona un archivo en el input, dispara la predicción
fileInput.addEventListener('change', predict);