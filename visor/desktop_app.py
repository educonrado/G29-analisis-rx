import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import os
import sys

class ModelPredictor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Analizador de RX - Enfermedades respiratorias")
        self.window.geometry("800x500")
        self.window.resizable(True, True)
        
        # Variables
        self.model = None
        self.current_image = None
        self.image_path = None
        
        # Cargar modelo al iniciar
        self.load_model()
        
        # Crear interfaz
        self.create_interface()
    
    def resource_path(self, relative_path):
        """Obtener ruta absoluta de recursos (necesario para PyInstaller)"""
        try:
            # PyInstaller crea una carpeta temporal y almacena la ruta en _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        
        return os.path.join(base_path, relative_path)
    
    def load_model(self):
        """Cargar el modelo Keras"""
        try:
            model_path = self.resource_path("best_model.keras")
            self.model = tf.keras.models.load_model(model_path)
            print("Modelo cargado exitosamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {str(e)}")
            self.window.destroy()
    
    def create_interface(self):
        """Crear la interfaz gráfica"""
        # Título
        title_label = tk.Label(
            self.window, 
            text="Análisis de RX", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Frame para la imagen
        self.image_frame = tk.Frame(self.window, bg="lightgray", width=300, height=300)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        # Label para mostrar imagen
        self.image_label = tk.Label(
            self.image_frame, 
            text="Selecciona una imagen", 
            bg="lightgray"
        )
        self.image_label.pack(expand=True)
        
        # Botón para seleccionar imagen
        select_btn = tk.Button(
            self.window,
            text="Seleccionar Imagen",
            command=self.select_image,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=5
        )
        select_btn.pack(pady=10)
        
        # Botón para predecir
        self.predict_btn = tk.Button(
            self.window,
            text="Realizar Predicción",
            command=self.make_prediction,
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=5,
            state="disabled"
        )
        self.predict_btn.pack(pady=5)
        
        # Frame para resultados
        result_frame = tk.Frame(self.window)
        result_frame.pack(pady=10, fill="x", padx=20)
        
        tk.Label(result_frame, text="Resultado:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        self.result_text = tk.Text(
            result_frame, 
            height=6, 
            font=("Arial", 10),
            state="disabled"
        )
        self.result_text.pack(fill="x", pady=5)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(
            self.window, 
            mode='indeterminate',
            length=200
        )
    
    def select_image(self):
        """Seleccionar imagen desde el explorador"""
        file_types = [
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Todos los archivos", "*.*")
        ]
        
        self.image_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=file_types
        )
        
        if self.image_path:
            self.display_image()
            self.predict_btn.config(state="normal")
    
    def display_image(self):
        """Mostrar imagen seleccionada"""
        try:
            # Cargar y redimensionar imagen para mostrar
            image = Image.open(self.image_path)
            
            # Redimensionar manteniendo proporción
            image.thumbnail((280, 280), Image.Resampling.LANCZOS)
            
            # Convertir para tkinter
            photo = ImageTk.PhotoImage(image)
            
            # Actualizar label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Mantener referencia
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocesar imagen para MobileNetV2 64x64"""
        try:
            # Cargar imagen con el tamaño correcto
            image = tf.keras.preprocessing.image.load_img(
                image_path, 
                target_size=(64, 64)  # Tu modelo usa 64x64
            )
            
            # Convertir a array
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            
            # Expandir dimensiones para batch
            image_array = np.expand_dims(image_array, axis=0)
            
            # Preprocesamiento específico para MobileNetV2 (rango [-1, 1])
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
            
            return image_array
            
        except Exception as e:
            raise Exception(f"Error al procesar imagen: {str(e)}")
    
    def make_prediction(self):
        """Realizar predicción con el modelo"""
        if not self.image_path or not self.model:
            return
        
        try:
            # Mostrar barra de progreso
            self.progress.pack(pady=10)
            self.progress.start()
            self.window.update()
            
            # Preprocesar imagen
            processed_image = self.preprocess_image(self.image_path)
            
            # Realizar predicción
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Formatear resultados
            self.display_results(predictions)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {str(e)}")
        
        finally:
            # Ocultar barra de progreso
            self.progress.stop()
            self.progress.pack_forget()
    
    def display_results(self, predictions):
        """Mostrar resultados de la predicción"""
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)

        # Mapeo de clases para mayor claridad
        # Puedes personalizar esto según tus etiquetas reales
        class_labels = {
            0: "Normal",
            1: "COVID",
            2: "Neumonía"
        }

        # Formatear según tu tipo de modelo
        if len(predictions[0]) == 1:
            # Regresión o clasificación binaria
            result = f"Predicción: {predictions[0][0]:.4f}\n"
        else:
            # Clasificación multi-clase
            result = "Probabilidades por clase:\n"
            for i, prob in enumerate(predictions[0]):
                label = class_labels.get(i, f"Clase {i}") # Usa el mapeo, si no, usa el número
                result += f"{label}: {prob:.4f} ({prob*100:.2f}%)\n"

            # Clase predicha
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_label = class_labels.get(predicted_class_index, f"Clase {predicted_class_index}")
            confidence = np.max(predictions[0])
            result += f"\nClase predicha: {predicted_class_label}\n"
            result += f"Confianza: {confidence:.4f} ({confidence*100:.2f}%)"

        self.result_text.insert(tk.END, result)
        self.result_text.config(state="disabled")
    
    def run(self):
        """Ejecutar la aplicación"""
        self.window.mainloop()

if __name__ == "__main__":
    app = ModelPredictor()
    app.run()