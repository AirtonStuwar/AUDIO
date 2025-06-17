import tkinter as tk
from tkinter import ttk
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image, ImageTk

# Cargar datos MNIST
(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.astype("float32") / 255.
x_test = np.expand_dims(x_test, -1)

# Añadir ruido
def add_noise(imgs, noise_factor=0.5):
    noisy_imgs = imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)
    return np.clip(noisy_imgs, 0., 1.)

x_test_noisy = add_noise(x_test)

# Cargar el modelo entrenado
model = load_model("autoencoder_mnist.h5")

# Predecir imágenes restauradas
decoded_imgs = model.predict(x_test_noisy)

# Función para convertir una imagen numpy a PhotoImage
def numpy_to_photo(img_array):
    img = (img_array * 255).astype(np.uint8).reshape(28, 28)
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((100, 100), Image.NEAREST)
    return ImageTk.PhotoImage(img_pil)

# Crear ventana
root = tk.Tk()
root.title("Autoencoder Denoising - MNIST")

# Índice de imagen mostrada
img_index = tk.IntVar(value=0)

# Etiquetas para mostrar imágenes
lbl_noisy = tk.Label(root, text="Ruidosa")
lbl_original = tk.Label(root, text="Original")
lbl_restored = tk.Label(root, text="Restaurada")
lbl_noisy.grid(row=0, column=0)
lbl_original.grid(row=0, column=1)
lbl_restored.grid(row=0, column=2)

canvas_noisy = tk.Label(root)
canvas_original = tk.Label(root)
canvas_restored = tk.Label(root)
canvas_noisy.grid(row=1, column=0)
canvas_original.grid(row=1, column=1)
canvas_restored.grid(row=1, column=2)

# Mostrar las imágenes
def show_images(index):
    i = index % len(x_test)
    noisy_img = numpy_to_photo(x_test_noisy[i])
    original_img = numpy_to_photo(x_test[i])
    restored_img = numpy_to_photo(decoded_imgs[i])
    canvas_noisy.config(image=noisy_img)
    canvas_noisy.image = noisy_img
    canvas_original.config(image=original_img)
    canvas_original.image = original_img
    canvas_restored.config(image=restored_img)
    canvas_restored.image = restored_img

# Botones para navegar
def next_image():
    img_index.set(img_index.get() + 1)
    show_images(img_index.get())

def prev_image():
    img_index.set(img_index.get() - 1)
    show_images(img_index.get())

btn_prev = ttk.Button(root, text="Anterior", command=prev_image)
btn_next = ttk.Button(root, text="Siguiente", command=next_image)
btn_prev.grid(row=2, column=0)
btn_next.grid(row=2, column=2)

# Mostrar primera imagen
show_images(0)

# Iniciar la interfaz
root.mainloop()
