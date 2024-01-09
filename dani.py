import cv2
import numpy as np
import os
from collections import Counter

# Definir la función largest_contour
def largest_contour(contours):
    return max(contours, key=cv2.contourArea)

def create_combined_sample(paths, samples_per_path=250):
    combined_images = []
    combined_labels = []

    for path in paths:
        # Lista para almacenar las imágenes
        images = []
        # Lista para almacenar las etiquetas de las imágenes
        labels = []

        # Iterar sobre los archivos en la ruta
        for filename in os.listdir(path):
            # Comprobar si el archivo es una imagen (ajusta las extensiones según tus necesidades)
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Obtener la etiqueta de la clase a partir del nombre del archivo
                label = filename.split('_')[0].lower()
                labels.append(label)

                # Construir la ruta completa del archivo
                filepath = os.path.join(path, filename)
                # Leer la imagen y agregarla a la lista
                img = cv2.imread(filepath)
                images.append(img)

        # Asegurarse de que las clases estén balanceadas
        class_counts = Counter(labels)
        min_class_count = min(class_counts.values())

        # Seleccionar aleatoriamente el número mínimo de instancias para cada clase
        selected_indices = []
        for class_name in class_counts.keys():
            class_indices = [i for i, label in enumerate(labels) if label == class_name]
            selected_indices.extend(np.random.choice(class_indices, min_class_count, replace=False))

        # Crear la muestra de 250 patrones para esta ruta
        sample_images = [images[i] for i in selected_indices[:samples_per_path]]
        sample_labels = [labels[i] for i in selected_indices[:samples_per_path]]

        combined_images.extend(sample_images)
        combined_labels.extend(sample_labels)

    # Lista para almacenar los momentos invariantes de Hu de cada imagen
    hu_moments_list = []

    # Iterar sobre las imágenes seleccionadas
    for img in combined_images:
        # Procesamiento previo
        img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Encontrar contornos
        contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Seleccionar el contorno más grande
        largest_contour_img = largest_contour(contours)

        # Calcular momentos de Hu
        moments = cv2.moments(largest_contour_img)
        hu_moments = cv2.HuMoments(moments)

        # Agregar los momentos invariantes de Hu a la lista
        hu_moments_list.append(hu_moments.flatten())

    # Convertir la lista a un array de numpy
    hu_moments_array = np.array(hu_moments_list)

    return hu_moments_array, combined_labels

# Rutas de las imágenes (no olviden cambiar la ruta)
paths = [
    r'.\archive\shapes\circle',
    r'.\archive\shapes\square',
    r'.\archive\shapes\star',
    r'.\archive\shapes\triangle'
]

# Crear la muestra de 250 patrones para cada ruta en un solo conjunto
hu_moments_array, combined_labels = create_combined_sample(paths)


