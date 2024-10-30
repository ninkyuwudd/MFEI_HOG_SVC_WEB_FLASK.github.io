from skimage.feature import hog
import pandas as pd
import numpy as np
def divide_into_cells(image):
    cells = []
    cells.append(image[0:image.shape[0]//2, 0:image.shape[1]//2])  # Cell 1
    cells.append(image[0:image.shape[0]//2, image.shape[1]//2:image.shape[1]])  # Cell 3
    cells.append(image[image.shape[0]//2:image.shape[0], 0:image.shape[1]//2])  # Cell 7
    cells.append(image[image.shape[0]//2:image.shape[0], image.shape[1]//2:image.shape[1]])  # Cell 9
    cells.append(image[0:image.shape[0]//2, image.shape[1]//4:3*image.shape[1]//4])  # Cell 2
    cells.append(image[image.shape[0]//2:image.shape[0], image.shape[1]//4:3*image.shape[1]//4])  # Cell 8
    cells.append(image[image.shape[0]//4:3*image.shape[0]//4, 0:image.shape[1]//2])  # Cell 4
    cells.append(image[image.shape[0]//4:3*image.shape[0]//4, image.shape[1]//2:image.shape[1]])  # Cell 6
    cells.append(image[image.shape[0]//4:3*image.shape[0]//4, image.shape[1]//4:3*image.shape[1]//4])  # Cell 5
    return cells


def compute_hog(image):
    
    divide_cells = divide_into_cells(image)
    # Daftar untuk menyimpan deskripsi HOG dari tiap cell
    hog_descriptors = []
    hog_images = []
    # Hitung HOG untuk setiap cell
    for i, cell in enumerate(divide_cells):
       
        # Hitung HOG menggunakan skimage untuk setiap cell
        hog_descriptor, hog_image = hog(
            cell,
            orientations=7,  # 7 bin histogram (sesuai paper)
            pixels_per_cell=(8,8), 
            cells_per_block=(1, 1),  # Normalisasi per cell
            block_norm='L2-Hys',  # Normalisasi menggunakan metode L2-Hys (perbandingan dan penjelasan)
            visualize=True,  # untuk mendapatkan gambar HOG
            feature_vector=True  # hasilkan vektor fitur
        )

    # Simpan deskripsi HOG dari cell ke dalam list
        print(hog_descriptor)
        hog_descriptors.append(hog_descriptor)
        hog_images.append(hog_image)
    faltten_hog = np.array(hog_descriptors).flatten()
    # data_item = [hog_descriptor, f"{idx}"]
    df = pd.DataFrame([faltten_hog], columns=[f'feature_{i}' for i in range(len(faltten_hog))])

    
    # Menyimpan DataFrame ke file CSV
    df.to_csv("testing_data.csv", index=False)
  
    return hog_descriptors



