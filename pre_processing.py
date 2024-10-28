import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def multiplyNumb(numbA,numbB):
    numbC = numbA + numbB
    return numbC



def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):#[1]
        img_path = os.path.join(folder_path, filename)#[2]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)#[3]
        if img is not None:
            images.append(img)
    return images

def findObject(frame):
    x, y, w, h = cv2.boundingRect(frame)#[1]
    img = frame[y:y+h,x:x+w]#[2]
    return img,h,w

def findObjectAndFillingHole(frame):
    img,h,w = findObject(frame)#[1]
    kernel1 = np.ones((3,3), np.uint8)#[2]
    kernel2 = np.ones((5,5), np.uint8)#[3]
    res = cv2.dilate(img, kernel=kernel2, iterations=1)#[4]
    res = cv2.erode(res, kernel=kernel2, iterations=1)#[5]
    return res,h,w 


def normalize_height(centered_frames,target_height,bounding_h,bounding_w,max_width):

    #RESIZE IMAGE 

    scale = target_height / bounding_h#[1]
    
    scaled_image = cv2.resize(centered_frames, (int(bounding_w * scale), target_height), interpolation=cv2.INTER_LINEAR)#[2]
    
    if scaled_image.shape[0] < target_height :#[3]
        padded_image = cv2.copyMakeBorder(scaled_image, 0, target_height - scaled_image.shape[0], 0, 0,borderType=cv2.BORDER_CONSTANT,value=0)#[4]

    elif scaled_image.shape[0] > target_height:#[5]
        padded_image = scaled_image[:target_height, :]#[6]
    else:
        padded_image = scaled_image#[7]

    return padded_image


def manual_bounding_rect(image):
    # Temukan piksel putih pada gambar biner
    white_pixels = np.argwhere(image == 255)
    
    # Jika tidak ada piksel putih, kembalikan None
    if white_pixels.shape[0] == 0:
        return None
    
    # Dapatkan batas bounding rectangle
    y_min, x_min = white_pixels.min(axis=0)  # Top-left corner
    y_max, x_max = white_pixels.max(axis=0)  # Bottom-right corner
    
    # Return koordinat bounding rect
    return (x_min, y_min, x_max, y_max)

def cut_upper(image):

    x_leftmost = None
    for x in range(image.shape[1]):
        if np.any(image[:, x] == 255):
            x_leftmost = x
            break

    return x_leftmost

def cut_upper_body(x,y,w,h,image):
   
    top_cutoff = y + h // 4
 
    upper_body = image[y:top_cutoff, 0:240]
    _, binary_image = cv2.threshold(upper_body, 127, 255, cv2.THRESH_BINARY)
    left_most = cut_upper(binary_image)
    coords = np.column_stack(np.where(binary_image > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    head  = upper_body[y_min:y_max+1, x_min:x_max+1]

    return head,left_most



def found_shift(cropped,bounding_subject,gab):
    aligned_frame = np.zeros((cropped.shape[0], 240), dtype=np.uint8)
   
    center_x = (aligned_frame.shape[1] - cropped.shape[1]) // 2
    return center_x -gab

def center_subject(bounding_subject,center_x):
    bg =  np.zeros((240,240))

    if bounding_subject.shape[1] > bg.shape[1]:
        scale_factor = bg.shape[1] / bounding_subject.shape[1]
        new_width = bg.shape[1]
        new_height = int(bounding_subject.shape[0] * scale_factor)
        bounding_subject = cv2.resize(bounding_subject, (new_width, new_height))

    if(center_x <= 0):
        left_shift = 0
    else:
        left_shift = center_x

    center_y = (bg.shape[0] - bounding_subject.shape[0]) // 2

    if(left_shift+bounding_subject.shape[1] > 240):
        left_shift = 0

    bg[center_y:center_y+bounding_subject.shape[0], left_shift:left_shift+bounding_subject.shape[1]] = bounding_subject

    return bg


def is_black_image(frame):
    # Mengecek apakah semua piksel di dalam gambar bernilai 0 (gambar hitam)
    return np.all(frame == 0)

def normalize_silhouettes(frames):
    aligned_frames = []
    for idx in range(0,len(frames)):#[1]
        frame = frames[idx]
        if is_black_image(frame):
            continue
       # melakukan find objek dan close processing
        filling_hole,h,w = findObjectAndFillingHole(frame)
        # Normalisasi tinggi
        height_normalize_frame = normalize_height(filling_hole,240,h,w,240)
        # melakukan maual bounding rect
        x, y, w, h = manual_bounding_rect(height_normalize_frame)
    
        bounding_subject = height_normalize_frame[y:y+h, x:x+w]
        # pemotongan objek dari badan bagian atas sampai kepala
        cropped_upper,gab = cut_upper_body(x,y,w,h,bounding_subject)
        # melakukan perhitungan total pergeseran kepala dari kiri 
        shift_founded = found_shift(cropped_upper,bounding_subject,gab)
        # pengengahan gambar
        subject_centered = center_subject(bounding_subject,shift_founded)  
        
        aligned_frames.append(subject_centered)#[4]
    return aligned_frames

