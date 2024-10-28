import cv2
import numpy as np

def calculateMIG(frames):
    
    mig = []
    for idx in range(len(frames) -1):#[1]
        count_between = frames[idx + 1] - frames[idx] #[1]
        _, motion_image = cv2.threshold(count_between, 0, 255, cv2.THRESH_BINARY) #[1]
        
        mig.append(motion_image)
        
    return mig

def mfeiCalculate(motion_images):
    mfei = np.sum(motion_images, axis=0) / (len(motion_images) - 1)
    
    return mfei

def execute_mfei(frame_normalized):
    # frame_normalized = normalize_silhouettes(frames)
    mig = calculateMIG(frame_normalized)
    mfei = mfeiCalculate(mig)
    return mfei