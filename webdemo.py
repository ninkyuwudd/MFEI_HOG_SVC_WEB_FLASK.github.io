import os
from flask import Flask, render_template, request, redirect, url_for
from pre_processing import *
from input_handle import *
from mfei_proccess import *
from hog_proccess import *
from clasification_proccess import *
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    # check_input_video(request,app)

    full_path = insert_input_to_folder(request)

    
    folder_path = os.path.join(os.path.dirname(__file__), full_path)

    frames = load_images_from_folder(folder_path)
    if(len(frames) != 0):

        aligned_frames = normalize_silhouettes(frames)

        mfei_image = execute_mfei(aligned_frames)

        compute_hog(mfei_image)

        compute_svc()
        

    res = multiplyNumb(1,2)
    return render_template('index.html',res=res)