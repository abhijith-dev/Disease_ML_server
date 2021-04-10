import os, shutil
from flask import Flask, flash, request, redirect, url_for, session,jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
app = Flask(__name__,static_folder="./static/images",static_url_path="/")
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('logging...')
result_classes=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___healthy',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___healthy',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper_bell___Bacterial_spot',
 'Pepper_bell___healthy',
 'Potato___Early_blight',
 'Potato___healthy',
 'Potato___Late_blight',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___healthy',
 'Strawberry___Leaf_scorch',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___healthy',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

 
def predict_image():
    Trained_model = load_model("model123.h5")
    batch_size = 128
    datagen_train = ImageDataGenerator()
    base_path = "C:/Users/hp/Desktop/server/static"
    test_generator = datagen_train.flow_from_directory(base_path,target_size=(100,100),color_mode="grayscale",batch_size=batch_size,class_mode='categorical',shuffle=True)
    resultss=result_classes[np.argmax(Trained_model.predict(test_generator))]
    return resultss
def clear():
    folder = "C:/Users/hp/Desktop/server/static/images"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER="C:/Users/hp/Desktop/server/static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload',methods=['POST'])
@cross_origin()
def upload_file():
    clear()
    target=os.path.join(UPLOAD_FOLDER,'images')
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("Server running...")
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    url="http://127.0.0.1:5000/"+filename
    result=predict_image()
    response={"url":url,"result":result}
    return jsonify(response)
   
if __name__ == '__main__':
   app.secret_key = os.urandom(24)
   app.run()


