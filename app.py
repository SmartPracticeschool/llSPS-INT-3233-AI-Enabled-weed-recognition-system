"""
web application for images
mail flask folder
static 
css
templates folder
app.py
.h5 file

"""
from flask import Flask,request,render_template
import os
import numpy as np
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
model=load_model("weed.h5")
import tensorflow as tf
global graph
graph=tf.get_default_graph()
app = Flask(__name__)
@app.route('/', methods = ["GET"] )
def index():
    return render_template("base.html")
@app.route('/predict', methods = ["GET","POST"])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        basepath = os.path.dirname(__file__)
        print("current path:",basepath)
        file_path = os.path.join(basepath,"uploads",secure_filename(f.filename))
        f.save(file_path)
        print("Joined path: ",file_path)
        img = image.load_img(file_path,target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        with graph.as_default():
            preds = model.predict_classes(x)
            print(preds)
        index = ["Broadleaf","Grass","Soil","Soybean"]
        text ="The detected type of weed is "+index[preds[0]]
        return text
if __name__ =='__main__':
    app.run(debug = True)