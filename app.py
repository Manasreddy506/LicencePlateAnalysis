import os

from flask import Flask, render_template, request, send_file
import io
import os
from PIL import Image
import cv2
import numpy as np
from torchvision.models import detection

import torch
from torchvision import models
from flask import Flask, render_template, request, redirect, Response
import sqlite3
from dl_prediction.predictor import Predictor
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from com_in_ineuron_ai_utils.utils import decodeImage
from Detector import Detector
from logger import getLog

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

CORS(app)

logger=getLog('clientApp.py')
class ClientApp:

    def __init__(self):

        try:

            self.filename = "inputImage.jpg"
            self.obj_detect = Detector()
            logger.info("ClientApp object initialized")

        except Exception as e:

            logger.exception(f"Failed to initialize App Object : \n{e}")
            raise Exception("Failed to initialize App Object")

predictor = Predictor()
output_car_file = 'static/output_car.jpg'
output_license_file = 'static/output_license.jpg'
output_license_original_file = 'static/output_license_original.jpg'
output_video_file = 'static/output_video.mp4'


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")



@app.route('/yolo')
def yolo():
    return render_template("yolo.html")



@app.route('/yolo-upload', methods=['POST'])
def yolo_upload():
    isImage = request.args.get('type') == 'image'
    isVideo = request.args.get('type') == 'video'
    file = request.files['file']
    file.save(file.filename)
    license_txt = predictor.predict(file.filename, output_car_file, output_license_original_file, output_license_file)
    
    os.remove(file.filename)
    return render_template("yolo.html")


@app.route('/cnn')
def cnn():
    return render_template("cnn.html")



@app.route('/cnn-upload', methods=['POST'])
def cnn_upload():
    isImage = request.args.get('type') == 'image'
    isVideo = request.args.get('type') == 'video'
    file = request.files['file']
    file.save(file.filename)
    
    license_txt = predictor.predict(file.filename, output_car_file, output_license_original_file, output_license_file, is_cnn=True)
    
    os.remove(file.filename)
    return render_template("cnn.html")

model = torch.hub.load("ultralytics/yolov5", "custom", path = "best.pt", force_reload=True)

model.eval()
model.conf = 0.5  
model.iou = 0.45  

from io import BytesIO

def gen():
    """
    The function takes in a video stream from the webcam, runs it through the model, and returns the
    output of the model as a video stream
    """
    cap=cv2.VideoCapture(0)
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            results.print()  
            img = np.squeeze(results.render()) 
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        else:
            break
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    """
    It returns a response object that contains a generator function that yields a sequence of images
    :return: A response object with the gen() function as the body.
    """
    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

                     

@app.route("/predict1", methods=["GET", "POST"])
def predict():
    """
    The function takes in an image, runs it through the model, and then saves the output image to a
    static folder
    :return: The image is being returned.
    """
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=640)
        results.render()  
        for img in results.render():
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")
    return render_template("index.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/notebook")
def notebook():
    return render_template("Notebook.html")

@app.route("/notebook1")
def notebook1():
    return render_template("Notebook1.html")

@app.route("/charyolo")
def charyolo():

    return render_template("index1.html")

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():

    try:

        image = request.json['image']
        logger.info("Image loaded")
        clApp = ClientApp()
        decodeImage(image, clApp.filename)
        result = clApp.obj_detect.run_inference()
        return jsonify(result)

    except Exception as e:

        return jsonify(e)

if __name__ == '__main__':
    app.run(debug=False)
