import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, CSVLogger, ModelCheckpoint
from tensorflow.keras.models import load_model
from paralleldots import set_api_key, get_api_key
import paralleldots
from flask import Flask, render_template, Response
import os
import operator

# Setting your API key
set_api_key( "qC31vQRZWwELxBGd1ArzTn41xjRxMImzZ5ZFLBnCAis" )
# Viewing your API key
get_api_key()

# text      = "Chipotle in the north of Chicago is a nice outlet. I went to this place for their famous burritos but fell in love with their healthy avocado salads. Our server Jessica was very helpful. Will pop in again soon!"
# aspect    = "food"
# lang_text = "C'est un environnement très hostile, si vous choisissez de débattre ici, vous serez vicieusement attaqué par l'opposition."
# category  = [ "travel","food","shopping", "market" ]
# data      =  [ "drugs are fun", "don\'t do drugs, stay in school", "lol you a fag son", "I have a throat infection" ]

# # get max emotion of text
# emotion = paralleldots.emotion( text )
# emotion = emotion['emotion']

# print( paralleldots.emotion( text ) )
# print( paralleldots.keywords( text ) )
# print(max(emotion.items(), key=operator.itemgetter(1))[0])

# sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# web_params = {
#     "emotion_key":"None",
# }

app = Flask(__name__)

global page_ID
global page_idx
global prev_emotion
global emotion_ctr

prev_emotion = 'None'


page_ID = ["sample.html", "first_page.html", "second_page.html", "sample.html"]
page_idx = 0
emotion_ctr = 0
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}

# help on Flask from: https://medium.com/datadriveninvestor/video-streaming-using-flask-and-opencv-c464bf8473d6
# pip install Flask

# defining face detector

ds_factor=0.6
class VideoCamera(object):
    ''' from link above '''
    def __init__(self):
       #capturing video
        self.video = cv2.VideoCapture(0)
        self.facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        model = load_model('model_tuned.h5')
        self.classifier = model
        self.zero_pad = np.ones((12,48))*255
    def __del__(self):
        #releasing camera
        self.video.release()

    def get_frame(self):
        #extracting frames
        global emotion_ctr

        global prev_emotion
        emotion_ctr += 1
        ret, frame = self.video.read()

        frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
        interpolation=cv2.INTER_AREA)                    
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_rects = self.facecasc.detectMultiScale(gray,1.3,5)
        curr_emotion = prev_emotion
        for (x,y,w,h) in face_rects:
            
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y + h, x:x + w]
            resize_im = cv2.resize(roi_gray, (48, 48))
            # only every twenty
            if emotion_ctr % 20 == 0:
                cropped_img = np.expand_dims(np.expand_dims(resize_im, -1), 0)
                prediction = self.classifier.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                # web_params["emotion_key"] = emotion_dict[maxindex]
                curr_emotion = emotion_dict[maxindex]
                print(prediction)
            
            break
        bordersize = 24
        frame = cv2.copyMakeBorder(
            frame,
            top=bordersize,
            bottom=0,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        cv2.putText(frame, 'Emotion:' + curr_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        prev_emotion = curr_emotion

        # encode OpenCV raw frame to jpg and display it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    

        # for (x, y, w, h) in faces:
        #     roi_gray = gray[y:y + h, x:x + w]
        #     cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        #     prediction = model.predict(cropped_img)
        #     maxindex = int(np.argmax(prediction))
        #     cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

@app.route('/')
def index():
    # rendering webpage
    # return render_template('first_page.html')
    return render_template(page_ID[page_idx])

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/emotion')
def emotion_feed():
    global curr_emotion
    return Response(curr_emotion)

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)