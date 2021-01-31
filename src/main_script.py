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
from flask import request, redirect
import os
import operator
import statistics
from statistics import mode


# Setting your API key
set_api_key( "qC31vQRZWwELxBGd1ArzTn41xjRxMImzZ5ZFLBnCAis" )
# Viewing your API key
get_api_key()



# http://127.0.0.1:5000/
app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
global emotion_arr
emotion_arr = []

global depression_score
depression_score = 0

global page_ID

page_ID = ["first_page.html", "second_page.html", "third_page.html", "fourth_page.html", "fifth_page.html", "sixth_page.html", "seventh_page.html", "eighth_page.html", "ninth_page.html", "tenth_page.html", "short_questions_1.html", "short_questions_2.html", "short_questions_3.html", "exit_page.html", "summary_page.html"]
# page_ID = ["first_page.html","exit_page.html"]

# help on Flask from: https://medium.com/datadriveninvestor/video-streaming-using-flask-and-opencv-c464bf8473d6
# pip install Flask

# defining face detector

global page_idx 
global prev_emotion
global emotion_ctr

prev_emotion = 'None'


page_idx = 0
emotion_ctr = 0
# emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}
emotion_dict = {0: "Sad", 1: "Sad", 2: "Sad", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Sad"}

# help on Flask from: https://medium.com/datadriveninvestor/video-streaming-using-flask-and-opencv-c464bf8473d6
# pip install Flask

def get_face_emotion_emoji(stamp):
    if stamp == 'Happy':
        return '&#128512'

    elif stamp == 'Sad':
        return '&#128557'

    elif stamp == 'Neutral':
        return '&#128566'

def get_text_emotion_emoji(stamp):
    if stamp == 'Happy':
        return '&#128512'

    elif stamp == 'Angry':
        return '&#128545'

    elif stamp == 'Excited':
        return '&#128540'

    elif stamp == 'Sad':
        return '&#128557'

    elif stamp == 'Fear':
        return '&#128561'

    elif stamp == 'Bored':
        return '&#128554'

    else:
        print(stamp)
        sys

# defining face detector
def most_common(lst):
    return max(set(lst), key=lst.count)

class html_output_writer:
    fname = 'N/A'
    lname = 'N/A'
    DOB = 'N/A'
    Depression_Result = 'N/A'
    Depression_emoji = 'N/A'
    Depression_Score = 'N/A'
    D1 = 'N/A'
    D2 = 'N/A'
    D3 = 'N/A'
    D4 = 'N/A'
    D5 = 'N/A'
    D6 = 'N/A'
    D7 = 'N/A'
    D8 = 'N/A'
    D9 = 'N/A'
    challenge_text = 'N/A'
    challenge_kw = 'N/A'
    challenge_face = 'N/A'
    challenge_text_emotion = 'N/A'

    coping_text = 'N/A'
    coping_kw = 'N/A'
    coping_face = 'N/A'
    coping_text_emotion = 'N/A'

    goal_text = 'N/A'
    goal_kw = 'N/A'
    goal_face = 'N/A'
    goal_text_emotion = 'N/A'


ds_factor=0.6
class VideoCamera(object):
    ''' from link above '''
    def __init__(self):
       #capturing video
        self.video = cv2.VideoCapture(0)
        self.facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # model = load_model('model_tuned.h5')
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        model.load_weights('model.h5')
        self.classifier = model

        self.zero_pad = np.ones((12,48))*255
    def __del__(self):
        #releasing camera
        self.video.release()

    def get_frame(self):
        #extracting frames
        global emotion_ctr
        global emotion_arr
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
                emotion_arr.append(curr_emotion)
            
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
    global page_idx
    # rendering webpage
    # return render_template('first_page.html')
    return render_template(page_ID[page_idx])

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/signup', methods = ['POST'])
def signup():
    global page_idx
    page_idx += 1
    fname = request.form['fname']
    print("The first name is '" + fname + "'")
    lname = request.form['lname']
    print("The last name is '" + lname + "'")
    dob = request.form['dob']
    print("The dob is '" + dob + "'")
    html_output_writer.fname = fname
    html_output_writer.lname = lname
    html_output_writer.DOB = dob

    render_template(page_ID[page_idx])
    return redirect('/')

@app.route('/emotion')
def emotion_feed():
    global curr_emotion
    return Response(curr_emotion)

@app.route('/page_two', methods = ['POST'])
def page_two():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3

    html_output_writer.D1 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')

@app.route('/page_three', methods = ['POST'])
def page_three():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3
        
    html_output_writer.D2 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')

@app.route('/page_four', methods = ['POST'])
def page_four():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3
        
    html_output_writer.D3 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')
    
@app.route('/page_five', methods = ['POST'])
def page_five():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3
        
    html_output_writer.D4 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')

@app.route('/page_six', methods = ['POST'])
def page_six():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3
        
    html_output_writer.D5 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')

@app.route('/page_seven', methods = ['POST'])
def page_seven():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3
        
    html_output_writer.D6 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')

@app.route('/page_eight', methods = ['POST'])
def page_eight():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3
        
    html_output_writer.D7 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')

@app.route('/page_nine', methods = ['POST'])
def page_nine():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3
        
    html_output_writer.D8 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')

@app.route('/page_ten', methods = ['POST'])
def page_ten():
    global depression_score
    global page_idx
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []
    page_idx += 1
    if request.form['submit'] == "Not at all":
        depression_score += 0
    elif request.form['submit'] == "Several days":
        depression_score += 1
    elif request.form['submit'] == "More than half the days":
        depression_score += 2
    elif request.form['submit'] == "Nearly every day":
        depression_score += 3
        
    html_output_writer.D9 = get_face_emotion_emoji(current_emotion)

    print(depression_score)
    render_template(page_ID[page_idx])

    return redirect('/')

@app.route('/short_questions_1', methods = ['POST'])
def short_questions_1():
    global page_idx
    global challenge
    global coping
    global goals
    global challenge_emotion
    global coping_emotion
    global goals_emotion
    global keyword_challenge
    global keyword_coping
    global keyword_goals
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []

    page_idx += 1
    challenge = '{}'.format(request.form['challenge'])

    challenge_emotion = paralleldots.emotion( challenge )
    challenge_emotion = challenge_emotion['emotion']
    challenge_emotion = max(challenge_emotion.items(), key=operator.itemgetter(1))[0]
    keyword_challenge = paralleldots.keywords( challenge )
    
    html_output_writer.challenge_text = challenge
    html_output_writer.challenge_kw = keyword_challenge['keywords']
    html_output_writer.challenge_face = get_face_emotion_emoji(current_emotion)
    html_output_writer.challenge_text_emotion = get_text_emotion_emoji(challenge_emotion)

    render_template(page_ID[page_idx])
    return redirect('/')

@app.route('/short_questions_2', methods = ['POST'])
def short_questions_2():
    global page_idx
    global challenge
    global coping
    global goals
    global challenge_emotion
    global coping_emotion
    global goals_emotion
    global keyword_challenge
    global keyword_coping
    global keyword_goals
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []

    page_idx += 1
    coping = '{}'.format(request.form['coping'])

    coping_emotion = paralleldots.emotion( coping )
    coping_emotion = coping_emotion['emotion']
    coping_emotion = max(coping_emotion.items(), key=operator.itemgetter(1))[0]
    keyword_coping = paralleldots.keywords( coping )

    html_output_writer.coping_text = coping
    html_output_writer.coping_kw = keyword_coping['keywords']
    html_output_writer.coping_face = get_face_emotion_emoji(current_emotion)
    html_output_writer.coping_text_emotion = get_text_emotion_emoji(coping_emotion)

    render_template(page_ID[page_idx])
    return redirect('/')

@app.route('/short_questions_3', methods = ['POST'])
def short_questions_3():
    global page_idx
    global challenge
    global coping
    global goals
    global challenge_emotion
    global coping_emotion
    global goals_emotion
    global keyword_challenge
    global keyword_coping
    global keyword_goals
    global emotion_arr
    current_emotion = most_common(emotion_arr)
    print('Submitted emotion:', current_emotion)
    emotion_arr = []

    page_idx += 1
    goals = '{}'.format(request.form['goals'])

    goals_emotion = paralleldots.emotion( goals )
    goals_emotion = goals_emotion['emotion']
    goals_emotion = max(goals_emotion.items(), key=operator.itemgetter(1))[0]
    keyword_goals = paralleldots.keywords( goals )

    html_output_writer.goal_text = goals
    html_output_writer.goal_kw = keyword_goals['keywords']
    html_output_writer.goal_face = get_face_emotion_emoji(current_emotion)
    html_output_writer.goal_text_emotion = get_text_emotion_emoji(goals_emotion)


    render_template(page_ID[page_idx])
    return redirect('/')

@app.route('/exit_page', methods = ['POST'])
def exit_page():
    global page_idx
    global challenge
    global coping
    global goals
    global challenge_emotion
    global coping_emotion
    global goals_emotion
    global keyword_challenge
    global keyword_coping
    global keyword_goals
    global depression_score

    if depression_score < 4:
        d_s = '&#128533'
        d_result = 'Minimum Depression'
    elif depression_score < 10:
        d_s = '&#128542'
        d_result = 'Mild Depression'
    elif depression_score < 15:
        d_s = '&#128542'
        d_result = 'Moderate Depression'
    elif depression_score < 20:
        d_s = '&#128542'
        d_result = 'Mod/Severe Depression'
    elif depression_score < 28:
        d_s = '&#128542'
        d_result = 'Severe Depression'    
    page_idx += 1

    html_output_writer.Depression_Score = depression_score
    html_output_writer.Depression_Result = d_result
    html_output_writer.Depression_emoji = d_s


    save_filename = html_output_writer.fname+html_output_writer.lname + '.html'
    # Read the HTML file
    HTML_File=open(os.path.join('C:\\Users\\Timot\\Documents\\GitHub\\Emotion-detection\\src\\templates', 'sample_summary.html'),'r')
    s = HTML_File.read().format(p=html_output_writer())
    Html_file= open(os.path.join('C:\\Users\\Timot\\Documents\\GitHub\\Emotion-detection\\src\\templates', save_filename),"w")
    Html_file.write(s)
    Html_file.close()
    print(s)
    return render_template(save_filename)


@app.route('/summary_page')
def summary_page():
    # global page_idx
    # global challenge
    # global coping
    # global goals
    # global challenge_emotion
    # global coping_emotion
    # global goals_emotion
    # global keyword_challenge
    # global keyword_coping
    # global keyword_goals
    # global depression_score

    # if depression_score < 4:
    #     d_s = '&#128533'
    #     d_result = 'Minimum Depression'
    # elif depression_score < 10:
    #     d_s = '&#128542'
    #     d_result = 'Mild Depression'
    # elif depression_score < 15:
    #     d_s = '&#128542'
    #     d_result = 'Moderate Depression'
    # elif depression_score < 20:
    #     d_s = '&#128542'
    #     d_result = 'Mod/Severe Depression'
    # elif depression_score < 28:
    #     d_s = '&#128542'
    #     d_result = 'Severe Depression'
    # else:
    #     d_s = 'INVALID'
    #     d_result = 'INVALID'

    # selected_1 = 
    print('asdf')

    return render_template(page_ID[page_idx], variable = "12345")
    print('completed')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    # your code
    # return a response

# # # def main():
# # #     # model.load_model('model_tuned.h5')

# # #     # prevents openCL usage and unnecessary logging messages
# # #     cv2.ocl.setUseOpenCL(False)

# # #     # dictionary which assigns each label an emotion (alphabetical order)
# # #     emotion_dict = {0: "Sad", 1: "Sad", 2: "Sad", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Sad"}

# # #     # start the webcam feed
# # #     cap = cv2.VideoCapture(0)
# # #     while True:
# # #         # Find haar cascade to draw bounding box around face
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break
# # #         facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# # #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #         faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

# # #         for (x, y, w, h) in faces:
# # #             roi_gray = gray[y:y + h, x:x + w]
# # #             cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

# # #             prediction = model.predict(cropped_img)
# # #             maxindex = int(np.argmax(prediction))
# # #             cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# # #         cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
# # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # #             break

# # #     cap.release()
# # #     cv2.destroyAllWindows()

if __name__ == "__main__":
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)