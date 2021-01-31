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
import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

path = 'C:\\Users\\Timot\\Documents\\GitHub\\Emotion-detection\\src\\data\\pre_proc'
save_path = 'C:\\Users\\Timot\\Documents\\GitHub\\Emotion-detection\\src'

im_filenames = os.listdir(path)
ctr = 0
# print(im_filenames)
for curr_im in im_filenames:
    ful_im_path = os.path.join(path,curr_im)
    img = cv2.imread(ful_im_path)
    # cv2.imshow('image',img)
    # k = cv2.waitKey(0)
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image',gray)
    # k = cv2.waitKey(0)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        img = cv2.resize(roi_gray, (48, 48))
        # plt.imshow(img)
        # plt.show
        cv2.imwrite(os.path.join(save_path, str(ctr) + ".png"),img)
        ctr += 1
