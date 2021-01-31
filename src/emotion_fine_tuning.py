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
import os

# training framework is heavily adapted from https://github.com/atulapra/Emotion-detection
# contributions include retraining, adding new data specific to our domain for visual purposes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode


def lr_scheduler(epoch, lr): # triangular2 (halved every full cycle)
    total_epochs = 150
    LR_Specs = [0.001, 0.0000001, 'step_decay', 1]
    
    if LR_Specs[2] == 'step_decay':
        step_size = 150/(LR_Specs[3]) # into 100/10 -> step size of 10
        delta_lr_per_step = (LR_Specs[0]-LR_Specs[1])/step_size
        return LR_Specs[0] - (delta_lr_per_step * ((epoch) % step_size))

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28759
num_val = 7178
batch_size = 64
num_epoch = 150

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=15,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
    )
val_datagen = ImageDataGenerator(rescale=1./255,
    )

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical', shuffle=True, seed=13)

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical', shuffle=True, seed=13)


# # # Create the model
# # model = Sequential()

# # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
# # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))

# # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
# # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"))
# # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"))
# # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.3))

# # model.add(Flatten())
# # model.add(Dense(512, activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(4, activation='softmax'))

model = load_model('model_tuned.h5')

save_path = 'C:\\Users\\Timot\\Documents\\GitHub\\Emotion-detection\\src'

checkpoint = ModelCheckpoint(os.path.join(save_path,'model_tuned.h5'), monitor = 'val_loss', verbose=1, save_best_only = True, save_weights_only= False, mode = 'min') 
# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            epochs=150,
            validation_data=validation_generator,
            callbacks=[checkpoint, LearningRateScheduler(lr_scheduler, verbose=1)])
    plot_model_history(model_info)

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model = load_model('model_tuned.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Sad", 1: "Sad", 2: "Sad", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Sad"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()