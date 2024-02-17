import cv2
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as keras
import pickle
import keras.utils as image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

## Preprocess Captured User Face Custom Images ##
def preprocess_faces():
    headshots_folder_name = 'static/face/'
    # dimension of images
    image_width = 224
    image_height = 224

    # for detecting faces
    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # set the directory containing the images
    images_dir = os.path.join(".", headshots_folder_name)
    current_id = 0
    label_ids = {}
    # iterates through all the files in each subdirectories
    for root, _, files in os.walk(images_dir):
        if os.path.basename(root) == 'train':
            continue
        for file in files:
            if file.endswith(("png", "jpg", "jpeg")):  # Use tuple to check multiple file extensions
                # path of the image
                path = os.path.join(root, file)
                # get the label name (name of the person)
                label = os.path.basename(root).replace(" ", ".").lower()

                # add the label (key) and its number (value)
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                # load the image
                imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
                image_array = np.array(imgtest, "uint8")

                # get the faces detected in the image
                faces = facecascade.detectMultiScale(imgtest,
                                                    scaleFactor=1.1, minNeighbors=5)

                # if not exactly 1 face is detected, skip this photo
                if len(faces) != 1:
                    print(f'---Photo skipped---\n')
                    print(f'One of the face not detected\n')
                    return False

                # save the detected face(s) and associate them with the label
                for (x_, y_, w, h) in faces:
                    # draw the face detected
                    face_detect = cv2.rectangle(imgtest,
                                                (x_, y_),
                                                (x_ + w, y_ + h),
                                                (255, 0, 255), 2)               

                    # resize the detected face to 224x224
                    size = (image_width, image_height)

                    # detected face region
                    roi = image_array[y_: y_ + h, x_: x_ + w]

                    # resize the detected head to target size
                    resized_image = cv2.resize(roi, size)
                    image_array = np.array(resized_image, "uint8")

                    # remove the original image
                    # os.remove(path)

                    # replace the image with only the face
                    im = Image.fromarray(image_array)
                    trainpath = os.path.join(root,'train')
                    os.makedirs(trainpath, exist_ok=True)
                    newjpgPath = os.path.join(trainpath,file)
                    im.save(newjpgPath)
    return True

#Check current user with (if exist) model to prevent from using same face
def check_current_user(user):
    # Check if model already trained
    h5_path = './transfer_learning_trained_face_cnn_model.h5'
    if os.path.exists(h5_path):
        current_user_path = './static/face/' + str(user) + '/train/' + str(user) + '.jpg'
        print(predict_faces(current_user_path))
        # Check if there are face is high probablity in the model
        if (predict_faces(current_user_path) > 0.8):
            return False

    else:
        return True


def train_save_model():
    ## Augmenting ##
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    train_generator = \
        train_datagen.flow_from_directory(
    './static/face',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

    train_generator.class_indices.values()
    NO_CLASSES = len(train_generator.class_indices.values())

    ## Building Model ##
    base_model = VGGFace(include_top=False,
        model='resnet50',
        input_shape=(224, 224, 3))
    base_model.summary()

    print(len(base_model.layers))
    # 26 layers in the original VGG-Face

    #Add layers for new training faces
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    # final layer with softmax activation
    preds = Dense(NO_CLASSES, activation='softmax')(x)
    # create a new model with the base model's original input and the 
    # new model's output
    model = Model(inputs = base_model.input, outputs = preds)
    model.summary()

    # don't train the first 19 layers - 0..18 (Since these layers trained by VGGFace)
    for layer in model.layers[:19]:
        layer.trainable = False

    # train the rest of the layers - 19 onwards
    for layer in model.layers[19:]:
        layer.trainable = True

    model.compile(optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    ## Training the Model
    model.fit(train_generator,
    batch_size = 1,
    verbose = 1,
    epochs = 20)

    ## Saving the Model ## 
    # creates a HDF5 file
    model.save(
        'transfer_learning_trained' +
        '_face_cnn_model.h5')
    
    # deletes the existing model
    del model

    ## Saving the Training Labels ##
    class_dictionary = train_generator.class_indices
    class_dictionary = {
        value:key for key, value in class_dictionary.items()
    }

    # save the class dictionary to pickle
    face_label_filename = 'face-labels.pickle'
    with open(face_label_filename, 'wb') as f: pickle.dump(class_dictionary, f)

    return True

def predict_faces(captured_user_image):
    # returns a compiled model identical to the previous one
    model = load_model(
        'transfer_learning_trained' +
        '_face_cnn_model.h5')
    # dimension of images
    image_width = 224
    image_height = 224

    # load the training labels
    face_label_filename = 'face-labels.pickle'
    with open(face_label_filename, "rb") as \
        f: class_dictionary = pickle.load(f)

    class_list = [value for _, value in class_dictionary.items()]

    # for detecting faces
    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # load the image
    imgtest = cv2.imread(captured_user_image, cv2.IMREAD_COLOR)
    image_array = np.array(imgtest, "uint8")

    # get the faces detected in the image
    faces = facecascade.detectMultiScale(imgtest, 
        scaleFactor=1.1, minNeighbors=5)

    # if not exactly 1 face is detected, skip this photo
    if len(faces) != 1: 
        print("---Please provide only 1 face photo, skipped---")
        return "face undetected"

    for (x_, y_, w, h) in faces:
        # draw the face detected
        face_detect = cv2.rectangle(imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)

        # resize the detected face to 224x224
        size = (image_width, image_height)
        roi = image_array[y_: y_ + h, x_: x_ + w]
        resized_image = cv2.resize(roi, size)

        # prepare the image for prediction
        x = image.img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)

        # making prediction
        predicted_prob = model.predict(x)
        print("Predicted Probability: " + predicted_prob)
        print(predicted_prob[0].argmax())
        print("Predicted face: " + class_list[predicted_prob[0].argmax()])
        print("============================\n")
        return class_list[predicted_prob[0].argmax()]
                
        







