from __future__ import division, print_function
from flask import Flask , redirect , render_template, request, url_for
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from PIL import Image
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
app = Flask(__name__)


kernel_size = (3,3)
pool_size = (2,2)
first_filters = 32
second_filters = 64
third_filters = 128
IMAGE_SIZE = 50
dropout_conv = 0.3
dropout_dense = 0.3

def model_predict(img_path,model):
    img = image.load_img(img_path, target_size=(50,50,3))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The Image does not indicate the presence of cancer"
    else:
        preds = "The Image indicates the presence of cancer"

    return preds


def create_model():
    model = Sequential()
    model.add(Conv2D(first_filters, kernel_size, activation='relu', input_shape= (IMAGE_SIZE, IMAGE_SIZE,3)))
    model.add(Conv2D(first_filters, kernel_size, activation='relu'))
    model.add(Conv2D(first_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_dense))
    model.add(Dense(2,activation='softmax'))

    model.load_weights('model_bc1.h5')
    return model

UPLOAD_FOLDER = 'C:/Users/nikki/Desktop/RegistrationForm/static'

model = create_model()
print(model.summary())

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)

            return render_template('devangi2.html', prediction=model_predict(image_location, model))
    return render_template('devangi2.html', prediction="Upload an image for prediction")

if __name__=='__main__':
    app.run(debug=True)