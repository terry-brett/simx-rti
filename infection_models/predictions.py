from keras.models import load_model
import numpy as np
import cv2
import scipy.misc
from PIL import Image

age_model = load_model('../.models/age-model.h5')
gender_model = load_model('../.models/gen-model.h5')
ethnicity_model = load_model('../.models/eth-model.h5')

gender_classes = ['male', 'female']
age_classes = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
               '55-59', '60-64', '65-69', '70-74', '75+']
ethinicity_classes = ['white', 'black', 'asian', 'other']

def predict(facial_image):
    img_arr = []
    image = cv2.resize(facial_image, (96, 96))
    img_arr.append(image)

    img_arr = np.asarray(img_arr)
    img_arr = img_arr / 255.0

    age_pred = age_model.predict(img_arr)
    age_index = np.argmax(age_pred, axis=1)

    gender_pred = gender_model.predict(img_arr)
    gender_index = np.argmax(gender_pred, axis=1)

    ethnicity_pred = ethnicity_model.predict(img_arr)
    ethnicity_index = np.argmax(ethnicity_pred, axis=1)

    return age_classes[age_index[0]], gender_classes[gender_index[0]], ethinicity_classes[ethnicity_index[0]]