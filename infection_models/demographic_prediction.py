from typing import Tuple
import cv2
import numpy as np
from keras.models import load_model

age_model = load_model(
    # These paths should probably be command line arguments instead of hardcoded paths.
    "/Users/terrybrett/Documents/GitHub/simx-rti/.models/age-model.h5"
)
gender_model = load_model(
    "/Users/terrybrett/Documents/GitHub/simx-rti/.models/gen-model.h5"
)
ethnicity_model = load_model(
    "/Users/terrybrett/Documents/GitHub/simx-rti/.models/eth-model.h5"
)

gender_classes = ["male", "female"]
age_classes = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75+",
]
ethinicity_classes = ["white", "black", "asian", "other"]


def predict(facial_image) -> Tuple[int, int, int]:
    if not facial_image.any():
        return None, None, None

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

    return age_index[0], gender_index[0], ethnicity_index[0]
