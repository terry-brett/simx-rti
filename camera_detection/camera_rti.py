"""
The Real Time Face Recognition

Sveleba, S., Katerynchuk, I., Karpa, I., Kunyo, I., Ugryn, S.
and Ugryn, V., 2019, July.
The real time face recognition.
In 2019 3rd International Conference on Advanced Information and
Communications Technologies (AICT) (pp. 294-297). IEEE.
"""


import datetime as dt
import logging as log
from time import sleep

import cv2
import networkx as nx

from infection_models import demographic_prediction, helpers

face_cascade = cv2.CascadeClassifier(
    r"cascades/haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

log.basicConfig(filename="webcam.log", level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# create an empty graph
G = nx.Graph()

while True:
    if not video_capture.isOpened():
        print("Unable to load camera.")
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    text = ""
    infection_text = ""

    face_img_list = []

    # Draw a rectangle around the faces
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y: y + h, x: x + w]
        if face_img not in face_img_list:
            face_img_list.append(face_img)
        age, gender, ethnicity = demographic_prediction.predict(face_img)
        if age is not None and gender is not None and ethnicity is not None:
            text = (
                str(demographic_prediction.gender_classes[gender])
                + " "
                + str(demographic_prediction.age_classes[age])
                + " "
                + str(demographic_prediction.ethinicity_classes[ethnicity])
            )

            if i in G:
                if "infection_rate" in G.nodes[i]:
                    infection_text += (
                        " "
                        + str(round(G.nodes[i]["infection_rate"] * 100, 2))
                        + "%"
                    )

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x, y + h + 25),
            font,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            infection_text,
            (x, y + h + 45),
            font,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        helpers.connect(frame, x, y, G, face_img_list, uses_camera=True)

    if anterior != len(faces):
        anterior = len(faces)
        log.info(
            "faces: " + str(len(faces)) + " at " + str(dt.datetime.now())
        )

    # Display the resulting frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Display the resulting frame
    cv2.imshow("Video", frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
