import cv2
import face_recognition
from infection_models import demographic_prediction
import math

def euclidean_dist(pA, pB):
    return math.sqrt((pA[0] - pB[0]) ** 2 + (pA[1] - pB[1]) ** 2)

cap= cv2.VideoCapture('../examples/example_video2.mp4')

faces = []

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    # Convert the image from BGR color (which OpenCV uses) to RGB
    # color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
    faces = face_recognition.face_locations(rgb_frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    points = []

    MIN_DIST = 120
    REF_DISTANCE = 50
    REF_WIDTH = 50
    REF_PIX = 306.36553955078125

    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y: y + h, x: x + w]
        age, gender, ethnicity = predictions.predict(face_img)

        #cv2.rectangle(frame, (h, x), (y, w), (0, 255, 0), 2)
        cv2.putText(frame, gender + ' ' + age + ' ' + ethnicity, (h,w+20), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        ratio_px_cm = REF_WIDTH / REF_PIX
        center = (h+20, w - 20)
        for p in points:
            ed = euclidean_dist(p, center) * ratio_px_cm
            color = (0, 255, 0)
            if ed < MIN_DIST:
                color = (0, 0, 255)
            # draw a rectangle over each detected face
            cv2.rectangle(frame, (h, x), (y, w), color, 2)
            # put the distance as text over the face's rectangle
            # draw a line between the faces detected
            cv2.line(frame, center, p, color, 5)
        points.append(center)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()