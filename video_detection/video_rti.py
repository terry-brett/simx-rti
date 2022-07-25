import cv2
import face_recognition
import networkx as nx

from infection_models import demographic_prediction, helpers

cap = cv2.VideoCapture("../examples/example_video.mp4")

faces = []

# create an empty graph
G = nx.Graph()

while True:
    # grab a single frame of video
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    # find all the faces in the current frame of video
    faces = face_recognition.face_locations(rgb_frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (x, y, w, h) in enumerate(faces):

        face_img = frame[y : y + h, x : x + w]
        # make prediction
        age, gender, ethnicity = demographic_prediction.predict(face_img)

        cv2.rectangle(frame, (h, x), (y, w), (0, 255, 0), 2)
        text = ""
        infection_text = ""

        # check if all demographic features has been found
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
                        " " + str(round(G.nodes[i]["infection_rate"] * 100, 2)) + "%"
                    )

        cv2.putText(frame, text, (h, w + 20), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            infection_text,
            (h - 10, w + 40),
            font,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        helpers.connect(frame, x, y, G, faces)

    # display the resulting image
    cv2.imshow("Video", frame)

    # stop if escape key is pressed
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
