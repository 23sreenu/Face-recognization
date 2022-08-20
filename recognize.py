import cv2
import os
import numpy as np

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# video_capture = cv2.VideoCapture(0)

# Call the trained model yml file to recognize faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

# Names corresponding to each id
names = []
for users in os.listdir("dataset"):
    names.append(users)
n = len(os.listdir("dataset"))
img = cv2.imread("test/paavan.jpeg")

while True:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img = video_capture.read()
    faces = faceCascade.detectMultiScale3(gray_image,
                                          scaleFactor=1.2,
                                          minNeighbors=5,
                                          minSize=(100, 100),
                                          outputRejectLevels=True)
    # Try to predict the face and get the id
    
    # Accordingly add the names
    for (x, y, w, h) in faces[0]:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray_image[y:y + h, x:x + w])
        
        if id <= n and confidence:
            cv2.putText(
                img,
                names[id - 1],
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                img,
                "Unknown",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Recognize", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# video_capture.release()
cv2.destroyAllWindows()