import numpy as numpy
import cv2
import os
import pickle

#
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
print(face_cascade)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("label.pickle", 'rb') as f:
    unpickle = pickle.Unpickler(f)
    og_labels = unpickle.load()
    labels = {v: k for k, v in og_labels.items()}

path = "Images/test/2.jpg"
image = cv2.imread(path, 0)
image1 = cv2.imread(path, cv2.IMREAD_COLOR)

#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=2)
for (x, y, h, w) in faces:
    print(x, y, h, w)
    roi = image[y:y + h, x:x + w]

id_, conf = recognizer.predict(roi)
if conf >= 0:
    print(id_)
    print(labels[id_])
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = labels[id_]
    color = (0, 0, 255)
    stroke = 1
    cv2.putText(image1, name, (x, y - 5), font, 0.5, color, stroke, cv2.LINE_AA)
color = (255, 0, 0)  # BGR
stroke = 2
cv2.rectangle(image1, (x, y), (x + w, y + h), color, stroke)


cv2.imshow('image', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
