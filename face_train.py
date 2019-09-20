import numpy as np
import cv2
import os
from PIL import Image
import pickle
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
print(face_cascade)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
image_dir = os.path.join(BASE_DIR, "images")
'''
c = 0
p = 0
for root, dirs, files in os.walk("images"):

    for file in files:
        if file.endswith("png") or file.endswith("jpg"):

            path = os.path.join(root, file).replace(" ", "-")
            path = path.replace("\\", "/")
            p = p + 1
            # print(path)
            image = cv2.imread(path, 0)
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=4)
            print(faces)
            
            if len(faces) > 0:
                c = c + 1
                for (x, y, h, w) in faces:
                    pass
                    print(x, y, h, w)
                color = (255, 0, 0)  # BGR
                stroke = 2
                cv2.rectangle(image, (x, y), (x + w, y + h), color, stroke)

                cv2.imshow('image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
print("accuracy:", (c / p) * 100)
'''
current_id = 0
label_ids = {}
x_train = []
y_label = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):

            path = os.path.join(root, file).replace(" ", "-")
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #print(label, path)
            if not label in label_ids:

                label_ids[label] = current_id
                current_id = current_id + 1
            id_ = label_ids[label]
            print(label_ids)
            imgpath = os.path.join(root, file).replace("\\", "/")
            pil_image = Image.open(imgpath).convert("L")
            # print(pil_image)
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=4)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_label.append(id_)
#print(y_label, x_train)
with open("label.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
