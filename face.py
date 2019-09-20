import numpy as numpy
import cv2


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
print(face_cascade)

#cap = cv2.VideoCapture(0)

while True:
    #ret, frame = cap.read()
    frame = cv2.imread('Images/Cris-Evans/1.jpg', 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, h, w) in faces:
        print(x, y, h, w)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        img_item = "my.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)  # BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
