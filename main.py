import os
import cv2
import numpy as np

haar_cascade = cv2.CascadeClassifier('haar_face.xml')

features = np.load('Results/features.npy', allow_pickle=True)
labels = np.load('Results/labels.npy', allow_pickle=True)
people = []
DIR = r'E:\pythonprojects\openCVtrainAdvanced\Resources\Faces\val'
for folder in os.listdir(DIR):
    people.append(folder)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('Results/face_trained.yml')

img = cv2.imread(r'E:\pythonprojects\openCVfaceRecognition\Resources\Faces\val\mindy_kaling\2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
cv2.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y + h, x:x + w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv2.putText(img, str(people[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)


cv2.imshow('Detected face', img)


def test_accuracy():
    result = 0
    test_len = 0
    for person in people:
        path = os.path.join(DIR, person)
        for img in os.listdir(path):
            test_len +=1
            im_path = os.path.join(path, img)
            im = cv2.imread(im_path)
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray_im, scaleFactor=1.1, minNeighbors=2)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                label, confidence = face_recognizer.predict(faces_roi)
                if people[label] == person:
                    result += 1
    return result/test_len


print(f"\nValidation accuracy = {test_accuracy()}. Random choice accuracy would be {1/len(people)}")

cv2.waitKey(0)
