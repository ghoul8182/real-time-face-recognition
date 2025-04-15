import os
import numpy as np
import cv2 as cv

lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)# Creates an LBPH (Local Binary Patterns Histograms) face recognizer with a threshold of 500, which is used to decide if a face is recognized or not.

def create_train():#: A function that creates training data by reading face images from the 'faces' directory.
    faces = []
    labels = []
    for id in os.listdir('faces'):
        path = os.path.join('faces', id)
        try:
            os.listdir(path)
        except:
            continue
        for img in os.listdir(path):
            try:
                face = cv.imread(os.path.join(path, img))#Reads the image from the file.
                face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)#Converts the image to grayscale.


                faces.append(face)#Adds the grayscale face image to the faces list
                labels.append(int(id))
            except:
                pass
    return np.array(faces), np.array(labels)

faces, labels = create_train()

print('Training Started')
lbph.train(faces, labels)#Trains the LBPH face recognizer using the arrays of face images and labels.
lbph.save('Classifiers/TrainedLBPH.yml')
print('Training Complete!')