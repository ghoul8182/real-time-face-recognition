import pandas as pd
import numpy as np
import cv2 as cv
from datetime import datetime
import os

# Check if 'id-names.csv' exists
if os.path.exists('id-names.csv'):
    id_names = pd.read_csv('id-names.csv')
    id_names = id_names[['id', 'name']]
else:
    id_names = pd.DataFrame(columns=['id', 'name'])
    id_names.to_csv('id-names.csv', index=False)

# Check if 'faces' directory exists
if not os.path.exists('faces'):
    os.makedirs('faces')

print('Welcome!')
print('\nPlease put in your ID.')
print('If this is your first time, choose a random ID between 1-10000')

id = int(input('ID: '))
name = ''

# Check if the ID already exists in the CSV
if id in id_names['id'].values:
    name = id_names[id_names['id'] == id]['name'].item()
    print(f'Welcome Back {name}!!')
else:
    name = input('Please Enter your name: ')
    os.makedirs(f'faces/{id}')
    # Create a new row DataFrame and concatenate it with the existing DataFrame
    new_row = pd.DataFrame([{'id': id, 'name': name}])
    id_names = pd.concat([id_names, new_row], ignore_index=True)
    id_names.to_csv('id-names.csv', index=False)

print("\nLet's capture!")
print("Now this is where you begin taking photos. Once you see a rectangle around your face, press the 's' key to capture a picture.", end=" ")
print("It is recommended to take at least 20-25 pictures, from different angles, in different poses, with and without specs, you get the gist.")
input("\nPress ENTER to start when you're ready, and press the 'q' key to quit when you're done!")

camera = cv.VideoCapture(0)#set up the camera
haar_cascade_path = 'Classifiers/haarface.xml'#Set up the haar cascade for face detection
face_classifier = cv.CascadeClassifier(haar_cascade_path)

if face_classifier.empty():
    raise IOError(f"Failed to load Haar cascade file from {haar_cascade_path}. Check the file path.")

photos_taken = 0

# Capture photos in a loop
while(cv.waitKey(1) & 0xFF != ord('q')):
    _, img = camera.read()# is used to discard the first returned value (a boolean indicating success), while img stores the captured image.
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#Converts the captured frame from BGR color space to grayscale. Face detection works better on grayscale images.
    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))#face_classifier.detectMultiScale: Detects faces in the grayscale image
    #scaleFactor=1.1: Specifies how much the image size is reduced at each image scale.
    #minNeighbors=5: Specifies how many neighbors each candidate rectangle should have to retain it.
    #minSize=(50, 50): Specifies the minimum size of detected faces.

    
    for (x, y, w, h) in faces:# Iterates through all the detected faces
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)#Draw rectangle among the detected faces 

        face_region = grey[y:y + h, x:x + w]
        if cv.waitKey(1) & 0xFF == ord('s') and np.average(face_region) > 50:
            face_img = cv.resize(face_region, (220, 220))
            img_name = f'face.{id}.{datetime.now().microsecond}.jpeg'
            cv.imwrite(f'faces/{id}/{img_name}', face_img)
            photos_taken += 1
            print(f'{photos_taken} -> Photos taken!')

    cv.imshow('Face', img)

camera.release()
cv.destroyAllWindows()
