import pandas as pd
import cv2 as cv

id_names = pd.read_csv('id-names.csv')
id_names = id_names[['id', 'name']]

haar_cascade_path = 'Classifiers/haarface.xml'
face_classifier = cv.CascadeClassifier(haar_cascade_path)

if face_classifier.empty():
    raise IOError(f"Failed to load Haar cascade file from {haar_cascade_path}. Check the file path.")

#local binary histogram pattern algorithm
lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)
lbph.read('Classifiers/TrainedLBPH.yml')

camera = cv.VideoCapture(0)

while cv.waitKey(1) & 0xFF != ord('q'):
    ret, img = camera.read()
    if not ret:# If the frame was not captured successfully, skip the rest of the loop and try again

        continue
    
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)#Detects faces in the grayscale image.

    for (x, y, w, h) in faces:# Iterates through all detected faces. Each face is represented by a rectangle with top-left corner (x, y), width (w), and height (h).
        face_region = grey[y:y + h, x:x + w]
        face_region = cv.resize(face_region, (220, 220))

        # Predict the ID and confidence (trust)
        label, trust = lbph.predict(face_region)#Uses the LBPH recognizer to predict the ID and confidence (trust) of the detected face.
        try:
            name = id_names[id_names['id'] == label]['name'].item()
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(img, name, (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))#Puts the predicted name below the rectangle.
        except:
            pass

    cv.imshow('Recognize', img)

camera.release()
cv.destroyAllWindows()