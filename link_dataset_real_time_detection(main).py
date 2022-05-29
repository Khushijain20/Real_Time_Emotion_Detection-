from keras.models import load_model #used to load model from the dataset i.e created in model building
from time import sleep 
from tensorflow.keras.utils import img_to_array #In order to import image as an array
from keras.preprocessing import image
import cv2 #for real time detection of image 
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\Khushi\Desktop\Real_Time_Emotion_Detector\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Khushi\Desktop\Real_Time_Emotion_Detector\Model_building.h5')

emotion_to_detect = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

video_cap = cv2.VideoCapture(0) #opens the camera to detect real time emotion



while True:
    _, frame = video_cap.read() 
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (a,b,c,d) in faces: #draws a rectangle around face
        cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,255),2)
        roi_gray = gray[b:b+d,a:a+c] #ROI=region of interest 
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA) #resizing the image to 48*48 as per trained model 



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_to_detect[prediction.argmax()]
            label_position = (a,b-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,254,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) #if no faces found 
    cv2.imshow('Real Time Emotion Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()