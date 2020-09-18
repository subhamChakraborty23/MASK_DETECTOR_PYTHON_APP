# plot photo with detected faces using opencv cascade classifier
import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import numpy as np
from keras.models import load_model
model=load_model(".\model1.h5")
results={0:'without mask',1:'mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}
rect_size = 4
# load the photograph
pixels = imread('test2.jpg')

im=pixels 

    
rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
faces = classifier.detectMultiScale(pixels,
    scaleFactor=1.05,
    minNeighbors=3)
# print bounding box for each detected face
for f in faces:
        #(x, y, w, h) = [v * rect_size for v in f] 
        x, y, w, h = f
        x2, y2 = x + w, y + h
        
        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(150,150))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
cv2.imshow('face detection',   im)
waitKey(0)
# close the window
destroyAllWindows()