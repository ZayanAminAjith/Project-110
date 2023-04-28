
# import the opencv library
import cv2
import numpy as np

import tensorflow as tf
  
model = tf.keras.models.load_model("keras_model.h5")
# define a video capture object
vid = cv2.VideoCapture(1)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    img = cv2.resize(frame,(224,224))
    img2 = np.array(img,dtype=np.float32)
    img2 = np.expand_dims(img2,0)
    normalize_img = img2/255.0
    prediction = model.predict(normalize_img)
    print("Prediction : ",prediction)
  
    # Display the resulting frame
    cv2.imshow("Rock...paper...Scissors...SHOOT!", frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()