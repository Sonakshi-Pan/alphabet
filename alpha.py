import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X = np.load('image3.npz')['arr_0']
y = pd.read_csv("labels2.csv")["labels"]
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n = len(classes)


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 7500 , test_size = 2500, random_state = 9)
X_train_scale = X_train/255
X_test_scale = X_test/255

lr=LogisticRegression(solver="saga",multi_class="multinomial")
lr.fit(X_train_scale,y_train)
pred = lr.predict(X_test_scale)

print(accuracy_score(y_test,pred))

cap = cv2.read()

while True:
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
         
        h,w = gray.shape 
        u_1 = (int(w/2-56),int(h/2-56))
        b_r = (int(w/2+56),int(h/2+56))

        cv2.rectangle(gray ,u_1,b_r ,(0,255,0),2)
        roi = gray[u_1[1]:b_r[1],u_1[0]:b_r[0]]

        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        pixel_filter = 20 
        min_pixel = np.percentile(image_bw_resized_inverted,pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel = np.max(image_bw_resized_inverted)

        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = lr.predict(test_sample)
        print("the predicted class is :",test_pred)

        cv2.imshow('sonakshi',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()        
