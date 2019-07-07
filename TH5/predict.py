from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
import sys
import os
from PIL import ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES=True
print("[+] Setup model")
base_model = VGG16(weights='imagenet', include_top=True)
out = base_model.get_layer("fc2").output
model = Model(inputs=base_model.input, outputs=out)

def save_feature(save_path, feature):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save extracted feature to file : ", save_path)
    np.save(save_path, feature)

def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((145.,100,100)), np.array((179., 255., 255.)))
    mask_red = cv2.inRange(hsv, np.array((0.,100,100)), np.array((10., 255., 255.)))
    dst = np.zeros(img.shape, img.dtype)
    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(mask[i, j] > 0):
                dst[i,j] = img[i,j]
                count = count+1
            else:
                dst[i, j] = 0
            if (mask_red[i, j])>0:
                dst[i, j] = img[i, j]
                count = count+1
    if count >= threshold_red_pixel:
        img_data = image.img_to_array(dst)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        print("The object is: " + clf.predict(feature)[0])
        return feature
    #save_feature(save_path, feature)
            

if __name__=="__main__":
    src = sys.argv[1]
    db = sys.argv[2]
    """open file mean.txt"""
    file1 = open("mean.txt", "r")
    threshold_red_pixel = int(file1.read())
    file1.close()
    #img = np.load(src)
    clf = joblib.load(db + '/model.joblib')
    feature = extract_features(src)
    # print("The object is: " + clf.predict(feature)[0])