from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
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

def extract_features(src):
    with open(src, "r") as file:
        for i,line in enumerate(file):
            img_path = line[:-1]
            print("[+] Read image  : ", img_path," id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
                save_path = img_path.replace("images", "features/gg16_fc2").replace(".jpg", ".npy")            
                      
                # img = image.load_img(img_path, target_size=(113, 113))
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array((145.,100,100)), np.array((179., 255., 255.))) # tai sao lai doan mau do nam trong khoang 145 den 179
                dst = np.zeros(img.shape, img.dtype)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if(mask[i, j] > 0):
                            dst[i,j] = img[i,j]
                        else:
                            dst[i, j] = 0
                img_data = image.img_to_array(dst)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)

                print("[+] Extract feature from image : ", img_path)
                feature = model.predict(img_data)

                save_feature(save_path, feature)
            

if __name__=="__main__":
    src = sys.argv[1]
    print("src: ", src)
    extract_features(src)
