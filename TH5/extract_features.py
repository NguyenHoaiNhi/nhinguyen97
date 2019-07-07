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
    average = []
    hand = []
    two = []
    three = []
    curve = []
    thumb = []
    sum_min = []
    with open(src, "r") as file:
        for i,line in enumerate(file):
            count = 0
            img_path = line[:-1]
            print("[+] Read image  : ", img_path," id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
                save_path = img_path.replace("images", "features/gg16_fc2").replace(".jpg", ".npy")
                lb = save_path.split("\\")[2]
                label = save_path.split("\\")[1]
                print("label: ", label)
                # img = image.load_img(img_path, target_size=(113, 113))
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array((145.,100,100)), np.array((179., 255., 255.))) # tai sao lai doan mau do nam trong khoang 145 den 179
                mask_red = cv2.inRange(hsv, np.array((0.,100,100)), np.array((10., 255., 255.)))
                dst = np.zeros(img.shape, img.dtype)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if(mask[i, j] > 0):
                            dst[i,j] = img[i,j]
                            count = count +1   
                        else:
                            dst[i, j] = 0
                        if (mask_red[i, j])>0:
                            dst[i, j] = img[i, j]
                            count = count+1
                if label == "hand":
                    hand.append(count)
                if label == "two":
                    # print("hello")
                    two.append(count)
                if label == "three":
                    # print("hi")
                    three.append(count)
                if label == "curve":
                    curve.append(count)
                if label == "thumb":
                    # print("ngoncai")
                    thumb.append(count)
                # if label == "unknown":
                    # unknown.append(count)
                # average.append(count)
                img_data = image.img_to_array(dst)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                print("[+] Extract feature from image : ", img_path)
                feature = model.predict(img_data)
                save_feature(save_path, feature)
    min_hand = np.amin(np.array(hand))
    # print("min_circle: ", min_circle)
    sum_min.append(min_hand)
    # print("two : ", two)
    min_two = np.amin(np.array(two))
    # print("min_two: ", min_two)
    sum_min.append(min_two)
    min_three = np.amin(np.array(three))
    sum_min.append(min_three)
    min_curve = np.amin(np.array(curve))
    sum_min.append(min_curve)
    min_thumb = np.amin(np.array(thumb))
    sum_min.append(min_thumb)
    # min_unknown = np.amin(np.array(unknown))
    # sum_min.append(min_unknown)
    pixel_red_density = np.amin(sum_min)
    print("mean_average: ", pixel_red_density)
    # file_result = open("mean.txt", "w")
    # file_result.write(str(pixel_red_density))
    # file_result.close()
    # return pixel_red_density

def get_label():
    temple_lb = ""
    data_label = []
    with open(src, "r") as file:
        for i,line in enumerate(file):
            img_path = line[:-1]
            print("[+] Read image  : ", img_path," id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
                save_path = img_path.replace("images", "features/vgg16_fc2").replace(".jpg", ".npy")
                lb = save_path.split("\\")[2]
                if temple_lb != lb:
                    data_label.append(lb)
                    temple_lb = lb
    return  data_label

if __name__=="__main__":
    src = sys.argv[1]
    print("src: ", src)
    extract_features(src)
