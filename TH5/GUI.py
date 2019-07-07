import tkinter as tk
from tkinter import messagebox
import colorsys
from GridEye import GridEYEKit
import random
import numpy as np
import math
import cv2
import os
from PIL import Image
from pylab import *
import predict as pr
from sklearn.externals import joblib
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from sklearn import svm
from sklearn.externals import joblib
from PIL import ImageFile
import time
import save as sa
import PIL.Image, PIL.ImageTk
# Grid Eye related numbers

class GridEYE_Viewer():

    def __init__(self, root):
        """Hien thi hinh lena"""
        self.flag = 0
        """ Initialize Window """
        self.tkroot = root
        self.tkroot.protocol('WM_DELETE_WINDOW', self.exitwindow)  # Close serial connection and close window

        """ Initialize variables for color interpolation """
        self.HUEstart = 0.5  # initial color for min temp (0.5 = blue)
        self.HUEend = 1  # initial color for max temp (1 = red)
        self.HUEspan = self.HUEend - self.HUEstart

        """ Grid Eye related variables"""
        self.MULTIPLIER = 0.25  # temp output multiplier

        """ Initialize Loop bool"""
        self.START = False

        """Initialize frame tor temperature array (tarr)"""
        # self.frameTarr = tk.Frame(master=self.tkroot, bg='white')
        # self.frameTarr.place(x=5, y=5, width=400, height=400)
        self.canvas = tk.Canvas(self.tkroot,width = 565, height = 565, bg = "white")
        self.canvas.place(x= 5, y =5, width = 565, height = 565)
        """Initialize pixels tor temperature array (tarr)"""
        # self.tarrpixels = []
        # for i in range(8):
        #     for j in range(8):
        #         pix = tk.Label(master=self.frameTarr, bg='gray')
        #         spacerx = 1
        #         spacery = 1
        #         pixwidth = 40
        #         pixheight = 40
        #         pix.place(x=j * pixwidth, y=i * pixheight, width=pixwidth,
        #                   height=pixheight)
        #         print(self.tarrpixels.append(pix))  # attache all pixels to tarrpixel list

        # """Initialize frame tor Elements"""
        self.frameElements = tk.Frame(master=self.tkroot, bg='white')
        # self.frameElements.place(x=410, y=5, width=100, height=400)
        self.frameElements.place(x=575, y=5, width=120, height=565)

        # """Initialize controll buttons"""
        self.buttonStart = tk.Button(master=self.frameElements, text='start', bg='white',
                                     command=self.start_update, width = 10)
        self.buttonStart.pack(padx = 5, pady = 5)
        self.buttonStop = tk.Button(master=self.frameElements, text='stop', bg='white',
                                    command=self.stop_update, width = 10)
        self.buttonStop.pack(padx = 5, pady = 5)
        # """Initialize button get image """
        self.buttonGetimage = tk.Button(master=self.frameElements, text='get', bg='white',
                                        command=self.get_image, width = 10)
        self.buttonGetimage.pack(padx = 5, pady = 5)

        # """Initialize temperature adjustment"""
        self.lableTEMPMAX = tk.Label(master=self.frameElements, text='Max Temp', width = 10, bg = 'white')
        self.lableTEMPMAX.pack(padx = 5, pady = 5)
        self.editTextMaxtemp =tk.Entry(master = self.frameElements, bd = 5, width = 11)
        self.editTextMaxtemp.pack(padx = 5, pady = 5)
        self.editTextMaxtemp.insert(0,"34")
        self.lableMINTEMP = tk.Label(master=self.frameElements, text='Min Temp', width = 10, bg = 'white')
        self.lableMINTEMP.pack(padx = 5, pady = 5)
        self.editTextMintemp = tk.Entry(master = self.frameElements, bd = 5, width = 11)
        self.editTextMintemp.pack(padx = 5, pady = 5)
        self.editTextMintemp.insert(0,"28")
        # """Recognize"""
        self.labelRecognize = tk.Label(master=self.frameElements, text = "Recognize", width = 10, bg = 'white')
        self.labelRecognize.pack(padx = 5, pady = 5)
        self.labelResultRecognize = tk.Label(master=self.frameElements,font = ("Arial", 14), width = 11, height = 4, bg = 'pink')
        self.labelResultRecognize.pack(padx = 5, pady = 5)
        
        self.kit = GridEYEKit()
       
    def exitwindow(self):
        """ if windwow is clsoed, serial connection has to be closed!"""
        self.kit.close()
        self.tkroot.destroy()

    def stop_update(self):
        """ stop button action - stops infinite loop """
        self.START = False
        self.kit._connected = False
        self.kit.flag_check_stop = True
        # self.kit.ser.close()
        print("END\n")
        # self.update_tarrpixels()

    def start_update(self):
        self.kit.flag_check_stop = False
        if self.kit.connect():
            """ start button action -start serial connection and start pixel update loop"""
            self.START = True
            """ CAUTION: Wrong com port error is not handled"""
            self.update_tarrpixels()
        else:
            messagebox.showerror("Not connected",
                                "Could not find Grid-EYE Eval Kit - please install driver and connect")
    def get_tarr(self):
        """ unnecessary function - only converts numpy array to tuple object"""
        tarr = []
        for temp in self.kit.get_temperatures():  # only fue to use of old rutines
            tarr.append(temp)
        return tarr

    def get_ther(self):
        ther = self.kit.get_thermistor()
        return ther
    def update_tarrpixels(self):
        """ Loop for updating pixels with values from funcion "get_tarr" - recursive function with exit variable"""
        if self.START == True:
            # print("update_tarrpixels")
            ther = self.kit.get_thermistor()
            """Bien dung hien thi anh"""
            array_numpy = []
            dst = np.zeros((113, 113, 3), dtype=np.uint8)
            tarr = self.get_tarr()  # Get temerature array
            tarr = np.asarray(tarr)
            tarr_shape = np.reshape(tarr, (113, 113))
            if self.flag is 0:
                self.flag = 1
                thermistor = self.get_ther()
                var_max = int(thermistor + 2)
                print("var_max: ", var_max)
                var_min = int(thermistor - 4)
                print("var_min: ", var_min)
                self.editTextMaxtemp.delete(0, "end")
                self.editTextMaxtemp.insert(0, str(var_max))
                self.editTextMintemp.delete(0, "end")
                self.editTextMintemp.insert(0, str(var_min))
                print("Nhiet do: ", thermistor)
            i = 0  # counter for tarr
            for i in range(0, 113):
                for j in range(0, 113):
                    BGR_opencv = []
                    if tarr_shape[i, j] < float(self.editTextMintemp.get()):
                        normtemp = 0
                    elif tarr_shape[i, j] > float(self.editTextMaxtemp.get()):
                        normtemp = 1
                    else:
                        TempSpan = float(self.editTextMaxtemp.get())-float(self.editTextMintemp.get())
                        if TempSpan <= 0:  # avoid division by 0 and negative values
                            TempSpan = 1
                        normtemp = (float(tarr_shape[i, j])- float(self.editTextMintemp.get()))/TempSpan
                    h = normtemp * self.HUEspan + self.HUEstart  # Convert to HSV colors (only hue used)
                    if h > 1:
                        pass
                    RGB = list(colorsys.hsv_to_rgb(h, 1, 1))
                    # dst = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)
                    BGR_opencv.append(RGB[2] * 255)
                    dst[i, j, 0] = BGR_opencv[0]
                    BGR_opencv.append(RGB[1] * 255)
                    dst[i, j, 1] = BGR_opencv[1]
                    BGR_opencv.append(RGB[0] * 255)
                    dst[i, j, 2] = BGR_opencv[2]
                    # print(BGR_opencv)
            """Show Video"""
            result = cv2.resize(dst, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
            cv_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            # cv2.imshow('image1', result)
            """predict image"""
            img_vgg16 = cv2.resize(dst, (224, 224))
            hsv = cv2.cvtColor(img_vgg16, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((145., 100, 100)), np.array((179., 255., 255.)))
            mask_red = cv2.inRange(hsv, np.array((0.,100,100)), np.array((10., 255., 255.)))
            dst_pr = np.zeros(img_vgg16.shape, img_vgg16.dtype)
            count_average = 0
            for i in range(img_vgg16.shape[0]):
                for j in range(img_vgg16.shape[1]):
                    if (mask[i, j] > 0):
                        dst_pr[i, j] = img_vgg16[i, j]
                        count_average = count_average+1
                    else:
                        dst_pr[i, j] = 0
                    if (mask_red[i, j])>0:
                        dst_pr[i, j] = img_vgg16[i, j]
                        count_average = count_average+1
            self.labelResultRecognize.configure(text ="")
            # print("Mat do cac pixel mau do la: ", count_average)
            if count_average >= threshold_red_pixel:
                img_data = image.img_to_array(dst_pr)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                # start_t = time.time()
                feature = model.predict(img_data)
                # print('execute_time: ' + str(time.time() - start_t))
                # print("The object is: " + clf.predict(feature)[0])
                self.labelResultRecognize.configure(text = clf.predict(feature)[0])
                sa.save_image(clf.predict(feature)[0], dst)
            """"""
            self.tkroot.after(5,self.update_tarrpixels)  # recoursive function call all 10 ms (get_tarr will need about 100 ms to respond)

    def get_image(self):
        if self.START == True:
            dst_get_image = np.zeros((113, 113, 3), dtype = uint8)
            tarr_get = self.get_tarr()  # Get temerature array
            # tarr_get = np.ndarray(tarr_get)
            tarr_get = np.asarray(tarr_get)
            tarr_get_shape = np.reshape(tarr_get, (113, 113))
            for i in range(0, 113):
                for j in range(0, 113):
                    BGR_opencv_get = []
                    # if tarr_get_shape[i, j] < self.MINTEMP.get():
                    if tarr_get_shape[i, j] < float(self.editTextMintemp.get()):
                        normtemp_get = 0
                    # elif tarr_get_shape[i, j] > self.MAXTEMP.get():
                    elif tarr_get_shape[i, j] > float(self.editTextMaxtemp.get()):
                        normtemp_get = 1
                    else:
                        # TempSpan_get = self.MAXTEMP.get() - self.MINTEMP.get()
                        TempSpan_get = float(self.editTextMaxtemp.get()) - float(self.editTextMintemp.get())
                        if TempSpan_get <= 0:
                            TempSpan_get = 1
                        # normtemp_get = (float(tarr_get_shape[i, j]) - self.MINTEMP.get()) / TempSpan_get
                        normtemp_get = (float(tarr_get_shape[i, j]) - float(self.editTextMintemp.get()))/ TempSpan_get
                    h = normtemp_get* self.HUEspan+ self.HUEstart
                    if h > 1:
                        pass
                    RGB_get = list(colorsys.hsv_to_rgb(h, 1, 1))
                    # dst_get_image= cv2.cvtColor(RGB_get, cv2.COLOR_RGB2BGR)
                    BGR_opencv_get.append(int(RGB_get[0]*255))
                    dst_get_image[i, j, 0] = (BGR_opencv_get[0])
                    BGR_opencv_get.append(int(RGB_get[1]*255))
                    dst_get_image[i, j, 1] = (BGR_opencv_get[1])
                    BGR_opencv_get.append(int(RGB_get[2]*255))
                    dst_get_image[i, j, 2] = (BGR_opencv_get[2])
                    # print("BGR_opencv: ", BGR_opencv_get)
            dst_get_image= cv2.cvtColor(dst_get_image, cv2.COLOR_RGB2BGR)
            path = 'D:/Python/gesture_recognition/TH3/nhinguyen_v20/test'
            path2, dirs, files = next(os.walk(path))
            file_count = len(files)
            name_image = str(file_count)+".png"
            cv2.imwrite(os.path.join(path, name_image), dst_get_image)
            path_image_to_read = path+"/"+name_image
            image_read = array(Image.open(path_image_to_read))
            result_compare = np.array_equal(image_read, dst_get_image)
            print("image_read: ", image_read)
            print("Result: ", result_compare)
            # gray()
            imshow(image_read)
            """extract feature"""
            img_vgg16_get = cv2.resize(dst_get_image, (224, 224))
            hsv_get = cv2.cvtColor(img_vgg16_get, cv2.COLOR_BGR2HSV)
            mask_get = cv2.inRange(hsv_get, np.array((145., 100, 100)), np.array((179., 255., 255.)))
            mask_red_get = cv2.inRange(hsv_get, np.array((0.,100,100)), np.array((10., 255., 255.)))
            dst_pr_get = np.zeros(img_vgg16_get.shape, img_vgg16_get.dtype)
            count_average = 0
            for i in range(img_vgg16_get.shape[0]):
                for j in range(img_vgg16_get.shape[1]):
                    if (mask_get[i, j] > 0):
                        dst_pr_get[i, j] = img_vgg16_get[i, j]
                        count_average = count_average+1
                    else:
                        dst_pr_get[i, j] = 0
                    if (mask_red_get[i, j])>0:
                        dst_pr_get[i, j] = img_vgg16_get[i, j]
                        count_average = count_average+1
            print("Mat do cac pixel mau do la: ", count_average)
            if count_average >= threshold_red_pixel:
                img_data_get = image.img_to_array(dst_pr_get)
                img_data_get = np.expand_dims(img_data_get, axis=0)
                img_data_get = preprocess_input(img_data_get)
                # feature = pr.extract_features(path_image_to_read)
                feature_get = model.predict(img_data_get)
                print("The object is: " + clf.predict(feature_get)[0])
            title('image')
            show()

root = tk.Tk()
root.title('Grid-Eye Viewer')
root.geometry('700x600')
"""Load model.joblib"""
db = sys.argv[1]
clf = joblib.load(db + '/model.joblib')
ImageFile.LOAD_TRUNCATED_IMAGES=True
print("[+] Setup model")
base_model = VGG16(weights='imagenet', include_top=True)
out = base_model.get_layer("fc2").output
model = Model(inputs=base_model.input, outputs=out)
"""open file mean.txt"""
file1 = open("mean.txt", "r")
# temp = file1.read
# print("Gia tri doc tu file la: ", temp)
threshold_red_pixel = int(file1.read())
print("threshold: ", threshold_red_pixel)
# print("type of threshold: ", type(threshold_red_pixel))
# print("Type of value from file: ", type(file1.read()))
file1.close()
# label_train = sys.argv[2]
# average = ex.extract_features(label_train)

Window = GridEYE_Viewer(root)
root.mainloop()