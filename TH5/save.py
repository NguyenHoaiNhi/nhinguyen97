import os
import cv2
def save_image(predict, image):
    path_two ='D:/Python/gesture_recognition/TH5/data/two'
    path_three = 'D:/Python/gesture_recognition/TH5/data/three'
    path_thumb = 'D:/Python/gesture_recognition/TH5/data/thumb'
    path_curve = 'D:/Python/gesture_recognition/TH5/data/curve'
    path_hand = 'D:/Python/gesture_recognition/TH5/data/hand'
    path_unknown = 'D:/Python/gesture_recognition/TH5/data/unknown'
    if predict == 'two':
         path, dirs, files = next(os.walk(path_two))
         file_count = len(files)
         name_image = str(file_count)+".jpg"
         cv2.imwrite(os.path.join(path, name_image), image)
    if predict == 'three':
        path, dirs, files = next(os.walk(path_three))
        file_count = len(files)
        name_image = str(file_count)+".jpg"
        cv2.imwrite(os.path.join(path, name_image), image)
    if predict == 'thumb':
        path, dirs, files = next(os.walk(path_thumb))
        file_count = len(files)
        name_image = str(file_count)+".jpg"
        cv2.imwrite(os.path.join(path, name_image), image)
    if predict == 'hand':
        path, dirs, files = next(os.walk(path_hand))
        file_count = len(files)
        name_image = str(file_count)+".jpg"
        cv2.imwrite(os.path.join(path, name_image), image)
    if predict == 'curve':
        path, dirs, files = next(os.walk(path_curve))
        file_count = len(files)
        name_image = str(file_count)+".jpg"
        cv2.imwrite(os.path.join(path, name_image), image)
    if predict == 'unknown':
        path, dirs, files = next(os.walk(path_unknown))
        file_count = len(files)
        name_image = str(file_count)+".jpg"
        cv2.imwrite(os.path.join(path, name_image), image)
    # print("save image")

