import os
import cv2
def save_image(predict, image):
    print("save image")
    path_two ='D:/Python/gesture_recognition/nhinguyen_v9/data/two'
    path_three = 'D:/Python/gesture_recognition/nhinguyen_v9/data/three'
    path_thumb = 'D:/Python/gesture_recognition/nhinguyen_v9/data/thumb'
    path_gun = 'D:/Python/gesture_recognition/nhinguyen_v9/data/gun'
    path_hand = 'D:/Python/gesture_recognition/nhinguyen_v9/data/hand'
    path_none = 'D:/Python/gesture_recognition/nhinguyen_v9/data/none'
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
    if predict == 'gun':
        path, dirs, files = next(os.walk(path_gun))
        file_count = len(files)
        name_image = str(file_count)+".jpg"
        cv2.imwrite(os.path.join(path, name_image), image)
    if predict == 'none':
        path, dirs, files = next(os.walk(path_none))
        file_count = len(files)
        name_image = str(file_count)+".jpg"
        cv2.imwrite(os.path.join(path, name_image), image)

