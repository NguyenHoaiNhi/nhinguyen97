from sklearn import svm
from sklearn.externals import joblib
import numpy as np
import os
import sys

def predict_test(src, db):
    if os.path.exists(db + "/ketqua.txt"):
        os.remove(db + "/ketqua.txt")
    print("[+] Load model....")
    clf = joblib.load(db + '/model.joblib')
    print("[+] Load model finished")
    print("[+] Predict test data....")
    file = open(db + "/ketqua.txt", "a")
    file.write("              Images                |               Predict               " + "\n")
    file.close()
    with open(src, "r") as file:
        for i,line in enumerate(file):
            img_path = line[:-1]
            #print("[+] Read image  : ", img_path," id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
                save_path = img_path.replace("images", "features\gg16_fc2").replace(".jpg", ".npy")
                file = open(db + "/ketqua.txt", "a")  
                file.write(str(save_path) + "     |      ")  
                data = np.load(save_path)
                file.write(str(clf.predict(data)[0]) + "\n")
                file.close()
    print("[+] Predict test data finished, write result to: " + db + "/ketqua.txt")

if __name__=='__main__':
    src = sys.argv[1]
    db = sys.argv[2]
    predict_test(src, db)
