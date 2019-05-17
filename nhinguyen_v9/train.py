from sklearn import svm
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np
import os
import sys

def load_features(src):
    print("[+] Load data....")
    data = []
    label = []
    with open(src, "r") as file:
        for i,line in enumerate(file):
            img_path = line[:-1]
            #print("[+] Read image  : ", img_path," id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
                save_path = img_path.replace("images", "features\gg16_fc2").replace(".jpg", ".npy")
                if os.path.isfile(save_path):
                    lb = save_path.split("\\")[2]
                    # lb1 = lb.split(".")[1]
                    # print (lb)
                    # print(save_path)
                    data.append(np.load(save_path)[0])
                    label.append(lb)
    print("[+] Load data finished")
    return data, label

def save_model(model, name, db):
    if not os.path.exists(db):
        os.makedirs(db)
    os.chdir(db)
    file_name = name + ".joblib"
    print("[+] Saving model to file : " ,file_name)
    joblib.dump(model, file_name)

def linear_svm(X_train, y_train, X_test, y_test):
    print("[!] SVM data...")
    clf = svm.SVC(kernel='linear').fit(X_train, y_train)
    # clf = svm.LinearSVC().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("[+] Finished")
    return clf

if __name__=='__main__':
    train = sys.argv[1]
    test = sys.argv[2]
    db = sys.argv[3]
    X_train, y_train = load_features(train)
    X_test, y_test = load_features(test)
    # print(X_test)
    clf = linear_svm(X_train, y_train, X_test, y_test)
    save_model(clf, "model", db)