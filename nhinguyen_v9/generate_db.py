import sys
import os
import random
import re

def generate_train(files, n):
    random.shuffle(files) # sap xep cac thanh phan trong list mot cach ngau nhien
    return random.sample(files, n)     #tai sao lai phai random cac du lieu

def generate_test(files, train_files):
    # print("Test: ", list(set(files)-set(train_files)))
    return list(set(files) - set(train_files))

'''def get_trailing_numbers(s):
    m = re.search(r'\d+$', s)
    return m.group() 
'''

def write_file(path, files):
    print("[+]Write ", path)
    with open(path, "w") as f:
        for file in files:
            f.write(file)
            f.write("\n")            

def generate_data(src, db):
    train_files = []
    test_files = []
    print("os: ", os.listdir(src))
    for folder in os.listdir(src):
        print("[+]Access folder ",folder)
        folder_path = os.path.join(src, folder)
        print("folder_path: ", folder_path)
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        # print("files: ", files)
        n = len(files)
        # print("n: ", n)
        n_train = int(n * 0.7)
        train_files.extend(generate_train(files, n_train))
        # print("train_file: ", train_files)
        test_files.extend(generate_test(files, train_files))
        # print("test file: ", test_files)
    
    print("[+]Create folder ", db)
    os.makedirs(db)
    print("[+]Change current wd to ", db)
    os.chdir(db)
    write_file("train.txt", train_files)
    write_file("test.txt", test_files)

def main():
    print("Danh sach tham so: ", str(sys.argv))
    src = sys.argv[1] # doi so thu nhat truyen vao
    print("src: ", src)
    db = sys.argv[2] # doi so thu hai truyen vao
    print("dc: ", db)
    generate_data(src, db)

if __name__=='__main__':
    main()
