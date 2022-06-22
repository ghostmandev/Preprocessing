import os
import glob
import numpy as np
import time
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.image import imread

import pickle
Dataset_Path = 'D:\\TSLT\\PyCharm\\SplitFrames\\clips\\'

def Delete_MP4(path):
    keyword = '.mp4'
    for folderName in os.listdir(path):
        if folderName[0] == 'D' :
            path_of_the_directory = path + folderName + '\\'
            #print(path_of_the_directory)

            for filename in os.listdir(path_of_the_directory):
                if filename != '.DS_Store':
                    for i in glob.glob(os.path.join(path_of_the_directory,'*.mp4')):
                        try :
                            os.chmod(i,0o777)
                            os.remove(i)
                        except OSError:
                            pass

def get_name_folder(CurrentPath):
    sub_folders = [name for name in os.listdir(CurrentPath) if os.path.isdir(os.path.join(CurrentPath, name))]
    #print(sub_folders)
    return sub_folders

def CountFolder(APP_FOLDER):
    totalFiles = 0
    totalDir = 0
    for base, dirs, files in os.walk(APP_FOLDER):
        print('Searching in : ',base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1
    print('Total number of files',totalFiles)
    print('Total Number of directories',totalDir)
    print('Total:',(totalDir + totalFiles))

####### Resized Files from original to 28x28  ################
def Resized_Files(DirectoryPath) :
    from PIL import Image
    import glob
    image_list = []
    resized_images = []

    for folderName in os.listdir(DirectoryPath):
        if folderName[0] == 'D' :
            path_of_the_directory = DirectoryPath + folderName + '\\'
            print(path_of_the_directory)

            for filename in os.listdir(path_of_the_directory):
                if filename != '.DS_Store' and filename != 'desktop.ini':
                    print(filename)

                    for image in glob.glob(path_of_the_directory + filename + '\\*.jpg'):
                        print(image)
                        img = Image.open(image)
                        image_list.append(img)

                    for im in image_list:
                        im = im.resize((28,28))
                        resized_images.append(im)

                    for (i, new) in enumerate(resized_images):
                        new.save(path_of_the_directory + filename + '\\' + filename + '-' + str(i+1) + '.jpg')

def Show_Image(path) :
    for folderName in os.listdir(path):
        if folderName[0] == 'D' :
            path_of_the_directory = path + folderName + '\\'
            #print(path_of_the_directory)
            name_folder = get_name_folder(path_of_the_directory)
            for i in range(len(name_folder)):
                filename = path_of_the_directory + str(name_folder[i]) + '\\'
                #print(filename)
                for k in range(3) :
                    pyplot.subplot(330 + 1 + k)
                    img_files = filename + str(name_folder[i]) + '-' + str(k+1) + '.jpg'
                    print(img_files)
                    image = imread(img_files)
                    pyplot.imshow(image)
                pyplot.show()
            print(image.shape)

photos, labels, seq, datas = list(), list(), list(), list()
count = 0
for folderName in os.listdir(Dataset_Path):
    start = time.time()

    if folderName[0] == 'D':
        path_of_the_directory = Dataset_Path + folderName + '\\'
        name_folder = get_name_folder(path_of_the_directory)

        for i in range(len(name_folder)):
            filename = path_of_the_directory + str(name_folder[i]) + '\\'
            y_data = name_folder[i]
            for base, dirs, files in os.walk(filename):

                for k in range(len(files)):
                    count += 1
                    try:
                        img_files = filename + str(name_folder[i]) + '-' + str(k + 1) + '.jpg'

                        photo = load_img(img_files, target_size=(28, 28))

                        photo = img_to_array(photo)

                        photos = np.append(photos, photo)

                        datas = photos.reshape(count, 28, 28, 3)
                        labels = np.append(labels, y_data)
                    except FileNotFoundError:
                        print('File Error:=>  ', img_files)
                print(datas.shape, labels.shape)

pre_data = datas
bk_data = open('cnn_mode_data_220622_1_to_9_2828.pkl','wb')
pickle.dump(pre_data,bk_data)
bk_data.close()

pre_labels = labels
bk_labels = open('cnn_mode_labels_220622_1_to_9_2828.pkl','wb')
pickle.dump(pre_labels, bk_labels)
bk_labels.close()

