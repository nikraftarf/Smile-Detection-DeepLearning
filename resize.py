import os
from PIL import Image
import cv2
import glob
import numpy as np
import argparse



path_laugh = "./maindataset1/val/laugh"
path1_laugh = "./maindataset/val/laugh"
dirs = os.listdir(path_laugh)
# Don't froget to change your path!
listing = os.listdir(path_laugh)
from PIL import Image
import os, sys




def resize():
    i=0
    for item in dirs:  # Iterates through each picture
        if os.path.isfile(path_laugh+item):
            im = cv2.imread(path_laugh + item)
            try:
                im.shape
            except AttributeError:
                print(item)
            if (len(im.shape) == 3) and (im.shape[2] == 3):
                imResize = cv2.resize(im, (224, 224))
                cv2.imwrite(os.path.join(path1_laugh ,'pic'+str(i)+'.jpg'),imResize)
            if len(im.shape) < 3 and (im.shape[2] != 3):
                imrgb = np.repeat(im.astype(np.uint8), 3, 2)
                cv2.imwrite(os.path.join(path1_laugh, 'pic' + str(i) + '.jpg'), imrgb)
        i+=1
resize()



path_smile = "D:/internship/mainproject/maindataset1/val/smile"
path1_smile = "D:/internship/mainproject/maindataset/val/smile"
dirs_smile = os.listdir(path_smile)
# Don't froget to change your path!
def resize1():
    i1=0
    for item in dirs_smile:  # Iterates through each picture
        if os.path.isfile(path_smile+item):
            im1 = cv2.imread(path_smile + item)
            try:
                im1.shape
            except AttributeError:
                print(item)
            if (len(im1.shape) == 3) and (im1.shape[2]==3):
                imResize1 = cv2.resize(im1, (224,224))
                cv2.imwrite(os.path.join(path1_smile ,'pic'+str(i1)+'.jpg'),imResize1)
            if len(im1.shape) < 3 and (im1.shape[2] != 3):
                imrgb1 = np.repeat(im1.astype(np.uint8), 3, 2)
                cv2.imwrite(os.path.join(path1_smile, 'pic' + str(i1) + '.jpg'), imrgb1)
        i1+=1
resize1()



path_poker = "D:/internship/mainproject/maindataset1/val/poker"
path1_poker = "D:/internship/mainproject/maindataset1/val/poker"
dirs_poker = os.listdir(path_poker)
# Don't froget to change your path!
def resize2():
    i2=0
    for item in dirs_poker:  # Iterates through each picture
        if os.path.isfile(path_poker+item):
            im2 = cv2.imread(path_poker + item)
            try:
                im2.shape
            except AttributeError:
                print(item)
            if (len(im2.shape) == 3) and (im2.shape[2]==3):
                imResize2 = cv2.resize(im2, (224,224))
                cv2.imwrite(os.path.join(path1_poker ,'pic'+str(i2)+'.jpg'),imResize2)
            if len(im2.shape) < 3 and (im2.shape[2] != 3):
                imrgb2 = np.repeat(im2.astype(np.uint8), 3, 2)
                cv2.imwrite(os.path.join(path1_poker, 'pic' + str(i2) + '.jpg'), imrgb2)
        i2+=1
resize2()
