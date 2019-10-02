import os
from PIL import Image
import cv2
import glob
import numpy as np
import argparse
import face_recognition



path_laugh = "data/train1/laugh2/"
path1_laugh = "data/train1/laugh3/"
# path2_laugh = "data/train1/laugh3"
dirs = os.listdir(path_laugh)
# Don't froget to change your path!
listing = os.listdir(path_laugh)


def resize():
    i=0
    for item in dirs:  # Iterates through each picture
        if os.path.isfile(path_laugh+item):
            # im = cv2.imread(path_laugh+item)
            image = face_recognition.load_image_file(path_laugh+item)
            face_locations = face_recognition.face_locations(image,number_of_times_to_upsample=1,model="hog")
            for face_location in face_locations:
                # Print the location of each face in this image
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image.save(os.path.join(path1_laugh ,'pic'+str(i)+'.jpg'))

        i+=1
resize()


#
path_smile = "data/train1/smile2/"
path1_smile = "data/train1/smile3/"
dirs_smile = os.listdir(path_smile)
# # Don't froget to change your path!
def resize1():
    i=0
    for item in dirs:  # Iterates through each picture
        if os.path.isfile(path_smile+item):
            # im = cv2.imread(path_laugh+item)
            image1 = face_recognition.load_image_file(path_smile+item)
            face_locations1 = face_recognition.face_locations(image1,number_of_times_to_upsample=1,model="hog")
            for face_location in face_locations1:
                # Print the location of each face in this image
                top, right, bottom, left = face_location
                face_image = image1[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image.save(os.path.join(path1_smile ,'pic'+str(i)+'.jpg'))

        i+=1
resize1()



path_poker = "data/train1/poker2/"
path1_poker = "data/train1/poker3/"
dirs_poker = os.listdir(path_poker)
# # Don't froget to change your path!
def resize2():
    i=0
    for item in dirs_poker:  # Iterates through each picture
        if os.path.isfile(path_poker+item):
            # im = cv2.imread(path_laugh+item)
            image2 = face_recognition.load_image_file(path_poker+item)
            face_locations2 = face_recognition.face_locations(image2,number_of_times_to_upsample=1,model="hog")
            for face_location in face_locations2:
                # Print the location of each face in this image
                top, right, bottom, left = face_location
                face_image = image2[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image.save(os.path.join(path1_poker ,'pic'+str(i)+'.jpg'))

        i+=1
resize2()
