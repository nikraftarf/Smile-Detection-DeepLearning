# Smile-Detection-DeepLearning
This project was my internship project that was about detecting smile, laugh and poker faces. I have used torch library and resnet34 pretrained model for training my model. You can find the main dataset which contains not-resized and non-cropped images hereunder.

https://drive.google.com/open?id=1qQgzHkYmKueY8tyh0f32EIq0Bohoyuvl

main.py contains main code for train the dataset. The dataset contains cropped faces from a main dataset that have croped with face_detection.py code. The dataset have 3 classes: laugh, poker, smile. Laugh and smile folders have about 900 images and poker folder has 373 images. Change your path for training the model. The resize.py is a code for resize all images in a folder. Enjoy Deep Learning!

![face](https://user-images.githubusercontent.com/41823988/66033776-d197bd80-e514-11e9-94d4-49c884606810.gif)
