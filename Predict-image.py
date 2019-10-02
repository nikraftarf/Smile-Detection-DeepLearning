from torch.autograd import Variable
from torchvision import transforms
import cv2
import torch
import dlib
import numpy as np
import face_recognition
from PIL import Image



label_map={0:'laugh',1:'poker',2:'smile'}

detector = dlib.get_frontal_face_detector()
color_green = (0,255,0)
line_width = 3
cuda=False

device = torch.device('cpu')
model=torch.load('D:/internship/mainproject/my_resnet101_lr2_SGD_model.pth',map_location=device)
model.eval()

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    # print(output)
    index = output.data.cpu().numpy().argmax()
    return index
def argmax(prediction):
    prediction = prediction.cpu()
    print('1',prediction)
    prediction = prediction.detach().numpy()
    print('p',prediction)
    top_1 = np.argmax(np.abs(prediction),axis=1)
    print(top_1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = label_map[prediction]

    return result,score


fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0
frame = cv2.imread('your image path',cv2.IMREAD_UNCHANGED)
to_pil = transforms.ToPILImage()
# image = to_pil(frame)
image = face_recognition.load_image_file("your image path")
imResize = cv2.resize(frame, (224, 224))
face_locations = face_recognition.face_locations(imResize,number_of_times_to_upsample=1,model="hog")
for face_location in face_locations:
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    face_image = imResize[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
frame1=np.array(pil_image)
index = predict_image(pil_image)
res = label_map[index]
print(res)
winname = 'smile-detection'
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname,500,250)
scale_percent = 300 # percent of original size
width = int(frame1.shape[1] * scale_percent / 100)
height = int(frame1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
cv2.putText(resized,'%s' %res,(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
cv2.imshow(winname,resized)
cv2.waitKey(0)
cv2.destroyAllWindows()




