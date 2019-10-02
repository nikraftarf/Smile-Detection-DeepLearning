import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as trans
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import os
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torch.autograd import Variable as V

# Import data
root = '/content/drive/My Drive/Colab Notebooks/maindataset/train'
train_trans = trans.Compose([
        trans.RandomResizedCrop(224),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val_trans = trans.Compose([
        trans.Resize(256),
        trans.CenterCrop(224),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_data = dset.ImageFolder(root=root, transform=train_trans,target_transform=None)
test_data = dset.ImageFolder(root='/content/drive/My Drive/Colab Notebooks/maindataset/val', transform=val_trans, target_transform=None)
print(type(train_data))

# Learning Parameters
bs = 27# Batch Size
learning_rate = 0.03
wd = 1e-4 # weight_decay
itr = 28
cuda = True
torch.manual_seed(0)

if torch.cuda.is_available() and cuda:
    torch.cuda.manual_seed_all(0)
    FloatType = torch.cuda.FloatTensor
    LongType = torch.cuda.LongTensor
else:
    FloatType = torch.FloatTensor
    LongType = torch.LongTensor





# Data Loader
kwargs = {'num_workers': 1}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_data,
                 batch_size=bs,
                 shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=bs,
                shuffle=False, **kwargs)




# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         torch.nn.init.kaiming_normal(m.weight.data)
#     elif isinstance(m, nn.Linear):
#         torch.nn.init.kaiming_normal(m.weight.data)
#         m.bias.data.normal_(mean=0,std=1e-2)
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.uniform_()
#         m.bias.data.zero_()


#model
model = models.resnet34(pretrained=True)
# print(model)
# fc = model.classifier._modules['6']
in_features = model.fc.in_features
num_class = len(train_loader.dataset.classes)
# model.classifier._modules['6'] = nn.Linear(4096, num_class)
model.fc = nn.Linear(in_features, num_class)


if __name__ == '__main__':
    # device = torch.device("cuda:0,1")
    # model.to(device)
    if cuda:
        model = model.cuda()

#     model.apply(weights_init)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)



    def train_model(model, optimizer, train_loader, criterion, epoch, vis_step=20):
        model.train(mode=True)
        num_hit = 0
        total = len(train_loader.dataset)
        num_batch = np.ceil(total / bs)
        # Training Phase on train dataset
        for batch_idx, (image, labels) in enumerate(train_loader):
            # Step 1
            optimizer.zero_grad()

            image, labels = V(image.type(FloatType)), V(labels.type(LongType))
            output = model(image)
            loss = criterion(output, labels)

            if batch_idx % vis_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(image),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.data))
            loss.backward()
            optimizer.step()
        # Validation Phase on train dataset
        for image, labels in train_loader:
            image, labels = Variable(image.type(FloatType)), Variable(labels.type(LongType))
            output = model(image)
            _, pred_label = output.data.max(dim=1)
#             print('predt',pred_label)
            num_hit += (pred_label == labels.data).sum()
        train_accuracy = (num_hit.float() / total)
        print("Epoch: {}, Training Accuracy: {:.2f}%".format(epoch, 100. * train_accuracy))

        return train_accuracy * 100.

    def eval_model(model, test_loader, epoch):
        model.train(mode = False)
        num_hit = 0
        total = len(test_loader.dataset)
        for batch_idx, (image, labels) in enumerate(test_loader):
            image, labels = V(image.type(FloatType)), V(labels.type(LongType))
            output = model(image)
            _ , pred_label = output.data.max(dim=1)
            num_hit += (pred_label == labels.data).sum()
        test_accuracy = (num_hit.float() / total)
        print("Epoch: {}, Testing Accuracy: {:.2f}%".format(epoch, 100. * test_accuracy))
        return 100. * test_accuracy


    train_acc = []
    test_acc = []

    for epoch in range(itr):
        tr_acc = train_model(model, optimizer, train_loader, criterion, epoch)
        ts_acc = eval_model(model, test_loader, epoch)

    model.class_to_idx = train_data.class_to_idx
    print(model.class_to_idx)
    torch.save(model,'/content/drive/My Drive/Colab Notebooks/my_resnet34_lr0.03_SGD_model.pth')
#     model.cpu()
#     torch.save({'arch': 'resnet34',
#                 'state_dict': model.state_dict(),
#                 'class_to_idx': model.class_to_idx},
#                 '/content/drive/My Drive/Colab Notebooks/classifier.pth')

#     torch.save(checkpoint, '/content/drive/My Drive/Colab Notebooks/my_model1.pth')
#     torch.save(model.state_dict(), '/content/drive/My Drive/Colab Notebooks/my_model.pkl')