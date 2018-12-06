# import modules
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from collections import OrderedDict

import model_functions
import utility_functions

# argumentparser for command argument
parser = argparse.ArgumentParser(description='Program to train deeplearning model')


parser.add_argument('images_dir', action="store", help="directory of images for training")
parser.add_argument('--save_path', action="store", help="path of the file where the checkpoint would be recprded(NOT THE FILE NAME)")
parser.add_argument('--arch', action="store", help ="architecture of the model. Select from 'vgg13' or 'vgg16'.")
parser.add_argument('--learning_rate', action="store", type=float, help="learning rate when traing the model")
parser.add_argument("--hidden_units", action="store", type=int, help="numer of hidden layers")
parser.add_argument("--epochs", action="store", type=int, help="epochs for training")
parser.add_argument("--gpu", action="store_true", help="whether using the gpu or not")

args = vars(parser.parse_args())

# image data directories
data_dir = args["images_dir"]
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

vandt_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=vandt_transforms)
test_data = datasets.ImageFolder(test_dir,  transform=vandt_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle= True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data,  batch_size=32, shuffle=True)



###  Build and train your network

# import pre-trained model
if args["arch"] is None:
    print("pick the model")
    exit()
elif (args["arch"] == "vgg13"):
    model = models.vgg13(pretrained=True)
elif (args["arch"] == "vgg16"):
    model = models.vgg16(pretrained=True)
else:
    print("pick the model from vgg13 or vgg16")
    exit()

if args["hidden_units"] is None:
    args["hidden_units"] = 1024
elif args["hidden_units"] < 1:
    print("Please set the number of hidden layer to be '2'..")
    exit()


#  set the adjusting classifier
classifier_dict = OrderedDict([("fc1",nn.Linear(25088, args["hidden_units"])),
                                ("relu1", nn.ReLU()),
                                ("fc2", nn.Linear(args["hidden_units"],102)),
                                ("p_out", nn.LogSoftmax(dim=1))])
model.classifier = nn.Sequential(classifier_dict)

# set grad_enabled setting
torch.set_grad_enabled(True)
for param in model.features.parameters():
    param.requires_grad = False


# set the default learning rate = 0.001
if (args["learning_rate"] is None ):
    lr = 0.001
elif (args["learning_rate"] <= 0 ):
    print("Please set the valid learning rate")
    exit()
else:
    lr = args["learning_rate"]

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=lr) ###0.002とか？？？？？

# set the gpu setting if specified
if (args["gpu"] == True):
    device = torch.device("cuda" if torch.cuda.is_available() == True else "cpu")
else:
    device = torch.device("cuda")

print("traing using {}".format(device))
model = model.to(device)


# set the default epoch = 10
if (args["epochs"] is None ):
    epochs = 10
elif (args["epochs"] <= 0 ):
    print("Please set the valid learning rate")
    exit()
else:
    epochs = args["epochs"]

model = model_functions.train_model(model, epochs, device, optimizer,
                                    criterion, train_loader, valid_loader)


model_functions.test_model(model, device, optimizer, criterion, test_loader)



# make the checpoint dictionary
checkpoint = {'epochs' : epochs,
                'arch' : args["arch"],
                'classifier_dict' : classifier_dict,
                'cumulative_epochs' : epochs,
                'state_dict' : model.state_dict(),
                'class_to_idx': train_data.class_to_idx,
                'optimizer_state_dict' : optimizer.state_dict()}

if (args["save_path"] is None):
    args["save_path"] = "checkpoint.pth"

# save the checkpoint in the file
torch.save(checkpoint, args["save_path"])

print("\ncheckpoint saved in {}".format(args["save_path"]))
