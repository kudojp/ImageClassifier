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
parser.add_argument('--save_path', action="store", help="path of the file where the checkpoint would be recprded", default = "checkpoint.pth")
parser.add_argument('--arch', action="store", help ="architecture of the model. Select from 'vgg13', 'vgg16', or 'densenet121'", default="vgg13" )
parser.add_argument('--learning_rate', action="store", type=float, help="learning rate when traing the model", default="0.03")
parser.add_argument("--hidden_units", action="store", type=int, help="numer of hidden layers", default=1028)
parser.add_argument("--epochs", action="store", type=int, help="epochs for training", default=10)
parser.add_argument("--gpu", action="store_true", help="whether using the gpu or not", default=True)

args = vars(parser.parse_args())


# check whether command arguments are valid values
model_functions.check_args(args)


# set the gpu setting if specified
if (args["gpu"] == True):
    device = torch.device("cuda" if torch.cuda.is_available() == True else "cpu")
else:
    device = torch.device("cpu")

print("traing using {}".format(device))



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




###  Build and train network
# import pre-trained model
model, classifier_dict = model_functions.generate_model(args["arch"], args["hidden_units"])


# set grad_enabled setting
torch.set_grad_enabled(True)
for param in model.features.parameters():
    param.requires_grad = False

# set loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=args["learning_rate"])

# tramsfer the model and train
model = model.to(device)
model = model_functions.train_model(model, args["epochs"], device, optimizer,
                                    criterion, train_loader, valid_loader)

# test the model with training dataset
model_functions.test_model(model, device, optimizer, criterion, test_loader)


# make the checpoint dictionary
checkpoint = {'epochs' : args["epochs"],
                'arch' : args["arch"],
                'classifier_dict' : classifier_dict,
                'cumulative_epochs' : args["epochs"],
                'state_dict' : model.state_dict(),
                'class_to_idx': train_data.class_to_idx,
                'optimizer_state_dict' : optimizer.state_dict()}


# save the checkpoint in the file
torch.save(checkpoint, args["save_path"])

print("\ncheckpoint saved in {}".format(args["save_path"]))
