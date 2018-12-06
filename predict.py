# import mapping of category label to category name
import json
import argparse
import torch
from torch import optim, nn
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image

import model_functions
import utility_functions


# set the parser for command arguments
parser = argparse.ArgumentParser(description="Program to predict the flower type")

parser.add_argument("image_path", action="store", help="path of the flower image to be predicted")
parser.add_argument("checkpoint_path", action="store", help="path of the checkpoint file for the model", default = "checkpoint.pth")
parser.add_argument("--topk", action="store", help="count of ranking", type=int, default = 5)
parser.add_argument("--category_names", help="path of the json file of mapping of categories to real name", default="cat_to_name.json")
parser.add_argument("--gpu", action="store_true", help="whether using gpu or not")

args = vars(parser.parse_args())


# check if the topk is the valid positive integer
if args["topk"] <= 0:
    print("Please set the valid top numbers")
    exit()


# make the dictionary from category(file) name to flower name
with open(args["category_names"]) as f:
    cat_to_name = json.load(f)

# create the torch.device object
if args["gpu"] == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")



# load the file and recreate the model
model, cat_to_idx = model_functions.load_checkpoint(args["checkpoint_path"], device)

# make the dictionary from idx (of output) to the category(file name)
idx_to_cat = {val: key for key, val in cat_to_idx.items()}



# transfer the model to gpu if availavle
model = model.to(device)

# predict the top k probability and categories
top_p, top_class = utility_functions.predict(args["image_path"], model, device, args["topk"])



# print out the prediction in the console
for i in range(args["topk"]):
    label = cat_to_name[  idx_to_cat[top_class[0][i].item()]  ]
    clas = top_class[0][i].item()
    prob = top_p[0][i].item()
    print( "No.{} prediction : {} [{}] ({}%)".format(i+1, label, clas, prob))
