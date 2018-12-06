# import mapping of category label to category name
import json
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from collections import OrderedDict

from PIL import Image

# process the argument of PIL Image object and returns numpy array
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    ## Process a PIL image for use in a PyTorch model

    # resize so that the shortest side is 256 pixels
    width, height = image.size
    longer_side = max(width,height) * 256 // min(width,height)
    if (width >= height):
        img2 = image.resize((longer_side, 256))
    else:
        img2 = image.resize((256, longer_side))

    # crop out the center 224x224 portion of the image
    width, height = img2.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img3 = img2.crop((left,top,right, bottom))

    # convert a PIL image to Numpy array
    ##### SET THE RANGE BETWEEN 0 AND 1before normalizing
    np_image3 = np.asarray(img3) / 255

    # normalize the array of the image
    mean_array = np.array([0.485, 0.456, 0.406])
    std_array = np.array([0.229, 0.224, 0.225])
    np_image3 = (np_image3 - mean_array) / std_array

    # reorder dimensions so that the color channel to be the first dimension
    np_image4 = np_image3.transpose(2,0,1)

    return np_image4





# returns the top few predictions of the picture on the filepath and returns the topk prediction prob and class names tensor
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''


    # Implement the code to predict the class from an image file
    image = Image.open(image_path).convert("RGB")
    image_pro = torch.from_numpy(process_image(image))
    image_pro = image_pro.view(1, image_pro.shape[0], image_pro.shape[1], image_pro.shape[2]).float()
    image_pro = image_pro.to(device)

    model.eval()

    y_pred = model.forward(image_pro)
    top_logp, top_class = y_pred.topk(topk, dim=1)

    model.train()
    return torch.exp(top_logp), top_class
