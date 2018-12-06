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

# returns the model newly made
def generate_model():
    model = models.vgg16(pretrained=True)
    classifier_dict = OrderedDict([("fc1",nn.Linear(25088,4096)), ("relu1", nn.ReLU()), ("fc2", nn.Linear(4096,102)), ("p_out", nn.LogSoftmax(dim=1))])
    model.classifier = nn.Sequential(classifier_dict)
    return model


# returns the model trained. also pronts the process and validation result
def train_model(model, epochs, device, optimizer, criterion, train_loader, valid_loader):

    # iterate the steps
    for epoch_count, e in enumerate(range(epochs)):

        print("\ncurently {}th epoch".format(epoch_count + 1))
        batch_step = 0

        loss_train_total = 0
        accuracy_train_total = 0

        # train model by training data
        for x_train, label_train in train_loader:
            batch_step += 1

            x_train, label_train = x_train.to(device), label_train.to(device)
            optimizer.zero_grad()
            output_train = model.forward(x_train)

            loss_train = criterion(output_train, label_train)
            loss_train_total += loss_train

            top_logp, top_class = output_train.topk(1, dim=1)
            equals = label_train.view(*top_class.shape) == top_class
            accuracy_train_total += equals.type(torch.FloatTensor).mean().item()

            loss_train.backward()
            optimizer.step()

        # testing by validation data
        else:
            model.eval()
            with torch.no_grad():

                loss_valid_total = 0
                accuracy_valid_total = 0

                for x_valid, label_valid in valid_loader:
                    x_valid, label_valid = x_valid.to(device), label_valid.to(device)
                    output_valid = model.forward(x_valid)

                    loss_valid_total += criterion(output_valid, label_valid)

                    top_logp, top_class = output_valid.topk(1, dim=1)
                    equals = label_valid.view(*top_class.shape) == top_class
                    accuracy_valid_total += equals.type(torch.FloatTensor).mean().item()

            print("  training   : loss = {}, accuracy = {}".format(loss_train_total/len(train_loader), accuracy_train_total/len(train_loader)  ))
            print("  validation : loss = {}, accuracy = {}".format(loss_valid_total/len(valid_loader), accuracy_valid_total/len(valid_loader)  ))
            model.train()

    print("\n\ntraining finished")
    return model;


# test the model and prints the results
def test_model(model, device, optimizer, criterion, test_loader):
    model.eval()
    with torch.no_grad():
        loss_test_total = 0
        accuracy_test_total = 0

        for x_test, label_test in test_loader:
            x_test, label_test = x_test.to(device), label_test.to(device)
            output_test = model.forward(x_test)

            loss_test_total = criterion(output_test, label_test)
            top_logp, top_class = output_test.topk(1,dim=1)
            equals = label_test.view(*top_class.shape) == top_class
            accuracy_test_total += equals.type(torch.FloatTensor).mean()

    model.train()
    print("  testing   : loss = {}, accuracy = {}".format(loss_test_total/len(test_loader), accuracy_test_total/len(test_loader) ))





#  return the model, cumulative epoch counts, and class_to_idx mapping
def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    epochs = checkpoint["epochs"]
    model = models.vgg16(pretrained=True)
    if checkpoint["arch"]=="vgg13":
        model = models.vgg13(pretrained=True)
    elif checkpoint["arch"]=="vgg16":
        model = models.vgg16(pretrained=True)
    else:
        print("model not suitable")
        exit()
    model.classifier = nn.Sequential(checkpoint["classifier_dict"])
    model.load_state_dict(checkpoint["state_dict"])
    class_to_idx = checkpoint["class_to_idx"]
    return model, class_to_idx
