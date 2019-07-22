import argparse

import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, transforms, models

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from collections import OrderedDict
import time
#import random, os

from PIL import Image

from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')
    return parser.parse_args()



# change to cuda
#model.to('cuda') # use cuda
#start = time.time()
#print('Initiating training')
def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    steps = 0
    print_everyafter = 10
    for i in range(epochs):
        loss_run = 0
    for ii, (inputs, labels) in enumerate(dataloaders[0]): # 0 = train
        steps += 1 
        #inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda
        if gpu == 'gpu':
            model.cuda()
            inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda
        else:
            model.cpu() # use a CPU if user says anything other than "gpu"
        #inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda # uncomment later
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_run += loss.item()
        
        if steps % print_everyafter == 0:
            model.eval()
            eval_loss = 0
            accuracy=0
        
            for ii, (inputs2,labels2) in enumerate(dataloaders[1]): # 1 = validation 
                    optimizer.zero_grad()
                    
                    if gpu == 'gpu':
                        inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # use cuda
                        model.to('cuda:0') # use cuda
                    #inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # use cuda
                    else:
                        pass
                  
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        eval_loss = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

            eval_loss = eval_loss / len(dataloaders[1])
            accuracy = accuracy /len(dataloaders[1])

            print("Epoch: {}/{}... ".format(i+1, epochs),
                  "Loss Training: {:.4f}".format(loss_run/print_everyafter),
                  "Loss Validation: {:.4f}".format(eval_loss),
                  "Accuracy: {:.4f}".format(accuracy),
                 )

            running_loss = 0
            
def main():
    args = parse_args()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224), 
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    validataion_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                 transforms.ToTensor(), 
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    testing_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 


    # TODO: Load the datasets with ImageFolder
    image_datasets = [datasets.ImageFolder(train_dir, transform=training_transforms), 
                      datasets.ImageFolder(valid_dir, transform=validataion_transforms), 
                      datasets.ImageFolder(test_dir, transform=testing_transforms)]
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True), 
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True), 
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]
    model = getattr(models, args.arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
        if args.arch == "vgg13":
            feature_num = model.classifier[0].in_features
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(feature_num, 1024)),
                ('drop', nn.Dropout(p=0.5)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(1024, 102)),
                ('output', nn.LogSoftmax(dim=1))]))
        elif args.arch == "densenet121":
                classifier = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(1024, 500)),
                    ('drop', nn.Dropout(p=0.6)),
                    ('relu', nn.ReLU()),
                    ('fc2', nn.Linear(500, 102)),
                    ('output', nn.LogSoftmax(dim=1))]))
                
        model.classifier = classifier
        criterion = nn.NLLLoss() 
        optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
        epochs = int(args.epochs)
        class_index = image_datasets[0].class_to_idx
        gpu = args.gpu # get gpu set
        train(model, criterion, optimizer, dataloaders, epochs, gpu)
        model.class_to_idx = class_index
        path = args.save_dir # new save location 
        save_checkpoint(path, model, optimizer, args, classifier)


if __name__ == "__main__":
    main()