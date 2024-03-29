import argparse

import numpy as np

import torchvision
from torchvision import datasets, transforms, models

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

import random, os
from PIL import Image
import json

from utils import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', default='flowers/test/1/image_06743.jpg')
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')  
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    current_width, current_height = image.size
    if current_width < current_height:
        new_height = int(current_height * 256 / current_width)
        image = image.resize((256 , new_height))
    else:
        new_width = int(current_width * 256 / current_height)
        image = image.resize((new_width, 256))

    precrop_width, precrop_height = image.size
    left = (precrop_width - 224)/2
    top = (precrop_height - 224)/2
    right = (precrop_width + 224)/2
    bottom = (precrop_height + 224)/2
    
    image = image.crop((left, top, right, bottom))

    # Convert to numpy array
    np_image = np.array(image)
    np_image = np_image/255.

    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])

    np_image = ((np_image - mean) / sd)

    np_image = np.transpose(np_image, (2, 0, 1)) 

    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    return tensor_image

def predict(image_path, model, topk=3, gpu='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
 
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    #model.to('cuda')
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
    
    image_t = Image.open(image_path) 
    image_t = process_image(image_t)
    #image_t = torch.from_numpy(image_t)
    
    image_t = image_t.unsqueeze_(0)
    image_t = image_t.float()

    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(image_t.cuda())
    else:
        with torch.no_grad():
            output = model.forward(image_t.cpu()) # cpu just modified
        
    probability = F.softmax(output.data,dim=1) # softmax f
    
    probs = np.array(probability.topk(topk)[0][0])
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()} 
    classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    return probs, classes

def main(): 
    args = parse_args()
    image_path = args.filepath
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    gpu = args.gpu
    
    probs, classes = predict(image_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + image_path)
    
    print(labels)
    print(probability)
    
    i=0 # this prints out top k classes and probs as according to user 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 # cycle through

if __name__ == "__main__":
    main()
