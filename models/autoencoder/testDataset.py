"""
from classifierBalancedDataset import HPylorisDatasetClassifier
from AutoEncoderModel import AutoEncoderHPyloris
from sklearn.metrics import accuracy_score
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageChops
import numpy as np
import pickle
import torch

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from ensembleLoss import LinearEnsembleLoss
import torch.optim as optim
from torch import nn
import random
import pickle
import torch


myTransforms = transforms.Compose( [transforms.RandomHorizontalFlip(),  
                                    transforms.RandomRotation(10),   
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),       
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])


labels_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'
image_folders_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/'

myDataset = HPylorisDatasetClassifier(image_folders_path, labels_path, infected=0, transform=myTransforms)
dataloader = DataLoader(myDataset, batch_size=1, shuffle=True)
"""

labels_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'
image_folders_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/'

from classifierBalancedDataset import HPylorisDatasetClassifier

specifyPatientsPath = '/fhome/gia01/vl_project2/randomDatasetCNN.txt'
from torchvision import transforms
myTransforms = transforms.Compose( [transforms.RandomHorizontalFlip(),  
                                    transforms.RandomRotation(10),   
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),       
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

myDataset = HPylorisDatasetClassifier(data_folder=image_folders_path, 
                                      annotations=None, 
                                      specifyPaths=specifyPatientsPath, 
                                      transform=myTransforms)

print(len(myDataset))
print(myDataset[0])

myDataset2 = HPylorisDatasetClassifier(data_folder=image_folders_path, 
                                      annotations=labels_path, 
                                      returnLabels=True)

print(len(myDataset2))
print(myDataset2[0])

from torch.utils.data import DataLoader

dataloader = DataLoader(myDataset, batch_size=1, shuffle=True)

# print(next(iter(dataloader)))