from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import os

class PatientDataset(Dataset):
    def __init__(self, data_folder):
        """
        Inputs: 
            DATA_FOLDER: folder that contains all folders with the annotated images.
            ANNOTATIONS: csv file that labels the input images
            RETRIEVEPATIENTS: dictionary with the patients to retrieve
            TRANSFORM: apply any kind of torchvision transforms to the images        
        """
        self.data_folder = data_folder
        self.images = self.obtain_all_images(self.data_folder)
        
    def obtain_all_images(self, data_folder):
        output = []        
        images_path = data_folder+'_1/'
        if os.path.exists(images_path):
            img_list = os.listdir(images_path)
            for idx in range(len(img_list)):
                output.append(images_path+img_list[idx])
        return output

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        try:
            image = Image.open(img_name).convert('RGB')
            return image
        
        except Exception as e:
            print(f"Error loading image '{img_name}': {str(e)}")
            return None