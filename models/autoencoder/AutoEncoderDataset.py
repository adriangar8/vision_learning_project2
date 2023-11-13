from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import os

class AutoEncoderDataset(Dataset):
    def __init__(self, data_folder, annotations, retrievePatients=None, transform=None, classe='NEGATIVA'):
        """
        Inputs: 
            DATA_FOLDER: folder that contains all folders with the annotated images.
            ANNOTATIONS: csv file that labels the input images
            RETRIEVEPATIENTS: dictionary with the patients to retrieve
            TRANSFORM: apply any kind of torchvision transforms to the images        
        """
        self.data_folder = data_folder
        self.annotations = annotations
        self.transform = transform
        self.images = self.obtain_all_images(self.annotations, 
                                             self.data_folder,
                                             retrievePatients,
                                             classe=classe)
        
    def obtain_all_images(self, annotations, data_folder, retrievePatients=None, classe='POSITIVA'):
        output = []        
        labels = pd.read_csv(annotations)
        
        if classe != 'POSITIVA':
            if retrievePatients is None:
                for _, row in labels.iterrows():
                    code = row['CODI'].split('.')
                    density = row['DENSITAT']  
                    if (density == 'NEGATIVA'):
                        images_path = data_folder+code[0]+'_1/'
                        if os.path.exists(images_path):
                            img_list = os.listdir(images_path)
                            for idx in range(len(img_list)//24):
                                output.append(images_path+img_list[idx])
            else:
                for _, row in labels.iterrows():
                    code = row['CODI'].split('.')
                    if code[0] in retrievePatients:
                        density = row['DENSITAT']  
                        if (density == 'NEGATIVA'):
                            images_path = data_folder+code[0]+'_1/'
                            if os.path.exists(images_path):
                                img_list = os.listdir(images_path)
                                for idx in range(len(img_list)): # 3/4 of the images
                                    output.append(images_path+img_list[idx])
        else:
            if retrievePatients is None:
                for _, row in labels.iterrows():
                    code = row['CODI'].split('.')
                    density = row['DENSITAT']  
                    if (density == 'NEGATIVA'):
                        images_path = data_folder+code[0]+'_1/'
                        if os.path.exists(images_path):
                            img_list = os.listdir(images_path)
                            for idx in range(len(img_list)//24):
                                output.append(images_path+img_list[idx])
            else:
                for _, row in labels.iterrows():
                    code = row['CODI'].split('.')
                    if code[0] in retrievePatients:
                        density = row['DENSITAT']  
                        if (density == 'ALTA' or density == 'BAIXA'):
                            images_path = data_folder+code[0]+'_1/'
                            if os.path.exists(images_path):
                                img_list = os.listdir(images_path)
                                for idx in range(len(img_list)): # 3/4 of the images
                                    output.append(images_path+img_list[idx])
        return output

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        try:
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        
        except Exception as e:
            print(f"Error loading image '{img_name}': {str(e)}")
            return None