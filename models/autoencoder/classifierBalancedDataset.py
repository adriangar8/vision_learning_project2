from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

# Define a custom dataset class for the Helicobacter pylori classifier
class HPylorisDatasetClassifier(Dataset):
    def __init__(self, data_folder, annotations, specifyPaths=None, infected=0, returnLabels=True, transform=None):
        """
        Inputs: 
            DATA_FOLDER: folder that contains all folders with the annotated images.
            ANNOTATIONS: csv file that labels the input images
            SPECIFYPATHS: txt file with the paths of the images to retrieve with its corresponding label
            INFECTED: add infected (1), non-infected (-1) or all images (0) to dataset
            RETURNLABELS: return image labels (1) or not (0)
            TRANSFORM: apply any kind of torchvision transforms to the images        
        """
        self.data_folder = data_folder
        self.annotations = annotations
        self.infected = infected
        self.returnLabels = returnLabels
        self.transform = transform
        # Obtain all images from the annotations file and store them in a list
        self.images = self.obtain_all_images(self.annotations, 
                                             self.data_folder,
                                             self.infected,
                                             specifyPaths)
        
    def obtain_all_images(self, annotations, data_folder, infected, specifyPatients):
        output = []        
        if annotations != None:
            labels = pd.read_csv(annotations)
        negative_samples, positive_samples = 0, 0
        
        if specifyPatients == None:
            # Iterate over all rows in the annotations file
            for _, row in labels.iterrows():
                image_names = row['ID'].split('.')
                presence = row['Presence']
                # If we want all images or only infected/non-infected images, add the image to the list
                if infected == 0 and presence != 0:
                    image_path = os.path.join(data_folder,image_names[0], image_names[1]+'.png')
                    if os.path.exists(image_path):
                        if presence == -1:
                            if negative_samples < positive_samples:
                                output.append((image_path, 0))
                                negative_samples += 1
                        else:
                            output.append((image_path, presence))
                            positive_samples += 1       
            # print(output)
            return output
        else:
            with open(specifyPatients, 'r') as file:
                specifyPatients = [tuple(line[:-1].split(',')) for line in file.readlines()]
            output = []
            for instance in specifyPatients:
                output.append((instance[0], int(instance[1])))
            # print(output)
            return output

    def get_path_label(self, idx):
        # Return the image name and label at the specified index
        img_path, label = self.images[idx]
        return img_path, label

    def __len__(self):
        # Return the length of the list of images
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image name and label at the specified index
        img_name, label = self.images[idx]
        try:
            # Open the image and convert it to RGB
            image = Image.open(img_name).convert('RGB')
            # Apply any specified transforms to the image
            if self.transform:
                image = self.transform(image)
            # else:
            #     image = transforms.ToTensor(image)
            # If we want to return the labels, return the image and label
            if self.returnLabels:
                return image, label
            # Otherwise, return just the image
            else:
                return image
        except Exception as e:
            # If there was an error loading the image, print an error message and return None
            print(f"Error loading image '{img_name}': {str(e)}")
            return None