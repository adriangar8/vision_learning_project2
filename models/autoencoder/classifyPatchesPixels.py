from AutoEncoderModel import AutoEncoderHPyloris
from patientID import retrieve_patients
from histogramDataset import DatasetForNotInfected
from AutoEncoderDataset import AutoEncoderDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageChops
import numpy as np
from PIL import Image
import pickle
import torch
import cv2

cropped_labels = '/fhome/mapsiv/QuironHelico/CroppedPatches/metadata.csv'
cropped_images = '/fhome/mapsiv/QuironHelico/CroppedPatches/'
annotated_labels = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'

not_used_patients_txt = '/fhome/gia01/vl_project2/scripts/not_used_patients.txt'
used_patients_txt = '/fhome/gia01/vl_project2/used_patients.txt'

# List of patients that are infected
infectedPatients = list(retrieve_patients(labels=cropped_labels, 
                                     data_path=cropped_images, 
                                     rm_labels=annotated_labels)['INFECTED'].keys())

# List of patients that are not infected, used for training the autoencoder (although the model hasnt seen the images)
with open(used_patients_txt, 'r') as file:
    healthyPatients = [line.split(',')[0] for line in file.readlines()]

# Dataset with all images from patients that have been already seen
notInfectedDataset = DatasetForNotInfected(data_folder=cropped_images,   
                                        annotations=cropped_labels, 
                                        retrievePatients=healthyPatients,
                                        transform=None)  

infectedDataset = AutoEncoderDataset(data_folder=cropped_images,
                                     annotations=cropped_labels,
                                     retrievePatients=infectedPatients[23],
                                     transform=None,
                                     classe='POSITIVA')

##########################################################################################################################

# Now I want to retrieve all histograms belonging to the subtraction of the original image - the reconstructed image

# First, I need to load the model
model_path = '/fhome/gia01/vl_project2/AE_10epoch_L1Loss.pkl'
model_path = '/fhome/gia01/vl_project2/autoencoder/model_pkl/10_EPOCH_MSE_AE.pkl'
with open(model_path, 'rb') as weights:
    state = pickle.load(weights)
    model = AutoEncoderHPyloris()
    model.load_state_dict(state)        

# Now, choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
inverse_transform = transforms.Compose([transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                                                std=[1/0.229, 1/0.224, 1/0.225]), 
                                        transforms.ToPILImage()])
       
with torch.no_grad():
    all_ratios = []
    for idx in range(3):#len(notInfectedDataset)):
        image = notInfectedDataset[idx]
        # image = infectedDataset[idx]
        
        original = transform(image).to(device).unsqueeze(0)
        reconstructed_image = inverse_transform(model(original).squeeze(0))
 
        hsv_original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        red_pixels_original = np.sum(np.logical_or((hsv_original_image[:, :, 0] < 20) & (hsv_original_image[:, :, 0] >= 0),
                                               (hsv_original_image[:, :, 0] > 160) & (hsv_original_image[:, :, 0] <= 180)))
        percentage_original = 100 * (red_pixels_original / (224 * 224))
        
        hsv_reconstructed_image = cv2.cvtColor(np.array(reconstructed_image), cv2.COLOR_RGB2HSV)
        red_pixels_reconstructed = np.sum(np.logical_or((hsv_reconstructed_image[:, :, 0] < 20) & (hsv_reconstructed_image[:, :, 0] >= 0), 
                                                        (hsv_reconstructed_image[:, :, 0] > 160) & (hsv_reconstructed_image[:, :, 0] <= 180)))
        percentage_reconstructed = 100 * (red_pixels_reconstructed / (224 * 224))
        
        lost_pixels_ratio = percentage_original-percentage_reconstructed if percentage_reconstructed != 0 else 0
        all_ratios.append(lost_pixels_ratio)
        
    print(sum(all_ratios)/len(all_ratios))
    
    