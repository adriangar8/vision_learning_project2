from classifierBalancedDataset import HPylorisDatasetClassifier
from AutoEncoderModel import AutoEncoderHPyloris
from sklearn.metrics import accuracy_score
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageChops
import numpy as np
import pickle
import torch.nn as nn
import torch

labels_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'
image_folders_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/'

myDataset = HPylorisDatasetClassifier(image_folders_path, labels_path, infected=0)
print(len(myDataset))

model_path = '/fhome/gia01/vl_project2/autoencoder/model_pkl/AE_10epoch_L1Loss.pkl'
with open(model_path, 'rb') as weights:
    state = pickle.load(weights)
    model = AutoEncoderHPyloris()
    model.load_state_dict(state)        
    
# Now, choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

transform = transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.ToTensor(), 
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

inverse_transform = transforms.Compose([transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                                                std=[1/0.229, 1/0.224, 1/0.225]), 
                                        transforms.ToPILImage()])

# histograms_list = []
gt = []
loss_func = nn.L1Loss()
with torch.no_grad():
    for idx in range(len(myDataset)):
        image, label = myDataset[idx]
        original = transform(image).to(device).unsqueeze(0)
        
        if label == 1:
            loss = loss_func(original, model(original))
            print(loss.item()); break
        
        reconstructed_image = inverse_transform(model(original).squeeze(0))
        
        # Convert PIL images to HSV color space
        original_brightness = image.convert('HSV').split()[0]
        reconstructed_brightness = reconstructed_image.convert('HSV').split()[0]
        
        image_difference = np.array(ImageChops.difference(original_brightness, reconstructed_brightness))
            
        gt.append(label)