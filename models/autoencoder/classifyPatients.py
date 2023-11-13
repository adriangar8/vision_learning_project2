from classify_patient_dataset import PatientDataset
from model import AutoEncoderHPyloris
from sklearn.metrics import accuracy_score
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageChops
import numpy as np
import pickle
import torch


def return_positive_rate(predictions):
    """
    This function returns the percentage of negative class predictions in a list of predictions.
    """
    positive = predictions.count(1)
    return positive / len(predictions)

def chi_square_similarity(hist1, hist2):
    # Ensure both histograms have the same length
    if len(hist1) != len(hist2):
        raise ValueError("Histograms must have the same length")
    # Calculate the Chi-Square statistic
    chi_square_value = np.sum((np.array(hist1) - np.array(hist2))**2 / (np.array(hist1) + np.array(hist2) + 1e-10))
    return chi_square_value

def return_predictions(model, dataset, transform, inverse_transform, num_bins, healthyHistogram, device):
    pred = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image = patient_images[idx]
            original = transform(image).to(device).unsqueeze(0)
            reconstructed_image = inverse_transform(model(original).squeeze(0))
            
            # Convert PIL images to HSV color space
            original_brightness = image.convert('HSV').split()[0]
            reconstructed_brightness = reconstructed_image.convert('HSV').split()[0]
            
            image_difference = np.array(ImageChops.difference(original_brightness, reconstructed_brightness))
            
            non_black_pixels = image_difference[image_difference > 55]
            hist, _ = np.histogram(non_black_pixels.ravel(), bins=num_bins, range=[0, 256])
            
            similarity = chi_square_similarity(hist, healthyHistogram)
            
            if similarity > 700:
                pred.append(1)
            elif similarity <= 700:
                pred.append(0)
    
    return pred


data_path = '/fhome/gia01/vl_project2/autoencoder/txt files/train_thresholds.txt'
with open(data_path, 'rb') as file:
    test_patients = [patient[:-1] for patient in file.readlines()]

model_path = '/fhome/gia01/vl_project2/autoencoder/model_pkl/AE_10epoch_L1Loss.pkl'
with open(model_path, 'rb') as weights:
    state = pickle.load(weights)
    model = AutoEncoderHPyloris()
    model.load_state_dict(state)    
    
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

healthyHistogram = np.loadtxt('/fhome/gia01/vl_project2/autoencoder/txt files/mean_negative_histogram.txt', dtype=int)

image_folders_path = '/fhome/mapsiv/QuironHelico/CroppedPatches/'
global_gt, global_pred = [], []
i = 0
for idx in range(5):
    patient = test_patients[idx]
    patient_id, _, label = patient.decode('utf-8').split(',')
    # print(f'Patient {i + 1}')
    print(f'\nStarting patient {patient_id}')
    
    patient_images = PatientDataset(data_folder=image_folders_path+f'{patient_id}') 
    
    pred = return_predictions(model=model, dataset=patient_images, transform=transform,
                              inverse_transform=inverse_transform, num_bins=20, healthyHistogram=healthyHistogram,
                              device=device)
    
    positive_rate = return_positive_rate(pred)
    
    # if len(patient_images) > 1200:
    #     if positive_rate > 0.165:
    #         global_pred.append(1)
    #     else:
    #         global_pred.append(0)
    # elif len(patient_images) > 700:
    #     if positive_rate > 0.02:
    #         global_pred.append(1)
    #     else:
    #         global_pred.append(0)
    # else:   
    #     if positive_rate > 0.1:
    #         global_pred.append(1)
    #     else:
    #         global_pred.append(0)
        
    if positive_rate > 0.03:
        global_pred.append(1)
    else:
        global_pred.append(0)
    
    global_gt.append(int(label))
    i += 1
    
    print(f'Processed patient {patient_id}.')
    print(f'Patient has {positive_rate*100}% of positive patches.')
    print(f'Classified as {global_pred[-1]}')
    print(f'Patient has {len(patient_images)} images to classify.')

accuracy = accuracy_score(global_gt, global_pred)

print(global_gt)
print(global_pred)

print(f'The accuracy over patients of our model is of {round(accuracy*100, 3)}%.')