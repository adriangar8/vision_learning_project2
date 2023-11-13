from classifierBalancedDataset import HPylorisDatasetClassifier
from AutoEncoderModel import AutoEncoderHPyloris
from sklearn.metrics import accuracy_score
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageChops
import numpy as np
import pickle
import torch


def chi_square_similarity(hist1, hist2):
    # Ensure both histograms have the same length
    if len(hist1) != len(hist2):
        raise ValueError("Histograms must have the same length")
    # Calculate the Chi-Square statistic
    chi_square_value = np.sum((np.array(hist1) - np.array(hist2))**2 / (np.array(hist1) + np.array(hist2) + 1e-10))
    return chi_square_value


def intersection_over_union(hist1, hist2):
    # Ensure both histograms have the same length
    if len(hist1) != len(hist2):
        raise ValueError("Histograms must have the same length")
    # Calculate Intersection (common elements between histograms)
    intersection = np.minimum(hist1, hist2)
    # Calculate Union (total unique elements in both histograms)
    union = np.maximum(hist1, hist2)
    # Calculate Intersection over Union
    iou = np.sum(intersection) / np.sum(union)
    return iou


labels_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'
image_folders_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/'

myDataset = HPylorisDatasetClassifier(image_folders_path, labels_path, infected=0)

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
pred_chi = []
pred_iou = []
num_bins = 20
healthyHistogram = np.loadtxt('/fhome/gia01/vl_project2/autoencoder/txt files/mean_negative_histogram.txt', dtype=int)

with torch.no_grad():
    for idx in range(len(myDataset)):
        image, label = myDataset[idx]
        original = transform(image).to(device).unsqueeze(0)
        reconstructed_image = inverse_transform(model(original).squeeze(0))
        
        # Convert PIL images to HSV color space
        original_brightness = image.convert('HSV').split()[0]
        reconstructed_brightness = reconstructed_image.convert('HSV').split()[0]
        
        image_difference = np.array(ImageChops.difference(original_brightness, reconstructed_brightness))
        
        non_black_pixels = image_difference[image_difference > 55]
        hist, bins = np.histogram(non_black_pixels.ravel(), bins=num_bins, range=[0, 256])
        
        similarity1 = chi_square_similarity(hist, healthyHistogram)
        similarity2 = intersection_over_union(hist, healthyHistogram)
        
        if similarity1 > 30000:
            pred_chi.append(1)
        elif similarity1 <= 30000:
            pred_chi.append(0)
        
        if similarity2 > 0:
            pred_iou.append(0)
        elif similarity2 >= 0:
            pred_iou.append(1)
            
        gt.append(label)
        
accuracy_chi = accuracy_score(gt, pred_chi)
accuracy_iou = accuracy_score(gt, pred_iou)

print('The accuracy of the classifier with CHI SQUARE is:', accuracy_chi*100)
print('The accuracy of the classifier with IOU is:', accuracy_iou*100)

# # Convert the list of histograms to a 2D NumPy array
# histograms_array = np.array(histograms_list)

# # Compute the mean of all histograms along the first axis (rows)
# mean_histogram = np.mean(histograms_array, axis=0)

# plt.hist(mean_histogram, num_bins, label=f'Test Histogram')
# plt.savefig('mean_histogram_nonInfected.png')

# np.savetxt('mean_negative_histogram.txt', mean_histogram, fmt='%d')