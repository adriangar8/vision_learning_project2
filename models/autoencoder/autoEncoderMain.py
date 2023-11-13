from patientID import retrieve_patients
from AutoEncoderDataset import AutoEncoderDataset
from AutoEncoderModel import AutoEncoderHPyloris
from trainAutoEncoder import train

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from ensembleLoss import LinearEnsembleLoss
import torch.optim as optim
from torch import nn
import random
import pickle
import torch

cropped_labels = '/fhome/mapsiv/QuironHelico/CroppedPatches/metadata.csv'
cropped_images = '/fhome/mapsiv/QuironHelico/CroppedPatches/'

annotated_labels = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'
annotated_images = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/'

patients = retrieve_patients(labels=cropped_labels, data_path=cropped_images, rm_labels=annotated_labels)

patients_for_autoencoder = '/fhome/gia01/vl_project2/autoencoder/txt files/used_patients.txt'

with open(patients_for_autoencoder, 'r') as file:
    patients_for_autoencoder = [patient.split(',')[0] for patient in file.readlines()]

# print(patients_for_autoencoder)

"""
for key in patients_for_autoencoder:
    del patients['NOT INFECTED'][key]
      
# Open a file in write mode
with open('not_used_patients.txt', 'w') as file:
    for patient_id, patient_path in patients['NOT INFECTED'].items():
        file.write(str(patient_id) + ',' + str(patient_path) + ',0' + '\n')
    for patient_id, patient_path in patients['INFECTED'].items():
        file.write(str(patient_id) + ',' + str(patient_path) + ',1' + '\n')
"""

myTransforms = transforms.Compose( [transforms.RandomHorizontalFlip(),  
                                    transforms.RandomRotation(10),   
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),       
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

AE_nonInfectedDataset = AutoEncoderDataset(data_folder=cropped_images, 
                                                annotations=cropped_labels, 
                                                retrievePatients=patients_for_autoencoder,
                                                transform=myTransforms)
                                        
print(len(AE_nonInfectedDataset))
                                        
train_ratio = 0.90
val_ratio = 0.10

train_size = int(train_ratio * len(AE_nonInfectedDataset))
val_size = len(AE_nonInfectedDataset) - train_size

AE_train_dataset, AE_val_dataset = random_split(AE_nonInfectedDataset, [train_size, val_size])

AE_train_loader = DataLoader(AE_train_dataset, batch_size=16, shuffle=True)
AE_val_loader = DataLoader(AE_val_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoEncoderHPyloris()

epochs = 10
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

train_loss, val_loss, outputs = train(model=model, 
                                      train_dataloader=AE_train_loader, 
                                      val_dataloader=AE_val_loader,
                                      epochs=epochs, 
                                      criterion=criterion, 
                                      optimizer=optimizer, 
                                      device=device)

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [12, 7]

plt.plot(train_loss, label='Train loss')
plt.plot(val_loss, label='Val loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.savefig('./model_training_losses_10_epochs.png')

with open('./10_EPOCH_MSE_AE.pkl', 'wb') as f:
    pickle.dump(model.state_dict(), f)