import pandas as pd
import numpy as np
import random
import os

# cropped_labels = '/fhome/mapsiv/QuironHelico/CroppedPatches/metadata.csv'
# cropped_images = '/fhome/mapsiv/QuironHelico/CroppedPatches/'

# annotated_labels_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'
# annotated_image_folders_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/'

def retrieve_patients(labels: str, data_path: str, rm_labels:str=None):
    """
    Inputs:
        LABELS: path to the labels file.
        DATA_PATH: path to the folder containing the images.
        REMOVE_LABELS: path to the file containing the patient IDs to remove.
    Output:
        Dictionary with INFECTED and NOT INFECTED as keys.
        Each key has a nested dictionary with PATIENT_ID as keys and path to the patient folder as values.
    """
    output = {}
    infected, notInfected = {}, {}
    
    if rm_labels is not None:
        rm_labels = pd.read_csv(rm_labels)
        rm_labels = list(set([patientID.split('_')[0] for patientID in rm_labels.ID]))
        
        labels = pd.read_csv(labels)
        for _, row in labels.iterrows():
            patient_id = row['CODI']
            if patient_id not in rm_labels:
                infecció = 0 if row['DENSITAT']=='NEGATIVA' else 1
                patient_folder = os.path.join(data_path, f'{patient_id}_1/')
                if os.path.exists(patient_folder):
                    if infecció:
                        infected[patient_id] = patient_folder
                    else:
                        notInfected[patient_id] = patient_folder
        output['INFECTED'] = infected; output['NOT INFECTED'] = notInfected
        
    else:
        labels = pd.read_csv(labels)
        for _, row in labels.iterrows():
            patient_id = row['CODI']
            infecció = 0 if row['DENSITAT']=='NEGATIVA' else 1
            patient_folder = os.path.join(data_path, f'{patient_id}_1/')
            if os.path.exists(patient_folder):
                if infecció:
                    infected[patient_id] = patient_folder
                else:
                    notInfected[patient_id] = patient_folder
        output['INFECTED'] = infected; output['NOT INFECTED'] = notInfected
        
    return output

# patients = retrieve_patients(labels=cropped_labels, data_path=cropped_images, rm_labels=annotated_labels_path, )

# for patient_id, patient_path in patients.items():
#     print(f'{patient_id}: {patient_path}')
    
# print(len(patients['INFECTED']))
# print(len(patients['NOT INFECTED']))

# random_keys = random.sample(patients['NOT INFECTED'].keys(), 21)

# for key in random_keys:
#     del patients['NOT INFECTED'][key]
    
# print(len(patients['INFECTED']))   
# print(len(patients['NOT INFECTED']))