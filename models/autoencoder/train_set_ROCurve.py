import random

with open('/fhome/gia01/vl_project2/autoencoder/txt files/not_used_patients.txt', 'r') as file:
    patients = [patient[:-1].split(',') for patient in file.readlines()]
    
infected = [patient for patient in patients if patient[2] == '1']
notInfected = [patient for patient in patients if patient[2] == '0']

infectedForTrain = random.sample(infected, 20)
notInfectedForTrain = random.sample(notInfected, 20)

for patient in infectedForTrain:
    patients.remove(patient)
    
for patient in notInfectedForTrain:
    patients.remove(patient)

with open('/fhome/gia01/vl_project2/autoencoder/txt files/train_thresholds.txt', 'w') as file:
    for patient in infectedForTrain:
        patient_id, patient_path, patient_class = patient
        file.write(str(patient_id) + ',' + str(patient_path) + ',' + str(patient_class) + '\n')
    for patient in notInfectedForTrain:
        patient_id, patient_path, patient_class = patient
        file.write(str(patient_id) + ',' + str(patient_path) + ',' + str(patient_class) + '\n')
            
with open('/fhome/gia01/vl_project2/autoencoder/txt files/test_aggregate.txt', 'w') as file:
    for patient in patients:
        patient_id, patient_path, patient_class = patient
        file.write(str(patient_id) + ',' + str(patient_path) + ',' + str(patient_class) + '\n')