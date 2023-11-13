from classifierBalancedDataset import HPylorisDatasetClassifier
import random

labels_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'
image_folders_path = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/'

myDataset = HPylorisDatasetClassifier(image_folders_path, labels_path, infected=0)

healthy, infected = [], []
for idx in range(len(myDataset)):
    path, label = myDataset.get_path_label(idx)
    if label == 1:
        infected.append((path, label))
    else:
        healthy.append((path, label))

# random_infected_CNN = random.sample(infected, len(infected)//2)
# random_healthy_CNN = random.sample(healthy, len(healthy)//2)


# print('CNN, INFECTED AND HEALTHY')
# print(len(random_infected_CNN))
# print(len(random_healthy_CNN))
# print('VGG, INFECTED AND HEALTHY')
# print(len(random_infected_VGG))
# print(len(random_healthy_VGG))

# with open('randomDatasetCNN.txt', 'w') as file:
#     for path, label in random_infected_CNN:
#         file.write(f'{path},{label}\n')
#     for path, label in random_healthy_CNN:
#         file.write(f'{path},{label}\n')

with open('randomDatasetCNN.txt', 'r') as file:
    all = [line[:-1].split(',') for line in file.readlines()]

# print(all)

random_infected_CNN = []
random_healthy_CNN = []
for patient in all:
    if patient[1] == '1':
        random_infected_CNN.append((patient[0], int(patient[1])))
    else:
        random_healthy_CNN.append((patient[0], int(patient[1])))
# print(infected)
# print(len(random_infected_CNN))
# print(len(random_healthy_CNN))

# print(random_infected_CNN)
# print(random_healthy_CNN)

random_infected_VGG = []
random_healthy_VGG = []
for patient in infected:
    if patient not in random_infected_CNN and patient[1] == 1:
        random_infected_VGG.append((patient[0], patient[1]))
for patient in healthy: 
    if patient not in random_healthy_CNN and patient[1] == 0:
        random_healthy_VGG.append((patient[0], patient[1]))

print(len(random_infected_VGG))
print(len(random_healthy_VGG))
with open('randomDatasetVGG.txt', 'w') as file:
    for path, label in random_infected_VGG:
        file.write(f'{path},{label}\n')
    for path, label in random_healthy_VGG:
        file.write(f'{path},{label}\n')