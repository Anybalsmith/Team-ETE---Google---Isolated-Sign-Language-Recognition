# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import pickle
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import random
import os
dataset_path = '/kaggle/input/asl-signs/train_landmark_files'
dataset_files = os.listdir(dataset_path)
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
       # print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
dataset_files

# %%
file_paths_dict = {}

# %%
for digit in dataset_files:
    # Chemin complet du répertoire du chiffre
    digit_path = os.path.join(dataset_path, digit)
    # Listes des fichiers dans le répertoire du chiffre
    digit_files = os.listdir(digit_path)
    # Chemins complets des sous-fichiers pour le chiffre
    digit_file_paths = [os.path.join(digit_path, file) for file in digit_files]
    # Stocker les chemins complets des sous-fichiers dans le dictionnaire
    file_paths_dict[digit] = digit_file_paths
    # print(file_paths_dict[digit])

# %%
# file_paths_dict

# %%
print(file_paths_dict['16069'][0])

# %%
metadata_table_test = pq.read_metadata(file_paths_dict['16069'][0])

# %%
metadata_table_test

# %% [markdown]
# dataset description : 
# train_landmark_files/[participant_id]/[sequence_id].parquet The landmark data. The landmarks were extracted from raw videos with the MediaPipe holistic model. Not all of the frames necessarily had visible hands or hands that could be detected by the model.
# 
# Landmark data should not be used to identify or re-identify an individual. Landmark data is not intended to enable any form of identity recognition or store any unique biometric identification.
# 
# frame - The frame number in the raw video.
# row_id - A unique identifier for the row.
# type - The type of landmark. One of ['face', 'left_hand', 'pose', 'right_hand'].
# landmark_index - The landmark index number. Details of the hand landmark locations can be found here.
# [x/y/z] - The normalized spatial coordinates of the landmark. These are the only columns that will be provided to your submitted model for inference. The MediaPipe model is not fully trained to predict depth so you may wish to ignore the z values.

# %%
table_test = pq.read_table(file_paths_dict['16069'][0])

# %%
# Show example
table_test.to_pandas()

# %% [markdown]
# Coordonnées relatives
# 
# Durée des séquences
# 
# Compter nombre de mots dit
# 
# Chaque utilisateur à un nombre de frame 
# 
# Utilisation de transformeur on traite des séquences (temporelles)
# 
# Puiser de l'inspiration dans l'autre Kaggle 
# 
# Preprocessing : NaN à virer
# 
# 250 classes
# 
# les distributions des types (face, left hand...)
# 
# éventuellement aussi les coordonnées (x,y, z peut être moin pertinent) et les coordonnées peuvent être croisé avec les types
# 
# Dans chaque séquence l'utilisateur dit un mot

# %%
#file_paths_dict

# %%
# Count mean frame
participants = file_paths_dict.keys()
print(participants)
print(len(participants))
nb_values_cumul = 0
for keys,values in file_paths_dict.items() : 
    nb_values = len(values)
    nb_values_cumul += nb_values
    print('number of values for keys', keys, ":", nb_values)
print('mean number of frame : ', nb_values_cumul/len(participants))

# %%
# Count duration of a sequence and then mean duration of a sequence assuming we have 24 fps
# table_test = pq.read_table(file_paths_dict['16069'][1]) pour ce cas là pourquoi ça commence à 100 frame ? 


table_test = pq.read_table(file_paths_dict['18796'][45])
table_test = table_test.to_pandas()
table_test['frame'].max()
# Idea : count max frame

# %%
# max_frame_list = []
# for keys,file_paths in file_paths_dict.items() :
#     for file_path in file_paths:
#         table = pq.read_table(file_path)
#         table = table.to_pandas()
#         max_frame_value = table['frame'].max()
#         max_frame_list.append(max_frame_value)

# %%
# output_file = "max_frame_list.txt"

# with open(output_file, 'w') as file:
#     for value in max_frame_list:
#         file.write(str(value) + '\n')

# %%
max_frame_list = []
with open('/kaggle/input/max-frame-list/max_frame_list.txt','r') as file : 
    for value in file :
        max_frame_list.append(value)

# %%
max_frame_list = [element.strip() for element in max_frame_list]

# %%
max_frame_list = [int(element) for element in max_frame_list]

# %%


# %%
import matplotlib.pyplot as plt
plt.hist(max_frame_list, bins = 40, color = 'skyblue', edgecolor='black')
plt.xlabel('Maximum frame value')
plt.ylabel('Frequency')
plt.title('Distribution of maximum frame values')
plt.grid(True)
plt.show()

# %%
mini = 30
maxi = 60
nb_value_min_max = sum(1 for value in max_frame_list if mini <= value <= maxi)
per_plage = (nb_value_min_max / len(max_frame_list)) * 100

# %%
per_plage

# %%
train_df = pd.read_csv('/kaggle/input/asl-signs/train.csv')

# %%
train_df

# %%
# occurence de chaque signe
occurence_sign = train_df['sign'].value_counts()
print(occurence_sign)

# %% [markdown]
# Retrieve image from dataset

# %%
table_test['type'].unique()

# %% [markdown]
# Preprocess Data - Fill NAN with 0, normalization

# %% [markdown]
# test

# %%
table_test = pq.read_table(file_paths_dict['16069'][0])

table_test = table_test.to_pandas()

table_test = table_test.fillna(0)

output_dir = '/kaggle/working/output_dir'


os.makedirs(output_dir, exist_ok=True)

dir_name, base_name = os.path.split(file_paths_dict['16069'][0])

new_file_path = os.path.join(output_dir, f"modified_{base_name}")

table_test.to_parquet(new_file_path, index = False)

# %% [markdown]
# Fill NAN Done below

# %%
# for keys,file_paths in file_paths_dict.items() :
#     for file_path in file_paths:
#         table = pq.read_table(file_path)
#         df = table.to_pandas()
#         df = df.fillna(0)
#         # Generate a new file name not to erase the original
#         base_name = os.path.basename(file_path)
#         new_file_path = os.path.join(output_dir, f"modified_{base_name}")
#         # Save DataFrame
#         df.to_parquet(new_file_path, index=False)
        
        

# %% [markdown]
# Index face

# %%
table_test = pq.read_table(file_paths_dict['16069'][0])
df = table_test.to_pandas()
face_df = df[df['row_id'].str.contains('face')]


# %%
face_df

# %%
face_df_first_img = face_df[face_df['frame'] == 25]

# %%
face_df_first_img

# %%
index_to_highlight = face_df_first_img.index[face_df_first_img['row_id'] == '25-face-0']

# %%
plt.scatter(face_df_first_img['x'],-face_df_first_img['y'])
plt.scatter(face_df_first_img.loc[index_to_highlight, 'x'], -face_df_first_img.loc[index_to_highlight, 'y'], color='red', label='25-face-17')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
# récupération d'une ligne sur 10
every_fifth_row = face_df_first_img.iloc[::2]
index_to_highlight2 = face_df_first_img.index[face_df_first_img['row_id'] == '25-face-0']

# %%
plt.scatter(every_fifth_row['x'],-every_fifth_row['y'])
# for i, row in every_fifth_row.iterrows():
#     plt.text(row['x'], row['y'], str(i), fontsize=5, ha='center', va='bottom')
plt.scatter(face_df_first_img.loc[index_to_highlight, 'x'], -face_df_first_img.loc[index_to_highlight, 'y'], color='red', label='25-face-17')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
face_df_second_img = face_df[face_df['frame'] == 46]
every_fifth_row2 = face_df_second_img.iloc[::2]
index_to_highlight2 = face_df_second_img.index[face_df_first_img['row_id'] == '46-face-0']

# %%
face_df_second_img

# %%
# second frame
plt.scatter(every_fifth_row2['x'],-every_fifth_row2['y'])
plt.scatter(face_df_second_img.loc[index_to_highlight2, 'x'], -face_df_second_img.loc[index_to_highlight2, 'y'], color='red', label='25-face-17')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %% [markdown]
# **Work on preprocess dataset**

# %%
COMPETITION_PATH = '/kaggle/input/asl-signs/'
dataset_path = '/kaggle/input/asl-signs/train_landmark_files'
user_ids = os.listdir('/kaggle/input/asl-signs/train_landmark_files')

# %%
ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

# %%
def select_random_sequence():
    usr = random.choice(user_ids)
    usr_sqc = os.listdir(os.path.join(dataset_path,usr))
    sqc = random.choice(usr_sqc)
    return os.path.join(dataset_path,usr,sqc)

# %%
import random
select_random_sequence()

# %%
maxX=[]
maxY=[]
maxZ=[]
for usr in user_ids:
    usr_sqc = os.listdir(os.path.join(dataset_path,usr))
    for sqc in usr_sqc:
        pth = os.path.join(dataset_path,usr,sqc)
        df = pd.read_parquet(pth, columns=['x', 'y', 'z'])
        maxX.append(np.max(df.x))
        maxY.append(np.max(df.y))
        maxZ.append(np.max(df.z))

print(f'max x: {np.max(maxX)}\nmax y: {np.max(maxY)}\nmax z: {np.max(maxZ)}')

'''
outputs:

max x: 2.9205052852630615
max y: 3.572496175765991
max z: 4.796591758728027
'''

# %%
cols = ['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']
pq_path = select_random_sequence()
df = pd.read_parquet(pq_path, columns=cols)
print(pq_path)
print(f'xmax: {np.max(df.x)}\nymax: {np.max(df.y)}\nxmin: {np.min(df.x)}\nymin: {np.min(df.y)}')

# %%
# lips idx
LIPS_IDXS0 = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])

# left hand, by taking account face from 0 to 468
LEFT_HAND_IDXS0 = np.arange(468,489)
RIGHT_HAND_IDXS0 = np.arange(522,543)
LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])

REDUCED_LANDMARKS = np.sort(np.concatenate([LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, LEFT_POSE_IDXS0, RIGHT_POSE_IDXS0]))
print(REDUCED_LANDMARKS)
print(len(REDUCED_LANDMARKS))

# %%
train_path = '/kaggle/input/asl-signs/train.csv'
train = pd.read_csv(train_path)
train.head()

# %%
import json
 
# Opening JSON file
f = open('/kaggle/input/asl-signs/sign_to_prediction_index_map.json')
 
# returns JSON object as 
# a dictionary
WORD2IDX = json.load(f)
# print(len(WORD2IDX), WORD2IDX)

# %%
random_word = random.choice(train.sign.unique())
print(f'idx for {random_word} is {WORD2IDX[random_word]}')

# %%
preprocess_path = '/kaggle/input/preprocess-dataset'
preprocess_files = os.listdir(preprocess_path)

# %%
import pickle
with open('/kaggle/input/preprocess-dataset/preprocess_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# %%
data

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

# %%
class ISLR(Dataset) : 
    def __init__(self, dataset,split):
        self.split = split
        self.dataset = dataset
        
        if split == 'trainval':
            self.islr_dataset = dataset[:int(0.8*len(dataset))]
        elif split == 'test':
            self.islr_dataset = dataset[int(0.8*len(dataset)):]
    
    def __len__(self):
        return len(self.islr_dataset)
    
    def __getitem__(self, index):
        sample = self.islr_dataset[index]
        features = torch.FloatTensor(sample[0])
        target = WORD2IDX[sample[1]]
        
        return features, target
    
    def get_sequences_by_target(self, target_value):
        sequences = []
        for sample in self.islr_dataset:
            features = torch.FloatTensor(sample[0])
            target = WORD2IDX[sample[1]]
            if target == target_value:
                sequences.append((features, target))
            if len(sequences) == 2:  # Stop after finding two sequences
                break
        return sequences

# %%
trainset = ISLR(data, split='trainval')
testset = ISLR(data, split='test')

# %%
len(testset)+len(trainset)

# %%
len(data)

# %%
features, target = trainset.__getitem__(0)

# %%
first_image = features[0]

# %%
first_image

# %% [markdown]
# **Augmentation des données**
#     
#     Augmentation spatiale
#     
# * Ajout de bruit gaussien (ok)
# * Translation (ok)
# * Resized crop (not sure given that we already normalized all data)
# * Masquage : random erasing (ok)
# * Mixup : Combine 2 images and their associated labels with a given ratio, can be done for example with a random body part (almost ok)
# 
#     Augmentation temporelle
# * Time Shift : shift temporal sequence (ok)
# 
# Mettre dans une classe (almost ok)

# %%
x_coords = first_image[:,0]
# print(x_coords1)
y_coords = first_image[:,1]
plt.scatter(x_coords, -y_coords)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot of First Image')
plt.show()

# %% [markdown]
# **Add Gaussian Noise**

# %%
def add_gaussian_noise(features, noise_std=0.01):
    noise = torch.randn_like(features) * noise_std
    noisy_features = features + noise
    noisy_features = torch.clamp(noisy_features, 0, 1)  # Clamping to keep values within [0, 1]
    return noisy_features

# %%
noised_features = add_gaussian_noise(features)

# %%
first_image_noised = noised_features[0]

# %%
first_image_noised

# %%
x_coords = first_image_noised[:,0]
# print(x_coords1)
y_coords = first_image_noised[:,1]
plt.scatter(x_coords, -y_coords)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot of First Image Noised')
plt.show()

# %% [markdown]
# **Add Translation**

# %%
def translation(features):
    tr_x = random.uniform(-0.1, 0.1)
    tr_y = random.uniform(-0.1, 0.1)
    translated_features = features.clone()  # Cloning to avoid modifying the original features
    translated_features[:, 0] += tr_x
    translated_features[:, 1] += tr_y
    translated_features = torch.clamp(translated_features, 0, 1)  # Clamping to keep values within [0, 1]
    return translated_features


# %%
first_image_translated = translation(first_image)

# %%
first_image_translated

# %%
x_coords = first_image_translated[:,0]
# print(x_coords1)
y_coords = first_image_translated[:,1]
plt.scatter(x_coords, -y_coords)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot of First Image Translated')
plt.show()

# %% [markdown]
# remark : we may need to drop all values above or under certain values to keep the same frame

# %% [markdown]
# **Random Erasing**

# %%
def random_erasing(points, probability = 0.9, erasing_value = (0,0)):
    if random.uniform(0,1)>probability:
        return points
    points_copy = points.clone()
    nb_points = len(points_copy)
    nb_points_erased = int(nb_points*random.uniform(0.9,0.9))
    erased_indices = random.sample(range(nb_points), nb_points_erased)
    for idx in erased_indices:
        points_copy[idx] = torch.tensor(erasing_value, dtype=points_copy.dtype)
    return points_copy

# %%
first_image_masked = random_erasing(first_image)

# %%
first_image_masked

# %%
x_coords = first_image_masked[:,0]
# print(x_coords1)
y_coords = first_image_masked[:,1]
plt.scatter(x_coords, -y_coords)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot of First Image Masked')
plt.show()

# %% [markdown]
# **Time Shift : shift temporal sequence**

# %%
def time_shift_images(sequence, max_shift=5):

    shift = random.randint(-max_shift, max_shift)
    seq_len, num_points, num_features = sequence.shape
    
    shifted_sequence = torch.zeros_like(sequence)  
    
    for i in range(seq_len):
        if i + shift < seq_len:
            shifted_sequence[i + shift] = sequence[i]
    
    return shifted_sequence

# %%
features, target

# %%
features[-2]

# %%
features_shifted = time_shift_images(features)

# %%
features_shifted

# %%
features_shifted[-1]

# %%
features_shifted[-1]

# %%
features_shifted[1]

# %%
# lips idx
LIPS_IDXS0 = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])

# left hand, by taking account face from 0 to 468
LEFT_HAND_IDXS0 = np.arange(468,489)
RIGHT_HAND_IDXS0 = np.arange(522,543)
LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])

REDUCED_LANDMARKS = np.sort(np.concatenate([LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, LEFT_POSE_IDXS0, RIGHT_POSE_IDXS0]))
print(REDUCED_LANDMARKS)

# Créer un dictionnaire de correspondance
index_map = {value: idx for idx, value in enumerate(REDUCED_LANDMARKS)}



# %%
index_map

# %%
def mixup_body_part_sequence(sequence1, sequence2, part_indices):
    # Obtenir les longueurs des séquences
    len1 = len(sequence1)
    len2 = len(sequence2)

    # Déterminer la longueur commune
    min_len = min(len1, len2)

    # Tronquer les séquences à la longueur commune
    sequence1 = sequence1[:min_len]
    sequence2 = sequence2[:min_len]

    # Cloner la seconde séquence pour éviter de la modifier directement
    mixed_sequence = sequence2.clone()

    # Appliquer le mixup pour chaque image dans la séquence
    for i in range(min_len):
        mixed_sequence[i][part_indices] = sequence1[i][part_indices]

    return mixed_sequence



# %%
# retrouver toutes les parties du corps dans REDUCED_LANDMARKS
# Trouver les indices des lèvres dans les landmarks réduits
# LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, LEFT_POSE_IDXS0, RIGHT_POSE_IDXS0
lips_indices_in_reduced = [index_map[idx] for idx in LIPS_IDXS0]
left_hand_indices_in_reduced = [index_map[idx] for idx in LEFT_HAND_IDXS0]
right_hand_indices_in_reduced = [index_map[idx] for idx in RIGHT_HAND_IDXS0]
left_pose_indices_in_reduced = [index_map[idx] for idx in LEFT_POSE_IDXS0]
right_pose_indices_in_reduced = [index_map[idx] for idx in RIGHT_POSE_IDXS0]

print("RIGHT_POSE_IDXS0:", RIGHT_POSE_IDXS0)
print("REDUCED_LANDMARKS:", REDUCED_LANDMARKS)
print("RIGHT POSE:", right_pose_indices_in_reduced)

# %%
# choisir une partie du corps au hasard

# Dictionnaire des parties du corps
BODY_PARTS = {
    'lips': lips_indices_in_reduced,
    'left_hand': left_hand_indices_in_reduced,
    'right_hand': right_hand_indices_in_reduced,
    'left_pose': left_pose_indices_in_reduced,
    'right_pose': right_pose_indices_in_reduced
}

# Fonction pour obtenir une partie du corps au hasard
def get_random_body_part():
    body_part_name = random.choice(list(BODY_PARTS.keys()))
    body_part_indices = BODY_PARTS[body_part_name]
    return body_part_name, body_part_indices

body_part_name, body_part_indices = get_random_body_part()
print(f"Applying mixup to: {body_part_name}")
print(body_part_indices)

# %% [markdown]
# Test de la fonction mixup pour le mot TV

# %%
# récupérer deux séquences dont le mot est TV

# # Fonction pour récupérer deux séquences avec target 0
# def get_sequences_with_target(dataset, target_value, num_sequences=2):
#     sequences = []
#     for i in range(len(dataset)):
#         features, target = dataset[i]
#         if target == target_value:
#             sequences.append((features, target))
#             if len(sequences) == num_sequences:
#                 break
#     return sequences

# Récupérer deux séquences dont la cible est 0
# target_value = 0
sequences = trainset.get_sequences_by_target(0)

# Afficher les séquences récupérées
for i, (features, target) in enumerate(sequences_with_target_0):
    print(f"Séquence {i+1} - Features: {features}, Target: {target}")

# %%
sequence_1 = sequences_with_target_0[0][0]
sequence_2 = sequences_with_target_0[1][0]

# %%
sequence_1[0]

# %%
# Get first image first sequence
first_image_sequence_1 = sequence_1[0]

x_coords = first_image_sequence_1[:,0]
y_coords = first_image_sequence_1[:,1]

# Tracer les points
plt.scatter(x_coords, -y_coords, label='Other Points', color='blue')
# Tracer les points des lèvres en rouge
plt.scatter(x_coords[body_part_indices], -y_coords[body_part_indices], label='Lips', color='red')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot of First Image First Sequence')
plt.show()

# %%
# Get first image second sequence
first_image_sequence_2 = sequence_2[0]

x_coords = first_image_sequence_2[:,0]
y_coords = first_image_sequence_2[:,1]

# Tracer les points
plt.scatter(x_coords, -y_coords, label='Other Points', color='blue')
# Tracer les points des lèvres en rouge
plt.scatter(x_coords[body_part_indices], -y_coords[body_part_indices], label='Lips', color='red')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot of First Image Second Sequence')
plt.show()

# %%
mixed_sequence = mixup_body_part_sequence(sequence_1,sequence_2,body_part_indices)

# %%
mixed_sequence[0]

# %%
first_image_mixed_sequence = mixed_sequence[0]
x_coords = first_image_mixed_sequence[:,0]
y_coords = first_image_mixed_sequence[:,1]

# Tracer les points
plt.scatter(x_coords, -y_coords, label='Other Points', color='blue')
# Tracer les points des lèvres en rouge
plt.scatter(x_coords[body_part_indices], -y_coords[body_part_indices], label='Lips', color='red')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter Plot of First Image Second Sequence')
plt.show()

# %% [markdown]
# **Data Augmentation Class**

# %%
class DataAugmentation:
    def __init__(self, dataset, noise_std=0.01, translation_range=0.1, erasing_probability_range=(0.1, 0.5), erasing_value=(0, 0), max_shift=5):
        self.noise_std = noise_std
        self.translation_range = translation_range
        self.erasing_probability_range = erasing_probability_range
        self.erasing_value = erasing_value
        self.max_shift = max_shift
        self.dataset = dataset
        
        # Define body part indices
        self.LIPS_IDXS0 = np.array([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ])
        self.LEFT_HAND_IDXS0 = np.arange(468, 489)
        self.RIGHT_HAND_IDXS0 = np.arange(522, 543)
        self.LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
        self.RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])

        self.REDUCED_LANDMARKS = np.sort(np.concatenate([self.LIPS_IDXS0, self.LEFT_HAND_IDXS0, self.RIGHT_HAND_IDXS0, self.LEFT_POSE_IDXS0, self.RIGHT_POSE_IDXS0]))
        self.index_map = {value: idx for idx, value in enumerate(self.REDUCED_LANDMARKS)}

        self.lips_indices_in_reduced = [self.index_map[idx] for idx in self.LIPS_IDXS0]
        self.left_hand_indices_in_reduced = [self.index_map[idx] for idx in self.LEFT_HAND_IDXS0]
        self.right_hand_indices_in_reduced = [self.index_map[idx] for idx in self.RIGHT_HAND_IDXS0]
        self.left_pose_indices_in_reduced = [self.index_map[idx] for idx in self.LEFT_POSE_IDXS0]
        self.right_pose_indices_in_reduced = [self.index_map[idx] for idx in self.RIGHT_POSE_IDXS0]

        self.BODY_PARTS = {
            'lips': self.lips_indices_in_reduced,
            'left_hand': self.left_hand_indices_in_reduced,
            'right_hand': self.right_hand_indices_in_reduced,
            'left_pose': self.left_pose_indices_in_reduced,
            'right_pose': self.right_pose_indices_in_reduced
        }

    def add_gaussian_noise(self, features):
        noise = torch.randn_like(features) * self.noise_std
        noisy_features = features + noise
        noisy_features = torch.clamp(noisy_features, 0, 1)  # Clamping to keep values within [0, 1]
        return noisy_features

    def translation(self, features):
        tr_x = random.uniform(-self.translation_range, self.translation_range)
        tr_y = random.uniform(-self.translation_range, self.translation_range)
        translated_features = features.clone()  # Cloning to avoid modifying the original features
        translated_features[:, 0] += tr_x
        translated_features[:, 1] += tr_y
        translated_features = torch.clamp(translated_features, 0, 1)  # Clamping to keep values within [0, 1]
        return translated_features

    def random_erasing(self, points):
        probability = random.uniform(*self.erasing_probability_range)
        if random.uniform(0, 1) > probability:
            return points
        points_copy = points.clone()
        nb_points = len(points_copy)
        nb_points_erased = int(nb_points * random.uniform(0.1, 0.5))
        erased_indices = random.sample(range(nb_points), nb_points_erased)
        for idx in erased_indices:
            points_copy[idx] = torch.tensor(self.erasing_value, dtype=points_copy.dtype)
        return points_copy

    def time_shift_images(self, sequence):
        shift = random.randint(-self.max_shift, self.max_shift)
        seq_len, num_points, num_features = sequence.shape
        shifted_sequence = torch.zeros_like(sequence)  
        for i in range(seq_len):
            if 0 <= i + shift < seq_len:
                shifted_sequence[i + shift] = sequence[i]
        return shifted_sequence

    def mixup_body_part_sequence(self, sequence1, sequence2, part_indices):
        # Obtenir les longueurs des séquences
        len1 = len(sequence1)
        len2 = len(sequence2)
        # Déterminer la longueur commune
        min_len = min(len1, len2)
        # Tronquer les séquences à la longueur commune
        sequence1 = sequence1[:min_len]
        sequence2 = sequence2[:min_len]
        # Cloner la seconde séquence pour éviter de la modifier directement
        mixed_sequence = sequence2.clone()
        # Appliquer le mixup pour chaque image dans la séquence
        for i in range(min_len):
            mixed_sequence[i][part_indices] = sequence1[i][part_indices]
        return mixed_sequence

    def get_random_body_part(self):
        body_part_name = random.choice(list(self.BODY_PARTS.keys()))
        body_part_indices = self.BODY_PARTS[body_part_name]
        return body_part_name, body_part_indices

    def get_sequences_by_target(self, target_value):
        sequences = [seq for seq in self.dataset if seq[1] == target_value]
        return sequences

    def scatter_plot(self, features, title='Scatter Plot', body_part_indices=None):
        first_image = features[0]
        x_coords = first_image[:, 0]
        y_coords = first_image[:, 1]
        plt.scatter(x_coords, -y_coords, label='Other Points', color='blue')
        if body_part_indices:
            plt.scatter(x_coords[body_part_indices], -y_coords[body_part_indices], label='Body Part', color='red')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title(title)
        plt.legend()
        plt.show()

    def augment_dataset(self, proportion=0.1, output_file='augmented_data.pkl'):
        num_samples = len(self.dataset)
        num_to_augment = int(proportion * num_samples)
        indices_to_augment = random.sample(range(num_samples), num_to_augment)
        new_sequences = []
        
        for idx in indices_to_augment:
            sequence, target = self.dataset[idx]
            augmentation_choice = random.choice([
                self.add_gaussian_noise,
                self.translation,
                self.random_erasing,
                self.time_shift_images,
                self.apply_mixup_to_random_body_part
            ])
            if augmentation_choice == self.apply_mixup_to_random_body_part:
                other_sequence, _ = random.choice(self.get_sequences_by_target(target))
                body_part_name, body_part_indices = self.get_random_body_part()
                augmented_sequence = augmentation_choice(sequence, other_sequence, body_part_indices)
            else:
                augmented_sequence = augmentation_choice(sequence)
            new_sequences.append((augmented_sequence, target))
        # Save the augmented data to a file
        with open(output_file, 'wb') as f:
            pickle.dump(new_sequences, f)
        print(f'Augmented data saved to {output_file}')
        
#         self.dataset.extend(new_sequences)  # Ajouter les nouvelles séquences augmentées au dataset

    def apply_mixup_to_random_body_part(self, sequence1, sequence2, part_indices):
        return self.mixup_body_part_sequence(sequence1, sequence2, part_indices)
    
    



# %%
# Exemple de séquence
sequences = augmentor.get_sequences_by_target(0)  # Récupérer des séquences avec la cible 0
sequence1 = sequences[0][0]  # Séquence de caractéristiques de la première séquence
sequence2 = sequences[1][0]  # Séquence de caractéristiques de la seconde séquence

# Appliquer plusieurs augmentations
augmentations = [augmentor.add_gaussian_noise, augmentor.translation, augmentor.random_erasing, augmentor.time_shift_images]
augmented_sequence = augmentor.apply_augmentations(sequence1, augmentations)

# Appliquer mixup à une partie du corps aléatoire
body_part_name, body_part_indices = augmentor.get_random_body_part()
mixed_sequence = augmentor.mixup_body_part_sequence(sequence1, sequence2, body_part_indices)


# %%
def scatter_plot(sequence, body_part_indices=None, title='Scatter Plot'):
    """
    Trace un scatter plot d'une séquence de points avec une partie spécifique du corps en couleur différente (optionnel).

    Args:
    - sequence (torch.Tensor): Une séquence de points à tracer.
    - body_part_indices (list, optional): Les indices de la partie spécifique du corps à colorer différemment.
    - title (str): Le titre du plot.
    """
    first_image = sequence[0]
    x_coords = first_image[:, 0]
    y_coords = first_image[:, 1]

    # Tracer les points
    plt.scatter(x_coords, -y_coords, label='Other Points', color='blue')
    
    # Tracer les points de la partie spécifique du corps en rouge, si fournis
    if body_part_indices is not None:
        plt.scatter(x_coords[body_part_indices], -y_coords[body_part_indices], label='Highlighted Points', color='red')
    
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title(title)
    plt.legend()
    plt.show()

# Utilisation de la fonction scatter_plot
first_image_mixed_sequence = mixed_sequence[0]


# Avec une partie du corps aléatoire
body_part_name, body_part_indices = augmentor.get_random_body_part()
scatter_plot(mixed_sequence, body_part_indices, title=f'Scatter Plot with {body_part_name}')

# %%
augmented_sequence

# %%
scatter_plot(sequence1)

# %%
scatter_plot(augmented_sequence)

# %%
with open(test_file, 'wb') as f:
    pickle.dump(new_sequences, f)
print(f'Augmented data saved to {output_file}')

# %%
augmentor = DataAugmentation(trainset)
# Appliquer l'augmentation à 10% du dataset
augmentor.augment_dataset(proportion=0.1, output_file='augmented_data.pkl')

# Visualiser une séquence augmentée
sequence, target = random.choice(trainset)
augmentor.scatter_plot(sequence, title='Scatter Plot of Augmented Sequence')

# %%
with open('/kaggle/input/augmented-data-from-islr/augmented_data.pkl', 'rb') as f:
    data_agmented_from_dataset = pickle.load(f)

# %%
len(data)

# %%
len(data_agmented_from_dataset)

# %%
augmented_dataset = data + data_agmented_from_dataset

# %%
len(augmented_dataset)

# %% [markdown]
# Dataloader

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

# %%
testset = ISLR(augmented_dataset, split='test')
trainvalset = ISLR(augmented_dataset, split='trainval')

# %%
len(augmented_dataset)

# %%
len(testset)

# %%
len(trainvalset)

# %%
trainset, valset = train_test_split(trainvalset,test_size=0.1, random_state=42)

# %%
len(trainset),len(valset)

# %%
len(testset)+len(trainset)+len(valset)

# %%
def custom_collate_fn(batch):
    padded_batch = []
    labels= []

    max_frame = max(len(sequence) for sequence,_ in batch)
#     print(max_frame)
    for sequence, label in batch:
        padding_array = -np.ones(((max_frame-len(sequence)), len(REDUCED_LANDMARKS), 2))
        padded_sequence = sequence.tolist()+padding_array.tolist()

        padded_batch.append(padded_sequence)
        labels.append(label)


    return torch.tensor(padded_batch), torch.tensor(labels)

# %%
train_loader = DataLoader(trainset, 
                          batch_size=8, 
                          collate_fn=custom_collate_fn, 
                          shuffle=True,
                          num_workers=4)

# %%
val_loader = DataLoader(valset,
                        batch_size=8,
                        collate_fn=custom_collate_fn,
                        shuffle=False,
                        num_workers=4)

# %%
test_loader = DataLoader(testset,
                         batch_size=8,
                         collate_fn=custom_collate_fn,
                         shuffle=False,
                         num_workers=4)

# %%
def select_random_sequence():
    usr = random.choice(user_ids)
    usr_sqc = os.listdir(os.path.join(dataset_path,usr))
    sqc = random.choice(usr_sqc)
    return os.path.join(dataset_path,usr,sqc)

# %%
user_ids = os.listdir('/kaggle/input/asl-signs/train_landmark_files')
select_random_sequence()

# %%
custom_it = enumerate(train_loader)

# %%
idx,(sqc,lb)=next(custom_it)
print(sqc.shape, lb)

# %% [markdown]
# # Model

# %%
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim, n_landmarks, max_seq_length=1000):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim*n_landmarks, hidden_dim) # change encoding 
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.positional_encoding = self.positional_encoding = self.create_positional_encoding(max_seq_length, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # activation softmax
        
        
    def forward(self, x):
        
        batch_size, n_frames, n_landmarks, input_dim = x.shape
        pad_mask = self.sequence_mask(x)
        pad_mask = pad_mask.to(device)
        
        
        # Flatten n_landmarks and input_dim for embedding
        x = x.view(batch_size, n_frames, -1)
        x = x.to(device)
        x = self.embedding(x)
        
        x = self.layer_norm1(x)
        x += self.positional_encoding[:, :n_frames, :].to(device)
        x = x.permute(1, 0, 2)  # Transformer expects sequence length first
                
        transformer_out = self.transformer_encoder(x,src_key_padding_mask=pad_mask)
        out = self.fc(transformer_out[-1, :, :])
        assert not torch.isnan(out).any(), "NaN in final output"
        
        
        return out
    
    def create_positional_encoding(self, max_seq_length, hidden_dim):
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        
        positional_encoding = torch.zeros(max_seq_length, hidden_dim)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return positional_encoding.unsqueeze(0)
    
    def sequence_mask(self, sequence):
        lengths = [self.valid_len(padded_sequence) for padded_sequence in sequence]
        
        mask = torch.zeros(sequence.size()[:2], dtype=torch.bool)  # shape: [batch_size, n_frames]
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        mask = ~mask # True values are ignored
        return mask

        
    def valid_len(self, padded_sequence):
        for idx, frame in  enumerate(padded_sequence):
            if -1 in frame:
                break

        return idx+1

# %% [markdown]
# Training phase

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
input_dim = 2  # (x, y)
num_heads = 4
num_layers = 2
hidden_dim = 64
output_dim = 250
n_landmarks = 92

model1 = TransformerModel(input_dim=input_dim,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         n_landmarks=n_landmarks)

model1 = model1.to(device)
print(model1)

# %%
o1 = model1(sqc)

# %%
loss_function = nn.CrossEntropyLoss()
optim1 = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)

# %%
num_epochs = 15

dataloader = train_loader

for epoch in range(num_epochs):

    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)
    
    model1.train()
    
    running_l1 = 0.0

    for sequence, label in dataloader:
        sequence, label = sequence.to(device), label.to(device)
        
        optim1.zero_grad()


        target = label
        
        out1 = model1(sequence)


        # Compute the loss, gradients, and update optimizer
        loss1 = loss_function(out1, target)

        loss1.backward()
        
        optim1.step()

    
        running_l1 += loss1.item()
        


    epoch_l1 = running_l1 / len(dataloader)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_l1:.4f}")

    
    # Validation step
    model1.eval()

    val_loss1 = 0.0
    correct_top5_mod1 = 0
    correct_mod1 = 0
    
    total = 0
    
    with torch.no_grad():
        for sequence, label in val_loader:
            sequence, label = sequence.to(device), label.to(device)
            
            out1 = model1(sequence)

            loss1 = loss_function(out1, label)

            val_loss1 += loss1.item() * sequence.size(0)

            
#           use top 5 pred
            _, pred1_top5 = torch.topk(out1, 5, dim=1)

            correct_top5_mod1 += (pred1_top5.to(device) == label.view(-1, 1)).sum().item()

#             top prediction
            _, pred1 = torch.max(out1, 1)

            correct_mod1 += (pred1 == label).sum().item()
            
            
            total += label.size(0)



    val_loss1 /= len(val_loader.dataset)

    val_acc1 = correct_mod1 /total

    val_acc1_top5 = correct_top5_mod1/total

    print(f'(model1) val loss: {val_loss1:.4f} Acc: {val_acc1:.4f} Top 5 Acc: {val_acc1_top5:.4f}')

    torch.save(model1.state_dict(), f'pointnet_transformer_model1_agmented_dataset_{epoch+1}.pth')


