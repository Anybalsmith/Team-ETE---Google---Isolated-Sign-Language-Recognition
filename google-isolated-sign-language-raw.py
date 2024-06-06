# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

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
        
        if split == 'train':
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

# %%
trainset = ISLR(data, split='train')
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
    tr_x = random.uniform(-0.1,0.1)
    tr_y = random.uniform(-0.1,0.1)
    features[:,0] += tr_x
    features[:,1] += tr_y
    return features


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
def mixup(image1, label1, image2, label2, alpha=0.2):

    
    # Sample lambda from the beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Create the mixed image
    mixed_image = lam * image1 + (1 - lam) * image2
    
    # Create the mixed label
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_image, mixed_label

# %%
class AugmentationPipeline:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, label=None):
        for aug in self.augmentations:
            if label is not None:
                image, label = aug(image, label)
            else:
                image = aug(image)
        if label is not None:
            return image, label
        return image

# Exemple d'utilisation des fonctions
def apply_augmentations_to_dataset(images, labels):
    # Définir les augmentations
    augmentations = [
        lambda img, lbl: (random_erasing(img), lbl),
        lambda img, lbl: (time_shift(img), lbl),
        lambda img, lbl: mixup_images(img, random.choice(images), lbl, random.choice(labels))
    ]

    pipeline = AugmentationPipeline(augmentations)
    augmented_images = []
    augmented_labels = []

    for img, lbl in zip(images, labels):
        aug_img, aug_lbl = pipeline(img, lbl)
        augmented_images.append(aug_img)
        augmented_labels.append(aug_lbl)

    return torch.stack(augmented_images), torch.stack(augmented_labels)


