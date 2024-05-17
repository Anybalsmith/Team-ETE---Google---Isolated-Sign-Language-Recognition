# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyarrow.parquet as pq

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
# 

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
max_frame_list = []
for keys,file_paths in file_paths_dict.items() :
    for file_path in file_paths:
        table = pq.read_table(file_path)
        table = table.to_pandas()
        max_frame_value = table['frame'].max()
        max_frame_list.append(max_frame_value)

# %%
output_file = "max_frame_list.txt"

with open(output_file, 'w') as file:
    for value in max_frame_list:
        file.write(str(value) + '\n')

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


