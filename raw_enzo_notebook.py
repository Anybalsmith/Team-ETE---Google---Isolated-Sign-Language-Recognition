# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T16:56:59.554309Z","iopub.execute_input":"2024-05-16T16:56:59.554889Z","iopub.status.idle":"2024-05-16T16:59:15.753656Z","shell.execute_reply.started":"2024-05-16T16:56:59.554859Z","shell.execute_reply":"2024-05-16T16:59:15.752482Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir('/kaggle/input'))

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     print(dirname)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:00:49.403358Z","iopub.execute_input":"2024-05-16T17:00:49.404524Z","iopub.status.idle":"2024-05-16T17:00:49.409669Z","shell.execute_reply.started":"2024-05-16T17:00:49.404481Z","shell.execute_reply":"2024-05-16T17:00:49.408711Z"}}
dataset_path = '/kaggle/input/asl-signs/train_landmark_files'
user_ids = os.listdir('/kaggle/input/asl-signs/train_landmark_files')

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:23.201378Z","iopub.execute_input":"2024-05-16T13:13:23.201788Z","iopub.status.idle":"2024-05-16T13:13:23.208335Z","shell.execute_reply.started":"2024-05-16T13:13:23.201741Z","shell.execute_reply":"2024-05-16T13:13:23.206884Z"}}
ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:12:56.420676Z","iopub.execute_input":"2024-05-16T13:12:56.420997Z","iopub.status.idle":"2024-05-16T13:12:56.425314Z","shell.execute_reply.started":"2024-05-16T13:12:56.420970Z","shell.execute_reply":"2024-05-16T13:12:56.424342Z"}}
test_path = '/kaggle/input/asl-signs/train_landmark_files/36257'

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2024-05-16T13:12:59.037623Z","iopub.execute_input":"2024-05-16T13:12:59.037994Z","iopub.status.idle":"2024-05-16T13:12:59.046677Z","shell.execute_reply.started":"2024-05-16T13:12:59.037967Z","shell.execute_reply":"2024-05-16T13:12:59.045369Z"}}
test_filenames = os.listdir(test_path)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:14.623465Z","iopub.execute_input":"2024-05-16T13:13:14.623820Z","iopub.status.idle":"2024-05-16T13:13:14.629568Z","shell.execute_reply.started":"2024-05-16T13:13:14.623794Z","shell.execute_reply":"2024-05-16T13:13:14.627673Z"}}
parquet_path = os.path.join(test_path, test_filenames[1])
print(parquet_path)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:25.543118Z","iopub.execute_input":"2024-05-16T13:13:25.543474Z","iopub.status.idle":"2024-05-16T13:13:25.734037Z","shell.execute_reply.started":"2024-05-16T13:13:25.543442Z","shell.execute_reply":"2024-05-16T13:13:25.733113Z"}}
mytest0 = load_relevant_data_subset(parquet_path)
mytest0.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:27.904630Z","iopub.execute_input":"2024-05-16T13:13:27.904962Z","iopub.status.idle":"2024-05-16T13:13:27.912785Z","shell.execute_reply.started":"2024-05-16T13:13:27.904933Z","shell.execute_reply":"2024-05-16T13:13:27.911139Z"}}
type(mytest0)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:28.243773Z","iopub.execute_input":"2024-05-16T13:13:28.244122Z","iopub.status.idle":"2024-05-16T13:13:28.253172Z","shell.execute_reply.started":"2024-05-16T13:13:28.244093Z","shell.execute_reply":"2024-05-16T13:13:28.251657Z"}}
mytest0[0]

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:36:27.481934Z","iopub.execute_input":"2024-05-16T15:36:27.482467Z","iopub.status.idle":"2024-05-16T15:36:27.518348Z","shell.execute_reply.started":"2024-05-16T15:36:27.482428Z","shell.execute_reply":"2024-05-16T15:36:27.517369Z"}}
parquet_path = os.path.join(test_path, test_filenames[0])
cols = ['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']
parquet_df = pd.read_parquet(parquet_path, columns=cols)
parquet_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:36:39.876836Z","iopub.execute_input":"2024-05-16T15:36:39.877401Z","iopub.status.idle":"2024-05-16T15:36:39.889422Z","shell.execute_reply.started":"2024-05-16T15:36:39.877359Z","shell.execute_reply":"2024-05-16T15:36:39.886884Z"}}
len(parquet_df)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T14:55:40.122023Z","iopub.execute_input":"2024-05-16T14:55:40.122427Z","iopub.status.idle":"2024-05-16T14:55:40.135453Z","shell.execute_reply.started":"2024-05-16T14:55:40.122396Z","shell.execute_reply":"2024-05-16T14:55:40.133754Z"}}
parquet_df[parquet_df.frame ==5].type.value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:37:07.478816Z","iopub.execute_input":"2024-05-16T15:37:07.479163Z","iopub.status.idle":"2024-05-16T15:37:07.488151Z","shell.execute_reply.started":"2024-05-16T15:37:07.479136Z","shell.execute_reply":"2024-05-16T15:37:07.486494Z"}}
parquet_df.frame.value_counts().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:35.119685Z","iopub.execute_input":"2024-05-16T13:13:35.120026Z","iopub.status.idle":"2024-05-16T13:13:35.331906Z","shell.execute_reply.started":"2024-05-16T13:13:35.119996Z","shell.execute_reply":"2024-05-16T13:13:35.330999Z"}}
train_path = '/kaggle/input/asl-signs/train.csv'
train = pd.read_csv(train_path)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:35.615863Z","iopub.execute_input":"2024-05-16T13:13:35.616628Z","iopub.status.idle":"2024-05-16T13:13:35.624455Z","shell.execute_reply.started":"2024-05-16T13:13:35.616584Z","shell.execute_reply":"2024-05-16T13:13:35.623148Z"}}
train.columns

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:36.111858Z","iopub.execute_input":"2024-05-16T13:13:36.112438Z","iopub.status.idle":"2024-05-16T13:13:36.126012Z","shell.execute_reply.started":"2024-05-16T13:13:36.112408Z","shell.execute_reply":"2024-05-16T13:13:36.125098Z"}}
train.sign.unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:39.410823Z","iopub.execute_input":"2024-05-16T13:13:39.411203Z","iopub.status.idle":"2024-05-16T13:13:39.418188Z","shell.execute_reply.started":"2024-05-16T13:13:39.411173Z","shell.execute_reply":"2024-05-16T13:13:39.417090Z"}}
len(os.listdir('/kaggle/input/asl-signs/train_landmark_files'))

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:41.081892Z","iopub.execute_input":"2024-05-16T13:13:41.082233Z","iopub.status.idle":"2024-05-16T13:13:41.093979Z","shell.execute_reply.started":"2024-05-16T13:13:41.082204Z","shell.execute_reply":"2024-05-16T13:13:41.092964Z"}}
train.participant_id.unique(), len(train.participant_id.unique())

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:13:52.371687Z","iopub.execute_input":"2024-05-16T13:13:52.372061Z","iopub.status.idle":"2024-05-16T13:13:52.386892Z","shell.execute_reply.started":"2024-05-16T13:13:52.372030Z","shell.execute_reply":"2024-05-16T13:13:52.385709Z"}}
train[train.participant_id == 16069]

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:14:00.529690Z","iopub.execute_input":"2024-05-16T13:14:00.530006Z","iopub.status.idle":"2024-05-16T13:14:00.567024Z","shell.execute_reply.started":"2024-05-16T13:14:00.529979Z","shell.execute_reply":"2024-05-16T13:14:00.565592Z"}}
d=dict(train.sign.value_counts(dropna=True))
print(train.sign.value_counts(dropna=True).mean())
print(train.sign.value_counts(dropna=True).std())
print(train.sign.value_counts(dropna=True).max())
print(train.sign.value_counts(dropna=True).min())

# word distribution is not too expended
# any words have close occurences 

# %% [markdown]
# #### **Some notes:**
# * each parquet contains markers position [x y z] and type (face, left_hand, pose, right_hand) for different frame
# * train dataset is composed of image path, participant id (folder name of parquet file) sequence id (filename) and word said
# * one sequence = numerous frames = 1 word
# * every frame has data for each type, but it is possible that one type has no value in a frame, it is setted to NaN
# 
# **Goal**: using hand position, be able to understand word said in the sequence
# * classification between 250 words using positions of body parts in video

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:14:18.073454Z","iopub.execute_input":"2024-05-16T13:14:18.073806Z","iopub.status.idle":"2024-05-16T13:14:18.091034Z","shell.execute_reply.started":"2024-05-16T13:14:18.073778Z","shell.execute_reply":"2024-05-16T13:14:18.089539Z"}}
import json
 
# Opening JSON file
f = open('/kaggle/input/asl-signs/sign_to_prediction_index_map.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)
print(len(data), data)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:14:20.734622Z","iopub.execute_input":"2024-05-16T13:14:20.736169Z","iopub.status.idle":"2024-05-16T13:14:20.749046Z","shell.execute_reply.started":"2024-05-16T13:14:20.736117Z","shell.execute_reply":"2024-05-16T13:14:20.747206Z"}}
train_words = train.sign.unique()
print(len(train_words))
# same length as sign to prediction index json 

# %% [code]


# %% [markdown]
# #### **Analysis Ideas**
# 
# * class embalencement (count words for each element in train dataset)
# * size analysis (lenght of sequence, linked to words ?)
# * position ranges (x y z)
# * number of sequence per participant 
# * train dataset will be splitted for train test val
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:14:25.368130Z","iopub.execute_input":"2024-05-16T13:14:25.368455Z","iopub.status.idle":"2024-05-16T13:14:27.011205Z","shell.execute_reply.started":"2024-05-16T13:14:25.368427Z","shell.execute_reply":"2024-05-16T13:14:27.009957Z"}}
landmark_folders = '/kaggle/input/asl-signs/train_landmark_files'

squences_per_user = {}

user_ids= os.listdir(landmark_folders)

for user_foler in user_ids:
    squences_per_user[user_foler] = len(os.listdir(os.path.join(landmark_folders,user_foler)))
    print(len(os.listdir(os.path.join(landmark_folders,user_foler))))
    

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:14:27.013532Z","iopub.execute_input":"2024-05-16T13:14:27.013988Z","iopub.status.idle":"2024-05-16T13:14:27.020718Z","shell.execute_reply.started":"2024-05-16T13:14:27.013915Z","shell.execute_reply":"2024-05-16T13:14:27.019461Z"}}
print(f'max number of sequence: {np.max(list(squences_per_user.values()))}\nmin number of sequence: {np.min(list(squences_per_user.values()))}\nmean of number of sequence per user: {int(np.mean(list(squences_per_user.values())))}\nstandard deviation: {int(np.std(list(squences_per_user.values())))}')

# %% [markdown]
# #### **Type analysis**

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:50:11.832551Z","iopub.execute_input":"2024-05-16T15:50:11.833159Z","iopub.status.idle":"2024-05-16T16:38:24.775901Z","shell.execute_reply.started":"2024-05-16T15:50:11.833116Z","shell.execute_reply":"2024-05-16T16:38:24.774136Z"}}
my_df = pd.DataFrame(index=user_ids,columns=['face', 'pose','l_hand', 'r_hand'])

my_df.reset_index(inplace=True)
display(my_df)
cols = ['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']
# types_dic = {'face': 0, 'pose': 0, 'left_hand': 0, 'right_hand': 0}
for n,user in enumerate(user_ids):
    len_seq = 0
    user_dic = {'face': 0, 'pose': 0, 'left_hand': 0, 'right_hand': 0}
    sequence_files = os.listdir(os.path.join(dataset_path,user))
    for idx,sequence in enumerate(sequence_files):
        parquet_path = os.path.join(dataset_path, user, sequence)
        parquet_df = pd.read_parquet(parquet_path, columns=cols)
        len_seq+=len(parquet_df)
        
        sequence_dic = dict(parquet_df.type.value_counts())
        
        for key in user_dic.keys():
            user_dic[key]+=sequence_dic[key]
        
        
    print(user_dic, len_seq)
    my_df.iloc[n, 1:]  = [el/len_seq for el in list(user_dic.values())]


            
        

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T16:40:03.132450Z","iopub.execute_input":"2024-05-16T16:40:03.132918Z","iopub.status.idle":"2024-05-16T16:40:03.453360Z","shell.execute_reply.started":"2024-05-16T16:40:03.132881Z","shell.execute_reply":"2024-05-16T16:40:03.451997Z"}}


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T16:43:40.223553Z","iopub.execute_input":"2024-05-16T16:43:40.223894Z","iopub.status.idle":"2024-05-16T16:43:40.231193Z","shell.execute_reply.started":"2024-05-16T16:43:40.223872Z","shell.execute_reply":"2024-05-16T16:43:40.229908Z"}}
my_df.columns

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T16:44:40.308817Z","iopub.execute_input":"2024-05-16T16:44:40.309231Z","iopub.status.idle":"2024-05-16T16:44:40.316942Z","shell.execute_reply.started":"2024-05-16T16:44:40.309197Z","shell.execute_reply":"2024-05-16T16:44:40.315971Z"}}
my_df.to_csv('/kaggle/working/types_distribution.csv', columns=my_df.columns, header=True, index=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T16:44:40.645955Z","iopub.execute_input":"2024-05-16T16:44:40.646336Z","iopub.status.idle":"2024-05-16T16:44:40.665746Z","shell.execute_reply.started":"2024-05-16T16:44:40.646308Z","shell.execute_reply":"2024-05-16T16:44:40.664013Z"}}
pd.read_csv('/kaggle/working/types_distribution.csv')

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:48:53.178177Z","iopub.execute_input":"2024-05-16T15:48:53.178691Z","iopub.status.idle":"2024-05-16T15:48:53.191530Z","shell.execute_reply.started":"2024-05-16T15:48:53.178653Z","shell.execute_reply":"2024-05-16T15:48:53.188933Z"}}
a=2699424
b=(60158592+4241952+2699424+2699424)

a/b

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:49:45.982095Z","iopub.execute_input":"2024-05-16T15:49:45.982695Z","iopub.status.idle":"2024-05-16T15:49:45.990323Z","shell.execute_reply.started":"2024-05-16T15:49:45.982645Z","shell.execute_reply":"2024-05-16T15:49:45.988679Z"}}
c = 1318779
d = 29389932+2072367+1318779+1318779
c/d


# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:34:29.657690Z","iopub.execute_input":"2024-05-16T15:34:29.658269Z","iopub.status.idle":"2024-05-16T15:34:29.679195Z","shell.execute_reply.started":"2024-05-16T15:34:29.658226Z","shell.execute_reply":"2024-05-16T15:34:29.676492Z"}}
my_df.iloc[0, 1:]  = list(user_dic.values())
my_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:06:07.863791Z","iopub.execute_input":"2024-05-16T15:06:07.864227Z","iopub.status.idle":"2024-05-16T15:06:07.870697Z","shell.execute_reply.started":"2024-05-16T15:06:07.864197Z","shell.execute_reply":"2024-05-16T15:06:07.869308Z"}}
len(parquet_df)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T15:18:25.573618Z","iopub.execute_input":"2024-05-16T15:18:25.574000Z","iopub.status.idle":"2024-05-16T15:18:25.584995Z","shell.execute_reply.started":"2024-05-16T15:18:25.573959Z","shell.execute_reply":"2024-05-16T15:18:25.584233Z"}}
u=pd.DataFrame(index=[1,2],columns=['a', 'b'])
u.head()
u
u.head()


# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:01:11.176359Z","iopub.execute_input":"2024-05-16T17:01:11.176691Z","iopub.status.idle":"2024-05-16T17:01:11.206079Z","shell.execute_reply.started":"2024-05-16T17:01:11.176666Z","shell.execute_reply":"2024-05-16T17:01:11.205025Z"}}
parquet_path = os.path.join(dataset_path, user_ids[0],test_filenames[1]) # only first sequence of user here
cols = ['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']
parquet_df = pd.read_parquet(parquet_path, columns=cols)
# type distribution for 1st seuence
dd= dict(parquet_df.type.value_counts())

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:25:38.017029Z","iopub.execute_input":"2024-05-16T13:25:38.017417Z","iopub.status.idle":"2024-05-16T13:25:38.034740Z","shell.execute_reply.started":"2024-05-16T13:25:38.017389Z","shell.execute_reply":"2024-05-16T13:25:38.033480Z"}}
# word signed in previsous sequence 
train.loc[train.path == parquet_path[24:]].sign

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:23:32.555322Z","iopub.execute_input":"2024-05-16T13:23:32.555700Z","iopub.status.idle":"2024-05-16T13:23:32.563173Z","shell.execute_reply.started":"2024-05-16T13:23:32.555669Z","shell.execute_reply":"2024-05-16T13:23:32.561662Z"}}
parquet_path[24:]

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T13:51:13.364577Z","iopub.execute_input":"2024-05-16T13:51:13.364937Z","iopub.status.idle":"2024-05-16T13:51:13.371111Z","shell.execute_reply.started":"2024-05-16T13:51:13.364907Z","shell.execute_reply":"2024-05-16T13:51:13.369482Z"}}
types_dic = {'face': 0, 'pose': 0, 'left_hand': 0, 'right_hand': 0}
for key in types_dic.keys():
     print(dd[key])

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:01:23.895705Z","iopub.execute_input":"2024-05-16T17:01:23.896391Z","iopub.status.idle":"2024-05-16T17:01:24.065039Z","shell.execute_reply.started":"2024-05-16T17:01:23.896362Z","shell.execute_reply":"2024-05-16T17:01:24.064290Z"}}
cols = ['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']
sequence_files = os.listdir(os.path.join(dataset_path,user_ids[0]))
parquet_path = os.path.join(dataset_path, user_ids[0], sequence_files[0])
parquet_df = pd.read_parquet(parquet_path, columns=cols)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:05:59.069308Z","iopub.execute_input":"2024-05-16T17:05:59.069670Z","iopub.status.idle":"2024-05-16T17:05:59.080009Z","shell.execute_reply.started":"2024-05-16T17:05:59.069641Z","shell.execute_reply":"2024-05-16T17:05:59.079054Z"}}
parquet_df.row_id.unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:05:17.768658Z","iopub.execute_input":"2024-05-16T17:05:17.769841Z","iopub.status.idle":"2024-05-16T17:05:17.792499Z","shell.execute_reply.started":"2024-05-16T17:05:17.769802Z","shell.execute_reply":"2024-05-16T17:05:17.791395Z"}}
parquet_df.groupby(['type']).head()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:07:52.528048Z","iopub.execute_input":"2024-05-16T17:07:52.529294Z","iopub.status.idle":"2024-05-16T17:07:52.553646Z","shell.execute_reply.started":"2024-05-16T17:07:52.529208Z","shell.execute_reply":"2024-05-16T17:07:52.552663Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:21:49.417057Z","iopub.execute_input":"2024-05-16T17:21:49.417833Z","iopub.status.idle":"2024-05-16T17:21:49.422387Z","shell.execute_reply.started":"2024-05-16T17:21:49.417801Z","shell.execute_reply":"2024-05-16T17:21:49.421244Z"}}
user_enum = enumerate(user_ids)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:22:24.602348Z","iopub.execute_input":"2024-05-16T17:22:24.603145Z","iopub.status.idle":"2024-05-16T17:22:24.644623Z","shell.execute_reply.started":"2024-05-16T17:22:24.603107Z","shell.execute_reply":"2024-05-16T17:22:24.643548Z"}}
_,user = next(user_enum)
sequence_files = os.listdir(os.path.join(dataset_path,user))
parquet_path = os.path.join(dataset_path, user, sequence_files[0])
parquet_df = pd.read_parquet(parquet_path, columns=cols)

type_mean = parquet_df.groupby(['type'])[['x', 'y', 'z']].mean().reset_index(drop=False)
type_mean.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-16T17:20:29.318953Z","iopub.execute_input":"2024-05-16T17:20:29.319837Z","iopub.status.idle":"2024-05-16T17:20:29.325803Z","shell.execute_reply.started":"2024-05-16T17:20:29.319802Z","shell.execute_reply":"2024-05-16T17:20:29.324722Z"}}
next(enumerate(user_ids))

# %% [code]
