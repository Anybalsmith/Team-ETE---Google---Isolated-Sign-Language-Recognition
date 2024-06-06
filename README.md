### **This project is inspired of this kaggle challenge:**
https://www.kaggle.com/competitions/asl-signs

### **Description**

**Goal of the Competition**
The goal of this competition is to classify isolated American Sign Language (ASL) signs. You will create a TensorFlow Lite model trained on labeled landmark data extracted using the MediaPipe Holistic Solution.

Your work may improve the ability of PopSign* to help relatives of deaf children learn basic signs and communicate better with their loved ones.

As we are more familiar with PyTorch, we realized this challenge using this frameworks instead of using TensorFlow. 

### **Data**

- _train_landmark_files/[participant_id]/[sequence_id].parquet_ The landmark data. The landmarks were extracted from raw videos with the MediaPipe holistic model. Not all of the frames necessarily had visible hands or hands that could be detected by the model.

- Landmark data should not be used to identify or re-identify an individual. Landmark data is not intended to enable any form of identity recognition or store any unique biometric identification.

- _frame_ - The frame number in the raw video.
- _row_id_ - A unique identifier for the row.
- _type_ - The type of landmark. One of ['face', 'left_hand', 'pose', 'right_hand'].
- _landmark_index_ - The landmark index number. Details of the hand landmark locations can be found here.
- _[x/y/z]_ - The normalized spatial coordinates of the landmark. These are the only columns that will be provided to your submitted model for inference. The MediaPipe model is not fully trained to predict depth so you may wish to ignore the z values.

**train.csv**

- path - The path to the landmark file.
- participant_id - A unique identifier for the data contributor.
- sequence_id - A unique identifier for the landmark sequence.
- sign - The label for the landmark sequence.
