**17/05**:
- analyzed dataset
- distribution of classes and types
- transformers ? (sort of time series)

TODO
- filter face points (keep mouth and eyes as a priority)
- remove z values
- use sequential architectures (start with a simple model)
- look for ideas on Kaggle 
- try to plot frames (using scatter plots)

**researchs until next meetings**:
- CoAtNet: combination of CNN and Transformer
- https://www.youtube.com/watch?v=VoRQiKQcdcI
- 

**28/05**

DONE
- preprocessed data (remove NaN, normalization, filter landmarks)
- removed z coordinates
- research on architectures (among top participants and on the internet)

QUESTIONS
- data augmentation ? (rotations, zoom)
- is normalization useful ?

TODO
Use attention mask

- check points range
- create batch, by padding to max length sequence, but ignore padded frames in loss calculation = mask
- rotation may not be useful, or only on a small range

**31/05**
  Combinations of parts - for example we took a hand from one sample and lips from another sample with the same label. This augmentation gave a significant increase in score (idea proposed in 3rd place solution) 
  Besides doing spatial augmentation such as adding noise, we can think about doing temporal augmentation (cf 2nd place solution)

DONE
- dataset with only (x,y) coordinates of most relevant landmarks (92 points) and normalized frames 
- dataloader using padding
- transformer using mask to ignore padded frames

**07/06** 

QUESTIONS
- transformer adapted ? (first epochs very bad accuracy)
- way to fine tune ? instead of training an entire model
- proportion of data augmentation

Start without augmentation
Then augmente a little bit, no general rule
Vision transformer, point net, combination between time sequence and encode our point cloud. Activation softmax

**14/06**

DONE
- implement different model (CNN1D, tranformer, pointnet+transformer)
- trained transformer, validation accuracy increase a lot using top 5 predictions for correct prediction
- trained only on 4 epochs and got 0.96 as validation accuracy

QUESTION
- top 5 maybe too large ?
- way to train faster ? (currently it takes around 1 hour per epochs)

smaller model same data
model with smaller data
smaller batch size current 32-> 8 or 16
retenter model avec top 1 puisque le model apprend

report July 12th
presentation 28th June


