# Dissertation
Repository containing the scripts which generate the training data as outlined in the methdology of the Dissertation.

# Pre-requisites:

Required packages to install:
NumPy

OpenCV: cv2

natsort


Add two empty folders: One folder called "EndoMapper" and the other folder called "UCLSyntheticDataset"

Download the EndoMapper data separately using the Synapse API from: https://doi.org/10.7303/syn26707219, this will require a Synapse account and submitting a request for access to the data. Since we will only be utilising the "Simulated Sequences" Folder the download can likely be terminated/cancelled early once this folder has downloaded. Place the "Simulated Sequences" folder inside the "EndoMapper" folder in this repository. 

Download the Depth From Colonoscopy data separately from http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/, this will require the password from the authors (email is listed at the bottom of the web page). Download the T1 and T2 zipped folders from Depth From Colonoscopy, unzip the folders and place the unzipped T1 and T2 folders in the "UCLSyntheticDataset" folder.


# Order to run the scripts:


1) In the "EndoMapper Scripts" folder, run: "main_pipeline_pre_augmentation.py". This will generate multiple new folders containing ground truth labels and image manipulations of the original synthetic data as outlined in the dissertation.  
2) In the "UCL Dataset Scripts" folder, run: "UCL_gt_extraction&_reformattiong.py". This will generate another folder which reformats the dataset and also includes the ground truth labels.
3) Back in the "root" directory (I.e. the repo directory) run: "data_preparation_script.py". This will sample the frames to generate the foundational datasets outlined in the dissertation, in addition to a folder which contains pkl files containing the lists of frames that were sampled from each sequence for each dataset. 
4) In the "EndoMapper Scripts" folder, run: "endomapper_augmentation_main.py". This and "endomapper_data_augmentation" are both adapted from https://github.com/xavysp/MBIPED. This will apply the data augmentations to the datasets sampled from EndoMapper, this takes quite a while to run.
5) In the "UCL Dataset Scripts" folder, run "ucl_augmentation_main.py". This and "ucl_data_augmentation" are both adapted from https://github.com/xavysp/MBIPED. This will apply the data augmentations to the dataset sampled from Depth From Colonoscopy, this takes quite a while to run.
6) Back in the "root" directory (the repo directory) run: "dataset_combination_pipeline.py". This will generate combined datasets outlined in the dissertation.

# NOTE:

There will be a couple of folders generated by deprecated code, which are datasets which were not used in the experiments in the dissertation. 

The final datasets which were used for training are the following, with their corresponding descriptions:

UCL Selected Data: This is evidently Dataset 1 in the Dissertation, with frames sampled only from Depth From Colonoscopy.

EndoMapper Selected Data Inpainted Only: This is Dataset 2 in the Dissertation, with frames sampled only from the version of Endomapper with only inpainting applied.

EndoMapper Selected Dataset Rescaled Specularity Only: This is Dataset 3 in the Dissertation, with frames sampled only from the version of EndoMapper with specularity retuning applied.

Combined Selected Data v2: This is Dataset 4 in the Dissertation, which combines Datasets 1 and 2.
Combined Selected Data v3: This is Dataset 5 in the Dissertation, which combines Datasets 1 and 3.


Deprecated datasets: Combined Selected Data and EndoMapper Selected Data. 
