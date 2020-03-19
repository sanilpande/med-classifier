This is an experiment for developing and training a classifier.

The task is to classify CT scans as positive, false-positive, or false-negative.
The data is present in the NIfTI (.nii) file format.

data_segmentation.py is used to obtain train-val-test splits.

med_dataset.py defined the data_loader, and includes data augmentation.

models.py stores the models which were used.

The self defined model is inspired by the following literature:
https://www2.informatik.uni-hamburg.de/wtm/publications/2018/NKMMW18/Ng_et_al_3DCNN_2018.pdf
https://www.researchgate.net/publication/331018114_Evaluation_of_Functional_Decline_in_Alzheimer's_Dementia_Using_3D_Deep_Learning_and_Group_ICA_for_rs-fMRI_Measurements/fulltext/5c617e75299bf1d14cbcb195/Evaluation-of-Functional-Decline-in-Alzheimers-Dementia-Using-3D-Deep-Learning-and-Group-ICA-for-rs-fMRI-Measurements.pdf

res3d.py has this 3D implementation of ResNet architectures:
https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py

test.py contains a script for obtaining the test and train accuracies.

main.py contains the code for selecting and training the models.
