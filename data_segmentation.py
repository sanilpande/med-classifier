import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle

# filenames should be segmented class-wise into .txt files
data_dir = "seg"
pos_files = os.path.join(data_dir, "pos.txt")
fp_files = os.path.join(data_dir, "fp.txt")
fn_files = os.path.join(data_dir, "fn.txt")

# Obtain list with filenames and size of the lists
with open(pos_files) as pos:
    pos_list = [line.strip() for line in pos]
pos_size = len(pos_list)

with open(fp_files) as fp:
    fp_list = [line.strip() for line in fp]
fp_size = len(fp_list)

with open(fn_files) as fn:
    fn_list = [line.strip() for line in fn]
fn_size = len(fn_list)

# create one list for filenames and another for labels
data = pos_list + fp_list + fn_list
labels = np.zeros(len(data))
labels[pos_size:pos_size+fp_size] = 1
labels[pos_size+fp_size:] = 2

# create random train-val-test splits and save
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

with open("seg/train_data.txt", "wb") as fp:
    pickle.dump(X_train, fp)
with open("seg/train_labels.txt", "wb") as fp:
    pickle.dump(y_train, fp)

with open("seg/val_data.txt", "wb") as fp:
    pickle.dump(X_val, fp)
with open("seg/val_labels.txt", "wb") as fp:
    pickle.dump(y_val, fp)

with open("seg/test_data.txt", "wb") as fp:
    pickle.dump(X_test, fp)
with open("seg/test_labels.txt", "wb") as fp:
    pickle.dump(y_test, fp)
