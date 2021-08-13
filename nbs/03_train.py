# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Notebook settings

# %%
# %load_ext autoreload
# %autoreload 2

# # %autosave 10

# #%load_ext lab_black

# %%
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..")))

# %% [markdown]
# ## Imports

# %%
from lfp_analysis.data import *
from lfp_analysis.process import *

from lfp_analysis.resnet1d import *
from lfp_analysis.svm import *

from fastai.vision import *
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor

# %%
import numpy as np
import pandas as pd
import h5py

from pathlib import Path

import matplotlib.pyplot as plt

# %matplotlib widget

# %%
ll = list(range(10))
ll

# %%
x = np.arange(51)
x

# %%
frame(x,5,5)

# %%
import math
def frame(array,win_len, hop_len, pad = True):
    nrows = ((array.size-win_len)//hop_len)+1
    
    if pad:
        n_reps = math.ceil(1 - (array.size-win_len)/hop_len % 1)
        print(n_reps)
    
    nrows = ((array.size-win_len)//hop_len)+1
    print(nrows)
    n = array.strides[0]
    return np.lib.stride_tricks.as_strided(array, shape=(nrows,win_len), strides=(hop_len*n,n))


# %%
frame(x,5,1)

# %%
sig

# %%

# %%
pad = True
array = np.arange(50)
win_len = 5
hop_len = 2
    
array1 = array
n_reps = ((1 - (array1.size-win_len)/hop_len % 1) * win_len)
print(n_reps)
array2 = np.concatenate((array,np.repeat(array[-1], n_reps)),0)

nrows = ((array2.size-win_len)//hop_len)+1
n = array.strides[0]
np.lib.stride_tricks.as_strided(array2, shape=(nrows,win_len), strides=(hop_len*n,n))

# %%
array2

# %%
(array.size-win_len)/hop_len % 1

# %%

# %%

# %%
1 - 2.25 % 1

# %%
import math
math.ceil(0.75 * 5)

# %%
frame(x,5,4)

# %%

# %% [markdown]
# # Import data:

# %%
# DATA_PATH = Path("./../data")
# PAT_ID = Path("ET2")
# DATASET_NAME = Path("Pegboard_off")

# LFP_SOURCE = DATA_PATH / "preprocessed" / PAT_ID / DATASET_NAME.with_suffix(".h5")
# H5_SOURCE = DATA_PATH / "TF_and_df" / PAT_ID / DATASET_NAME.with_suffix(".h5")
# CSV_SOURCE = DATA_PATH / "TF_and_df" / PAT_ID / DATASET_NAME.with_suffix(".csv")

dataset1 = Patient(PatID.ET2).datasets[Task.Pegboard][Stim.OFF].load_1d()
dataset2 = Patient(PatID.ET2).datasets[Task.Pegboard][Stim.ON].load_1d()

# %%
dataset1.LFP.fs

# %%
dataset1.concatenate(dataset2)

# %%

# %% [markdown]
# ### Read TF

# %%
h5_LFP = h5py.File(LFP_SOURCE, "r")

LFP = h5_LFP["LFP"][:]
label = h5_LFP["label"][:]

# %%
# LFP = (LFP - np.mean(LFP,-1,keepdims=True)) / np.std(LFP,-1,keepdims=True)

# %%
LFP[0].std()

# %%
h5 = h5py.File(H5_SOURCE, "r")
TF = h5["TF"][:]
# h5.close()

# %%
TF.shape

# %% [markdown]
# ### Read df_data

# %%
df_data = make_label_df(label, 0.500)
df_data

# %%
LFP.shape

# %%
train_end = df_data[df_data["is_valid"] == 0]["id_end"].iloc[-1]


def get_norm_stats(LFP, train_end):
    return np.mean(LFP[:, :train_end], axis=-1), np.std(LFP[:, :train_end], axis=-1)


def norm_with_stats(LFP, stats):
    means, stds = stats[0], stats[1]
    return (LFP - means[:, np.newaxis]) / stds[:, np.newaxis]


# %%
stats = get_norm_stats(LFP, train_end)
LFP = norm_with_stats(LFP, stats)

# %%
fig, ax = plt.subplots()
ax.plot(LFP[0])


# %%
means, stds = torch.tensor(xxx[0]), torch.tensor(xxx[1])

# %%
means

# %%
X = torch.tensor(LFP)
X.shape

# %%
(X - means[:, None]).shape

# %%
df_data["label"].value_counts(normalize=True)

# %%
df_data["label"].value_counts()

# %%
df_data

# %% [markdown]
# # Baseline Classifier:

# %%
{"a": [33, 33], "b": [22, 22]}

# %%
bl_test = BLClassifier(LFP, df_data, extract_method="hilbert").get_feat_mats()
bl_test.X_train

# %%
bl_cls = BLClassifier(LFP, df_data, extract_method="periodogram")
bl_cls.classify(method="SVM")

# %%
np.argmin(np.abs(3 - np.array([0, 2, 4, 6, 8])))

# %%
list(range(1))

# %%
out = bl_cls.classify_many(method="LDA")

# %%
out[1]

# %%
pd.DataFrame(out[0])

# %%
out[0][0]

# %%

# %%
bl_y_pred = bl_cls.y_pred.astype(float)

# %%
bl_cls.classify_lda()

# %%
bl_lda_y_pred = bl_cls.lda_y_pred

# %%

# %%

# %%
y_pred = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)

y_pred_2d = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)

# %%
fig, ax = plt.subplots()
ax.plot(raw_label, label="Ground Truth")
ax.plot(
    df_data[df_data["is_valid"] == 1]["id_end"].values, y_pred * 0.8, label="1D-CNN"
)
ax.plot(
    df_data[df_data["is_valid"] == 1]["id_end"].values, y_pred_2d * 0.7, label="2D-CNN"
)

ax.plot(
    df_data[df_data["is_valid"] == 1]["id_end"].values, bl_y_pred * 0.5, label="SVM"
)
ax.plot(
    df_data[df_data["is_valid"] == 1]["id_end"].values, bl_lda_y_pred * 0.4, label="LDA"
)

ax.legend(loc="upper left")


# %%

# %%

# %%

# %%

# %%
# TF = LFP

# %%

# %% [markdown]
# # Prepare learner:

# %% [markdown]
# ## Dataloader prep

# %% [markdown]
# ### Data stream:

# %%
def get_x(row):
    return torch.tensor(TF[:, :, row["id_start"] : row["id_end"]])


def get_y(row):
    return row["label"]


def splitter(df):
    train = df.index[df["is_valid"] == 0].tolist()
    valid = df.index[df["is_valid"] == 1].tolist()
    return train, valid


# %% [markdown]
# ### Data transforms:

# %%
class Resizer(Transform):
    def __init__(self, target_size):
        self.target_size = target_size

    def encodes(self, X):
        if isinstance(X, TensorCategory):
            return X
        return torch.stack(
            [ToTensor()(ToPILImage()(x).resize(self.target_size)) for x in X]
        ).squeeze()


class LFPNormalizer(Transform):
    def __init__(self, means, stds):
        self.means, self.stds = means, stds

    def encodes(self, X):
        if isinstance(X, TensorCategory):
            return X
        xs = torch.unbind(X, 1)

        return torch.stack(
            [(x - self.means[ii]) / self.stds[ii] for ii, x in enumerate(xs)], 1
        )


class LFPNormalizer1d(Transform):
    def __init__(self, stats):
        self.means, self.stds = stats[0], stats[1]

    def encodes(self, X):
        if isinstance(X, TensorCategory):
            return X

        return


# %%
def LFP_block():
    return TransformBlock(
        item_tfms=[Resizer((160, 160)), IntToFloatTensor],
        batch_tfms=LFPNormalizer(
            [0.5, 0.5, 0.5, 0.5, 0.5], [0.098, 0.098, 0.098, 0.098, 0.098]
        ),
    )


# %%


def LFP_block1d():
    return TransformBlock(batch_tfms=(LFPNormalizer1d(get_norm_stats(LFP, train_end))))


# %%

# %% [markdown]
# ### Dataloader:

# %%
dblock = DataBlock(
    blocks=(LFP_block, CategoryBlock),
    get_x=get_x,
    get_y=get_y,
    splitter=splitter,
)

# %%
# dblock.summary(df_data)

# %%
dls = dblock.dataloaders(df_data, bs=32)

# %% [markdown]
# # Learner

# %%
resnet = myResNet(5, 2, [2, 3, 4, 3], 4)

# %%
resnet

# %%
learn = Learner(
    dls,
    resnet,
    metrics=[accuracy],
    loss_func=F.cross_entropy,
    # cbs=[ActivationStats(with_hist=True)],
)

# %%
xb, yb = dls.one_batch()
yb.shape, xb.shape

# %%
init_loss = learn.loss_func(learn.model(xb), yb)
init_loss

# %%

# %%
del xb, yb

# %%
learn.lr_find()

# %%
learn.fit_one_cycle(5, 4 * 10e-4)

# %%
preds_val = learn.get_preds()

# %%
labels_pred = torch.argmax(preds_val[0], -1).numpy()
labels_pred.shape

# %%
y = df_data[df_data["is_valid"] == 1]["label"].astype(float).values

# %%
y

# %%
(labels_pred == y).mean()

# %%
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pandas as pd
import seaborn as sns

# %%
display_labels = ["Still", "Mvmt"]
cm = confusion_matrix(y, labels_pred)
cm_norm = confusion_matrix(y, labels_pred, normalize="true")
df_cm = pd.DataFrame(cm_norm, index=display_labels, columns=display_labels)

# %%
plt.figure()
sns.heatmap(df_cm, annot=True)
plt.show()
cm

# %%
plt.show()

# %%
