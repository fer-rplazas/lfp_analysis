# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%


#%load_ext lab_black


# %%
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..")))


# %%


# %%
from lfp_analysis.data import *
from lfp_analysis.process import *
from lfp_analysis.resnet2d import *
from lfp_analysis.resnet1d import *
from lfp_analysis.svm import *
from lfp_analysis.report import *

from fastai.vision.all import *
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor


# %%
import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_hex
from icecream import ic

import seaborn as sns


def train1d():

    trainer = (
        Trainer1d(log_wandb=False, layers=[2], wd=25, experiment=EXPERIMENT)
        .prepare_dls(data1, windower)
        .prepare_learner()
    )

    trainer.train(70, 0.5 * 1e-2)
    trainer.save_model()
    cnn_1d_scores = trainer.score()

    return cnn_1d_scores


def train2d():
    trainer2 = (
        Trainer2d(log_wandb=False, wd=25, experiment=EXPERIMENT)
        .prepare_dls(data2, windower)
        .prepare_learner()
    )

    trainer2.train(70, 0.5 * 1e-2)
    trainer2.save_model()
    cnn_2d_scores = trainer2.score()

    return cnn_2d_scores


# %%
EXPERIMENT = "train_all_v3"
WIN_LEN_SEC = 0.750


if __name__ == "__main__":  # %%
    for pat_id in range(1, 9):
        ic(f"Starting Patient {pat_id}")
        for task in ["Pegboard", "Pouring", "Posture"]:
            ic(f"Starting task {task} for Pat {pat_id}")
            data1 = getattr(Patient(pat_id), task + "_off")
            if data1 is None:
                continue
            else:
                try:
                    Reporter.from_pickle(pat_id, task, EXPERIMENT)
                except:
                    pass
                else:
                    continue

                data1 = data1.load_1d()

                windower = Windower(WIN_LEN_SEC).window(data1.label)

                bl_cls = BLClassifier(
                    data1.LFP, windower.data_df, extract_method="periodogram"
                )
                svm_scores = bl_cls.classify_many(method="SVM")
                b_scores = bl_cls.classify_many(method="beta")
                th_scores = bl_cls.classify_many(method="theta")

                cnn_1d_scores = train1d()
                torch.cuda.empty_cache()

                ic(f"Starting 2D-CNN")
                data2 = getattr(Patient(pat_id), task + "_off").load_2d()
                cnn_2d_scores = train2d()

                rec = Reporter(
                    svm_scores,
                    cnn_1d_scores,
                    cnn_2d_scores,
                    b_scores,
                    th_scores,
                    data1,
                    windower,
                    experiment=EXPERIMENT,
                )

                rec.save_plots()
                rec.save()
