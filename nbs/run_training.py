import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..")))

from lfp_analysis.data import *
from lfp_analysis.process import *

#  from lfp_analysis.resnet2d import *
from lfp_analysis.resnet1d import *
from lfp_analysis.svm import *

from fastai.vision.all import *
from fastai.callback.wandb import *
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor

import numpy as np
import pandas as pd
import h5py

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib.colors import to_hex


def plot_LFPs(ts, LFPs, label):

    fig, ax = plt.subplots(LFPs.shape[0], 1)

    for i in range(LFPs.shape[0]):
        ax[i].plot(ts, LFPs[i])
        ax[i].plot(ts, label * 0.5 * np.max(LFPs[i]))

    return fig, ax


def get_metric(results, metric):
    return {
        "svm": results["svm_scores"][metric],
        "b-lda": results["b_lda_scores"][metric],
        "th-lda": results["th_lda_scores"][metric],
        "1d-cnn": results["1d_cnn_scores"][metric],
        "2d-cnn": results["2d_cnn_scores"][metric],
    }


def plot_roc_instance(roc_curve_cont, ax=None, color=None, label=None):

    mean_fpr = np.linspace(0, 1, 100)
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], color="k", alpha=0.5, linestyle="--", linewidth=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positve Rate")

    ax.plot(mean_fpr, roc_curve_cont["mean"], color=color, label=label)
    ax.fill_between(
        mean_fpr,
        roc_curve_cont["mean"] - roc_curve_cont["std"],
        roc_curve_cont["mean"] + roc_curve_cont["std"],
        alpha=0.2,
        color=color,
    )


def plot_conf_mats(conf_mats):

    fig, ax = plt.subplots(5, 1, figsize=(4, 18))

    ml_keys = ["svm", "b-lda", "th-lda"]
    nn_keys = ["1d-cnn", "2d-cnn"]

    max_val = max(
        [np.max(conf_mats[key]["mean"]) for key in ml_keys]
        + [np.max(conf_mats[key]) for key in nn_keys]
    )
    min_val = min(
        [np.min(conf_mats[key]["mean"]) for key in ml_keys]
        + [np.min(conf_mats[key]) for key in nn_keys]
    )

    for ii, cls_name in enumerate(ml_keys):
        means, stds = conf_mats[cls_name]["mean"], conf_mats[cls_name]["std"]
        annot = pd.DataFrame(
            [
                [
                    f"{mean:.1f} +/- {std:.2f}"
                    for mean, std in zip(mean_outer, std_outer)
                ]
                for mean_outer, std_outer in zip(means, stds)
            ]
        )

        df = pd.DataFrame(conf_mats[cls_name]["mean"], index=[0, 1], columns=[0, 1])

        sns.heatmap(
            df,
            annot=annot,
            vmin=min_val,
            vmax=max_val,
            ax=ax[ii],
            fmt="s",
            cmap="rocket",
        )
        ax[ii].set_title(cls_name)
        ax[ii].set_ylabel("True Label")

    for ii, cls_name in enumerate(nn_keys):
        df = pd.DataFrame(conf_mats[cls_name], index=[0, 1], columns=[0, 1])

        sns.heatmap(
            df,
            annot=True,
            vmin=min_val,
            vmax=max_val,
            ax=ax[3 + ii],
            fmt=".2f",
            cmap="rocket",
        )
        ax[3 + ii].set_title(cls_name)
        ax[3 + ii].set_ylabel("True Label")

    ax[-1].set_xlabel("Predicted Label")
    return fig, ax


def plot_svm_coefs(svm_coefs, fig=None, label=None, n_chan=4, **kwargs):

    color_dict = {0: to_hex("C0"), 1: to_hex("C1"), 2: to_hex("C2")}

    cls_id = 0 if label == "SVM" else (1 if label == "b-LDA" else 2)

    if fig is None:
        fig = make_subplots(rows=n_chan, cols=1)

    coefs_df = pd.DataFrame(
        np.stack([svm_coefs["mean"], svm_coefs["std"]]).T,
        columns=["mean", "std"],
        index=svm_coefs["names"],
    )

    df_cont = [
        coefs_df.iloc[[f"pow{i}" in coef for coef in coefs_df.index], :]
        for i in range(n_chan)
    ]

    for i in range(n_chan):
        this_df = df_cont[i]

        fig.add_trace(
            go.Scatter(
                name=label,
                showlegend=True if i == 0 else False,
                x=this_df.index.values,
                y=this_df["mean"],
                mode="markers",
                marker={"color": color_dict[cls_id]},
                error_y=dict(type="data", array=this_df["std"], visible=True),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(showlegend=True)

    return fig


def mainloop(PAT_ID, DATASET_NAME):
    DATA_PATH = Path("./../data")
    PAT_ID = Path(PAT_ID)
    DATASET_NAME = Path(DATASET_NAME)

    LFP_SOURCE = DATA_PATH / "preprocessed" / PAT_ID / DATASET_NAME.with_suffix(".h5")
    H5_SOURCE = DATA_PATH / "TF_and_df" / PAT_ID / DATASET_NAME.with_suffix(".h5")
    CSV_SOURCE = DATA_PATH / "TF_and_df" / PAT_ID / DATASET_NAME.with_suffix(".csv")

    h5_TF = h5py.File(H5_SOURCE, "r")
    h5_LFP = h5py.File(LFP_SOURCE, "r")

    TF = h5_TF["TF"][:]
    LFP = h5_LFP["LFP"][:]
    label = h5_LFP["label"][:]

    n_chan = LFP.shape[0]

    import wandb

    wandb.init(project="lfp-decoding", entity="fer-rplazas")
    wandb.run.name = str(PAT_ID) + "/" + str(DATASET_NAME)
    config = wandb.config

    config.DATASET_NAME = DATASET_NAME
    config.TRAIN_DATA_LEN_SECS = LFP.shape[1] / 2048.0 / 60 * 0.8
    config.n_chan = n_chan

    WIN_LEN_SEC = 0.750
    df_data = make_label_df(label, WIN_LEN_SEC)

    train_end = df_data[df_data["is_valid"] == 0]["id_end"].iloc[-1]

    def get_norm_stats(LFP, train_end):
        return np.mean(LFP[:, :train_end], axis=-1), np.std(LFP[:, :train_end], axis=-1)

    def norm_with_stats(LFP, stats):
        means, stds = stats[0], stats[1]
        return (LFP - means[:, np.newaxis]) / stds[:, np.newaxis]

    bl_cls = BLClassifier(LFP, df_data, extract_method="periodogram")

    svm_scores = bl_cls.classify_many(method="SVM")
    y_hat_svm = bl_cls.y_pred

    b_lda_scores = bl_cls.classify_many(method="b-LDA")
    y_hat_blda = bl_cls.y_pred

    th_lda_scores = bl_cls.classify_many(method="th-LDA")
    y_hat_thlda = bl_cls.y_pred

    fig = make_subplots(
        rows=n_chan,
        cols=1,
        shared_yaxes=True,
        subplot_titles=[f"LFP{i}" for i in range(n_chan)],
    )
    plot_svm_coefs(svm_scores["svm_coef"], fig=fig, n_chan=n_chan, label="SVM")
    plot_svm_coefs(b_lda_scores["svm_coef"], fig=fig, n_chan=n_chan, label="b-LDA")
    plot_svm_coefs(th_lda_scores["svm_coef"], fig=fig, n_chan=n_chan, label="th-LDA")

    fig.update_yaxes(matches="y")
    fig.write_html("temp.html", include_plotlyjs="cdn")

    wandb.log({"svm_coefs": wandb.Html(open("temp.html"))}, commit=False)

    from scipy.signal import decimate

    Q_factor = 10

    LFP_decim = decimate(LFP, Q_factor)
    label_decim = decimate(label, Q_factor)

    df_data_decim = make_label_df(label_decim, WIN_LEN_SEC, fs=204.8)

    # Log LFPs and label to WandB:
    ts = np.arange(0, len(label_decim)) / (2048.0 / Q_factor)
    fig, ax = plot_LFPs(ts, LFP_decim, label_decim)
    wandb.log({"LFPs": wandb.Image(plt)}, commit=False)

    def get_x(row):
        return torch.tensor(
            LFP_decim[:, row["id_start"] : row["id_end"]].copy()
        ).float()

    def get_y(row):
        return row["label"]

    def splitter(df):
        train = df.index[df["is_valid"] == 0].tolist()
        valid = df.index[df["is_valid"] == 1].tolist()
        return train, valid

    def LFP_block1d():
        return TransformBlock(
            batch_tfms=(LFPNormalizer1d(get_norm_stats(LFP, train_end)))
        )

    dblock = DataBlock(
        blocks=(LFP_block1d, CategoryBlock),
        get_x=get_x,
        get_y=get_y,
        splitter=splitter,
    )

    dls = dblock.dataloaders(
        df_data_decim,
        bs=256,
    )
    xb, yb = dls.one_batch()

    resnet = ResNet1d(LFP.shape[0], 2, [1, 1])

    learn = Learner(
        dls,
        resnet,
        metrics=[accuracy],
        loss_func=F.cross_entropy,
        cbs=[WandbCallback()],
        wd=0.025,
    )
    preds_val = learn.get_preds()

    learn.recorder.train_metrics = True

    learn.fit_one_cycle(14, 1 / 40)
    learn.add_cb(EarlyStoppingCallback(min_delta=0.01, patience=3))

    learn.fit_one_cycle(8, 10e-4)
    learn.fit_one_cycle(10, 5 * 10e-5)
    learn.fit_one_cycle(10, 10e-5)
    learn.fit_one_cycle(8, 3 * 10e-6)
    learn.fit_one_cycle(8, 10e-6)

    [learn.remove_cb(cb) for cb in learn.cbs[3:]]

    # print(learn.cbs)

    preds_val = learn.get_preds()
    y = df_data_decim[df_data_decim["is_valid"] == 1]["label"].astype(float).values
    y_pred = torch.argmax(preds_val[0], -1).numpy()
    y_score = preds_val[0][:, 1].numpy()

    cnn1d_scores = get_scores(y, y_pred, y_score)

    from lfp_analysis.resnet2d import ResNet2d, LFPNormalizer2d, Resizer

    df_data = make_label_df(label, WIN_LEN_SEC)

    def get_x(row):
        return torch.tensor(TF[:, :, row["id_start"] : row["id_end"]]).float()

    def get_y(row):
        return row["label"]

    def splitter(df):
        train = df.index[df["is_valid"] == 0].tolist()
        valid = df.index[df["is_valid"] == 1].tolist()
        return train, valid

    def LFP_block2d():
        return TransformBlock(
            item_tfms=[Resizer((160, 160)), IntToFloatTensor],
            batch_tfms=LFPNormalizer2d(
                (np.repeat(0.5, TF.shape[0]), np.repeat(0.098, TF.shape[0]))
            ),
        )

    dblock = DataBlock(
        blocks=(LFP_block2d, CategoryBlock),
        get_x=get_x,
        get_y=get_y,
        splitter=splitter,
    )

    dls = dblock.dataloaders(df_data, bs=32)
    xb, yb = dls.one_batch()

    wandb.run.finish()

    wandb.init(project="lfp-decoding", entity="fer-rplazas")
    wandb.run.name = str(PAT_ID) + "/" + str(DATASET_NAME) + "_2D"
    resnet2d = ResNet2d(TF.shape[0], 2, [2, 2, 1])

    learn2d = Learner(
        dls,
        resnet2d,
        wd=0.3,
        metrics=[accuracy],
        loss_func=F.cross_entropy,
        cbs=[WandbCallback(), EarlyStoppingCallback(min_delta=0.01, patience=4)],
    )

    learn2d.recorder.train_metrics = True

    learn2d.fit_one_cycle(8, 10e-4)
    learn2d.fit_one_cycle(8, 10e-6)

    [learn2d.remove_cb(cb) for cb in learn2d.cbs[3:]]

    preds_2d_val = learn2d.get_preds()
    y_2d_pred = torch.argmax(preds_2d_val[0], -1).numpy()
    y_2d_score = preds_2d_val[0][:, 1].numpy()

    y = df_data[df_data["is_valid"] == 1]["label"].astype(float).values

    cnn2d_scores = get_scores(y, y_2d_pred, y_2d_score)

    import pickle

    PKL_TARGET = DATA_PATH / "results" / PAT_ID / DATASET_NAME.with_suffix(".p")
    FIG_TARGET = DATA_PATH / "results" / PAT_ID / DATASET_NAME.with_suffix(".png")

    data_container = {
        "win_len_sec": WIN_LEN_SEC,
        "svm_scores": svm_scores,
        "b_lda_scores": b_lda_scores,
        "th_lda_scores": th_lda_scores,
        "1d_cnn_scores": cnn1d_scores,
        "2d_cnn_scores": cnn2d_scores,
    }

    pickle.dump(data_container, open(PKL_TARGET, "wb"))

    wandb.log(data_container)

    fig, ax = plt.subplots()
    plot_roc_instance(
        svm_scores["roc_curve"],
        ax=ax,
        label=f"SVM (AUC: {svm_scores['AUC']['valid']['mean']:0.2f} +/- {svm_scores['AUC']['valid']['std']:0.2f})",
    )
    plot_roc_instance(
        b_lda_scores["roc_curve"],
        ax=ax,
        label=f"b-LDA (AUC: {b_lda_scores['AUC']['valid']['mean']:0.2f} +/- {b_lda_scores['AUC']['valid']['std']:0.2f})",
    )
    plot_roc_instance(
        th_lda_scores["roc_curve"],
        ax=ax,
        label=f"th-LDA (AUC: {th_lda_scores['AUC']['valid']['mean']:0.2f} +/- {th_lda_scores['AUC']['valid']['std']:0.2f})",
    )

    ax.plot(
        cnn1d_scores["roc_curve"][0],
        cnn1d_scores["roc_curve"][1],
        label=f"1d-CNN (AUC: {cnn1d_scores['AUC']['valid']:.2f})",
    )
    ax.plot(
        cnn2d_scores["roc_curve"][0],
        cnn2d_scores["roc_curve"][1],
        label=f"2d-CNN (AUC: {cnn2d_scores['AUC']['valid']:.2f})",
    )

    ax.legend()

    fig_c, ax_c = plot_conf_mats(get_metric(data_container, "conf_mat_norm"))

    wandb.log({"conf_mat": wandb.Image(fig_c)})
    wandb.log({"roc_curves": wandb.Image(fig)})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y, label="Ground Truth")
    ax.plot(y_pred * 0.8, label="1D-CNN")
    ax.plot(y_2d_pred * 0.6, label="2D-CNN")
    ax.plot(y_hat_svm * 0.4, label="SVM")
    ax.plot(y_hat_blda * 0.2, label="b-LDA")
    ax.plot(y_hat_thlda * 0.1, label="th-LDA")

    ax.legend(bbox_to_anchor=(1, 1))
    # plt.legend(handles=[p1, p2], title='title', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
    plt.tight_layout()

    wandb.log({"timeseries_preds": wandb.Image(fig)})
    wandb.run.finish()

    plt.close("all")


if __name__ == "__main__":

    missing_datasets = {3: "Pouring", 5: "Pegboard", 6: "Pegboard", 8: "Pegboard"}

    for ii in range(4, 9):

        print(f"  ###  Beginning ET{ii}")

        for set_id, dataset in enumerate(["Pegboard", "Pouring", "Posture"]):
            print(f"  ---  Beginning {dataset}")

            try:
                missing_for_pat = missing_datasets[ii]
            except:
                missing_for_pat = dict()

            if dataset not in missing_for_pat:
                mainloop(f"ET{ii}", dataset + "_off")
