import os
from pathlib import Path

import numpy as np

from fastai.callback.wandb import *
from fastai.vision.all import *
from fastai.vision.all import nn, ConvLayer, F

import wandb

from .score import Scorer, CnnScores
from dataclasses import dataclass

DATA_PATH = Path("./../data")


@dataclass
class SilenceRecorder(Callback):
    learn: Learner

    def __post_init__(self):
        self.learn.recorder.silent = True


def _conv_block1d(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni, nf // 4, 1, ndim=1),
        ConvLayer(nf // 4, nf // 4, 3, stride=stride, ndim=1),
        ConvLayer(nf // 4, nf, 1, act_cls=None, norm_type=NormType.BatchZero, ndim=1),
    )


class ResBlock1d(Module):
    def __init__(self, ni, nf, stride=1):
        self.convs = _conv_block1d(ni, nf, stride)
        self.idconv = noop if ni == nf else ConvLayer(ni, nf, 1, act_cls=None, ndim=1)
        self.pool = noop if stride == 1 else nn.AvgPool1d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


def _resnet_stem1d(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i + 1], 3, stride=2 if i == 0 else 1, ndim=1)
        for i in range(len(sizes) - 1)
    ] + [nn.MaxPool1d(kernel_size=3, stride=2, padding=1)]


class ResNet1d(nn.Sequential):
    def __init__(self, n_in, n_out, layers, expansion=1):
        stem = _resnet_stem1d(n_in, 32, 32, 64)
        self.block_szs = [64, 64, 128, 256, 256, 512, 512]
        for i in range(1, len(self.block_szs)):
            self.block_szs[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]
        super().__init__(
            *stem,
            *blocks,
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.block_szs[len(layers)], n_out),
            Flatten(),
        )

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx == 0 else 2
        ch_in, ch_out = self.block_szs[idx : idx + 2]
        # print(f"{ch_in,ch_out}")
        return nn.Sequential(
            *[
                ResBlock1d(ch_in if i == 0 else ch_out, ch_out, stride if i == 0 else 1)
                for i in range(n_layers)
            ]
        )


###############################################################################
# FastAI pipeline:
##############################################################################


def get_norm_stats(LFP, train_end):
    return np.mean(LFP[:, :train_end], axis=-1), np.std(LFP[:, :train_end], axis=-1)


def norm_with_stats(LFP, stats):
    means, stds = stats[0], stats[1]
    return (LFP - means[:, np.newaxis]) / stds[:, np.newaxis]


class LFPNormalizer1d(Transform):
    def __init__(self, stats):
        self.means, self.stds = (
            torch.tensor(stats[0]).float(),
            torch.tensor(stats[1]).float(),
        )

        if torch.cuda.is_available():
            self.means = self.means.cuda()
            self.stds = self.stds.cuda()

    def encodes(self, X):
        if isinstance(X, TensorCategory):
            return X

        return (X - self.means[:, None]) / self.stds[:, None]

    def decodes(self, X):
        if isinstance(X, TensorCategory):
            return X

        return X * self.stds[:, None] + self.means[:, None]


class Trainer:
    def __init__(self, log_wandb=True, experiment=None):
        self.log_wandb = log_wandb
        self.experiment = experiment

        if self.log_wandb:
            if experiment is None:
                self.run = wandb.init()
            else:
                self.run = wandb.init(project=experiment)

    def save_model(self):
        self.learn.model_dir = self.model_path

        self.learn.save("model")

    def score(self):

        # Train:
        times_train = self.data_df[self.data_df["is_valid"] == False]["t"].values
        y_scores, y, losses = self.learn.get_preds(ds_idx=0, with_loss=True)
        y_hat = torch.argmax(y_scores, -1).numpy()
        y_scores = y_scores[:, 1].numpy()

        train_scores = Scorer(ds_type="train").get_scores(
            y, y_hat, y_scores, losses.numpy(), times=times_train
        )

        # Valid:
        times_valid = self.data_df[self.data_df["is_valid"] == True]["t"].values
        y_scores, y, losses = self.learn.get_preds(with_loss=True)
        y_hat = torch.argmax(y_scores, -1).numpy()
        y_scores = y_scores[:, 1].numpy()

        valid_scores = Scorer().get_scores(
            y, y_hat, y_scores, losses.numpy(), times=times_valid
        )

        self.scores = CnnScores(train_scores, valid_scores)

        return self.scores


class Trainer1d(Trainer):
    def __init__(
        self, layers=[1, 1], wd=0.025, log_wandb=True, silent=False, experiment=None
    ):

        self.layers, self.wd = layers, wd
        self.silent = silent
        super().__init__(log_wandb, experiment)

    def prepare_dls(self, dataset, windower, bs=256):

        if self.log_wandb:
            self.run.name = f"{dataset.pat_id}/{dataset.task}_{dataset.stim}_1d"

        if self.experiment is not None:
            self.model_path = (
                DATA_PATH
                / "results"
                / f"ET{dataset.pat_id}"
                / self.experiment
                / f"{dataset.task}"
                / "trained"
            )
            self.model_path.mkdir(parents=True, exist_ok=True)

        self.dataset, self.windower = dataset, windower

        self.data_df = self.windower.df

        def get_x(row):
            return torch.tensor(
                dataset.LFP.data[:, int(row["id_start"]) : int(row["id_end"])].copy()
            ).float()

        def get_y(row):
            return row["label"]

        def splitter(df):
            train = df.index[df["is_valid"] == 0].tolist()
            valid = df.index[df["is_valid"] == 1].tolist()
            return train, valid

        train_end = self.data_df[self.data_df["is_valid"] == 0]["id_end"].iloc[-1]

        def LFP_block1d():
            return TransformBlock(
                batch_tfms=(
                    LFPNormalizer1d(get_norm_stats(dataset.LFP.data, train_end))
                )
            )

        # def LFP_block1d():
        #     return TransformBlock(batch_tfms=())

        self.dblock = DataBlock(
            blocks=(LFP_block1d, CategoryBlock),
            get_x=get_x,
            get_y=get_y,
            splitter=splitter,
        )

        self.dls = self.dblock.dataloaders(self.data_df, bs=bs)

        return self

    def prepare_learner(self, dls=None, wd=None):

        cbs = [WandbCallback()] if self.log_wandb else []

        # if self.silent:
        #     cbs.append(SilenceRecorder)

        dls = self.dls if dls is None else dls
        wd = self.wd if wd is None else wd
        loss = LabelSmoothingCrossEntropy(eps=0.2)
        self.resnet = ResNet1d(self.dataset.LFP.data.shape[0], 2, self.layers).cuda()
        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()

        self.learn = Learner(
            dls,
            self.resnet.cuda(),
            metrics=[
                accuracy,
            ],
            loss_func=loss,
            cbs=cbs,
            wd=float(wd),
        )
        self.learn.recorder.train_metrics = True
        # self.learn.recorder.silent = True

        return self

    def train(self, n_epochs=45, lr_div=1):

        self.learn.fit_one_cycle(n_epochs, lr_div)
        self.learn.fit_one_cycle(n_epochs, lr_div / 2)
        self.learn.fit_one_cycle(n_epochs, lr_div / 4)
        self.learn.fit_one_cycle(n_epochs, lr_div / 8)

        self.learn.add_cb(EarlyStoppingCallback(min_delta=0.001, patience=3))

        # self.learn.fit_one_cycle(14, 10e-4)
        # self.learn.fit_one_cycle(25, 5 * 10e-5)
        self.learn.fit_one_cycle(35, 10e-5)
        self.learn.fit_one_cycle(35, 3 * 10e-6)
        self.learn.fit_one_cycle(35, 10e-6)
        self.learn.fit_one_cycle(35, 10e-7)

        [self.learn.remove_cb(cb) for cb in self.learn.cbs[3:]]
