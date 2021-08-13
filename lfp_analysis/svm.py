from functools import partial

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .feature_extractor import SignalFeatureExtractor, TfFeatureExtractor
from .score import Scorer, SvmScores

scalar_metric_func_container = {
    "accuracy": accuracy_score,
    "AUC": roc_auc_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

other_metric_func_container = {
    "conf_mat": confusion_matrix,
    "conf_mat_norm": partial(confusion_matrix, normalize="true"),
    "roc_curve": roc_curve,
}


def scale_feat_mats(X_train, X_valid):
    sc = StandardScaler().fit(X_train)

    if X_valid is None:
        return sc.transform(X_train), None
    return sc.transform(X_train), sc.transform(X_valid)


def rebalance_data(X, y):
    return RandomUnderSampler().fit_resample(X, y)


class BLClassifier:
    def __init__(
        self,
        data: np.ndarray,
        df: pd.DataFrame,
        fs: float = 2048.0,
        extract_method: str = "periodogram",
        select_band=None,
    ):

        if extract_method == "periodogram":
            self.extractor = SignalFeatureExtractor(method="periodogram", fs=fs)
        elif extract_method == "hilbert":
            self.extractor == SignalFeatureExtractor(method="hilbert", fs=fs)
        elif extract_method == "TF":
            self.extractor == TfFeatureExtractor()
        else:
            raise ValueError("extract_method not recognized")

        self.data, self.df, self.extract_method = data, df, extract_method
        self.select_band = select_band

        self.get_feat_mats()

        if select_band is not None:
            self.select_bands()

        self.train_valid_split()

    @classmethod
    def from_dataset_and_windower(cls, dataset, windower, **kwargs):
        return cls(dataset.LFP.data, windower.df, fs=dataset.LFP.fs, **kwargs)

    def select_bands(self):

        if self.select_band == "beta":
            chan_idx = np.where(
                [("13_20Hz" in el) or ("21_30Hz" in el) for el in list(self.X.columns)]
            )[0]

        elif self.select_band == "theta":
            chan_idx = np.where(["2_7Hz" in el for el in list(self.X.columns)])[0]

        else:
            raise ValueError("select_band not recognized")

        self.X = self.X[:, chan_idx]

    def get_feat_mats(self):

        self.X, self.y = self.extractor.get_feats(self.data, self.df)
        return self

    def train_valid_split(self):
        train_idx, valid_idx = self.df["is_valid"] == 0, self.df["is_valid"] == 1
        self.X_train, self.y_train = self.X[train_idx], self.y[train_idx]

        self.X_valid, self.y_valid = self.X[valid_idx], self.y[valid_idx]
        if self.X_valid.empty:
            self.X_valid, self.y_valid = None, None

        return self

    def cross_task(self, other):

        X_other = self.scaler.transform(other.X_train)
        y_hat = self.clf.predict(X_other)
        y_score = self.clf.decision_function(X_other)

        return Scorer().get_scores(
            other.y_train, y_hat, y_score, times=other.times_train
        )

    def classify(
        self,
        cls_method="SVM",
        verbose=False,
    ):

        self.cls_method = cls_method
        if not hasattr(self, "X_train"):
            self.get_feat_mats()
            if self.select_band is not None:
                self.select_bands()
            self.train_valid_split()

        # z-score features:
        self.scaler = StandardScaler().fit(self.X_train)
        X_train_sc = self.scaler.transform(self.X_train)
        X_valid_sc = self.scaler.transform(self.X_valid)

        # Rebalance training data for equal number of samples in each class:
        X_train_bal, y_train_bal = rebalance_data(X_train_sc, self.y_train)

        # Balance validation data:
        # X_valid_bal, y_valid_bal = RandomUnderSampler().fit_resample(
        #     X_valid_sc, self.y_valid
        # )

        # Train & Score SVMs:
        if self.cls_method == "SVM":
            self.cls = SVC().fit(X_train_bal, y_train_bal)

            def get_losses(X):
                return None

        elif self.cls_method == "LDA":
            self.cls = LinearDiscriminantAnalysis().fit(X_train_bal, y_train_bal)

            def get_losses(X):
                return self.cls.predict_log_proba(X)

        else:
            raise ValueError("cls_method not recognized, should be LDA or SVM")

        # Performance on Train:
        self.y_pred_train = self.cls.predict(X_train_sc)
        self.y_score_train = self.cls.decision_function(X_train_sc)

        self.times_train = self.df[self.df["is_valid"] == False]["t"].values
        self.train_scores = Scorer(
            ds_type="train", cls_type=self.cls_method
        ).get_scores(
            self.y_train,
            self.y_pred_train,
            self.y_score_train,
            times=self.times_train,
            losses=get_losses(X_train_sc),
        )

        # Performance on Valid:
        self.y_pred = self.cls.predict(X_valid_sc)
        self.y_score = self.cls.decision_function(X_valid_sc)
        self.times_valid = self.df[self.df["is_valid"] == True]["t"].values
        self.valid_scores = Scorer(
            ds_type="valid", cls_type=self.cls_method
        ).get_scores(
            self.y_valid,
            self.y_pred,
            self.y_score,
            times=self.times_valid,
            losses=get_losses(X_valid_sc),
        )

        self.linearSVM = LinearSVC(max_iter=10e4).fit(X_train_bal, y_train_bal)
        self.SVM_coefs = pd.Series(
            self.linearSVM.coef_.squeeze(), index=list(self.X.columns)
        )
        self.linearSVM_acc = self.linearSVM.score(X_valid_sc, self.y_valid)

        if verbose:
            print("On Train: ")
            print(self.train_scores)
            print("")
            print("On Valid:")
            print(self.valid_scores)

        return (
            self.train_scores,
            self.valid_scores,
            (self.SVM_coefs, self.linearSVM_acc),
        )

    def classify_many(
        self,
        n=10,
        **kwargs,
    ):

        train_scores, valid_scores, svm_coef_cont, lin_svm_acc_cont = [], [], [], []

        for i in range(n):
            res = self.classify(
                verbose=False,
                **kwargs,
            )

            train_scores.append(res[0])
            valid_scores.append(res[1])

            svm_coef_cont.append(res[2][0])
            lin_svm_acc_cont.append(res[2][1])

        self.train_scores_cum = Scorer(
            ds_type="train", cls_type=self.cls_method
        ).accumulate(train_scores)
        self.valid_scores_cum = Scorer(
            ds_type="valid", cls_type=self.cls_method
        ).accumulate(valid_scores)

        svm_coef_mat = np.zeros((len(self.SVM_coefs.index), n)) * np.nan
        for i in range(n):
            svm_coef_mat[:, i] = svm_coef_cont[i].values

        svm_coefs_mean, svm_coefs_std = np.mean(svm_coef_mat, axis=-1), np.std(
            svm_coef_mat, axis=-1
        )

        self.svm_coef_df = pd.DataFrame(
            np.c_[(svm_coefs_mean, svm_coefs_std)],
            columns=["mean", "std"],
            index=list(self.X.columns),
        )
        self.lin_svm_acc = np.mean(np.array(lin_svm_acc_cont)), np.std(
            np.array(lin_svm_acc_cont)
        )

        return SvmScores(
            self.train_scores_cum,
            self.valid_scores_cum,
            (self.svm_coef_df, self.lin_svm_acc),
        )
