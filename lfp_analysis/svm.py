
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class BLClassifier:
    def __init__(self, data, df_data):

        self.data, self.df_data = data, df_data

        frequencies = np.arange(2, 101, 1)
        self.freq_ranges = [
            [2, 7],
            [8, 12],
            [13, 20],
            [21, 30],
            [31, 45],
            [46, 60],
            [61, 75],
            [76, 100],
        ]

        self.feat_names = []
        for ch in range(self.data.shape[0]):
            self.feat_names += [f"pow{ch}_{el[0]}_{el[1]}Hz" for el in self.freq_ranges]

        self.freq_idx = np.array(
            [
                [np.where(frequencies == el[0]),
                 np.where(frequencies == el[1])]
                for el in self.freq_ranges
            ]
        ).squeeze()

    def get_feat_mats(self):

        self.X_train, self.y_train = self.get_feats(
            self.df_data[self.df_data["is_valid"] ==
                         False].drop(columns=["is_valid"]),
        )
        self.X_valid, self.y_valid = self.get_feats(
            self.df_data[self.df_data["is_valid"]
                         == True].drop(columns=["is_valid"]),
        )

        return self

    def get_feats(self, df_labels):

        feat_mat = (
            np.zeros(
                (len(df_labels), self.data.shape[0] * len(self.freq_idx))) * np.nan
        )
        label_mat = df_labels["label"].to_numpy()

        for ii, row in df_labels.reset_index().iterrows():
            this_data = self.data[:, :, row["id_start"]: row["id_end"]]

            for ch in range(self.data.shape[0]):

                for kk, freq_lims in enumerate(self.freq_idx):
                    feat_mat[ii, (ch * len(self.freq_ranges)) + kk] = np.mean(
                        this_data[ch, freq_lims[0]: freq_lims[1]]
                    )

        return pd.DataFrame(feat_mat, columns=self.feat_names), label_mat

    def classify(self, metric="accuracy"):

        if not hasattr(self, "X_train"):
            self = self.get_feat_mats()

        X_train_sc = StandardScaler().fit_transform(self.X_train)
        X_valid_sc = StandardScaler().fit_transform(self.X_valid)

        X_train_bal, y_train_bal = RandomUnderSampler().fit_resample(
            X_train_sc, self.y_train
        )
        X_valid_bal, y_valid_bal = RandomUnderSampler().fit_resample(
            X_valid_sc, self.y_valid
        )

        clf = SVC().fit(X_train_bal, y_train_bal)

        y_pred_train = clf.predict(X_train_sc)
        y_pred = clf.predict(X_valid_sc)

        print("On Train: ")
        print(classification_report(self.y_train, y_pred_train))
        print(confusion_matrix(self.y_train, y_pred_train))
        print("")
        print("On Valid:")
        print(classification_report(self.y_valid, y_pred))
        print(confusion_matrix(self.y_valid, y_pred))

        if metric == "accuracy":
            return (
                accuracy_score(self.y_train, y_pred_train),
                accuracy_score(self.y_valid, y_pred),
            )
