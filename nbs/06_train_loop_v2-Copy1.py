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

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %autosave 10

# #%load_ext lab_black

# %%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join("..")))

# %%
import logging

logging.basicConfig(filename="mylog.log", format="%(message)s", level=logging.INFO)
logging.info("-- Starting run")

# %%
from lfp_analysis.data import *
from lfp_analysis.feature_extractor import *
from lfp_analysis.windower import *
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
import h5py
import json

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_hex

# %matplotlib widget

import matplotlib

font = {"size": 9}
matplotlib.rc("font", **font)

import seaborn as sns

WIN_LEN_SEC = 0.750


# %%
def pr(*args):
    print(args)

    logging.info("")
    logging.info(args)
    logging.info("--------")


# %%
data1 = Patient(PatID.ET1).datasets[Task.Pegboard][Stim.OFF].load_1d(decimate_on_import=False)
windower1 = Windower.from_dataset(data1)
data2 = Patient(PatID.ET1).datasets[Task.Pegboard][Stim.ON].load_1d()

# %%
X,y = SignalFeatureExtractor(method='hilbert').get_feats(data1.LFP.data, windower1.df)

# %%
X.head()

# %%

# %%

# %%

# %% [markdown]
# # Prepare Data:

# %% [markdown]
# ### Channel consolidation
#
# - P1: Drop 2nd chan from StimOFF -- All tasks
# - P2: OK -- All tasks
# - P3: OK -- No Pouring
# - P4: OK (L23, R23 zeroed in Pouring OFF -> remove?) -- All tasks
# - P5: Drop dorsal-most channels from Pouring (not present if Posture) -- No Pegboard
# - P6: Drop dorsal-most channels from Pouring (not present if Posture) -- No Pegboard
# - P7: OK -- All tasks
# - P8: OK -- No Pegboard

# %%
data = Patient(7).Posture_off.load_1d()

data.plot()

# %% [markdown]
# # Training within-stim:
#
#
# ## SVM & LDA:

# %%

# %%
within_stim_res = create_pat_task_stim_dict()

# %% jupyter={"outputs_hidden": true} tags=[]
# Cross-Val loop:

for pat_num in pat_nums:
    for task in tasks:
        for stim_cond in stim_conds:
            
            dataset = getattr(Patient(pat_num),f'{task}_{stim_cond.lower()}')
            if dataset is None:
                continue
            else:
                data = dataset.load_1d()

            AUCs, F1s, accs = [], [], []
            for jj in range(5):
                windower = Windower(WIN_LEN_SEC,fold_id=jj).window(data.label)

                cls = BLClassifier(data.LFP, windower.data_df, extract_method='periodogram')
                sc = cls.classify_many(n=2,method='LDA')
                AUCs.append(sc.valid.AUC['mean']); F1s.append(sc.valid.f1['mean']); accs.append(sc.valid.accuracy['mean'])

            AUCs = np.array(AUCs)
            F1s = np.array(F1s)
            accs = np.array(accs)

            res = {
                'AUC':{'folds':list(AUCs),'mean':AUCs.mean(),'std':AUCs.std()},
                'f1':{'folds':list(F1s),'mean':F1s.mean(),'std':F1s.std()},
                'acc':{'folds':list(accs),'mean':accs.mean(),'std':accs.std()}
            }
                   
            within_stim_res[pat_num][task][stim_cond] = res

# %% jupyter={"outputs_hidden": true} tags=[]
within_stim_res

# %%
with open('within_cond_lda.json', 'w') as f:
    json.dump(within_stim_res, f)

# %%

# %%
with open('within_cond_svm.json', 'r') as f:
    svm_res = json.load(f)
    
with open('within_cond_lda.json', 'r') as f:
    lda_res = json.load(f)

# %%
svm_res['1']['Pegboard']['ON']

# %%
lda_res['1']['Pegboard']['ON']

# %%
fig, ax = plt.subplots()

for pat_num in pat_nums:
    for task in tasks:
        for stim_cond in stim_conds:
            lda_val = lda_res[str(pat_num)][task][stim_cond]
            if lda_val is None:
                continue
            lda_val = lda_val['AUC']['mean']
            svm_val = svm_res[str(pat_num)][task][stim_cond]['AUC']['mean']
            if lda_val is not None and svm_val is not None:
                ax.plot([0,1],[svm_val, lda_val],c='k')

# %%
fig, ax = plt.subplots()

for pat_num in pat_nums:
    for task in tasks:
        for stim_cond in stim_conds:
            lda_val = lda_res[str(pat_num)][task][stim_cond]
            if lda_val is None:
                continue
            lda_val = lda_val['f1']['mean']
            svm_val = svm_res[str(pat_num)][task][stim_cond]['f1']['mean']
            if lda_val is not None and svm_val is not None:
                ax.plot([0,1],[svm_val, lda_val],c='k')

# %%
fig, ax = plt.subplots()

for pat_num in pat_nums:
    for task in tasks:
        for stim_cond in stim_conds:
            lda_val = lda_res[str(pat_num)][task][stim_cond]
            if lda_val is None:
                continue
            lda_val = lda_val['acc']['mean']
            svm_val = svm_res[str(pat_num)][task][stim_cond]['acc']['mean']
            if lda_val is not None and svm_val is not None:
                ax.plot([0,1],[svm_val, lda_val],c='k')

# %%

# %%

# %% [markdown]
# ## CNN:

# %%
within_stim_res = create_pat_task_stim_dict()

# %% jupyter={"outputs_hidden": true} tags=[]
# Cross-Val loop:

for pat_num in pat_nums:
    for task in tasks:
        for stim_cond in stim_conds:
            
            dataset = getattr(Patient(pat_num),f'{task}_{stim_cond.lower()}')
            if dataset is None:
                continue
            else:
                data = dataset.load_1d()

            AUCs, F1s, accs = [], [], []
            for jj in range(5):
                windower = Windower(WIN_LEN_SEC,fold_id=jj).window(data.label)

                trainer = Trainer1d(log_wandb=False, layers=[2,2,1],wd=15, silent=True).prepare_dls(data,windower).prepare_learner()
                trainer.train(15, 0.5 * 1e-2)
                sc = trainer.score()

                AUCs.append(sc.valid.AUC); F1s.append(sc.valid.f1); accs.append(sc.valid.accuracy)

            AUCs = np.array(AUCs)
            F1s = np.array(F1s)
            accs = np.array(accs)

            res = {
                'AUC':{'folds':list(AUCs),'mean':AUCs.mean(),'std':AUCs.std()},
                'f1':{'folds':list(F1s),'mean':F1s.mean(),'std':F1s.std()},
                'acc':{'folds':list(accs),'mean':accs.mean(),'std':accs.std()}
            }
                   
            within_stim_res[pat_num][task][stim_cond] = res
            
            pr('Pat', pat_num, task, stim_cond)
            pr(res)

# %% jupyter={"outputs_hidden": true} tags=[]
within_stim_res

# %%
with open('within_cond_cnn.json', 'w') as f:
    json.dump(within_stim_res, f)

# %%

# %% [markdown]
# # Training across stim:
#
# ## LDA & SVM:

# %%
across_stim_res = create_pat_task_stim_dict()

# %% jupyter={"outputs_hidden": true} tags=[]
norm = False
method = "LDA"

for pat_num in pat_nums:
    for task in tasks:
        for stim_cond in stim_conds:

            # Load data:
            dataset = getattr(Patient(pat_num), f"{task}_{stim_cond.lower()}")

            if dataset is None:
                continue

            data1 = dataset.load_1d()

            stim_other = "ON" if stim_cond == "OFF" else "OFF"
            data2 = getattr(Patient(pat_num), f"{task}_{stim_other.lower()}").load_1d()

            # Get feat mats:
            AUCs, F1s, accs = [], [], []
            for jj in range(5):
                windower1 = Windower(WIN_LEN_SEC, fold_id=jj).window(data1.label)
                windower2 = Windower(WIN_LEN_SEC, fold_id=None).window(data2.label)

                cls1 = BLClassifier(data1.LFP, windower1.data_df)
                cls2 = BLClassifier(data2.LFP, windower2.data_df)

                X_train1, y_train1 = cls1.X_train, cls1.y_train
                X_train2, y_train2 = cls2.X_train, cls2.y_train

                X_valid, y_valid = cls1.X_valid, cls1.y_valid

                # Normalize separately:
                if norm:
                    X_train1, X_valid = scale_feat_mats(X_train1, X_valid)
                    X_train2, _ = scale_feat_mats(X_train2, None)

                # Concatenate
                X_train, y_train = np.concatenate((X_train1, X_train2)), np.concatenate(
                    (y_train1, y_train2)
                )

                # Normalize together
                if not norm:
                    X_train, X_valid = scale_feat_mats(X_train, X_valid)

                # Rebalance:
                X_train, y_train = rebalance_data(X_train, y_train)

                if method == "SVM":
                    cls = SVC()
                elif method == "LDA":
                    cls = LinearDiscriminantAnalysis()

                cls.fit(X_train, y_train)

                y_pred = cls.predict(X_valid)
                y_scores = cls.decision_function(X_valid)

                sc = Scorer(ds_type="valid").get_scores(y_valid, y_pred, y_scores)

                AUCs.append(sc.AUC), F1s.append(sc.f1), accs.append(sc.accuracy)

            AUCs, F1s, accs = tuple(list(map(np.array, (AUCs, F1s, accs))))

            res = {
                "AUC": {"folds": list(AUCs), "mean": AUCs.mean(), "std": AUCs.std()},
                "f1": {"folds": list(F1s), "mean": F1s.mean(), "std": F1s.std()},
                "acc": {"folds": list(accs), "mean": accs.mean(), "std": accs.std()},
            }

            across_stim_res[pat_num][task][stim_cond] = res
            pr(res)

# %% jupyter={"outputs_hidden": true} tags=[]
across_stim_res

# %%
with open("across_cond_lda_same_norm.json", "w") as f:
    json.dump(across_stim_res, f)

# %%

# %%

# %%
# sc.valid.AUC

# %%
data = Patient(1).Pegboard_on.load_1d()
jj = 4
windower = Windower(WIN_LEN_SEC,fold_id=jj).window(data.label)

# %%
trainer = Trainer1d(log_wandb=False, layers=[2,2,1],wd=15).prepare_dls(data,windower).prepare_learner()

# %%
trainer.train(15, 0.5 * 1e-2)

# %%
cnn_1d_scores = trainer.score()
cnn_1d_scores

# %%

# %%

# %%
data.plot()

# %%

for jj in range(5):
data1 = Patient(7).Pegboard_on.load_1d()
data1.zscore();

data2 = Patient(7).Pegboard_off.load_1d()
data2 = data2.zscore()

windower1 = Windower(WIN_LEN_SEC,fold_id=jj).window(data1.label)
windower2 = Windower(WIN_LEN_SEC,fold_id=None).window(data2.label)
    
end_samp = windower1.data_df.id_end.iloc[-1]
windower2.data_df.id_start += end_samp
windower2.data_df.id_end += end_samp
    
data = data1.concatenate(data2)
df_data = pd.concat([windower1.data_df, windower2.data_df])


windower1 = Windower(WIN_LEN_SEC,fold_id=0,n_folds=500).window(data1.label)
cls = BLClassifier(data1.LFP, windower1.data_df, extract_method='periodogram')


# %%
cls.y_train

# %%
sc = StandardScaler().fit(cls.X_train)

X_train_sc = sc.transform(cls.X_train)

# %%
X_train_sc.shape

# %%
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(X_train_sc)

# %%
X_embedded

# %%
fig,ax = plt.subplots()

ax.scatter(X_embedded[:,0],X_embedded[:,1],c=cls.y_train)

# %%

# %%
# Cross-Val loop across stim cond:

norm = False

AUCs = []
for jj in range(5):
    
    data1 = Patient(7).Pegboard_off.load_1d()
    data2 = Patient(7).Pegboard_on.load_1d()
    
    #data1 = data1.drop((2,5))
    #data2 = data2.drop((2,5))
    
    if norm:
        data1 = data1.zscore()
        data2 = data2.zscore()

    windower1 = Windower(WIN_LEN_SEC,fold_id=0,n_folds=600).window(data1.label)
    windower2 = Windower(WIN_LEN_SEC,fold_id=0,n_folds=600).window(data2.label)
    
    #end_samp = windower1.data_df.id_end.iloc[-1]
    #windower2.data_df.id_start += end_samp
    #windower2.data_df.id_end += end_samp
    
    data = data1.concatenate(data2)
    df_data = pd.concat([windower1.data_df, windower2.data_df])
    
    cls1 = BLClassifier(data1.LFP, windower1.df_data, extract_method='periodogram')
    cls2 = BLClassifier(data2.LFP, windower2.df_data, extract_method='periodogram')
    
    sc = cls.classify_many(n=2)
    AUCs.append(sc.valid.AUC['mean'])
    
AUCs = np.array(AUCs)

AUCs, AUCs.mean(), AUCs.std()

# %%
windower1.data_df[windower1.data_df["is_valid"]==False]

# %%
windower2.data_df[windower2.data_df["is_valid"]==False]

# %%

# %%
idx = np.concatenate((np.zeros((628,)),np.ones((524,))))
idx.shape

# %%
sc = StandardScaler().fit(cls.X_train)
X_train_sc = sc.transform(cls.X_train)

X_embedded = TSNE(n_components=2).fit_transform(X_train_sc)

fig,ax = plt.subplots()

ax.scatter(X_embedded[:628,0],X_embedded[:628,1],label='OFF data')
ax.scatter(X_embedded[628:,0],X_embedded[628:,1],label='ON data')
ax.legend()

# %% tags=[]
AUCs = []

for jj in range(5):

    data1 = Patient(7).Pegboard_on.load_1d()
    data2 = Patient(7).Pegboard_off.load_1d()

    #data1 = data1.drop((2,5))
    #data2 = data2.drop((2,5))

    norm = True
    if norm:
        data1 = data1.zscore()
        data2 = data2.zscore()

    windower1 = Windower(WIN_LEN_SEC,fold_id=0,n_folds=600).window(data1.label)
    windower2 = Windower(WIN_LEN_SEC,fold_id=jj).window(data2.label)

    cls1 = BLClassifier(data1.LFP, windower1.data_df, extract_method='periodogram')
    cls2 = BLClassifier(data2.LFP, windower2.data_df, extract_method='periodogram')
    
    X_train_sc1, X_valid_sc1 = scale_feat_mats(cls1.X_train,cls1.X_valid)
    X_train_sc2, X_valid_sc2 = scale_feat_mats(cls2.X_train,cls2.X_valid)
    
    X_train = np.concatenate((X_train_sc1,X_train_sc2))
    y_train = np.concatenate((cls1.y_train,cls2.y_train))
    
    X_valid = X_valid_sc2
    y_valid = cls2.y_valid
    
    cls = SVC().fit(X_train,y_train)
    y_score = cls.decision_function(X_valid_sc2)
    y_pred = cls.predict(X_valid_sc2)

    AUCs.append(Scorer(ds_type='valid').get_scores(y_valid,y_pred,y_score).AUC)


# %%
np.mean(AUCs)

# %%
sc1 = StandardScaler().fit(cls1.X_train)
sc2 = StandardScaler().fit(cls2.X_train)

X_train_sc1 = sc1.transform(cls1.X_train)
X_train_sc2 = sc2.transform(cls2.X_train)


# %%

# %%
X_embedded = TSNE(n_components=2).fit_transform(np.concatenate((X_train_sc1,X_train_sc2)))

# %%
idx = np.concatenate((np.zeros((X_train_sc1.shape[0],)),np.ones((X_train_sc2.shape[0],))))

fig,ax = plt.subplots()

ax.scatter(X_embedded[:X_train_sc1.shape[0],0],X_embedded[:X_train_sc1.shape[0],1],label='OFF data',alpha=1)
ax.scatter(X_embedded[X_train_sc1.shape[0]:,0],X_embedded[X_train_sc1.shape[0]:,1],label='ON data',alpha=1)
ax.legend()

# %%
plt.close('all')

# %%

# %%

# %%

# %%

# %%

# %%
# Cross-Val loop across stim cond:

norm = True

AUCs = []
for jj in range(5):
    
    data1 = Patient(7).Pegboard_on.load_1d()
    data2 = Patient(7).Pegboard_off.load_1d()
    
    #data1 = data1.drop((2,5))
    #data2 = data2.drop((2,5))
    
    if norm:
        data1 = data1.zscore()
        data2 = data2.zscore()

    windower1 = Windower(WIN_LEN_SEC,fold_id=jj).window(data1.label)
    windower2 = Windower(WIN_LEN_SEC,fold_id=None).window(data2.label)
    
    end_samp = windower1.data_df.id_end.iloc[-1]
    windower2.data_df.id_start += end_samp
    windower2.data_df.id_end += end_samp
    
    data = data1.concatenate(data2)
    df_data = pd.concat([windower1.data_df, windower2.data_df])
    
    cls = BLClassifier(data.LFP, df_data, extract_method='periodogram')
    sc = cls.classify_many(n=2)
    AUCs.append(sc.valid.AUC['mean'])
    
AUCs = np.array(AUCs)

AUCs, AUCs.mean(), AUCs.std()

# %%
data.plot()

# %%
windower.data_df

# %%

# %%
data1 = data1.drop(1)
data3 = data3.drop(1)
data5 = data5.drop(1)

# %%
data1, data2, data3, data4, data5, data6

# %%
data1.plot()

# %%
cls1 = BLClassifier(data1.LFP, windower1.data_df, extract_method='periodogram')
cls1.classify_many()

# %%
data2.plot()

# %%
cls2 = BLClassifier(data2.LFP, windower2.data_df, extract_method='periodogram')
cls2.classify_many()

# %%
data3.plot()

# %%
cls3 = BLClassifier(data3.LFP, windower3.data_df, extract_method='periodogram')
cls3.classify_many()

# %%
data4.plot()

# %%
cls4 = BLClassifier(data4.LFP, windower4.data_df, extract_method='periodogram')
cls4.classify_many()

# %%
data5.plot()

# %%
cls5 = BLClassifier(data5.LFP, windower5.data_df, extract_method='periodogram')
cls5.classify_many()

# %%
data6.plot()

# %%
cls6 = BLClassifier(data6.LFP, windower6.data_df, extract_method='periodogram')
cls6.classify_many()

# %%

# %%
data_concat1 = data6.concatenate(data5)
windower_concat1 = Windower(WIN_LEN_SEC).window(data_concat1.label)

# %%
data_concat1

# %%
data_concat1.plot()

# %%
cls_concat1 = BLClassifier(data_concat1.LFP, windower_concat1.data_df, extract_method='periodogram')
cls_concat1.classify_many()

# %%
data_concat1.plot()

# %%
fig, ax = plt.subplots(1,2,figsize=(8,6),gridspec_kw={'wspace':.3})

res_off = np.array([[.77,.7,.76],[.79,.79,.8],[.74,.73,.76],[.96,.97,.97],[.97,.97,.98],[.78,.81,.82],[.85,.82,.85],[.74,.68,.75],[.93,.91,.93],[.8,.79,np.nan],[.83,.84,.87],[.92,.92,.93],[.8,.81,.81],[.88,.89,.90],[.76,.80,.78],[.79,.83,.82],[.61,.69,.67],[.71,.75,.81],[.93,.92,.94],[.84,.85,.87]])

res_on = np.array([[.55,.62,.60],[.8,.79,.76],[.55,.58,.60],[.94,.94,.96],[.95,.90,.92],[.5,.5,.63],[.73,.69,.60],[.85,.68,.81],[.58,.58,.53],[.7,.7,np.nan],[.51,.49,.54],[.95,.94,.93],[.87,.78,.68],[.84,.86,.87],[.61,.65,.69],[.85,.66,.67],[.81,.66,.73],[.53,.49,.56],[.95,.86,.89],[.83,.58,.79]])



# OFF:
ax[0].plot([0,1],res_off[:,[0,1]].T,c='C0')
ax[0].scatter(np.tile(np.array([0,1]),res_off[:,[0,1]].shape[0]),res_off[:,[0,1]].flatten(),c='C0',label='single norm')

ax[0].plot([0,1],res_off[:,[0,2]].T,c='C1')
ax[0].scatter(np.tile(np.array([0,1]),res_off[:,[0,2]].shape[0]),res_off[:,[0,2]].flatten(),c='C1',label='condition-specific norm')

ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['trained \nwithin-condition', 'trained \ncross-condition'])
ax[0].set_ylabel('AUC')
ax[0].legend()
ax[0].set_title('OFF Stim')



# ON:
ax[1].plot([0,1],res_on[:,[0,1]].T,c='C0')
ax[1].scatter(np.tile(np.array([0,1]),res_on[:,[0,1]].shape[0]),res_on[:,[0,1]].flatten(),c='C0',label='single norm')

ax[1].plot([0,1],res_on[:,[0,2]].T,c='C1')
ax[1].scatter(np.tile(np.array([0,1]),res_on[:,[0,2]].shape[0]),res_on[:,[0,2]].flatten(),c='C1',label='condition-specific norm')

ax[1].set_xticks([0,1])
ax[1].set_xticklabels(['trained \nwithin-condition', 'trained \ncross-condition'])
ax[1].legend()
ax[1].set_title('ON Stim')


# %%
x.shape

# %%
y.shape

# %%
fig, ax = plt.subplots(1,2,figsize=(8,6),gridspec_kw={'wspace':.3},sharey=True)

x = np.concatenate((np.zeros(res_off.shape[0]),np.ones(res_off.shape[0])))
y = np.concatenate((res_off[:,1]-res_off[:,0],res_off[:,2]-res_off[:,0]))

sns.violinplot(x=x,y=y,ax=ax[0])
ax[0].set_xticklabels(['single norm','condition-specific norm'])
ax[0].set_ylabel(r'$\Delta$ AUC')
ax[0].set_title('Stim OFF')

x = np.concatenate((np.zeros(res_on.shape[0]),np.ones(res_on.shape[0])))
y = np.concatenate((res_on[:,1]-res_off[:,0],res_on[:,2]-res_on[:,0]))

sns.violinplot(x=x,y=y,ax=ax[1])
ax[1].set_xticklabels(['single norm','condition-specific norm'])
ax[1].set_title('Stim ON')


# %%
plt.close('all')

# %%
fig, ax = plt.subplots()

res = np.array([[0.63,0.66],[0.78,0.77],[.66,.68],[.96,.96],[.94,.95],[.69,.74]])

ax.plot([0,1],res.T,c='C0')
ax.scatter(np.tile(np.array([0,1]),res.shape[0]),res.flatten())

ax.set_xticks([0,1])
ax.set_xticklabels(['pooled stats', 'per-condition stats'])
ax.set_ylabel('AUC')
ax.set_ylim([0.5,1])

# %%

# %%

# %%

# %%

# %%

# %%
trainer = Trainer1d(log_wandb=False, layers=[2,2,1],wd=15).prepare_dls(data6,windower6).prepare_learner()

# %%

# %%
trainer.train(120, 0.5 * 1e-2)

# %%
cnn_1d_scores = trainer.score()
cnn_1d_scores

# %%
trainer2 = Trainer1d(log_wandb=False, layers=[2,2],wd=50, experiment='concat').prepare_dls(data2,windower2).prepare_learner()
trainer.learn.dls = trainer2.dls

# %%
trainer.learn.lr_find()

# %%
trainer.train(30,slice(1e-5,0.5*1e-2))

# %%
cnn_1d_scores = trainer.score()
cnn_1d_scores

# %%
trainer.train(30,0.1*1e-2)

# %%

# %% [raw]
#         if data1 is None or data2 is None:
#             continue
#         else:
#             
#             data1 = data1.load_1d()
#             data2 = data2.load_1d()
#             
#             if data1.n_chan != data2.n_chan:
#                 continue
#             else:
#                 windower1 = Windower(WIN_LEN_SEC).window(data1.label)
#                 windower2 = Windower(WIN_LEN_SEC).window(data2.label)
#                 
#                 cls1 = BLClassifier(data1.LFP, windower1.data_df, extract_method='periodogram')
#                 cls1.classify_many()
#                 cls2 = BLClassifier(data2.LFP, windower2.data_df, extract_method='periodogram')
#                 cls2.classify_many()
#                 
#                 ret = cls1.cross_task(cls2)
#                 
#                 AUCs.append((ii,jj,ret.AUC))

# %%
import itertools

# %% jupyter={"outputs_hidden": true} tags=[]
WIN_LEN_SEC = 0.750

task_train_cont = [["Pouring","Posture"],["Pegboard","Posture"],["Pegboard","Pouring"]]
task_valid_cont = ["Pegboard","Pouring","Posture"]
#task_combs = list(itertools.product(["Pegboard","Pouring","Posture"],["Pegboard","Pouring","Posture"]))
#tasks = ["Pegboard","Pouring","Posture"]

res_cont = []
for ii,pat_id in enumerate(range(1,9)):
    for jj,(task_train,task_valid) in enumerate(zip(task_train_cont,task_valid_cont)):
        
        data1 = getattr(Patient(pat_id),task_train[0]+"_off")
        data2 = getattr(Patient(pat_id),task_train[1]+"_off")
        data_valid = getattr(Patient(pat_id),task_valid+"_off")
        
        if data_valid is None: 
            continue
        else:
            data_valid.load_1d()
            windower_valid = Windower(WIN_LEN_SEC).window(data_valid.label)
            data_concat = None
            if data1 is not None and data1.load_1d().n_chan == data_valid.n_chan:
                data_concat = data1
            if data2 is not None and data2.load_1d().n_chan == data_valid.n_chan:
                if data_concat is not None:
                    data_concat = data_concat.concatenate(data2)
                else:
                    data_concat = data2
            
            if data_concat is None:
                continue
            
            
            # SVMs:
            data_concat_all = data_concat.concatenate(data_valid)
            windower_all = Windower(WIN_LEN_SEC).window(data_concat_all.label)
            
            svm_scores_valid = BLClassifier(data_valid.LFP, windower_valid.data_df, extract_method='periodogram').classify_many()
            svm_scores_all = BLClassifier(data_concat_all.LFP, windower_all.data_df, extract_method='periodogram').classify_many()
            
            # CNNs:
            
            trainer_valid = Trainer1d(log_wandb=False, layers=[2],wd=30, experiment='concat').prepare_dls(data_valid,windower_valid).prepare_learner()
            trainer_valid.train(100,0.5*1e-2)
            cnn_scores_valid = trainer_valid.score()
            
            windower_concat = Windower(WIN_LEN_SEC).window(data_concat.label)
            trainer_concat = Trainer1d(log_wandb=False, layers=[2],wd=30, experiment='concat').prepare_dls(data_concat,windower_concat).prepare_learner()
            trainer_concat.train(100,0.5*1e-2)
            
            trainer_concat.learn.dls = trainer_valid.dls
            trainer_concat.train(70,slice(1e-6,0.1*1e-2))
            cnn_scores_concat = trainer_valid.score()
            
            res_cont.append((ii,jj,svm_scores_valid.valid.AUC['mean'],svm_scores_all.valid.AUC['mean'],cnn_scores_valid.valid.AUC, cnn_scores_concat.valid.AUC))



# %%
AUCs

# %%
df = pd.DataFrame(AUCs)
df[df[0]==2]

# %%
task_combs

# %%
cont = np.zeros((8*3,8*3))
cont;

# %%
cont[0,0] = 

# %%
temp.shape

# %%
temp = np.zeros((8,9))

for jj,row in df.iterrows():
    temp[int(row[0]),int(row[1])] = row[2]

# %%
for ii,el in enumerate(temp):
    el = el.reshape((3,3))
    
    cont[ii*3:(ii+1)*3,ii*3:(ii+1)*3] = el
    


# %%
df_heat = pd.DataFrame(cont)

# %%
df_heat

# %%
fig, ax = plt.subplots(figsize=(10,7))
mask=df_heat.values==0
sns.heatmap(df_heat, mask=mask, vmax=1.2,vmin=0.4,cmap="rocket",ax=ax, cbar_kws={'label': 'AUC'})

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("SVM Task Cross-Validation")

# %%
task_combs = list(itertools.product(["Pegboard","Pouring","Posture"],["Pegboard","Pouring","Posture"]))

# %%

# %%

# %%
EXPERIMENT = "train_all_v3"

WIN_LEN_SEC = 0.750

# %% tags=[]
AUCs, Accs = [], []
durations = []
for pat_id in range(1,9):
    for task in ["Pegboard","Pouring","Posture"]:
        data1 = getattr(Patient(pat_id),task+"_off")
        if data1 is None:
            continue
        else:
            data1 = data1.load_1d()
            
            durations.append(data1.LFP.shape[-1]/2048.0/60)
            #rep = Reporter.from_pickle(pat_id, task, EXPERIMENT)
            #tasks.append(f"{rep.dataset.pat_id}/{rep.dataset.task}")
            #AUCs.append((rep.svm.valid.AUC['mean'],rep.beta.valid.AUC['mean'],rep.theta.valid.AUC['mean'],rep.cnn_1d.valid.AUC,rep.cnn_2d.valid.AUC))
            #Accs.append((rep.svm.valid.accuracy['mean'],rep.beta.valid.accuracy['mean'],rep.theta.valid.accuracy['mean'],rep.cnn_1d.valid.accuracy,rep.cnn_2d.valid.accuracy))


# %%
durations

# %%
np.mean(durations)

# %%
fig,ax = plt.subplots()

ax.hist(durations, bins=10)
ax.set_xlabel('Duration [min]')
ax.set_ylabel('# of occurrences')

# %%
python

# %%
tasks

# %%
aucs = pd.DataFrame(AUCs,columns=["svm","beta","theta","cnn_1d","cnn_2d"])
accs = pd.DataFrame(Accs,columns=["svm","beta","theta","cnn_1d","cnn_2d"])

# %%
aucs = aucs[["svm","cnn_1d","cnn_2d","beta","theta"]]
aucs.index =tasks
accs = accs[["svm","cnn_1d","cnn_2d","beta","theta"]]
accs.index = tasks


# %%
aucs

# %%
import matplotlib
font = {'size'   :14}
matplotlib.rc('font', **font)

fig,ax = plt.subplots(figsize=(10,7))
sns.violinplot(x=np.repeat([0,1,2,3,4],20),y=aucs.values.T.flatten(),inner='point',ax=ax)
ax.plot(aucs.values.T,c='k',linewidth=0.2)
ax.set_xticklabels(["SVM","1D-CNN","2D-CNN","b-SVM","th-SVM"]);

ax.set_ylabel("AUCs")
ax.set_title("AUCs")

# %%
fig,ax = plt.subplots(figsize=(10,7))
sns.violinplot(x=np.repeat([0,1,2,3,4],20),y=accs.values.T.flatten(),inner='point',ax=ax)
ax.plot(list(range(0,5)),accs.values.T,c='k',linewidth=0.2)
ax.set_xticklabels(["SVM","1D-CNN","2D-CNN","b-SVM","th-SVM"]);
ax.set_ylabel("Accuracy")
ax.set_title("Accuracies")

# %%
aucs[aucs["svm"]-aucs["cnn_1d"]>0.02]

# %%
accs[accs["svm"]-accs["cnn_1d"]>0.02]

# %%
accs[accs["svm"]-accs["cnn_1d"]<-0.06]

# %%
from scipy.stats import ttest_rel

# %%
ttest_rel(accs['svm'].values,accs['cnn_1d'].values,alternative='less')

# %%
ttest_rel(aucs['svm'].values,aucs['cnn_1d'].values,alternative='less')

# %%
np.mean(accs['svm'].values < accs['cnn_1d'].values)

# %%

# %%

# %%
Reporter.from_pickle(4, "Posture", EXPERIMENT).svm.svm_coef

# %%
fig, ax = plt.subplots(1,3,figsize=(12,6),)

coefs_pegboard, coefs_pouring, coefs_posture = [],[],[]

for pat_id in range(1,9):
    coefs_pegboard, coefs_pouring, coefs_posture = None, None, None
    for task in ["Pegboard","Pouring","Posture"]:
        data1 = getattr(Patient(pat_id),task+"_off")
        if data1 is None:
            continue
        else:
            if pat_id == 4:
                continue
            rep = Reporter.from_pickle(pat_id, task, EXPERIMENT)
            if task == "Pegboard":
                coefs_pegboard=[rep.svm.svm_coef['mean'].values,rep.beta.svm_coef['mean'].values,rep.theta.svm_coef['mean'].values]
                
            if task == "Pouring":
                coefs_pouring = [rep.svm.svm_coef['mean'].values,rep.beta.svm_coef['mean'].values,rep.theta.svm_coef['mean'].values]
                
            if task == "Posture":
                coefs_posture = [rep.svm.svm_coef['mean'].values,rep.beta.svm_coef['mean'].values,rep.theta.svm_coef['mean'].values]
    
    if (coefs_pegboard is not None) and (coefs_pouring is not None):
        ax[0].scatter(coefs_pegboard[0],coefs_pouring[0],marker='o',c='C0',s=5, )
        ax[0].scatter(coefs_pegboard[1],coefs_pouring[1],marker='o',c='C3',s=5, )
        ax[0].scatter(coefs_pegboard[2],coefs_pouring[2],marker='o',c='C4',s=5, )

        
    if (coefs_pegboard is not None) and (coefs_posture is not None):
        ax[1].scatter(coefs_pegboard[0],coefs_posture[0],marker='o',c='C0',s=5)
        ax[1].scatter(coefs_pegboard[1],coefs_posture[1],marker='o',c='C3',s=5)
        ax[1].scatter(coefs_pegboard[2],coefs_posture[2],marker='o',c='C4',s=5)
        
    if (coefs_pouring is not None) and (coefs_posture is not None):
        el1=ax[2].scatter(coefs_pouring[0],coefs_posture[0],marker='o',c='C0',s=5,label="SVM")
        el2=ax[2].scatter(coefs_pouring[1],coefs_posture[1],marker='o',c='C3',s=5,label="b-SVM")
        el3=ax[2].scatter(coefs_pouring[2],coefs_posture[2],marker='o',c='C4',s=5,label="th-SVM")

        
ax[0].set_xlabel('Pegboard'); ax[0].set_ylabel('Pouring'); 
ax[1].set_xlabel('Pegboard'); ax[1].set_ylabel('Posture'); 
ax[2].set_xlabel('Pouring'); ax[2].set_ylabel('Posture'); 

ax[0].plot((-1,3),(-1,3),'k--')
ax[1].plot((-1,3),(-1,3),'k--')
ax[2].plot((-1,3),(-1,3),'k--')

for ii in range(3):
    ax[ii].axhline(color='k',linewidth=0.2)
    ax[ii].axvline(color='k',linewidth=0.2)
    ax[ii].set_ylim((-2, 3.5))
    ax[ii].set_xlim((-1.5, 4.5))

ax[2].legend([el1,el2,el3],["SVM","b-SVM","th-SVM"])

plt.tight_layout()

# %%
plt.close('all')

# %%

# %%
