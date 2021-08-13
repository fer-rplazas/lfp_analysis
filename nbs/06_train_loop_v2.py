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

# %%
# %load_ext autoreload
# %autoreload 2

# %autosave 10

# #%load_ext lab_black

# %%
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..")))

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
import h5py

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_hex
# %matplotlib widget

import matplotlib
font = {'size'   :9}
matplotlib.rc('font', **font)

import seaborn as sns
WIN_LEN_SEC = 0.750

# %%
EXPERIMENT = "train_all_v3"
corr_coefs = []
for pat_id in range(1,9):
    for task in ["Pegboard","Pouring","Posture"]:
        data1 = getattr(Patient(pat_id),task+"_off")
        if data1 is None:
            continue
        else:
            data1 = data1.load_1d()
            windower1 = Windower(WIN_LEN_SEC).window(data1.label)
            cls1 = BLClassifier(data1.LFP, windower1.data_df, extract_method='periodogram')
            cls1.classify()
            
            gamma_chan_idx = np.where(
                [("31_45" in el) or ("46_60Hz" in el) or ("61_75" in el) or ("76_100" in el) for el in cls1.X_train.columns]
            )[0]
            
            gammas = cls1.X_train.iloc[:,gamma_chan_idx].values
            
            gammas_local = []
            for ii in range(gammas.shape[1]):
                gammas_local.append(np.corrcoef(gammas[:,ii],windower1.data_df[windower1.data_df['is_valid']==False]['label'])[0,1])
            
        
            rep = Reporter.from_pickle(pat_id, task, EXPERIMENT)
            
            corr_coefs.append([gammas_local,rep.svm.valid.AUC['mean']])
            #tasks.append(f"{rep.dataset.pat_id}/{rep.dataset.task}")
            #AUCs.append((rep.svm.valid.AUC['mean'],rep.beta.valid.AUC['mean'],rep.theta.valid.AUC['mean'],rep.cnn_1d.valid.AUC,rep.cnn_2d.valid.AUC))
            #Accs.append((rep.svm.valid.accuracy['mean'],rep.beta.valid.accuracy['mean'],rep.theta.valid.accuracy['mean'],rep.cnn_1d.valid.accuracy,rep.cnn_2d.valid.accuracy))
            

# %%
[[np.corrcoef(a) for el in all_data]for all_data,perfs in zip(corr_coefs,corr_coefs[1])]

# %%
rec = Reporter.from_pickle(1,'Posture',"train_all_v3")

# %%
rec.cnn_1d

# %%
data1.LFP.shape

# %%
data1 = Patient(7).Posture_off.load_1d()
windower1 = Windower(WIN_LEN_SEC).window(data1.label)


# %%
cls1 = BLClassifier(data1.LFP, windower1.data_df, extract_method='periodogram')
cls1.classify_many()

# %%

# %%
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

X_train_raw = cls1.X_train
X_valid_raw = cls1.X_valid

sc = StandardScaler().fit(X_train_raw)
X_train = sc.transform(X_train_raw)
X_valid = sc.transform(X_valid_raw)

forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, cls1.y_train)

svc = SVC().fit(X_train,cls1.y_train)
svc.score(X_valid,cls1.y_valid), forest.score(X_valid,cls1.y_valid)

# %%

# %%
importances = forest.feature_importances_
std = np.std([
    tree.feature_importances_ for tree in forest.estimators_], axis=0)

# %%
forest_importances = pd.Series(importances, index=X_train_raw.columns)

# %%
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# %%
svc.score(X_valid,cls1.y_valid)

# %%
cls1.X_valid

# %%
fig,ax = plt.subplots(3,1,sharex=True,figsize=(14,5))

#ax[0].plot(windower1.data_df['t'],np.concatenate([cls1.X_train["pow0_61_75Hz"].values, cls1.X_valid["pow0_61_75Hz"].values]))
ax[0].plot(windower1.data_df['t'],StandardScaler().fit_transform(np.concatenate([cls1.X_train["pow2_13_20Hz"].values, cls1.X_valid["pow2_13_20Hz"].values]).reshape(-1,1)))
ax[0].plot(windower1.data_df['t'],StandardScaler().fit_transform(np.concatenate([cls1.X_train["pow0_21_30Hz"].values, cls1.X_valid["pow2_21_30Hz"].values]).reshape(-1,1)))

ax[0].plot(windower1.data_df['t'],windower1.data_df['label'].values*4)
ax[-1].set_xlabel("Time [s]")
ax[0].set_title("Power in 46 - 60 Hz band")

ax[1].plot(windower1.data_df['t'],StandardScaler().fit_transform(np.concatenate([cls1.X_train["pow1_46_60Hz"].values, cls1.X_valid["pow1_46_60Hz"].values]).reshape(-1,1)))
ax[2].plot(windower1.data_df['t'],StandardScaler().fit_transform(np.concatenate([cls1.X_train["pow2_46_60Hz"].values, cls1.X_valid["pow2_46_60Hz"].values]).reshape(-1,1)))
ax[2].plot(windower1.data_df['t'],StandardScaler().fit_transform(np.concatenate([cls1.X_train["pow3_46_60Hz"].values, cls1.X_valid["pow3_46_60Hz"].values]).reshape(-1,1)))

#ax[2].plot(windower1.data_df['t'],np.concatenate([cls1.X_train["pow0_31_45Hz"].values, cls1.X_valid["pow0_31_45Hz"].values]))
#ax[2].plot(windower1.data_df['t'],np.concatenate([cls1.X_train["pow0_61_75Hz"].values, cls1.X_valid["pow0_61_75Hz"].values]))


#ax[2].plot(windower1.data_df['t'],np.concatenate([cls1.X_train["pow0_13_20Hz"].values, cls1.X_valid["pow0_13_20Hz"].values]))

#ax[3].plot(data1.t_decim(),data1.LFP_decim.T[:,1])

# %%

# %%

# %%
data1 = Patient(1).Pegboard_off.load_1d()
data2 = Patient(1).Pegboard_on.load_1d()
data3 = Patient(8).Pouring_off.load_1d()

windower1 = Windower(WIN_LEN_SEC).window(data1.label)
windower2 = Windower(WIN_LEN_SEC).window(data2.label)
windower3 = Windower(WIN_LEN_SEC).window(data3.label)

# %%
data1, data2

# %%
cls1 = BLClassifier(data1.LFP, windower1.data_df, extract_method='periodogram')
cls1.classify_many()

# %%
cls2 = BLClassifier(data2.LFP, windower2.data_df, extract_method='periodogram')
cls2.classify_many()

# %% [raw]
# cls3 = BLClassifier(data3.LFP, windower3.data_df, extract_method='periodogram')
# cls3.classify_many()

# %%

# %%

# %%

# %%
data_concat = data1.concatenate(data2)
windower_concat = Windower(WIN_LEN_SEC).window(data_concat.label)

# %%
data_concat

# %%
cls_concat = BLClassifier(data_concat.LFP, windower_concat.data_df, extract_method='periodogram')
cls_concat.classify_many()

# %%

# %%

# %%

# %%

# %%

# %%
trainer = Trainer1d(log_wandb=False, layers=[2],wd=50, experiment='concat').prepare_dls(data1,windower1).prepare_learner()

# %%
ResNet2d(4,2,[2,2,1])

# %%
trainer.learn.summary()

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
