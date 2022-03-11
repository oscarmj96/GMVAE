


import pickle
from scipy.io import arff
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from sklearn.cluster import AgglomerativeClustering
from model.GMVAE import *
import util
import scipy.io as sio
from scipy.fftpack import fft, dct, idct
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.stats import entropy

labels_mnist = np.load('data_MNIST/labels_pred_varx_0.1_K_10.npy')


def print_metrics(pred_label, true_label, n_clusters, features, model = 'KMEANS', name = 'time'):

    if model == 'KMEANS':
        #pred = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)
        if pred_label == 0:
            #pred = kmeans.predict(features)
            pred = AgglomerativeClustering(n_clusters=n_clusters, ).fit_predict(features)
    else:
        pred = pred_label

    ri = metrics.rand_score(true_label, pred)
    ari = metrics.adjusted_rand_score(true_label, pred)
    ami = metrics.adjusted_mutual_info_score(true_label, pred)
    homog = metrics.homogeneity_score(true_label, pred)
    complet = metrics.completeness_score(true_label, pred)
    silhouette = metrics.silhouette_score(features, true_label, metric='euclidean')

    print(model+': '+name)
    print('---------------------')
    print('Rand Index: ', ri)
    print('Adjusted Rand Index: ', ari)
    print('Adjusted Mutual Info: ', ami)
    print('Homogeneity: ', homog)
    print('Completeness: ', complet)
    print('Silhouette: ', silhouette)
    print('---------------------')
    print('\n')

def path_builder(dataset, cte_psd, transitions, K, var_x, version):
    str_dataset = 'data_'+dataset
    str_K = 'K_'+str(K)
    str_var_x = 'varx_'+str(var_x)
    str_version = 'version_'+str(version)
    if cte_psd:
        str_psd = 'cte_psd_'
    else:
        str_psd = ''

    if transitions:
        str_trans = 'transitions_'
    else:
        str_trans = ''

    path_features = str_dataset+'/features_'+str_psd+str_trans+ str_var_x+'_'+str_K+'_'+str_version+'.npy'
    path_labels = str_dataset + '/labels_pred_' + str_psd + str_trans + str_var_x + '_' + str_K + '_' + str_version + '.npy'

    return path_features, path_labels





'HAR_ext'
X_data_ext = pd.read_csv('Train_HAR_ext/X_train.txt', sep = ' ', header=None)
Y_data_ext = pd.read_csv('Train_HAR_ext/y_train.txt', header=None)

X = X_data_ext.values
labels = Y_data_ext.values.reshape((-1,))

X_trans_time_ext = X
labels_trans_time_ext = labels

mask = np.where(labels <=6,True, False)
X_time_ext = X[mask,:]
labels_time_ext = labels[mask]

X_dct_trans_ext = np.ones_like(X_trans_time_ext)
for i in range(X_trans_time_ext.shape[0]):
    X_dct_trans_ext[i, :] = dct(X_trans_time_ext[i, :], 1)

X_dct_ext = np.ones_like(X_time_ext)
for i in range(X_time_ext.shape[0]):
    X_dct_ext[i, :] = dct(X_time_ext[i, :], 1)

'HAR'
data = sio.loadmat('database2Sens_Preproc.mat')['dataPreproc']

X_time_trans, labels_time_trans, _, _ = util.database_creator_unsupervised(data, 8)
X_time, Y_time, labels_time, _, _, _ = util.database_creator(data, 8)

X_dct_trans = np.ones_like(X_time_trans)
for i in range(X_time_trans.shape[0]):
    X_dct_trans[i, :] = dct(X_time_trans[i, :], 1)

X_dct = np.ones_like(X_time)
for i in range(X_time.shape[0]):
    X_dct[i, :] = dct(X_time[i, :], 1)



#  HAR Time and DCT
print('HAR Time and DCT')
print('\n')

print_metrics(pred_label = 0, true_label = labels_time , n_clusters = 5, features = X_time, model = 'KMEANS', name = 'HAR_time_raw')
print_metrics(pred_label = 0, true_label = labels_time , n_clusters = 5, features = X_dct, model = 'KMEANS', name = 'HAR_dct_raw')
print_metrics(pred_label = 0, true_label = labels_time_trans , n_clusters = 10, features = X_time_trans, model = 'KMEANS', name = 'HAR_time_raw (transitions)')
print_metrics(pred_label = 0, true_label = labels_time_trans , n_clusters = 10, features = X_dct_trans, model = 'KMEANS', name = 'HAR_dct_raw (transitions)')



#  HAR_ext Time and DCT
print('HAR_ext Time and DCT')
print('\n')

print_metrics(pred_label = 0, true_label = labels_time_ext , n_clusters = 6, features = X_time_ext, model = 'KMEANS', name = 'HAR_ext_time_raw')
print_metrics(pred_label = 0, true_label = labels_time_ext , n_clusters = 6, features = X_dct_ext, model = 'KMEANS', name = 'HAR_ext_dct_raw')
print_metrics(pred_label = 0, true_label = labels_trans_time_ext , n_clusters = 12, features = X_trans_time_ext, model = 'KMEANS', name = 'HAR_ext_time_raw (transitions)')
print_metrics(pred_label = 0, true_label = labels_trans_time_ext , n_clusters = 12, features = X_dct_trans_ext, model = 'KMEANS', name = 'HAR_ext_dct_raw (transitions)')


# HAR GMVAE

print('HAR GMVAE  comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = False, K = 5, var_x = 0.1, version = 1)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time , n_clusters = 5, features = features, model = 'KMEANS', name = 'HAR_GMVAE_version 1')
path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = False, K = 5, var_x = 0.1, version = 2)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time , n_clusters = 5, features = features, model = 'KMEANS', name = 'HAR_GMVAE version 2')



# HAR_ext GMVAE

print('HAR_ext GMVAE  comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR_ext', cte_psd = False, transitions = False, K = 6, var_x = 0.1, version = 1)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time_ext , n_clusters = 6, features = features, model = 'KMEANS', name = 'HAR_ext_GMVAE_version 1')
path_features, path_labels = path_builder('HAR_ext', cte_psd = False, transitions = False, K = 6, var_x = 0.1, version = 2)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time_ext , n_clusters = 6, features = features, model = 'KMEANS', name = 'HAR_ext_GMVAE version 2')


# HAR GMVAE transitions

print('HAR with transitions GMVAE  comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = True, K = 10, var_x = 0.1, version = 1)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time_trans, n_clusters = 10, features = features, model = 'KMEANS', name = 'HAR_GMVAE_version 1 (transitions)')
path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = True, K = 10, var_x = 0.1, version = 2)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time_trans , n_clusters = 10, features = features, model = 'KMEANS', name = 'HAR_GMVAE version 2 (transitions)')



# HAR ext GMVAE transitions

print('HAR_ext with transitions GMVAE  comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR_ext', cte_psd = False, transitions = True, K = 12, var_x = 0.1, version = 1)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_trans_time_ext, n_clusters = 12, features = features, model = 'KMEANS', name = 'HAR_ext_GMVAE_version 1 (transitions)')
path_features, path_labels = path_builder('HAR_ext', cte_psd = False, transitions = True, K = 12, var_x = 0.1, version = 2)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_trans_time_ext , n_clusters = 12, features = features, model = 'KMEANS', name = 'HAR_ext_GMVAE version 2 (transitions)')




# HAR GMVAE with constant psd

print('HAR GMVAE (constant_psd) comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR', cte_psd = True, transitions = False, K = 5, var_x = 0.1, version = 1)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time , n_clusters = 5, features = features, model = 'KMEANS', name = 'HAR_GMVAE_version 1 (constant_psd)')
path_features, path_labels = path_builder('HAR', cte_psd = True, transitions = False, K = 5, var_x = 0.1, version = 2)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time , n_clusters = 5, features = features, model = 'KMEANS', name = 'HAR_GMVAE version 2 (constant_psd)')


# HAR_ext GMVAE with constant psd

print('HAR_ext GMVAE (constant_psd) comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR_ext', cte_psd = True, transitions = False, K = 6, var_x = 0.1, version = 1)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time_ext , n_clusters = 6, features = features, model = 'KMEANS', name = 'HAR_ext_GMVAE_version 1 (constant_psd)')
path_features, path_labels = path_builder('HAR_ext', cte_psd = True, transitions = False, K = 6, var_x = 0.1, version = 2)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time_ext , n_clusters = 6, features = features, model = 'KMEANS', name = 'HAR_ext_GMVAE version 2 (constant_psd)')


# HAR GMVAE transitions (constant_psd)

print('HAR with transitions (constant_psd) GMVAE  comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR', cte_psd = True, transitions = True, K = 10, var_x = 0.1, version = 1)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time_trans, n_clusters = 10, features = features, model = 'KMEANS', name = 'HAR_GMVAE_version 1 (transitions) (constant_psd)')
path_features, path_labels = path_builder('HAR', cte_psd = True, transitions = True, K = 10, var_x = 0.1, version = 2)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_time_trans , n_clusters = 10, features = features, model = 'KMEANS', name = 'HAR_GMVAE version 2 (transitions) (constant_psd)')



# HAR ext GMVAE transitions (constant_psd)

print('HAR_ext with transitions (constant_psd) GMVAE  comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR_ext', cte_psd = True, transitions = True, K = 12, var_x = 0.1, version = 1)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_trans_time_ext, n_clusters = 12, features = features, model = 'KMEANS', name = 'HAR_ext_GMVAE_version 1 (transitions) (constant_psd)')
path_features, path_labels = path_builder('HAR_ext', cte_psd = True, transitions = True, K = 12, var_x = 0.1, version = 2)
features = np.load(path_features)
#labels = np.load(path_labels)
print_metrics(pred_label = 0, true_label = labels_trans_time_ext , n_clusters = 12, features = features, model = 'KMEANS', name = 'HAR_ext_GMVAE version 2 (transitions) (constant_psd)')











# HAR predicted labels from GMVAE comparison version 1 and 2
print('HAR predicted labels from GMVAE comparison version 1 and 2')
print('\n')


path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = False, K = 5, var_x = 0.1, version = 1)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time , n_clusters = 5, features = features, model = 'GMVAE', name = 'HAR_GMVAE_version 1')
path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = False, K = 5, var_x = 0.1, version = 2)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time , n_clusters = 5, features = features, model = 'GMVAE', name = 'HAR_GMVAE version 2')



# HAR_ext predicted labels from GMVAE comparison version 1 and 2
print('HAR_ext predicted labels from GMVAE comparison version 1 and 2')
print('\n')


path_features, path_labels = path_builder('HAR_ext', cte_psd = False, transitions = False, K = 6, var_x = 0.1, version = 1)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time_ext , n_clusters = 6, features = features, model = 'GMVAE', name = 'HAR_ext_GMVAE_version 1')
path_features, path_labels = path_builder('HAR_ext', cte_psd = False, transitions = False, K =6, var_x = 0.1, version = 2)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time_ext , n_clusters = 6, features = features, model = 'GMVAE', name = 'HAR_ext_GMVAE version 2')


# HAR predicted labels from GMVAE comparison version 1 and 2 (transitions)
print('HAR predicted labels from GMVAE comparison version 1 and 2 (transitions)')
print('\n')

path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = True, K = 10, var_x = 0.1, version = 1)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time_trans, n_clusters = 10, features = features, model = 'GMVAE', name = 'HAR_GMVAE_version 1 (transitions)')
path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = True, K = 10, var_x = 0.1, version = 2)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time_trans , n_clusters = 10, features = features, model = 'GMAVAE', name = 'HAR_GMVAE version 2 (transitions)')



# HAR_ext predicted labels from GMVAE comparison version 1 and 2 (transitions)

print('HAR_ext GMVAE  comparison version 1 and 2 (transitions)')
print('\n')

path_features, path_labels = path_builder('HAR_ext', cte_psd = False, transitions = True, K = 12, var_x = 0.1, version = 1)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_trans_time_ext, n_clusters = 12, features = features, model = 'GMVAE', name = 'HAR_ext_GMVAE_version 1 (transitions)')
path_features, path_labels = path_builder('HAR_ext', cte_psd = False, transitions = True, K = 12, var_x = 0.1, version = 2)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_trans_time_ext , n_clusters = 12, features = features, model = 'GMVAE', name = 'HAR_ext_GMVAE version 2 (transitions)')




# HAR GMVAE with constant psd

print('HAR GMVAE (constant_psd) comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR', cte_psd = True, transitions = False, K = 5, var_x = 0.1, version = 1)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time , n_clusters = 5, features = features, model = 'GMVAE', name = 'HAR_GMVAE_version 1 (constant_psd)')
path_features, path_labels = path_builder('HAR', cte_psd = True, transitions = False, K = 5, var_x = 0.1, version = 2)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time , n_clusters = 5, features = features, model = 'GMVAE', name = 'HAR_GMVAE version 2 (constant_psd)')


# HAR_ext GMVAE with constant psd

print('HAR_ext GMVAE (constant_psd) comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR_ext', cte_psd = True, transitions = False, K = 6, var_x = 0.1, version = 1)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time_ext , n_clusters = 6, features = features, model = 'GMVAE', name = 'HAR_ext_GMVAE_version 1 (constant_psd)')
path_features, path_labels = path_builder('HAR_ext', cte_psd = True, transitions = False, K = 6, var_x = 0.1, version = 2)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time_ext , n_clusters = 6, features = features, model = 'GMVAE', name = 'HAR_ext_GMVAE version 2 (constant_psd)')


# HAR GMVAE transitions (constant_psd)

print('HAR with transitions (constant_psd) GMVAE  comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR', cte_psd = True, transitions = True, K = 10, var_x = 0.1, version = 1)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time_trans, n_clusters = 10, features = features, model = 'GMVAE', name = 'HAR_GMVAE_version 1 (transitions) (constant_psd)')
path_features, path_labels = path_builder('HAR', cte_psd = True, transitions = True, K = 10, var_x = 0.1, version = 2)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_time_trans , n_clusters = 10, features = features, model = 'GMVAE', name = 'HAR_GMVAE version 2 (transitions) (constant_psd)')



# HAR ext GMVAE transitions (constant_psd)

print('HAR_ext with transitions (constant_psd) GMVAE  comparison version 1 and 2')
print('\n')

path_features, path_labels = path_builder('HAR_ext', cte_psd = True, transitions = True, K = 12, var_x = 0.1, version = 1)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_trans_time_ext, n_clusters = 12, features = features, model = 'GMVAE', name = 'HAR_ext_GMVAE_version 1 (transitions) (constant_psd)')
path_features, path_labels = path_builder('HAR_ext', cte_psd = True, transitions = True, K = 12, var_x = 0.1, version = 2)
features = np.load(path_features)
labels = np.load(path_labels)
print_metrics(pred_label = np.argmax(labels,axis = 1), true_label = labels_trans_time_ext , n_clusters = 12, features = features, model = 'GMVAE', name = 'HAR_ext_GMVAE version 2 (transitions) (constant_psd)')







#Entropy HAR

K_list = [1,2,3,4,5,6,7]

for K in K_list:
    path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = False, K = K, var_x = 0.1, version = 1)
    features = np.load(path_features)
    labels = np.load(path_labels)

    entropy_v1 = entropy(labels, axis = 0)



    path_features, path_labels = path_builder('HAR', cte_psd = False, transitions = False, K = K, var_x = 0.1, version = 2)
    features = np.load(path_features)
    labels = np.load(path_labels)

    entropy_v2 = entropy(labels, axis=0)


    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    plt.title('version 1 '+ str(K)+' clusters')
    plt.hist(entropy_v1, K, density=True, facecolor='b', alpha=0.75)
    plt.ylabel('prob')
    plt.xlabel('entropy')
    plt.subplot(1, 2, 2)
    plt.title('version 2 ' + str(K) + ' clusters')
    plt.hist(entropy_v2, K, density=True, facecolor='b', alpha=0.75)
    plt.savefig('images_HAR/hist_entropy_K_' + str(K) + '_version_1_2.jpg')

# Entropy HAR_ext

K_list = [1, 2, 3, 4, 5, 6, 7]

for K in K_list:
    path_features, path_labels = path_builder('HAR_ext', cte_psd=False, transitions=False, K=K, var_x=0.1, version=1)
    features = np.load(path_features)
    labels = np.load(path_labels)

    entropy_v1 = entropy(labels, axis=0)

    path_features, path_labels = path_builder('HAR_ext', cte_psd=False, transitions=False, K=K, var_x=0.1, version=2)
    features = np.load(path_features)
    labels = np.load(path_labels)

    entropy_v2 = entropy(labels, axis=0)

    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    plt.title('version 1 ' + str(K) + ' clusters')
    plt.hist(entropy_v1, K, density=True, facecolor='b', alpha=0.75)
    plt.ylabel('prob')
    plt.xlabel('entropy')
    plt.subplot(1, 2, 2)
    plt.title('version 2 ' + str(K) + ' clusters')
    plt.hist(entropy_v2, K, density=True, facecolor='b', alpha=0.75)
    plt.savefig('images_HAR_ext/hist_entropy_K_' + str(K) + '_version_1_2.jpg')





