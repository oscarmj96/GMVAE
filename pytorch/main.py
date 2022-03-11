"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Main file to execute the model on the MNIST dataset

"""
import pickle
from scipy.io import arff
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib
from sklearn.cluster import AgglomerativeClustering
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
from model.GMVAE import *
import util
import scipy.io as sio
from scipy.fftpack import fft, dct, idct
from sklearn.preprocessing import MinMaxScaler, StandardScaler




#########################################################
## Input Parameters
#########################################################
parser = argparse.ArgumentParser(description='PyTorch Implementation of DGM Clustering')

## Used only in notebooks
parser.add_argument('-f', '--file',
                    help='Path for input file. First line should contain number of lines to search in')

## Dataset
parser.add_argument('--dataset', type=str, choices=['mnist', 'HAR', 'HAR_ext'],
                    default='HAR', help='dataset (default: mnist)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

## GPU
parser.add_argument('--cuda', type=int, default=0,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

## Training
##mnist, HAR, worms -------- batch size = 10,1890,128
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=10000, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=200, type=int,
                    help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay_epoch', default=-1, type=int, 
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')

## Architecture
#mnist, HAR, worms -------- num_classes = 10,5,5
#mnist, HAR, worms -------- gaussian size = 64,4,6
#mnist, HAR, HAR_ext -------- input_size = 784,128 (8 sec),561

parser.add_argument('--num_classes', type=int, default=5,
                    help='number of classes (default: 10)')
parser.add_argument('--gaussian_size', default= 20, type=int,
                    help='gaussian size (default: 64)')
parser.add_argument('--input_size', default=128, type=int,
                    help='input size (default: 784)')

## Partition parameters
parser.add_argument('--train_proportion', default=1.0, type=float,
                    help='proportion of examples to consider for training only (default: 1.0)')

## Gumbel parameters
parser.add_argument('--init_temp', default=1.0, type=float,
                    help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
parser.add_argument('--decay_temp', default=1, type=int, 
                    help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
parser.add_argument('--hard_gumbel', default=0, type=int, 
                    help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
parser.add_argument('--min_temp', default=0.5, type=float, 
                    help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)' )
parser.add_argument('--decay_temp_rate', default=0.013862944, type=float,
                    help='Temperature decay rate at every epoch (default: 0.013862944)')

## Loss function parameters
parser.add_argument('--w_gauss', default=1, type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_categ', default=1000, type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='mse', help='desired reconstruction loss function (default: bce)')

## Others
parser.add_argument('--verbose', default=0, type=int,
                    help='print extra information at every epoch.(default: 0)')
parser.add_argument('--random_search_it', type=int, default=20,
                    help='iterations of random search (default: 20)')

args = parser.parse_args()

if args.cuda == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

## Random Seed
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if args.cuda:
  torch.cuda.manual_seed(SEED)

#########################################################
## Read Data
#########################################################

K_list = [5]
for K in K_list:
    print(K)                                                                       
    print('\n')
    if args.dataset == "mnist":
      print("Loading mnist dataset...")
      # Download or load downloaded MNIST dataset
      train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
      test_dataset = datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor())
      psd = torch.ones(1, args.input_size)
      version = '1'

    if args.dataset == 'HAR':
        print("Loading HAR...")


        constant_psd = False
        transitions = False
        version = '3'
        data = sio.loadmat('database2Sens_Preproc.mat')['dataPreproc']

        if transitions:
            X, labels, list_len, list_seq = util.database_creator_unsupervised(data, 8)
        else:
            X, Y, labels, labels_name, count_disc, _ = util.database_creator(data, 8)

        print(X.shape, len(labels), labels[0])


        signals, label_sig = util.label_sample(labels, X)

        s1 = signals[0, 1:]  # tipo 4
        s2 = signals[1, 1:]  # tipo 2
        s3 = signals[2, 1:]  # tipo 3
        s4 = signals[3, 1:]  # tipo 1
        s5 = signals[4, 1:]

        if transitions:

            s6 = signals[5, :]
            s7 = signals[6, :]  # tipo 2
            s8 = signals[7, :]  # tipo 3
            s9 = signals[8, :]  # tipo 1
            s10 = signals[9, :]

        label_name1 = util.name_mapper(label_sig[0])
        label_name2 = util.name_mapper(label_sig[1])
        label_name3 = util.name_mapper(label_sig[2])
        label_name4 = util.name_mapper(label_sig[3])
        label_name5 = util.name_mapper(label_sig[4])

        if transitions:

            label_name6 = util.name_mapper(label_sig[5])
            label_name7 = util.name_mapper(label_sig[6])
            label_name8 = util.name_mapper(label_sig[7])
            label_name9 = util.name_mapper(label_sig[8])
            label_name10 = util.name_mapper(label_sig[9])

        if transitions == False:
            plt.figure(figsize=(15, 10))
            plt.subplot(3, 2, 1)
            plt.title(label_name1)
            plt.plot(s1)
            plt.subplot(3, 2, 2)
            plt.title(label_name2)
            plt.plot(s2)
            plt.subplot(3, 2, 3)
            plt.title(label_name3)
            plt.plot(s3)
            plt.subplot(3, 2, 4)
            plt.title(label_name4)
            plt.plot(s4)
            plt.subplot(3, 2, 5)
            plt.title(label_name5)
            plt.plot(s5)

            plt.savefig("images_HAR/signals.jpg")
            plt.show()
        else:
            plt.figure(figsize=(15, 10))
            plt.subplot(5, 2, 1)
            plt.title(label_name1)
            plt.plot(s1)
            plt.subplot(5, 2, 2)
            plt.title(label_name2)
            plt.plot(s2)
            plt.subplot(5, 2, 3)
            plt.title(label_name3)
            plt.plot(s3)
            plt.subplot(5, 2, 4)
            plt.title(label_name4)
            plt.plot(s4)
            plt.subplot(5, 2, 5)
            plt.title(label_name5)
            plt.plot(s5)
            plt.subplot(5, 2, 6)
            plt.title(label_name6)
            plt.plot(s6)
            plt.subplot(5, 2, 7)
            plt.title(label_name7)
            plt.plot(s7)
            plt.subplot(5, 2, 8)
            plt.title(label_name8)
            plt.plot(s8)
            plt.subplot(5, 2, 9)
            plt.title(label_name9)
            plt.plot(s9)
            plt.subplot(5, 2, 10)
            plt.title(label_name10)
            plt.plot(s10)

            plt.savefig("images_HAR/signals_transitions.jpg")
            plt.show()

        scaler_dct = MinMaxScaler()
        #scaler_dct = StandardScaler()

        x_min = np.amin(X)
        x_max = np.amax(X)

        X_norm = (X-x_min)/(x_max-x_min)
        #X_norm = X_std* (x_max-x_min)+ x_min
        #X_norm = scaler_dct.fit_transform(X)
        #X_norm = np.transpose(X_norm)

        signals, label_sig = util.label_sample(labels, X_norm)

        s1 = signals[0, :]  # tipo 4
        s2 = signals[1, :]  # tipo 2
        s3 = signals[2, :]  # tipo 3
        s4 = signals[3, :]  # tipo 1
        s5 = signals[4, :]
        '''
        s6 = signals[5, :]  # tipo 4
        s7 = signals[6, :]  # tipo 2
        s8 = signals[7, :]  # tipo 3
        s9 = signals[8, :]  # tipo 1
        s10 = signals[9, :]
        '''
        label_name1 = util.name_mapper(label_sig[0])
        label_name2 = util.name_mapper(label_sig[1])
        label_name3 = util.name_mapper(label_sig[2])
        label_name4 = util.name_mapper(label_sig[3])
        label_name5 = util.name_mapper(label_sig[4])
        '''
        label_name6 = util.name_mapper(label_sig[5])
        label_name7 = util.name_mapper(label_sig[6])
        label_name8 = util.name_mapper(label_sig[7])
        label_name9 = util.name_mapper(label_sig[8])
        label_name10 = util.name_mapper(label_sig[9])
        '''

        plt.figure(figsize=(15, 10))
        plt.subplot(3, 2, 1)
        plt.title(label_name1)
        plt.plot(s1)
        plt.subplot(3, 2, 2)
        plt.title(label_name2)
        plt.plot(s2)
        plt.subplot(3, 2, 3)
        plt.title(label_name3)
        plt.plot(s3)
        plt.subplot(3, 2, 4)
        plt.title(label_name4)
        plt.plot(s4)
        plt.subplot(3, 2, 5)
        plt.title(label_name5)
        plt.plot(s5)

        plt.savefig("images_HAR/signals_norm.jpg")
        plt.show()

        X_norm = X

        X_th = torch.from_numpy(X_norm.astype(np.float32))
        X_th = X_th.unsqueeze(-1)
        X_Rx, _, _ = util.get_Rx(X_th)

        psd_time = X_Rx.numpy()

        if constant_psd == True:
            psd_dct = np.ones((128,))
        else:
            psd_dct = dct(psd_time, 1)

        X_dct = np.ones_like(X_norm)
        for i in range(X.shape[0]):
            X_dct[i, :] = dct(X_norm[i, :], 1)






        signals, label_sig = util.label_sample(labels, X_dct)

        s1 = signals[0, 1:]  # tipo 4
        s2 = signals[1, 1:]  # tipo 2
        s3 = signals[2, 1:]  # tipo 3
        s4 = signals[3, 1:]  # tipo 1
        s5 = signals[4, 1:]  # tipo 1
        if transitions:
            s6 = signals[5, 1:]  # tipo 4
            s7 = signals[6, 1:]  # tipo 2
            s8 = signals[7, 1:]  # tipo 3
            s9 = signals[8, 1:]  # tipo 1
            s10 = signals[9, 1:]  # tipo 1

        if transitions == False:
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.title(util.name_mapper(label_sig[0]))
            plt.plot(s1)
            plt.subplot(2, 3, 2)
            plt.title(util.name_mapper(label_sig[1]))
            plt.plot(s2)
            plt.subplot(2, 3, 3)
            plt.title(util.name_mapper(label_sig[2]))
            plt.plot(s3)
            plt.subplot(2, 3, 4)
            plt.title(util.name_mapper(label_sig[3]))
            plt.plot(s4)
            plt.subplot(2, 3, 5)
            plt.title(util.name_mapper(label_sig[4]))
            plt.plot(s5)

            plt.savefig("images_HAR/signals_dct.jpg")
            plt.show()
        else:
            plt.figure(figsize=(15, 10))
            plt.subplot(5, 2, 1)
            plt.title(util.name_mapper(label_sig[0]))
            plt.plot(s1)
            plt.subplot(5, 2, 2)
            plt.title(util.name_mapper(label_sig[1]))
            plt.plot(s2)
            plt.subplot(5, 2, 3)
            plt.title(util.name_mapper(label_sig[2]))
            plt.plot(s3)
            plt.subplot(5, 2, 4)
            plt.title(util.name_mapper(label_sig[3]))
            plt.plot(s4)
            plt.subplot(5, 2, 5)
            plt.title(util.name_mapper(label_sig[4]))
            plt.plot(s5)
            plt.subplot(5, 2, 6)
            plt.title(util.name_mapper(label_sig[5]))
            plt.plot(s6)
            plt.subplot(5, 2, 7)
            plt.title(util.name_mapper(label_sig[6]))
            plt.plot(s7)
            plt.subplot(5, 2, 8)
            plt.title(util.name_mapper(label_sig[7]))
            plt.plot(s8)
            plt.subplot(5, 2, 9)
            plt.title(util.name_mapper(label_sig[8]))
            plt.plot(s9)
            plt.subplot(5, 2, 10)
            plt.title(util.name_mapper(label_sig[9]))
            plt.plot(s10)

            plt.savefig("images_HAR/signals_dct_transitions.jpg")
            plt.show()


        #psd_dct = np.mean(X_dct,0)
        #psd_time = idct(psd_dct, 1)

        print(X_dct.shape)  # Datos
        print(psd_dct.shape)  # psd

        plt.figure(figsize=(15, 10))
        plt.subplot(1, 1, 1)
        plt.title('PSD_dct')
        plt.plot(psd_dct[1:])
        if constant_psd:
            if transitions:
                plt.savefig("images_HAR/psd_cte_psd_transitions.jpg")
            else:
                plt.savefig("images_HAR/psd_cte_psd.jpg")
        else:
            if transitions:
                plt.savefig("images_HAR/psd_transitions.jpg")
            else:
                plt.savefig("images_HAR/psd.jpg")
        plt.show()

        X_dct = torch.from_numpy(X_dct.astype(np.float32))
        psd = torch.from_numpy(psd_dct.astype(np.float32))

        training_dataset = torch.utils.data.TensorDataset(X_dct, torch.from_numpy(np.asarray(labels).astype(int)))
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    if args.dataset == 'HAR_ext':
        print('loading HAR_ext...')
        X_data_ext = pd.read_csv('Train_HAR_ext/X_train.txt', sep = ' ', header=None)
        Y_data_ext = pd.read_csv('Train_HAR_ext/y_train.txt', header=None)
                                                                                   


        X = X_data_ext.values
        labels = Y_data_ext.values.reshape((-1,))

        constant_psd = True
        transitions = False
        version = '2'

        if transitions == False:
            mask = np.where(labels <=6,True, False)
            X = X[mask,:]
            labels = labels[mask]

        print(X.shape, len(labels), labels[0])

        signals, label_sig = util.label_sample(labels, X)

        s1 = signals[0, :]  # tipo 4
        s2 = signals[1, :]  # tipo 2
        s3 = signals[2, :]  # tipo 3
        s4 = signals[3, :]  # tipo 1
        s5 = signals[4, :]
        s6 = signals[5, :]

        if transitions:
            s7 = signals[6, :]  # tipo 4
            s8 = signals[7, :]  # tipo 2
            s9 = signals[8, :]  # tipo 3
            s10 = signals[9, :]  # tipo 1
            s11 = signals[10, :]
            s12 = signals[11, :]

        label_name1 = util.name_mapper_HAR_ext(label_sig[0])
        label_name2 = util.name_mapper_HAR_ext(label_sig[1])
        label_name3 = util.name_mapper_HAR_ext(label_sig[2])
        label_name4 = util.name_mapper_HAR_ext(label_sig[3])
        label_name5 = util.name_mapper_HAR_ext(label_sig[4])
        label_name6 = util.name_mapper_HAR_ext(label_sig[5])

        if transitions:
            label_name6 = util.name_mapper_HAR_ext(label_sig[5])
            label_name7 = util.name_mapper_HAR_ext(label_sig[6])
            label_name8 = util.name_mapper_HAR_ext(label_sig[7])
            label_name9 = util.name_mapper_HAR_ext(label_sig[8])
            label_name10 = util.name_mapper_HAR_ext(label_sig[9])
            label_name11 = util.name_mapper_HAR_ext(label_sig[10])
            label_name12 = util.name_mapper_HAR_ext(label_sig[11])

        if transitions == False:
            plt.figure(figsize=(15, 10))
            plt.subplot(3, 2, 1)
            plt.title(label_name1)
            plt.plot(s1)
            plt.subplot(3, 2, 2)
            plt.title(label_name2)
            plt.plot(s2)
            plt.subplot(3, 2, 3)
            plt.title(label_name3)
            plt.plot(s3)
            plt.subplot(3, 2, 4)
            plt.title(label_name4)
            plt.plot(s4)
            plt.subplot(3, 2, 5)
            plt.title(label_name5)
            plt.plot(s5)
            plt.subplot(3, 2, 6)
            plt.title(label_name6)
            plt.plot(s6)

            plt.savefig("images_HAR_ext/signals.jpg")
            plt.show()
        else:
            plt.figure(figsize=(15, 10))
            plt.subplot(6, 2, 1)
            plt.title(label_name1)
            plt.plot(s1)
            plt.subplot(6, 2, 2)
            plt.title(label_name2)
            plt.plot(s2)
            plt.subplot(6, 2, 3)
            plt.title(label_name3)
            plt.plot(s3)
            plt.subplot(6, 2, 4)
            plt.title(label_name4)
            plt.plot(s4)
            plt.subplot(6, 2, 5)
            plt.title(label_name5)
            plt.plot(s5)
            plt.subplot(6, 2, 6)
            plt.title(label_name6)
            plt.plot(s6)
            plt.subplot(6, 2, 7)
            plt.title(label_name7)
            plt.plot(s7)
            plt.subplot(6, 2, 8)
            plt.title(label_name8)
            plt.plot(s8)
            plt.subplot(6, 2, 9)
            plt.title(label_name9)
            plt.plot(s9)
            plt.subplot(6, 2, 10)
            plt.title(label_name10)
            plt.plot(s10)
            plt.subplot(6, 2, 11)
            plt.title(label_name11)
            plt.plot(s11)
            plt.subplot(6, 2, 12)
            plt.title(label_name12)
            plt.plot(s12)

            plt.savefig("images_HAR_ext/signals_transitions.jpg")
            plt.show()

        X_th = torch.from_numpy(X.astype(np.float32))
        X_th = X_th.unsqueeze(-1)
        X_Rx, _, _ = util.get_Rx(X_th)

        psd_time = X_Rx.numpy()

        if constant_psd:
            psd_dct = np.ones((561,))
        else:
            psd_dct = dct(psd_time, 1)

        X_dct = np.ones_like(X)
        for i in range(X.shape[0]):
            X_dct[i, :] = dct(X[i, :], 1)

        signals, label_sig = util.label_sample(labels, X_dct)

        s1 = signals[0, 1:]  # tipo 4
        s2 = signals[1, 1:]  # tipo 2
        s3 = signals[2, 1:]  # tipo 3
        s4 = signals[3, 1:]  # tipo 1
        s5 = signals[4, 1:]
        s6 = signals[5, 1:]

        if transitions:
            s7 = signals[6, 1:]  # tipo 4
            s8 = signals[7, 1:]  # tipo 2
            s9 = signals[8, 1:]  # tipo 3
            s10 = signals[9, 1:]  # tipo 1
            s11 = signals[10, 1:]
            s12 = signals[11, 1:]

        label_name1 = util.name_mapper_HAR_ext(label_sig[0])
        label_name2 = util.name_mapper_HAR_ext(label_sig[1])
        label_name3 = util.name_mapper_HAR_ext(label_sig[2])
        label_name4 = util.name_mapper_HAR_ext(label_sig[3])
        label_name5 = util.name_mapper_HAR_ext(label_sig[4])
        label_name6 = util.name_mapper_HAR_ext(label_sig[5])

        if transitions:
            label_name6 = util.name_mapper_HAR_ext(label_sig[5])
            label_name7 = util.name_mapper_HAR_ext(label_sig[6])
            label_name8 = util.name_mapper_HAR_ext(label_sig[7])
            label_name9 = util.name_mapper_HAR_ext(label_sig[8])
            label_name10 = util.name_mapper_HAR_ext(label_sig[9])
            label_name11 = util.name_mapper_HAR_ext(label_sig[10])
            label_name12 = util.name_mapper_HAR_ext(label_sig[11])

        if transitions == False:
            plt.figure(figsize=(15, 10))
            plt.subplot(3, 2, 1)
            plt.title(label_name1)
            plt.plot(s1)
            plt.subplot(3, 2, 2)
            plt.title(label_name2)
            plt.plot(s2)
            plt.subplot(3, 2, 3)
            plt.title(label_name3)
            plt.plot(s3)
            plt.subplot(3, 2, 4)
            plt.title(label_name4)
            plt.plot(s4)
            plt.subplot(3, 2, 5)
            plt.title(label_name5)
            plt.plot(s5)
            plt.subplot(3, 2, 6)
            plt.title(label_name6)
            plt.plot(s6)

            plt.savefig("images_HAR_ext/signals_dct.jpg")
            plt.show()
        else:
            plt.figure(figsize=(15, 10))
            plt.subplot(6, 2, 1)
            plt.title(label_name1)
            plt.plot(s1)
            plt.subplot(6, 2, 2)
            plt.title(label_name2)
            plt.plot(s2)
            plt.subplot(6, 2, 3)
            plt.title(label_name3)
            plt.plot(s3)
            plt.subplot(6, 2, 4)
            plt.title(label_name4)
            plt.plot(s4)
            plt.subplot(6, 2, 5)
            plt.title(label_name5)
            plt.plot(s5)
            plt.subplot(6, 2, 6)
            plt.title(label_name6)
            plt.plot(s6)
            plt.subplot(6, 2, 7)
            plt.title(label_name7)
            plt.plot(s7)
            plt.subplot(6, 2, 8)
            plt.title(label_name8)
            plt.plot(s8)
            plt.subplot(6, 2, 9)
            plt.title(label_name9)
            plt.plot(s9)
            plt.subplot(6, 2, 10)
            plt.title(label_name10)
            plt.plot(s10)
            plt.subplot(6, 2, 11)
            plt.title(label_name11)
            plt.plot(s11)
            plt.subplot(6, 2, 12)
            plt.title(label_name12)
            plt.plot(s12)

            plt.savefig("images_HAR_ext/signals_dct_transitions.jpg")
            plt.show()

        print(X_dct.shape)  # Datos
        print(psd_dct.shape)  # psd

        plt.figure(figsize=(15, 10))
        plt.subplot(1, 1, 1)
        plt.title('PSD_dct')
        plt.plot(psd_dct[1:])
        if constant_psd:
            if transitions:
                plt.savefig("images_HAR_ext/psd_cte_psd_transitions.jpg")
            else:
                plt.savefig("images_HAR_ext/psd_cte_psd.jpg")
        else:
            if transitions:
                plt.savefig("images_HAR_ext/psd_transitions.jpg")
            else:
                plt.savefig("images_HAR_ext/psd.jpg")
        plt.show()

        X_dct = torch.from_numpy(X_dct.astype(np.float32))
        psd = torch.from_numpy(psd_dct.astype(np.float32))

        training_dataset = torch.utils.data.TensorDataset(X_dct, torch.from_numpy(np.asarray(labels).astype(int)))
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)










    #########################################################
    ## Data Partition
    #########################################################
    def partition_dataset(n, proportion=0.8):
      train_num = int(n * proportion)
      indices = np.random.permutation(n)
      train_indices, val_indices = indices[:train_num], indices[train_num:]
      return train_indices, val_indices

    if args.dataset == "mnist":

        if args.train_proportion == 1.0:
          train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
          test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
          val_loader = test_loader
        else:
          train_indices, val_indices = partition_dataset(len(train_dataset), args.train_proportion)
          # Create data loaders for train, validation and test datasets
          train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_indices))
          val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_val, sampler=SubsetRandomSampler(val_indices))
          test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)

    ## Calculate flatten size of each input data
        args.input_size = np.prod(train_dataset[0][0].size())
        print(args.input_size)
    #########################################################
    ## Train and Test Model
    #########################################################

    var_x = 0.001
    network = GMVAENet(args.input_size, args.gaussian_size, K, psd, var_x, version = version)
    gmvae = GMVAE(args, psd, network, var_x)

    val_loader = 0
    ## Training Phase
    history_loss = gmvae.train(train_loader, val_loader, False)

    ## Testing Phase
    #accuracy, nmi = gmvae.test(test_loader)
    #print("Testing phase...")
    #print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi) )




    with open('history_loss_'+args.dataset+'.pickle', 'wb') as handle:
        pickle.dump(history_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open('history_loss_'+args.dataset+'.pickle', 'rb') as handle:
        history_loss = pickle.load(handle)

    plt.figure(figsize=(15, 3))
    plt.subplot(2, 2, 1)
    plt.title('Reconstruction')
    plt.plot(history_loss['train_history_rec'])
    plt.subplot(2, 2, 2)
    plt.title('Cat_loss')
    plt.plot(history_loss['train_history_cat'])
    plt.subplot(2, 2, 3)
    plt.title('Gauss_loss')
    plt.plot(history_loss['train_history_gauss'])
    plt.subplot(2, 2, 4)
    plt.title('Total loss')
    plt.plot(history_loss['train_history_loss'])
    if args.dataset == "mnist":
        plt.savefig('images_MNIST/losses.jpg')
    elif args.dataset == "HAR":
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR/losses_constant_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR/losses_constant_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR/losses_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR/losses_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
    elif args.dataset == "HAR_ext":
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR_ext/losses_constant_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR_ext/losses_constant_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR_ext/losses_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR_ext/losses_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')

    if args.dataset == "mnist":
        torch.save(gmvae.network.state_dict(), 'models_MNIST/GM_SGVAE_MNIST.pt')
    elif args.dataset == "HAR":
        if constant_psd:
            if transitions:
                torch.save(gmvae.network.state_dict(), 'models_HAR/GM_SGVAE_HAR_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt')
            else:
                torch.save(gmvae.network.state_dict(), 'models_HAR/GM_SGVAE_HAR_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt')

        else:
            if transitions:
                torch.save(gmvae.network.state_dict(), 'models_HAR/GM_SGVAE_HAR_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt')
            else:
                torch.save(gmvae.network.state_dict(), 'models_HAR/GM_SGVAE_HAR_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt')
    elif args.dataset == "HAR_ext":
        if constant_psd:
            if transitions:
                torch.save(gmvae.network.state_dict(), 'models_HAR_ext/GM_SGVAE_HAR_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt')
            else:
                torch.save(gmvae.network.state_dict(), 'models_HAR_ext/GM_SGVAE_HAR_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt')

        else:
            if transitions:
                torch.save(gmvae.network.state_dict(), 'models_HAR_ext/GM_SGVAE_HAR_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt')
            else:
                torch.save(gmvae.network.state_dict(), 'models_HAR_ext/GM_SGVAE_HAR_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt')

    network = GMVAENet(args.input_size, args.gaussian_size, K, psd, var_x,version = version)

    if args.dataset == "mnist":
        network.load_state_dict(torch.load('models_MNIST/GM_SGVAE_MNIST.pt'))
    elif args.dataset == "HAR":
        if constant_psd:
            if transitions:
                network.load_state_dict(torch.load('models_HAR/GM_SGVAE_HAR_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt'))
            else:
                network.load_state_dict(torch.load('models_HAR/GM_SGVAE_HAR_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt'))
        else:
            if transitions:
                network.load_state_dict(torch.load('models_HAR/GM_SGVAE_HAR_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt'))
            else:
                network.load_state_dict(torch.load('models_HAR/GM_SGVAE_HAR_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt'))
    elif args.dataset == "HAR_ext":
        if constant_psd:
            if transitions:
                network.load_state_dict(torch.load('models_HAR_ext/GM_SGVAE_HAR_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt'))
            else:
                network.load_state_dict(torch.load('models_HAR_ext/GM_SGVAE_HAR_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt'))
        else:
            if transitions:
                network.load_state_dict(torch.load('models_HAR_ext/GM_SGVAE_HAR_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt'))
            else:
                network.load_state_dict(torch.load('models_HAR_ext/GM_SGVAE_HAR_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.pt'))
    gmvae = GMVAE(args, psd, network, var_x)

    train_features, pred_labels, train_labels = gmvae.latent_features(train_loader, True)
    pred_labels_max = np.argmax(pred_labels, axis=1)

    if args.dataset == 'HAR' or args.dataset == 'HAR_ext':

        pred_cluster = AgglomerativeClustering(n_clusters=K).fit_predict(train_features)

        pred_labels_max = np.argmax(pred_labels, axis=1)



    if args.dataset == "HAR":
        if constant_psd:
            if transitions:
                np.save('data_HAR/features_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_features)
                np.save('data_HAR/labels_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_labels)
                np.save('data_HAR/labels_pred_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, pred_labels)
            else:
                np.save('data_HAR/features_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_features)
                np.save('data_HAR/labels_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_labels)
                np.save('data_HAR/labels_pred_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, pred_labels)
        else:
            if transitions:
                np.save('data_HAR/features_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_features)
                np.save('data_HAR/labels_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_labels)
                np.save('data_HAR/labels_pred_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, pred_labels)
            else:
                np.save('data_HAR/features_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_features)
                np.save('data_HAR/labels_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_labels)
                np.save('data_HAR/labels_pred_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, pred_labels)
    elif args.dataset == "HAR_ext":
        if constant_psd:
            if transitions:
                np.save('data_HAR_ext/features_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_features)
                np.save('data_HAR_ext/labels_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_labels)
                np.save('data_HAR_ext/labels_pred_cte_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, pred_labels)
            else:
                np.save('data_HAR_ext/features_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_features)
                np.save('data_HAR_ext/labels_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_labels)
                np.save('data_HAR_ext/labels_pred_cte_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, pred_labels)
        else:
            if transitions:
                np.save('data_HAR_ext/features_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_features)
                np.save('data_HAR_ext/labels_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_labels)
                np.save('data_HAR_ext/labels_pred_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, pred_labels)
            else:
                np.save('data_HAR_ext/features_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_features)
                np.save('data_HAR_ext/labels_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, train_labels)
                np.save('data_HAR_ext/labels_pred_varx_'+str(var_x)+'_K_'+str(K)+'_version_'+version, pred_labels)
    elif args.dataset == 'mnist':
        np.save('data_MNIST/features_varx_' + str(var_x) + '_K_' + str(K)+'_version_'+version, train_features)
        np.save('data_MNIST/labels_varx_' + str(var_x) + '_K_' + str(K)+'_version_'+version, train_labels)
        np.save('data_MNIST/labels_pred_varx_' + str(var_x) + '_K_' + str(K)+'_version_'+version, pred_labels)


    if args.gaussian_size >2:
        tsne_features = TSNE(n_components=2).fit_transform(train_features)

        fig = plt.figure(figsize=(10, 6))

        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=train_labels, marker='o',
                    edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
        plt.grid(False)
        plt.axis('off')
        plt.colorbar()
    elif args.gaussian_size ==2:
        fig = plt.figure(figsize=(10, 6))

        plt.scatter(train_features[:, 0], train_features[:, 1], c=train_labels, marker='o',
                    edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
        plt.grid(True)
        plt.colorbar()
    else:
        fig = plt.figure(figsize=(10, 6))

        plt.scatter(train_features, np.zeros_like(train_features), c=train_labels, marker='o',
                    edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
        plt.grid(True)
        plt.colorbar()




    if args.dataset == "mnist":
        plt.savefig('images_MNIST/latent_space.jpg')
    elif args.dataset == "HAR":
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR/latent_space_constant_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR/latent_space_constant_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR/latent_space_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR/latent_space_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
    elif args.dataset == "HAR_ext":
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR_ext/latent_space_constant_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR_ext/latent_space_constant_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR_ext/latent_space_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR_ext/latent_space_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')

    if args.dataset == 'HAR' or args.dataset == 'HAR_ext':

        if args.gaussian_size > 2:
            #tsne_features = TSNE(n_components=2).fit_transform(train_features)

            fig = plt.figure(figsize=(10, 6))

            plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=pred_cluster, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(False)
            plt.axis('off')
            plt.colorbar()
        elif args.gaussian_size == 2:
            fig = plt.figure(figsize=(10, 6))

            plt.scatter(train_features[:, 0], train_features[:, 1], c=pred_cluster, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(True)
            plt.colorbar()
        else:
            fig = plt.figure(figsize=(10, 6))

            plt.scatter(train_features, np.zeros_like(train_features), c=pred_cluster, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(True)
            plt.colorbar()

    if args.dataset == 'mnist':

        if args.gaussian_size > 2:
            # tsne_features = TSNE(n_components=2).fit_transform(train_features)

            fig = plt.figure(figsize=(10, 6))

            plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=pred_labels_max, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(False)
            plt.axis('off')
            plt.colorbar()
        elif args.gaussian_size == 2:
            fig = plt.figure(figsize=(10, 6))

            plt.scatter(train_features[:, 0], train_features[:, 1], c=pred_labels_max, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(True)
            plt.colorbar()
        else:
            fig = plt.figure(figsize=(10, 6))

            plt.scatter(train_features, np.zeros_like(train_features), c=pred_labels_max, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(True)
            plt.colorbar()

    if args.dataset == "mnist":
        plt.savefig('images_MNIST/latent_space_CLUSTER_GMVAE.jpg')
    elif args.dataset == "HAR":
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR/latent_space_constant_psd_transitions_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER.jpg')
            else:
                plt.savefig('images_HAR/latent_space_constant_psd_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR/latent_space_transitions_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER.jpg')
            else:
                plt.savefig(
                    'images_HAR/latent_space_varx_' + str(var_x) + '_K_' + str(K) + '_version' + version + 'CLUSTER.jpg')
    elif args.dataset == "HAR_ext":
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR_ext/latent_space_constant_psd_transitions_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER.jpg')
            else:
                plt.savefig('images_HAR_ext/latent_space_constant_psd_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR_ext/latent_space_transitions_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER.jpg')
            else:
                plt.savefig(
                    'images_HAR_ext/latent_space_varx_' + str(var_x) + '_K_' + str(K) + '_version' + version + 'CLUSTER.jpg')

    if args.dataset == 'HAR' or args.dataset == 'HAR_ext':
        if args.gaussian_size > 2:
            #tsne_features = TSNE(n_components=2).fit_transform(train_features)

            fig = plt.figure(figsize=(10, 6))

            plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=pred_labels_max, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(False)
            plt.axis('off')
            plt.colorbar()
        elif args.gaussian_size == 2:
            fig = plt.figure(figsize=(10, 6))

            plt.scatter(train_features[:, 0], train_features[:, 1], c=pred_labels_max, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(True)
            plt.colorbar()
        else:
            fig = plt.figure(figsize=(10, 6))

            plt.scatter(train_features, np.zeros_like(train_features), c=pred_labels_max, marker='o',
                        edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
            plt.grid(True)
            plt.colorbar()

    if args.dataset == "mnist":
        plt.savefig('images_MNIST/latent_space_void.jpg')
    elif args.dataset == "HAR":
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR/latent_space_constant_psd_transitions_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER_gmvae.jpg')
            else:
                plt.savefig('images_HAR/latent_space_constant_psd_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER_gmvae.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR/latent_space_transitions_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER_gmvae.jpg')
            else:
                plt.savefig(
                    'images_HAR/latent_space_varx_' + str(var_x) + '_K_' + str(K) + '_version' + version + 'CLUSTER_gmvae.jpg')
    elif args.dataset == "HAR_ext":
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR_ext/latent_space_constant_psd_transitions_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER_gmvae.jpg')
            else:
                plt.savefig('images_HAR_ext/latent_space_constant_psd_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER_gmvae.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR_ext/latent_space_transitions_varx_' + str(var_x) + '_K_' + str(
                    K) + '_version' + version + 'CLUSTER_gmvae.jpg')
            else:
                plt.savefig(
                    'images_HAR_ext/latent_space_varx_' + str(var_x) + '_K_' + str(K) + '_version' + version + 'CLUSTER_gmvae.jpg')

    if args.dataset == "HAR":
        original,recon, labels_recon = gmvae.reconstruct_data(train_loader)

        plt.figure(figsize=(20, 15))
        plt.subplot(3, 2, 1)
        plt.title('Signal: '+ util.name_mapper(labels_recon[0].numpy()))
        plt.plot(original[0, 1:], label = 'original')
        plt.plot(recon[0, 1:], label = 'recon')
        plt.legend()
        # plt.ylim((-4,4))
        plt.subplot(3, 2, 2)
        plt.title('Signal: '+ util.name_mapper(labels_recon[1].numpy()))
        plt.plot(original[1, 1:], label='original')
        plt.plot(recon[1, 1:], label='recon')
        plt.legend()
        plt.subplot(3, 2, 3)
        plt.title('Signal: '+ util.name_mapper(labels_recon[2].numpy()))
        plt.plot(original[2, 1:], label='original')
        plt.plot(recon[2, 1:], label='recon')
        plt.legend()
        plt.subplot(3, 2, 4)
        plt.title('Signal: '+ util.name_mapper(labels_recon[3].numpy()))
        plt.plot(original[3, 1:], label='original')
        plt.plot(recon[3, 1:], label='recon')
        plt.legend()
        plt.subplot(3, 2, 5)
        plt.legend()
        plt.title('Signal: '+ util.name_mapper(labels_recon[4].numpy()))
        plt.plot(original[4, 1:], label='original')
        plt.plot(recon[4, 1:], label='recon')
        plt.legend()
        plt.subplot(3, 2, 6)
        plt.title('Signal: '+ util.name_mapper(labels_recon[5].numpy()))
        plt.plot(original[5, 1:], label='original')
        plt.plot(recon[5, 1:], label='recon')
        plt.legend()
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR/recontruction_constant_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR/recontruction_constant_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR/recontruction_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR/recontruction_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')

    elif args.dataset == "HAR_ext":
        original, recon, labels_recon = gmvae.reconstruct_data(train_loader)

        plt.figure(figsize=(20, 15))
        plt.subplot(3, 2, 1)
        plt.title('Signal: ' + util.name_mapper_HAR_ext(labels_recon[0].numpy()))
        plt.plot(original[0, 1:], label='original')
        plt.plot(recon[0, 1:], label='recon')
        plt.legend()
        # plt.ylim((-4,4))
        plt.subplot(3, 2, 2)
        plt.title('Signal: ' + util.name_mapper_HAR_ext(labels_recon[1].numpy()))
        plt.plot(original[1, 1:], label='original')
        plt.plot(recon[1, 1:], label='recon')
        plt.legend()
        plt.subplot(3, 2, 3)
        plt.title('Signal: ' + util.name_mapper_HAR_ext(labels_recon[2].numpy()))
        plt.plot(original[2, 1:], label='original')
        plt.plot(recon[2, 1:], label='recon')
        plt.legend()
        plt.subplot(3, 2, 4)
        plt.title('Signal: ' + util.name_mapper_HAR_ext(labels_recon[3].numpy()))
        plt.plot(original[3, 1:], label='original')
        plt.plot(recon[3, 1:], label='recon')
        plt.legend()
        plt.subplot(3, 2, 5)
        plt.legend()
        plt.title('Signal: ' + util.name_mapper_HAR_ext(labels_recon[4].numpy()))
        plt.plot(original[4, 1:], label='original')
        plt.plot(recon[4, 1:], label='recon')
        plt.legend()
        plt.subplot(3, 2, 6)
        plt.title('Signal: ' + util.name_mapper_HAR_ext(labels_recon[5].numpy()))
        plt.plot(original[5, 1:], label='original')
        plt.plot(recon[5, 1:], label='recon')
        plt.legend()
        if constant_psd:
            if transitions:
                plt.savefig('images_HAR_ext/recontruction_constant_psd_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR_ext/recontruction_constant_psd_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
        else:
            if transitions:
                plt.savefig('images_HAR_ext/recontruction_transitions_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')
            else:
                plt.savefig('images_HAR_ext/recontruction_varx_'+str(var_x)+'_K_'+str(K)+'_version'+version+'.jpg')



