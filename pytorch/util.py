from statistics import mode

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler


def sample_MNIST(X, labels, n_cat):

    for i in np.unique(labels):
        mask = labels == i
        labels_masked = labels[mask]
        digits_masked = X[mask, :]
        randomlist = np.random.randint(low=0, high=digits_masked.shape[0], size=n_cat).tolist()

        if i == 0:
            photo = digits_masked[randomlist, :]
            labels_batch = labels_masked[randomlist]
        else:
            photo = np.vstack((photo, digits_masked[randomlist, :]))
            labels_batch = np.hstack((labels_batch, labels_masked[randomlist]))

    return photo, labels_batch












def plot_4x4_grid(
        images: np.ndarray,
        shape: tuple = (28, 28),
        cmap="gray",
        figsize=(10, 10)
) -> None:
    """
    Plot multiple images in subplot grid.
    :param images: tensor with MNIST images with shape [16, *shape]
    :param shape: shape of the images
    """
    assert images.shape[0] >= 16
    dist_samples_np = images[:16, ...].reshape([4, 4, *shape])
    x_ticks_labels = ['orig', 'reco', 'filt', 'samp']

    plt.figure(figsize=figsize)
    for i in range(4):
        for j in range(4):
            ax = plt.subplot(4, 4, i * 4 + j + 1)
            plt.imshow(dist_samples_np[i, j], cmap=cmap)
            plt.yticks(np.arange(10, 20, step=3), x_ticks_labels[i], fontsize=18)
            plt.xticks([])

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()


def get_Rx(x):
    x = x.permute(0, -1, 1)
    B = x.shape[0]
    F = x.shape[1]
    L = x.shape[-1]
    Rx = torch.empty((B, L), dtype= torch.float32)



    b = torch.clone(x)  # (B, L) -> i feature for all the B signals
    for j in range(L):
        Rx[:, j] = ((x * b.transpose(0, 0)).sum(-1)).mean(-1) / L
        b = torch.roll(b, 1, dims=-1)

    Rx_f = Rx[~torch.all(Rx.isnan(), dim=1)]
    E_Rx = torch.mean(Rx_f,0)

    return E_Rx, Rx_f, Rx


def mul_n(len, n):
    num_sec = 16 * n
    return len - len % num_sec, int((len - len % num_sec) / num_sec)


def label_mapper(raw_label):
    if raw_label == 12:
        label = 1
        label_name = 'sitting'
    elif raw_label == 13:
        label = 2
        label_name = 'lying'
    elif raw_label == 111:
        label = 3  # standing
        label_name = 'standing'
    elif raw_label == 2111:
        label = 4  # walking
        label_name = 'walking'
    elif raw_label == 2112:
        label = 5  # running
        label_name = 'running'
    elif raw_label == 31:
        label = 6
        label_name = 'stand_2_sitt'
    elif raw_label == 32:
        label = 7
        label_name = 'sitt_2_stand'
    elif raw_label == 34:
        label = 8
        label_name = 'ly_2_stand'
    elif raw_label == 35:
        label = 9
        label_name = 'sitt_2_ly'
    else:
        label = 10
        label_name = 'ly_2_sitt'

    return label, label_name


def name_mapper(raw_label):
    if raw_label == 1:
        label_name = 'sitting'
    elif raw_label == 2:

        label_name = 'lying'
    elif raw_label == 3:

        label_name = 'standing'
    elif raw_label == 4:

        label_name = 'walking'
    elif raw_label == 5:

        label_name = 'running'
    elif raw_label == 6:

        label_name = 'stand_2_sitt'
    elif raw_label == 7:

        label_name = 'sitt_2_stand'
    elif raw_label == 8:

        label_name = 'ly_2_stand'
    elif raw_label == 9:

        label_name = 'sitt_2_ly'
    else:

        label_name = 'ly_2_sitt'

    return label_name


def name_mapper_HAR_ext(raw_label):
    if raw_label == 1:
        label_name = 'walking'
    elif raw_label == 2:

        label_name = 'walking_upstairs'
    elif raw_label == 3:

        label_name = 'walking_downstairs'
    elif raw_label == 4:

        label_name = 'sitting'
    elif raw_label == 5:

        label_name = 'standing'
    elif raw_label == 6:

        label_name = 'laying'
    elif raw_label == 7:

        label_name = 'stand_2_sitt'
    elif raw_label == 8:

        label_name = 'sitt_2_stand'
    elif raw_label == 9:

        label_name = 'sitt_2_ly'
    elif raw_label == 10:
        label_name = 'ly_2_sitt'
    elif raw_label == 11:
        label_name = 'stand_2_ly'
    else:
        label_name = 'ly_2_stand'

    return label_name


def name_mapper_melbourne(raw_label):
    if raw_label == 1:
        label_name = 'Bourke Street Mall (North)'
    elif raw_label == 2:

        label_name = 'Southern Cross Station'
    elif raw_label == 3:

        label_name = 'New Quay'
    elif raw_label == 4:

        label_name = 'Flinders St Station Underpass'
    elif raw_label == 5:

        label_name = 'QV Market-Elizabeth (West)'
    elif raw_label == 6:

        label_name = 'Convention/Exhibition Centre'
    elif raw_label == 7:

        label_name = 'Chinatown-Swanston St (North)'
    elif raw_label == 8:

        label_name = 'Webb Bridge'
    elif raw_label == 9:

        label_name = 'Tin Alley-Swanston St (West)'
    else:

        label_name = 'Southbank'

    return label_name

def database_creator(data, num_sec):  # base de datos HAR
    # 16 Hz

    list_len = []
    list_seq = []

    list_pac = []

    for i in range(8):
        len, n = mul_n(data[i][1].shape[1], num_sec)
        list_len = list_len + [len]
        list_seq = list_seq + [n]

        dat_pac = data[i][0][0, :len]
        activities = data[i][1][0, :len]

        #scaler = MinMaxScaler()
        #dat_pac_norm = scaler.fit_transform(dat_pac.reshape(-1, 1))

        if i == 0:
            x_np = np.reshape(dat_pac, (n, 16 * num_sec))
            y_np = np.reshape(activities, (n, 16 * num_sec))
        else:
            x_np = np.vstack((x_np, np.reshape(dat_pac, (n, 16 * num_sec))))
            y_np = np.vstack((y_np, np.reshape(activities, (n, 16 * num_sec))))

    count_disc = 0
    labels = []
    labels_name = []
    for j in range(y_np.shape[0]):
        act_pac = y_np[j, :]
        sig_pac = x_np[j, :]
        uni = np.unique(act_pac)

        if j == 0:

            if uni.shape[0] == 1:
                if uni[0] == 111 or uni[0] == 2111 or uni[0] == 2112 or uni[0] == 12 or uni[0] == 13:
                    X = x_np[j, :]
                    Y = y_np[j, :]
                    lab, lab_name = label_mapper(uni[0])
                    labels = labels + [lab]
                    labels_name = labels_name + [lab_name]
            else:
                count_disc = count_disc + 1
        else:
            if uni.shape[0] == 1:
                if uni[0] == 111 or uni[0] == 2111 or uni[0] == 2112 or uni[0] == 12 or uni[0] == 13:
                    X = np.vstack((X, sig_pac))
                    Y = np.vstack((Y, act_pac))
                    lab, lab_name = label_mapper(uni[0])
                    labels = labels + [lab]
                    labels_name = labels_name + [lab_name]
            else:
                count_disc = count_disc + 1

    return X, Y, labels, labels_name, count_disc, x_np.shape[0]


def database_creator_unsupervised(data, num_sec):
    # 16 Hz

    list_len = []
    list_seq = []

    list_pac = []

    for i in range(8):
        len, n = mul_n(data[i][1].shape[1], num_sec)
        list_len = list_len + [len]
        list_seq = list_seq + [n]

        dat_pac = data[i][0][0, :len]
        activities = data[i][1][0, :len]

        if i == 0:
            x_np = np.reshape(dat_pac, (n, 16 * num_sec))
            y_np = np.reshape(activities, (n, 16 * num_sec))
        else:
            x_np = np.vstack((x_np, np.reshape(dat_pac, (n, 16 * num_sec))))
            y_np = np.vstack((y_np, np.reshape(activities, (n, 16 * num_sec))))

    labels_mode = []

    for j in range(y_np.shape[0]):
        act_pac = y_np[j, :]
        sig_pac = x_np[j, :]

        m = mode(act_pac)

        lab, lab_name = label_mapper(m)
        labels_mode = labels_mode + [lab]

    return x_np, labels_mode, list_len, list_seq


def label_sample(labels, X):
    uni = np.unique(labels)
    labels_np = np.asarray(labels)

    signals = np.ones((uni.shape[0], X.shape[1]))
    labels_sig = []

    for i in range(uni.shape[0]):
        mask = labels_np == uni[i]

        signals[i, :] = X[mask][0, :]
        labels_sig = labels_sig + [labels_np[mask][0]]

    return signals, labels_sig


def sample_balanced(labels, n):
    # labels: targets (numpy)
    # n: number of elements per class

    # returns a list of lists (indexes) that retrieves n elements of each class

    uni = np.unique(labels)
    index = np.arange(labels.shape[0])

    counts_class = []

    labels_list = labels.tolist()

    for i in uni:
        counts_class = counts_class + [labels_list.count(i)]

    min_count = min(counts_class)

    if n > min_count:
        sep = min_count
        print('Separator set to min_count')
    else:
        sep = n

    index_list = []

    for i in uni:
        mask = labels == i

        indexes = index[mask][:sep]

        index_list = index_list + [indexes]

    return index_list
