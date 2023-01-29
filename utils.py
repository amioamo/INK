import pickle
import random
import os
from itertools import cycle
import itertools
from statistics import mode


def calculate_phi(df, user1, user2, userList):
    """

    :param df: dataframe of genomic data, rows: SNPs, columns: users
    :param user1: userID
    :param user2: userID
    :param userList: list of users contained in df
    :return: KING coefficient of user1 and user 2
    """
    n11 = 0
    n02 = 0
    n20 = 0
    n_1 = 0
    n1_ = 0
    index1 = userList.index(user1)
    index2 = userList.index(user2)
    for row in df.itertuples(index=False):
        if row[index1] == 1 and row[index2] == 1:
            n11 += 1
        if row[index1] == 0 and row[index2] == 2:
            n02 += 1
        if row[index1] == 2 and row[index2] == 0:
            n20 += 1
        if row[index1] == 1:
            n1_ += 1
        if row[index2] == 1:
            n_1 += 1
    if n1_ != 0:
        phi = (2 * n11 - 4 * (n02 + n20) - n_1 + n1_) / (4 * n1_)
    else:
        phi = -999
    return phi, n1_


def randomized_response(val, p, q):
    """
    LDP variant from PPKI
    :param val: SNP value
    :param p:
    :param q:
    :return:
    """
    rand_val = np.random.uniform(0, 1)
    new_val = val
    if rand_val > p:
        if val == 0:
            new_val = 1
        elif val == 2:
            new_val = 1
        elif val == 1:
            if rand_val > p + q:
                new_val = 0
            else:
                new_val = 2
    return new_val


def select_threshold(dis, alpha):
    """
    :param dis: ndarray, KING coef distribution for each kinship degree
    :param alpha: selected quantile
    :return: cutoff values of each degree of kinship relationship
    """
    thresholds = []
    for k in range(dis.shape[0]):
        if k == 0:
            thresholds.append([np.quantile(dis[k], alpha), 0.5, np.mean(dis[k])])
        else:
            thresholds.append([np.quantile(dis[k], alpha), np.quantile(dis[k], 1 - alpha), np.mean(dis[k])])
    return thresholds


def calculate_coef(df, ids):
    """
    return dict coef{(useri, userj):coef}

    """
    userList = list(df.columns)
    phis_ = {}
    for couple in ids.values:
        phi_val_left = calculate_phi(df, str(couple[0]), str(couple[1]), userList)
        phi_val_right = calculate_phi(df, str(couple[1]), str(couple[0]), userList)
        phi_val = max(phi_val_left[0], phi_val_right[0])  # keeping the largest \phi

        phis_[(couple[0], couple[1])] = phi_val
    return phis_


def classification(thresholds, x, sim=True):
    """
    thresholds: K*2 arrays, k row is the threshold for the k-order degree relatives (unrelated as 4th order)
    x: the coef computed on real pair of samples
    sim: flag for using simulated set
    return: res, the class label assigned to this pair
    """
    #  classification based on simulated set
    if sim:
        res = []
        dis_to_center = []
        for k in range(len(thresholds)):
            if thresholds[k][0] < x and x < thresholds[k][1]:
                dis_to_center.append(np.abs(x - thresholds[k][2]))
                res.append(k + 1)
        # print(res)
        if len(res) == 0:  # handle exceptions: for some batches, the coef of some un-related individuals are too small
            return 4
        # chose one label based on the ditance to the center
        return res[np.argmin(dis_to_center)]
    else:
        res = []
        for k in range(len(thresholds)):
            if k != len(thresholds) - 1:
                if x >= thresholds[k]:
                    res.append(k + 1)
            else:
                if x < thresholds[k]:
                    res.append(k + 1)
            # print(res)
            return np.min(res)  # chose the smallest label

