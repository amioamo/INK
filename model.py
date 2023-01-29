import numpy as np

import utils
from utils import *

def INK_coeffcients(ref_list, real_list, df_ref, df_real, K, eps, output_dir, seed, eps_flag=False):
    """
    :param ref_list: list of simulated user IDs
    :param real_list: list of real (interested) user IDs
    :param df_real: dataframe of real users genomic data, SNP (row) * User (column)
    :param df_ref: dataframe of simulated users genomic data
    :param K: batch size, # of SNPs used for each iteration
    :param eps: noise level in PPKI
    :param output_dir: path to save results
    :param seed:
    :param eps_flag: flag for noise
    :return: None
    """
    T = len(df_real) // K  # number of iterations
    real = df_real.copy()
    ref = df_ref.copy()

    labels = {}
    if not os.path.isdir(output_dir + str(K)):
        os.makedirs(output_dir + str(K))

    for i_iter in range(T):
        if not eps_flag:
            copy_real = real.sample(int(K * (i_iter + 1)), random_state=seed)
            copy_ref = ref.sample(int(K * (i_iter + 1)), random_state=seed)

        else:  # introduce LDP to the training (simulated SNPs) and testing (real SNPs) set
            p = np.exp(eps) / (np.exp(eps) + 2)
            q = 1 / (np.exp(eps) + 2)

            copy_real = real.sample(int(K * (i_iter + 1)), random_state=seed)
            copy_ref = ref.sample(int(K * (i_iter + 1)), random_state=seed)

            user_id_real = copy_real.columns
            user_id_ref = copy_ref.columns

            for j in range(len(user_id_real)):
                copy_real[str(user_id_real[j])] = copy_real.apply(lambda x: randomized_response(x[str(user_id_real[j])],
                                                                                                p, q), axis=1)
            for j in range(len(user_id_ref)):
                copy_ref[str(user_id_ref[j])] = copy_ref.apply(lambda x: randomized_response(x[str(user_id_ref[j])],
                                                                                             p, q), axis=1)

        # First, compute the coefficients on training (simulated) set and obtain the thresholds
        dis = []
        for order, ids in enumerate(ref_list):
            coef_ref = calculate_coef(copy_ref, ids)
            dis.append(list(coef_ref.values()))

        threshold = select_threshold(np.array(dis), 0)
        if not eps_flag:
            np.savetxt(output_dir + str(K) + '/thresholds' + str(i_iter) + '.txt', threshold)
        else:
            np.savetxt(output_dir + str(K) + '/thresholds' + str(i_iter) + '_with_noise.txt', threshold)

        # then, compute the coefficients on testing set and save the results
        for order, ids in enumerate(real_list):
            coef_real = calculate_coef(copy_real, ids)

            for i, couple in enumerate(ids.values):
                if (couple[0], couple[1]) in labels:
                    labels[(couple[0], couple[1])].append(coef_real[(couple[0], couple[1])])
                else:
                    labels[(couple[0], couple[1])] = [coef_real[(couple[0], couple[1])]]
    if not eps_flag:
        with open(output_dir + str(K) + '/results.pkl', 'wb') as f:
            pickle.dump(labels, f)
    else:
        with open(output_dir + str(K) + '/results_with_noise.pkl', 'wb') as f:
            pickle.dump(labels, f)

    return labels, threshold

def PPKI_coef(ref_list, real_list, df_real, df_ref, K, eps, output_dir, seed, eps_flag=False):
    """
    Implementation for PPKI
    """
    T = len(df_real) // K
    real = df_real.copy()
    ref = df_ref.copy()

    labels = {}
    if not os.path.isdir(output_dir + str(K)):
        os.makedirs(output_dir + str(K))

    for i_iter in range(T):
        if not eps_flag:
            copy_real = real.sample(int(K * (i_iter + 1)), random_state=seed)
            copy_ref = ref.sample(int(K * (i_iter + 1)), random_state=seed)

        else:  # introduce LDP to the testing set (true SNPs)
            p = np.exp(eps) / (np.exp(eps) + 2)
            q = 1 / (np.exp(eps) + 2)

            copy_real = real.sample(int(K * (i_iter + 1)), random_state=seed)
            copy_ref = ref.sample(int(K * (i_iter + 1)), random_state=seed)

            user_id_real = copy_real.columns
            user_id_ref = copy_ref.columns

            for j in range(len(user_id_real)):
                copy_real[str(user_id_real[j])] = copy_real.apply(lambda x: randomized_response(x[str(user_id_real[j])],
                                                                                                p, q), axis=1)
            for j in range(len(user_id_ref)):
                copy_ref[str(user_id_ref[j])] = copy_ref.apply(lambda x: randomized_response(x[str(user_id_ref[j])],
                                                                                             p, q), axis=1)

        # first, compute the coefficients on training set and obtain the thresholds
        dis = []
        for order, ids in enumerate(ref_list):
            coef_ref = calculate_coef(copy_ref, ids)
            dis.append(list(coef_ref.values()))

        threshold = select_threshold(np.array(dis), 0)
        if not eps_flag:
            np.savetxt(output_dir + str(K) + '/thresholds' + str(i_iter) + '.txt', threshold)
        else:
            np.savetxt(output_dir + str(K) + '/thresholds' + str(i_iter) + '_with_noise.txt', threshold)
        # then, compute the coefficients on testing set and do the classification
        for order, ids in enumerate(real_list):
            coef_real = calculate_coef(copy_real, ids)

            for i, couple in enumerate(ids.values):
                if (couple[0], couple[1]) in labels:
                    labels[(couple[0], couple[1])].append(coef_real[(couple[0], couple[1])])
                else:
                    labels[(couple[0], couple[1])] = [coef_real[(couple[0], couple[1])]]
    if not eps_flag:
        with open(output_dir + str(K) + '/results.pkl', 'wb') as f:
            pickle.dump(labels, f)
    else:
        with open(output_dir + str(K) + '/results_with_noise.pkl', 'wb') as f:
            pickle.dump(labels, f)

    return labels, threshold

def fit_iter(coefs, threshold, method, T, real_list, out_dir, sim=True, eps_flag=False):
    """
    Given the coefficients and thresholds at each iteration and algorithms
    Output the labels at each iteration
    """
    # load the coefficients obtained from INK_coeffcients
   # if not eps_flag:
    #    with open(in_dir + 'results.pkl', 'rb') as f:
    #        coefs = pickle.load(f)
   # else:
    #    with open(in_dir + 'results_with_noise.pkl', 'rb') as f:
     #       coefs = pickle.load(f)

    labels = {}
    for i_iter in range(T):

        for order, ids in enumerate(real_list):

            for i, couple in enumerate(ids.values):
                if sim:
                    if (couple[0], couple[1]) in labels:
                        labels[(couple[0], couple[1])].append(
                            utils.classification(threshold, coefs[(couple[0], couple[1])][i_iter]))
                    else:
                        labels[(couple[0], couple[1])] = [
                            utils.classification(threshold, coefs[(couple[0], couple[1])][i_iter])]
                else:
                    if (couple[0], couple[1]) in labels:
                        labels[(couple[0], couple[1])].append(
                            utils.classification(threshold, coefs[(couple[0], couple[1])][i_iter], sim=False))
                    else:
                        labels[(couple[0], couple[1])] = [
                            utils.classification(threshold, coefs[(couple[0], couple[1])][i_iter], sim=False)]

    with open(out_dir + method + '_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    return labels

def fit(labels, weight, out_dir):
    """
    Given the labels at each iteration
    Calculate the final prediciton with different methods, such as, 'majority', 'weighted_average','recent'

    return the final predictions for each pair of individuals
    """
   # with open(in_dir, 'rb') as f:
   #     labels = pickle.load(f)

    T = len(list(labels.items())[0][1])
    final_res = []

    if weight == 'majority':
        for i_iter in range(T):
            res = [mode(list(labels.values())[i][:(i_iter + 1)]) for i in range(len(labels.values()))]
            final_res.append(res)
    elif weight == 'recent':
        for i_iter in range(T):
            res = [list(labels.values())[i][i_iter] for i in range(len(labels.values()))]
            final_res.append(res)
    else: # 'weighted_average'
        for i_iter in range(T):
            w = [1 / (1 + n) for n in range(i_iter + 1)][::-1]
            wg = [k / np.sum(w) for k in w]
            res = [np.argmax(np.bincount(list(labels.values())[i][:(i_iter + 1)], weights=wg)) for i in
                   range(len(labels.values()))]
            final_res.append(res)

    np.savetxt(out_dir + 'predictions.txt', np.array(final_res))

   # return final_res
