import pandas as pd

import model
from model import *

def run_experiments(K, seed):
    # load data
    first_ref = pd.read_csv('./reference/first_degree_ids.csv', sep=",", index_col=0)
    second_ref = pd.read_csv('./reference/second_degree_ids.csv', sep=",", index_col=0)
    third_ref = pd.read_csv('./reference/third_degree_ids.csv', sep=",", index_col=0)
    unrelated_ref = pd.read_csv('./reference/unrelated_ids.csv', sep=",", index_col=0)

    first_degree = pd.read_csv('./Relationships/first_degree_ids.csv', sep=",",
                               index_col=0)
    second_degree = pd.read_csv('./Relationships/second_degree_ids.csv', sep=",",
                                index_col=0)
    third_degree = pd.read_csv('./Relationships/third_degree_ids.csv', sep=",",
                               index_col=0)
    unrelated_degree = pd.read_csv('./Relationships/unrelated_ids.csv', sep=",",
                                   index_col=0)

    real_ids = [first_degree, second_degree, third_degree, unrelated_degree]
    ref_ids = [first_ref, second_ref, third_ref, unrelated_ref]

    df_real = pd.read_csv(
        "./Relationships/mixture_with_all_relatives_3000_SNPs.csv", sep=",",
        index_col=0)
    df_ref = pd.read_csv('./reference/mixture_with_all_relatives_3000_SNPs.csv', index_col=0)
    df_ref = df_ref.reindex(df_real.index)
    t = len(df_real)//K

    # obtain coefficients
    INK_coefs, ink_threshold = model.INK_coeffcients(ref_ids, real_ids, df_real, df_ref, K, 0, './results/INK/coefs/', seed, eps_flag=False)
    PPKI_coefs, ppki_threshold = model.PPKI_coef(ref_ids, real_ids, df_real, df_ref, K, 5, './results/PPKI/coefs/', seed, eps_flag=True)

    # obtain the label at each iteration
    INK_label = fit_iter(INK_coefs, ink_threshold, 'incre_simu', t, real_ids, './results/INK/labels/')
    PPKI_label = fit_iter(PPKI_coefs, ppki_threshold, 'ppki_simu', t, real_ids, './results/PPKI/labels/')

    # obtain the final label
    fit(INK_label, 'weighted_average', './results/INK/labels/')
    fit(PPKI_label, 'recent', './results/PPKI/labels/')

if __name__ == '__main__':
    run_experiments(100, 42)

