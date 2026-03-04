#!/usr/bin/env python3

import numpy as np
from scipy.stats import skewnorm


def seq_to_dig(read_seq):
    base_map = {'A': 0, 'T': 1, 'U': 1, 'G': 2, 'C': 3}
    return [base_map.get(base.upper(), -1) for base in read_seq]


def weibull_geom(a, pr, weight, size=10000):
    n_weibull = int(size * weight)
    n_geom = size - n_weibull

    weibull_samples = np.ceil(np.random.weibull(a, n_weibull)).astype(int)
    geom_samples = np.random.geometric(pr, n_geom)

    combined = np.concatenate([weibull_samples, geom_samples])
    np.random.shuffle(combined)
    return combined


def pois_geom(lam, pr, weight, size=10000):
    n_pois = int(size * weight)
    n_geom = size - n_pois

    pois_samples = 1 + np.random.poisson(lam, n_pois)
    geom_samples = np.random.geometric(pr, n_geom)

    combined = np.concatenate([pois_samples, geom_samples])
    np.random.shuffle(combined)
    return combined


def mix_weibul_distr_fit(params, real_distr):
    if len(real_distr) == 0 or np.all(real_distr <= 0):
        return 1e6

    a, pr, weight = params

    max_val = max(30, int(np.percentile(real_distr[real_distr > 0], 99)) + 5)
    bins = np.arange(0, max_val + 1)

    real_hist, _ = np.histogram(real_distr, bins=bins, density=True)

    sim_distr = weibull_geom(a, pr, weight, size=max(10000, len(real_distr) * 10))
    sim_hist, _ = np.histogram(sim_distr, bins=bins, density=True)

    min_len = min(len(real_hist), len(sim_hist))
    return np.mean(np.abs(real_hist[:min_len] - sim_hist[:min_len]))


def mix_pois_distr_fit(params, real_distr):
    if len(real_distr) == 0 or np.all(real_distr <= 0):
        return 1e6

    lam, pr, weight = params

    max_val = max(20, int(np.percentile(real_distr[real_distr > 0], 99)) + 5)
    bins = np.arange(0, max_val + 1)

    real_hist, _ = np.histogram(real_distr, bins=bins, density=True)
    sim_distr = pois_geom(lam, pr, weight, size=max(10000, len(real_distr) * 10))
    sim_hist, _ = np.histogram(sim_distr, bins=bins, density=True)

    min_len = min(len(real_hist), len(sim_hist))
    return np.mean(np.abs(real_hist[:min_len] - sim_hist[:min_len]))


def digs_to_err(digit):
    mapping = {'0': 'match', '1': 'mis', '2': 'ins', '3': 'del',
               0: 'match', 1: 'mis', 2: 'ins', 3: 'del'}
    return mapping.get(digit, 'unknown')


def err_type(df_row):
    base = str(df_row['base']).lower()
    if base in ['a', 't', 'g', 'c', 'u']:
        return 1
    elif df_row['read'] > 1:
        return 2
    elif df_row['ref'] > 1:
        return 3
    else:
        return 0


def err_seq(df_row):
    base_upper = str(df_row['base']).upper()
    base_lower = str(df_row['base']).lower()

    if df_row['ref'] == 1 and df_row['read'] == 1 and base_upper in ['A', 'T', 'G', 'C', 'U']:
        return '0'

    if df_row['ref'] == 1 and df_row['read'] == 1 and base_lower in ['a', 't', 'g', 'c', 'u']:
        return '1'

    if df_row['read'] > 1:
        ins_len = int(df_row['read'] - 1)
        suffix = '1' if base_lower in ['a', 't', 'g', 'c', 'u'] else ''
        return '2' * ins_len + suffix

    if df_row['ref'] > 1:
        del_len = int(df_row['ref'] - 1)
        suffix = '1' if base_lower in ['a', 't', 'g', 'c', 'u'] else ''
        return '3' * del_len + suffix

    return '0'


def mix_hp_distr_fit(params, ref_len, observed_shifts):
    if len(observed_shifts) == 0:
        return 1e6

    mu_scale, sigma_base = params

    sigma = sigma_base * np.sqrt(max(1.0, float(ref_len)))
    mu = mu_scale * np.log1p(float(ref_len))

    max_shift = max(20, int(0.4 * ref_len) + 10)
    bins = np.arange(-max_shift, max_shift + 1)

    real_hist, _ = np.histogram(observed_shifts, bins=bins, density=True)

    sim_shifts = np.round(np.random.normal(mu, sigma, size=max(5000, len(observed_shifts) * 5))).astype(int)
    sim_shifts = sim_shifts[(sim_shifts >= -max_shift) & (sim_shifts <= max_shift)]

    if len(sim_shifts) == 0:
        return 1e6

    sim_hist, _ = np.histogram(sim_shifts, bins=bins, density=True)

    min_len = min(len(real_hist), len(sim_hist))
    return np.mean(np.abs(real_hist[:min_len] - sim_hist[:min_len]))


def hp_qual_distr_fit_skewed_normal(params, real_q_arr):
    if len(real_q_arr) == 0:
        return 1e6

    valid_q = real_q_arr[(real_q_arr >= 0) & (real_q_arr <= 40)]
    if len(valid_q) == 0:
        return 1e6

    loc, scale, shape = params

    bins = np.arange(0, 41)

    real_hist, _ = np.histogram(valid_q, bins=bins, density=True)

    sim_distr = skewnorm.rvs(shape, loc=loc, scale=scale, size=max(5000, len(valid_q) * 5))
    sim_distr = np.clip(sim_distr, 0, 40)  # Обрезка вне диапазона

    sim_hist, _ = np.histogram(sim_distr, bins=bins, density=True)

    min_len = min(len(real_hist), len(sim_hist))
    return np.mean(np.abs(real_hist[:min_len] - sim_hist[:min_len]))
