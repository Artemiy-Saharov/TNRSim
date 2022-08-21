import numpy as np
import pandas as pd
import pysam
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.stats import geom
from pydtmc import MarkovChain
import random
from scipy.optimize import minimize
from math import ceil
from re import search
import gffutils
from Bio.Seq import reverse_complement, complement
from multiprocessing import Pool
from time import perf_counter
import sys
from statistics import mean
from scipy.stats import skewnorm


def seq_to_dig(read_seq):
    def base_to_dig(letter):
        if letter == 'A': return 0
        if letter == 'T': return 1
        if letter == 'G': return 2
        if letter == 'C': return 3

    return list(map(base_to_dig, list(read_seq)))

def weibull_geom(a, pr, weight):
    rand_arr = np.append(np.ceil(np.random.weibull(a, round(10000*weight))),
                                 np.random.geometric(pr, round(10000*((1-weight)))))
    np.random.shuffle(rand_arr)
    return rand_arr

def pois_geom(lam, pr, weight):
    rand_arr = np.append(1 + np.random.poisson(lam, round(10000*weight)),
                                 np.random.geometric(pr, round(10000*((1-weight)))))
    np.random.shuffle(rand_arr)
    return rand_arr


def mix_weibul_distr_fit(pl, real_distr):
    loss = 0
    if pl[0] <= 0: pl[0] = 0.001
    #pl[0] = 0.99 if pl[0] >= 1
    if pl[1] <= 0: pl[1] = 0.001
    if pl[1] >= 1: pl[1] = 0.99
    if pl[2] < 0: pl[2] = 0
    if pl[2] > 1: pl[2] = 1
    for i in range(30):
        loss += np.sum(np.abs(np.histogram(real_distr,
                                    density=True)[0] - np.histogram(weibull_geom(pl[0], pl[1], pl[2]), density=True)[0]))
    return loss/30


def mix_pois_distr_fit(pl, real_distr):
    loss = 0
    if pl[0] <= 0: pl[0] = 0.001
    #pl[0] = 0.99 if pl[0] >= 1
    if pl[1] <= 0: pl[1] = 0.001
    if pl[1] >= 1: pl[1] = 0.99
    if pl[2] < 0: pl[2] = 0
    if pl[2] > 1: pl[2] = 1
    for i in range(30):
        loss += np.sum(np.abs(np.histogram(real_distr,
                                    density=True, bins=5)[0] - np.histogram(pois_geom(pl[0], pl[1], pl[2]), density=True, bins=5)[0]))
    return loss/30

def digs_to_err(diget):
    if diget == '0':
        return 'match'
    if diget == '1':
        return 'mis'
    if diget == '2':
        return 'ins'
    if diget == '3':
        return 'del'

def err_type(df_row):
    err = 10
    if df_row['base'] in ['a', 't', 'g', 'c']: #mismatch
        err = 1
    elif df_row['read'] > 1: #insertion
        err = 2
    elif df_row['ref'] > 1: #deletion
        err = 3
    else: #match
        err = 0
    assert(err != 10)
    #return [df_row['read'], df_row['ref'], df_row['base'], err] #index=['read', 'ref', 'base', 'error'])
    return err


def err_seq(df_row):
    err = 10
    if df_row['ref'] == 1 and df_row['read'] == 1 and df_row['base'] in ['A', 'T', 'G', 'C']:
        return '0'
    if df_row['ref'] == 1 and df_row['read'] == 1 and df_row['base'] in ['a', 't', 'g', 'c']:
        return '1'
    if df_row['ref'] > 1 and df_row['base'] in ['A', 'T', 'G', 'C']:  # insertion
        return '2'*int((df_row['ref']-1))
    if df_row['read'] > 1 and df_row['base'] in ['A', 'T', 'G', 'C']:  # deletion
        return '3'*int((df_row['read']-1))
    if df_row['ref'] > 1 and df_row['base'] in ['a', 't', 'g', 'c']:  # insertion and mismatch after
        return '2'*int((df_row['ref']-1)) + '1'
    if df_row['read'] > 1 and df_row['base'] in ['a', 't', 'g', 'c']:  # deletion and mismatch after
        return '3'*int((df_row['read']-1)) + '1'





def mix_hp_distr_fit(pl, real_hp_distr):
    loss = 0
    if pl[1] <= 0: pl[1] = 0.001
    for i in range(30):
        loss += np.sum(np.abs(np.histogram(real_hp_distr,
            density=True, bins=60)[0] - np.histogram(np.random.normal(pl[0], pl[1], 300), density=True, bins=60)[0]))
    return loss/30



def hp_qual_distr_fit_skewed_normal(pl, real_q_arr):
    loss = 0
    if pl[1] < 0: pl[1] = 0
    for i in range(30):
        loss += np.sum(np.abs(np.histogram(real_q_arr, density=True,
                                           bins=60)[0] - np.histogram(skewnorm.rvs(pl[2],
                                           loc=pl[0], scale=pl[1], size=300), density=True, bins=60)[0]))
    return loss/30


def parse_ano(ano_fname, ano_format):
    inp_ano = pd.read_csv(ano_fname, sep=';', header=None,
                          comment='#', usecols=[0, 1], names=['merged', 'parent'], dtype=np.str_)
    inp_ano[['chr', 'ano_type', 'class', 'start', 'end', 'dot', 'strand', 'dot2', 'ID']] = inp_ano['merged'].str.split(
        '\t', expand=True)
    inp_ano.drop(['merged', 'ano_type', 'dot', 'dot2'], axis=1, inplace=True)
    inp_ano[['start', 'end']] = inp_ano[['start', 'end']].astype(np.int32)
    inp_ano['strand'] = inp_ano['strand'].astype('category')
    inp_ano.drop(inp_ano[~inp_ano['class'].isin(['gene', 'exon', 'transcript'])].index, inplace=True)

    if ano_format == 'gtf':
        genes_ano = inp_ano[inp_ano['class'] == 'gene'].copy()
        genes_ano.drop(['parent'], axis=1, inplace=True)
        genes_ano['ENSG'] = genes_ano['ID'].apply(lambda x: x[9:-1])
        genes_ano.drop(['ID'], axis=1, inplace=True)

        exons = inp_ano[inp_ano['class'] == 'exon'].copy()
        exons['ENSG'] = exons['ID'].apply(lambda x: x[9:-1])
        exons.drop(['parent', 'ID'], axis=1, inplace=True)
        exons.drop_duplicates(inplace=True)

        tran_lens = inp_ano[inp_ano['class'] == 'transcript'].copy()
        tran_lens['ENSG'] = tran_lens['ID'].apply(lambda x: x[9:-1])
        tran_lens['ENST'] = tran_lens['parent'].apply(lambda x: x[16:-1])
        tran_lens.drop(['parent', 'ID', 'chr', 'class', 'start', 'end', 'strand'], axis=1, inplace=True)
        tran_lens = tran_lens.groupby(by='ENSG', as_index=False).agg(lambda x: list(x))

        exons_df = inp_ano[inp_ano['class'] == 'exon'].copy()
        exons_df['ENST'] = exons_df['parent'].apply(lambda x: x[16:-1])
        exons_df['length'] = exons_df['end'] - exons_df['start']
        exons_df.drop(['ID', 'chr', 'class', 'start', 'end', 'strand', 'parent'], axis=1, inplace=True)
        exons_df = exons_df.groupby(by='ENST', as_index=False).sum()

        dict_exons = dict(zip(exons_df['ENST'], exons_df['length']))
        tran_lens['Length'] = tran_lens['ENST'].apply(lambda x: [dict_exons[i] for i in x])

    elif ano_format == 'gff3':
        genes_ano = inp_ano[inp_ano['class'] == 'gene'].copy()
        genes_ano['ENSG'] = genes_ano['ID'].apply(lambda x: x[3:]).copy()
        genes_ano.drop(['parent', 'ID'], axis=1, inplace=True)


        tran_lens = inp_ano[inp_ano['class'] == 'transcript'].copy()
        tran_lens['ENST'] = tran_lens['ID'].apply(lambda x: x[3:])
        tran_lens['ENSG'] = tran_lens['parent'].apply(lambda x: x[7:])
        tran_lens.drop(['parent', 'ID', 'chr', 'class', 'start', 'end', 'strand'], axis=1, inplace=True)

        exons = inp_ano[inp_ano['class'] == 'exon'].copy()
        exons['ENST'] = exons['parent'].apply(lambda x: x[7:])
        exons.drop(['parent', 'ID'], axis=1, inplace=True)
        exons.drop_duplicates(inplace=True)
        tx2gen = dict(zip(tran_lens['ENST'], tran_lens['ENSG']))
        exons['ENSG'] = exons['ENST'].apply(lambda x: tx2gen[x])

        tran_lens = tran_lens.groupby(by='ENSG', as_index=False).agg(lambda x: list(x))

        exons_df = inp_ano[inp_ano['class'] == 'exon'].copy()
        exons_df['length'] = exons_df['end'] - exons_df['start']
        exons_df['ID'] = exons_df['ID'].apply(lambda x: x[3:])
        exons_df['parent'] = exons_df['parent'].apply(lambda x: x[7:])
        exons_df.drop(['ID', 'chr', 'class', 'start', 'end', 'strand'], axis=1, inplace=True)
        exons_df = exons_df.groupby(by='parent', as_index=False).sum()

        dict_exons = dict(zip(exons_df['parent'], exons_df['length']))
        tran_lens['Length'] = tran_lens['ENST'].apply(lambda x: [dict_exons[i] for i in x])

    else:
        genes_ano, tran_lens = None, None
    del inp_ano

    return genes_ano, tran_lens, exons
