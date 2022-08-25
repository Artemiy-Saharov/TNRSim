#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pysam
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.optimize import curve_fit
from scipy import cluster
from scipy.stats import geom
from pydtmc import MarkovChain
import random
from scipy.optimize import minimize
from math import ceil
from re import search
from Bio.Seq import complement
from multiprocessing import Pool
from time import perf_counter
import sys
from statistics import mean
from scipy.stats import skewnorm
import argparse
from fitting_functions import *


def get_hp_qual_from_reads(hp_pos):
    hp_start, hp_end = int(hp_pos.split(':')[0]), int(hp_pos.split(':')[1])
    chrom, strand, cur_base = hp_pos.split(':')[2], hp_pos.split(':')[3], hp_pos.split(':')[4]
    #print('started::')
    hp_base = cur_base if strand == '+' else complement(cur_base)
    hp_quals = str(hp_end-hp_start) + ':' + hp_base + ';'
    bam_open_file = open(bam_fname, 'rb')
    ali = pysam.AlignmentFile(bam_open_file, mode='rb')
    for read in ali.fetch(chrom, hp_start-5, hp_end+5):
        if read.seq == None:
            continue
        arr_pairs = np.array(read.get_aligned_pairs(matches_only=True))
        pre_hp_pos = arr_pairs[:,0][(arr_pairs[:,1]<hp_start)&(arr_pairs[:,1]>hp_start-15)]
        if len(pre_hp_pos) < 2:
            continue
        hp_reg_seq = read.seq[pre_hp_pos[-1]-3: pre_hp_pos[-1]+30]
        for hp_in_read_len in range(50):
            if search(cur_base*hp_in_read_len, hp_reg_seq):
                continue
            else:
                break
        for hp_in_read_start in range(pre_hp_pos[-1]-3, pre_hp_pos[-1]+30):
            #print('HP is {}'.format(cur_base*(hp_in_read_len-1)))
            #print(read.seq[hp_in_read_start:hp_in_read_start+hp_in_read_len-1])
            if read.seq[hp_in_read_start:hp_in_read_start+hp_in_read_len-1] == cur_base*(hp_in_read_len-1):
                hp_quals += ''.join(map(lambda x: str(x)+':', [read.query_qualities[i] for i
                             in range(hp_in_read_start, hp_in_read_start+hp_in_read_len-1)])) + ';'
                break
    #print('processed all reads::')
    ali.close()
    bam_open_file.close()
    return hp_quals[:-1]


def get_hp_from_block(inp_hp_block):
    block_output = ''
    bam_open_file = open(bam_fname, 'rb')
    ali = pysam.AlignmentFile(bam_open_file, mode='rb')
    for hp_pos in inp_hp_block.split(';'):

        hp_start, hp_end = int(hp_pos.split(':')[0]), int(hp_pos.split(':')[1])
        chrom, strand, ref_base = hp_pos.split(':')[2], hp_pos.split(':')[3], hp_pos.split(':')[4]
        # print('started::')
        hp_base = ref_base if strand == '+' else complement(ref_base)
        hp_stats = str(hp_end - hp_start) + ':' + hp_base + ':'
        for read in ali.fetch(chrom, hp_start - 5, hp_end + 5):
            if read.seq == None:
                continue
            arr_pairs = np.array(read.get_aligned_pairs(matches_only=True))
            pre_hp_pos = arr_pairs[:, 0][(arr_pairs[:, 1] < hp_start) & (arr_pairs[:, 1] > hp_start - 15)]
            if len(pre_hp_pos) < 2:
                continue
            hp_reg_seq = read.seq[pre_hp_pos[-1] - 3: pre_hp_pos[-1] + 40]
            for hp_in_read_len in range(50):
                if search(ref_base * hp_in_read_len, hp_reg_seq):
                    continue
                else:
                    hp_stats += str(hp_in_read_len - 1) + ':'
                    break
        block_output += hp_stats[:-1] + ';'
        # print('processed all reads::')
    ali.close()
    bam_open_file.close()
    return block_output[:-1]

def estim_peak(N, p):
    return np.sum(N*(np.cumprod(np.full(1000, 1-step*p))))

parser = argparse.ArgumentParser(description='Characterization script')
parser.add_argument('-a', '--annotation', type=str,
                    help='path to .gff3 or .gff file with genome annotation')
#parser.add_argument('-e', '--expression', type=str, help='path to output of featureCounts')
parser.add_argument('-g', '--genome', type=str, help='fasta file with genome sequences')
parser.add_argument('-O', '--output', type=str, default='TNRSim_model.tsv', help='Output model file')
parser.add_argument('-b', '--bam_file', type=str,
                    help='path to input bam file with reads aligned to genome, bam file must be sorted, indexed and has MD tag')
parser.add_argument('--threads', type=int, default=2, help='Number of threads to use')
parser.add_argument('-f', '--fragmentation_only', action='store_true',
                    help='If this flag is specified, estimate only fragmentation probability and exit')
args = parser.parse_args()

threads_num = args.threads
bam_fname = args.bam_file
ano_fname = args.annotation
inp_ano_format = None
if 'gtf' in ano_fname.split('.'):
    inp_ano_format = 'gtf'
elif 'gff3' in ano_fname.split('.'):
    inp_ano_format = 'gff3'
else:
    print('Unrecognized annotation file format. Supported formats are gtf and gff3')
ano, tran_len, exon_ano = parse_ano(ano_fname, inp_ano_format)


os.system('mkdir tmp_tnrsim_files')

stdout = os.system('featureCounts -T {cpu} -a {ant} -t exon -g gene_id -G {gen} --primary -L -o ./tmp_tnrsim_files/sample_exp.txt {bam}'.format(cpu=threads_num, ant=ano_fname, gen=args.genome, bam=bam_fname))


hep_exp = pd.read_csv('./tmp_tnrsim_files/sample_exp.txt', sep='\t', skiprows=1)
hep_exp.rename({hep_exp.columns[6]: 'counts'}, axis='columns', inplace=True)

top_exp = hep_exp[hep_exp['counts'] > 300]['Geneid'].to_numpy()
#print(exon_ano)
#exon_df = exon_ano[exon_ano['ENSG'].isin(top_exp)].copy()
#print(exon_df)

ali = pysam.AlignmentFile(bam_fname, 'rb')
genes_p = {}

#print(set(top_exp) - set(ano['ENSG']))

print('fitting fragmentation probability')

for ensg in top_exp:
    # break
    new_isoform = False
    len_arr = np.array([])
    for read in ali.fetch(ano[ano['ENSG'] == ensg]['chr'].values[0], ano[ano['ENSG'] == ensg]['start'].values[0],
                          ano[ano['ENSG'] == ensg]['end'].values[0]):
        len_arr = np.append(len_arr, read.query_length)

    try:
        isoforms = np.array(tran_len[tran_len['ENSG'] == ensg]['Length'].values[0])
    except:
        continue
    min_len = max(isoforms)
    if min_len < 2000:
        continue
    # print(isoforms)
    n_bins1 = 100
    # print(min_len)
    hist1, bin_edges1 = np.histogram(len_arr[((len_arr < min_len + 500) & (len_arr > 10))], bins=n_bins1)
    bin_w1 = np.diff(bin_edges1)[0]
    step = (min_len + 490)/n_bins1
    # print('Mean: {}'.format(hist1.mean()))
    peaks, _ = find_peaks(hist1, prominence=[50, 1000], width=[1, 15], distance=5)
    # peaks, _ = find_peaks(hist1, threshold=0.1*hist1.mean(), width=[40/bin_w1, 300/bin_w1], prominence=[3, 1000])

    if len(peaks) > 0:
        min_len = bin_edges1[peaks[0]]
        # print(peaks)
        # print('max peak is {}'.format(bin_w1*peaks[-1]))
    else:
        continue
    if bin_w1 * peaks[-1] < 1400:
        continue

    peaks_width_res = peak_widths(hist1, [peaks[-1]], rel_height=0.85)
    popt, pcov = curve_fit(estim_peak,
                           xdata=[0.95 * hist1[round(peaks_width_res[2][0]) - 5:round(peaks_width_res[2][0])].mean()],
                           ydata=[1.1 * hist1[round(peaks_width_res[2][0]):round(peaks_width_res[3][0])].sum()],
                           p0=[0.0004])
    estimated_p = float(*popt)
    zero_N = hist1[round(peaks_width_res[2][0]) - 5:round(peaks_width_res[2][0])].mean() / (
                (1 - (estimated_p * step)) ** round(peaks_width_res[2][0]))
    genes_p.update({ensg: estimated_p})

ali.close()


clust_res = cluster.vq.kmeans2(list(genes_p.values()), k=3, iter=5000, minit='points')
cls, num_points = np.unique(clust_res[1], return_counts=True)
cls = np.delete(cls, np.argmin(num_points))
num_points = np.delete(num_points, np.argmin(num_points))
frag_p = 0
for cl, cl_weigth in zip(cls, num_points):
    frag_p += clust_res[0][cl]*(cl_weigth**2)
frag_p = frag_p/np.square(num_points).sum()

if args.fragmentation_only:
    print('Characterization is completed, fragmentation probability is {}'.format(round(frag_p, 6)))
    os.system('rm -r tmp_tnrsim_files')
    sys.exit(0)






model_dict = {}

print('fitting error profile')

ensg = top_exp[0]

ali = pysam.AlignmentFile(bam_fname, 'rb')
len_arr = np.array([])
sum_match_len_arr = np.array([])
low_match = 0
iters = 0
read_seq = ''
sum_err_arr = np.array([])
ins1_arr = np.array([])
del1_arr = np.array([])
mis_seq = np.array([])
for read in ali.fetch(ano[ano['ENSG'] == ensg]['chr'].values[0], ano[ano['ENSG'] == ensg]['start'].values[0],
                      ano[ano['ENSG'] == ensg]['end'].values[0]):
    try:
        pairs = read.get_aligned_pairs(matches_only=True, with_seq=True)
    except:
        continue
    if len(pairs) / (read.query_length) < 0.94:
        low_match += 1
        continue
    if read.query_length < 7000:
        continue
    len_arr = np.append(len_arr, read.query_length)
    # blocks = read.get_blocks()
    # print(blocks)
    df_pairs = pd.DataFrame(pairs, columns=['read', 'ref', 'base'])
    df_pairs['read'] = df_pairs['read'].diff()
    df_pairs['ref'] = df_pairs['ref'].diff()
    df_pairs.drop(index=0, inplace=True)
    # df_pairs['error'] = np.full(df_pairs.shape[0], 0)
    df_pairs['error'] = df_pairs['base'].apply(lambda x: 0 if x in ['A', 'T', 'C', 'G'] else 1)
    mis_seq = np.append(mis_seq, df_pairs['error'].to_numpy())
    df_pairs['ref'] = df_pairs['ref'].apply(lambda x: 1 if x > 10 else x)
    # df_pairs['ref'][df_pairs['ref']>1].hist(bins=30)
    err_arr = df_pairs.apply(err_type, axis=1).to_numpy()
    sum_err_arr = np.append(sum_err_arr, err_arr)
    match_len_arr = np.diff(np.where(err_arr != 0)[0]) - 1
    match_len_arr = match_len_arr[match_len_arr != 0]
    sum_match_len_arr = np.append(sum_match_len_arr, match_len_arr)

    del1_arr = np.append(del1_arr, df_pairs[df_pairs['ref'] > 1]['ref'].to_numpy() - 1)
    ins1_arr = np.append(ins1_arr, df_pairs[df_pairs['read'] > 1]['read'].to_numpy() - 1)

    iters += 1
    # print(iters)
    read_seq += df_pairs.apply(err_seq, axis=1).sum()
    if iters == 500:
        break
ali.close()
mis_seq = mis_seq.astype(int)

mis1_len = np.array([])
mismatch_len = 0
for i in range(len(mis_seq)):
    if mis_seq[i] == 0:
        if mismatch_len == 0:
            continue
        else:
            mis1_len = np.append(mis1_len, mismatch_len)
            mismatch_len = 0
    else:
        mismatch_len += 1


ins_res = minimize(mix_weibul_distr_fit, args=(ins1_arr),
        x0=[0.7, 0.5, 0.7], bounds=((0, 2), (0.1, 0.95), (0.1, 0.95)), method='Powell',
               tol=0.000001, options={'maxiter':1000})
del_res = minimize(mix_weibul_distr_fit, args=(del1_arr),
        x0=[0.7, 0.5, 0.7], bounds=((0, 2), (0.1, 0.95), (0.1, 0.95)), method='Powell',
               tol=0.000001, options={'maxiter':1000})
mis_res = minimize(mix_pois_distr_fit, args=(mis1_len),
        x0=[0.7, 0.65, 0.8], bounds=((0, 2), (0.1, 0.95), (0.1, 0.95)), method='Powell',
               tol=0.000001, options={'maxiter':1000})
hist_match = np.histogram(sum_match_len_arr[sum_match_len_arr<200], density=True, bins=198)[0]
model_dict.update({'mis_params': list(mis_res.x),
                   'del_params': list(del_res.x),
                   'ins_params': list(ins_res.x),
                   'hist_match': list(hist_match)})

reduced_err_arr = np.array([sum_err_arr[0]])
cur_sym = sum_err_arr[0]
for i in range(1, len(sum_err_arr)):
    if cur_sym==sum_err_arr[i]:
        continue
    else:
        reduced_err_arr = np.append(reduced_err_arr, sum_err_arr[i])
        cur_sym = sum_err_arr[i]

mc = MarkovChain.fit_walk(fitting_type='mle', possible_states=['match', 'mis', 'ins', 'del'],
                          walk=list(map(digs_to_err, list(map(lambda x: str(x)[0], reduced_err_arr)))))

model_dict.update({'err_prob': list(mc.p.round(3)[0, 1:]),
                   'mis_after_ins': [float(mc.p.round(3)[2, 1])],
                   'mis_after_del': [float(mc.p.round(3)[3, 1])]})  # order is mis, ins, del

ali = pysam.AlignmentFile(bam_fname, mode='rb')
ref_genome_seq = pysam.FastaFile(args.genome)
mis_matrix = pd.DataFrame(index=['A', 'T', 'G', 'C'], columns=['A', 'T', 'G', 'C'], dtype='Int64').fillna(0)
iters = 0
chr_seq = ref_genome_seq.fetch(ano[ano['ENSG']==ensg]['chr'].values[0])
for pileup in ali.pileup(ano[ano['ENSG']==ensg]['chr'].values[0], ano[ano['ENSG']==ensg]['start'].values[0],
                         ano[ano['ENSG']==ensg]['end'].values[0], truncate=True,
                         max_depth=8000, min_base_quality=0, fastafile=ref_genome_seq): #ref_genome_seq
    if pileup.n < 20:
        continue
    p_col = list(map(str.upper, pileup.get_query_sequences(mark_matches=True, add_indels=False)))
    p_col = list(map(lambda x: '.' if x == ',' else x, p_col))
    for sym in ['A', 'T', 'G', 'C']:
        mis_matrix.at[chr_seq[pileup.reference_pos], sym] += p_col.count(sym)
    mis_matrix.at[chr_seq[pileup.reference_pos], chr_seq[pileup.reference_pos]] += p_col.count('.')
    iters += 1

ali.close()
ref_genome_seq.close()
mis_matrix= mis_matrix.apply(lambda x: x/x.sum())

model_dict.update({'base_stability': list(mis_matrix.to_numpy(dtype=float).flatten())})

print('firring quality profile')

mis_q = np.array([], dtype=np.int64)
ali = pysam.AlignmentFile(bam_fname, 'rb')
iters = 0
for read in ali.fetch(ano[ano['ENSG'] == ensg]['chr'].values[0], ano[ano['ENSG'] == ensg]['start'].values[0],
                      ano[ano['ENSG'] == ensg]['end'].values[0]):

    try:
        pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    except:
        continue
    if len(pairs)/read.query_length < 0.95:
        continue
    if read.query_length < 6000:
        continue
    df_pairs = pd.DataFrame(pairs, columns=['read', 'ref', 'base'])
    mis_q = np.append(mis_q,
                      [read.query_qualities[i] for i in df_pairs[df_pairs['base'].isin(['a', 't', 'g', 'c'])]['read']])
    iters += 1
    if iters > 300:
        break
ali.close()
mis_q_hist = list(np.histogram(mis_q[(mis_q>0)&(mis_q<41)], density=True, bins=39)[0])


ali = pysam.AlignmentFile(bam_fname, 'rb')
len_arr = np.array([])
low_match = 0
iters = 0
ins_qual_arr = np.array([], dtype=np.int64)
for read in ali.fetch(ano[ano['ENSG'] == ensg]['chr'].values[0], ano[ano['ENSG'] == ensg]['start'].values[0],
                      ano[ano['ENSG'] == ensg]['end'].values[0]):
    try:
        pairs = read.get_aligned_pairs(matches_only=True, with_seq=True)
    except:
        continue
    if len(pairs) / (read.query_length) < 0.94:
        low_match += 1
        continue
    if read.query_length < 7000:
        continue
    len_arr = np.append(len_arr, read.query_length)
    # blocks = read.get_blocks()
    # print(blocks)
    df_pairs = pd.DataFrame(pairs, columns=['read', 'ref', 'base'])

    df_pairs['ref'] = df_pairs['ref'].diff()
    df_pairs.drop(index=0, inplace=True)
    # df_pairs['error'] = np.full(df_pairs.shape[0], 0)
    df_pairs['ref'] = df_pairs['ref'].apply(lambda x: 1 if x > 10 else x)
    # df_pairs['ref'][df_pairs['ref']>1].hist(bins=30)
    iters += 1
    # print(iters)
    df_pairs['read_diff'] = df_pairs['read'].diff()
    # df_pairs.drop(index=0, inplace=True)
    df_ins = df_pairs[df_pairs['read_diff'] > 1]
    for ins_end, ins_len in zip(df_ins['read'], list(map(int, df_ins['read_diff']))):
        ins_qual_arr = np.append(ins_qual_arr, read.query_qualities[ins_end - ins_len + 1:ins_end])

    if iters == 300:
        break
ali.close()

ins_q_hist = list(np.histogram(ins_qual_arr[(ins_qual_arr>0)&(ins_qual_arr<41)], density=True, bins=39)[0])


ali = pysam.AlignmentFile(bam_fname, 'rb')
homopolymers = [[], [], [], []]  # lenght of hp for A, T, G and C
hp_starts = [[], [], [], []]
iters = 0
len_list = []
short_seq_unalig = ''
long_seq_unalig = ''
unalig_reg_list = [[], []]
hps_in_unalig = []
long_unalig_num = 0
bases = ['A', 'T', 'G', 'C']
short_quals = np.array([], dtype=np.int64)
long_quals = np.array([], dtype=np.int64)
start_quals = np.array([], dtype=np.int64)
for read in ali.fetch(ano[ano['ENSG'] == ensg]['chr'].values[0], ano[ano['ENSG'] == ensg]['start'].values[0],
                      ano[ano['ENSG'] == ensg]['end'].values[0]):
    try:
        pairs = read.get_aligned_pairs(with_seq=True)
    except:
        continue
    if len(pairs) / (read.query_length) < 0.9:
        continue
    if read.query_length < 500:
        continue

    len_list.append(read.query_length)
    pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    # print(read.seq)
    # print(pairs[1:10])
    df_pairs = pd.DataFrame(pairs, columns=['read', 'ref', 'base'])
    # orig_orintation = read.get_forward_sequence()
    if pairs[0][0] < 40:
        short_seq_unalig += complement(read.seq[:pairs[0][0]])
        short_quals = np.append(short_quals, read.query_qualities[:pairs[0][0]])
        # if len(read.seq[:pairs[0][0]]) > 5:
    elif 50 < pairs[0][0] < 120:
        long_unalig_num += 1
        long_seq_unalig += complement(read.seq[:pairs[0][0]])
        long_quals = np.append(long_quals, read.query_qualities[:pairs[0][0]])
    if 20 > read.query_length - pairs[-1][0] > 3:
        start_quals = np.append(start_quals, read.query_qualities[pairs[-1][0]:])
    iters += 1
    if iters > 4000:
        break
len_arr = np.array(len_list)
ali.close()

start_unalig_q_hist = list(np.histogram(start_quals[(start_quals>0)&(start_quals<36)], density=True, bins=34)[0])
end_unalig_q_hist = list(np.histogram(short_quals[(short_quals>0)&(short_quals<31)], density=True, bins=29)[0])

model_dict.update({'mis_q_distr': mis_q_hist, 'ins_q_distr':ins_q_hist,
                   'end_q_distr': end_unalig_q_hist, 'start_q_distr': start_unalig_q_hist})

ali = pysam.AlignmentFile(bam_fname, 'rb')
homopolymers = [[], [], [], []]  # lenght of hp for A, T, G and C
hp_starts = [[], [], [], []]
iters = 0
len_list = []
short_seq_unalig = ''
long_seq_unalig = ''
unalig_reg_list = [[], []]
hps_in_unalig = []
long_unalig_num = 0
bases = ['A', 'T', 'G', 'C']
match_bases_quals = [np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                     np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
match_quals = np.array([], dtype=np.int64)
regs_match = []
for read in ali.fetch(ano[ano['ENSG'] == ensg]['chr'].values[0], ano[ano['ENSG'] == ensg]['start'].values[0],
                      ano[ano['ENSG'] == ensg]['end'].values[0]):
    try:
        pairs = read.get_aligned_pairs(with_seq=True)
    except:
        continue
    if len(pairs) /read.query_length < 0.9:
        continue
    if read.query_length < 500:
        continue

    len_list.append(read.query_length)
    pairs = read.get_aligned_pairs(with_seq=True, matches_only=True)
    # print(read.seq)
    # print(pairs[1:10])
    df_pairs = pd.DataFrame(pairs, columns=['read', 'ref', 'base'])
    # match_quals = np.append(match_quals,
    # np.array(read.query_qualities)[df_pairs[df_pairs['base'].isin(['A', 'T', 'G', 'C'])]['read']])
    for i in range(4):
        match_bases_quals[i] = np.append(match_bases_quals[i],
                                         np.array(read.query_qualities)[df_pairs[df_pairs['base'] == bases[i]]['read']])
    df_pairs.drop(df_pairs[df_pairs['base'].isin(['a', 't', 'g', 'c'])].index, inplace=True)
    df_pairs['read_dif'] = df_pairs['read'].diff()
    df_pairs['ref'] = df_pairs['ref'].diff()
    df_pairs.drop(index=0, inplace=True)
    match_reg = []
    for read_pos, ref_dif, read_dif in zip(df_pairs['read'], df_pairs['ref'], df_pairs['read_dif']):
        # print(read_pos)
        if ref_dif == 1 and read_dif == 1:
            match_reg.append(read_pos)
        else:
            if match_reg == []:
                continue
            regs_match.append(np.array(read.query_qualities)[match_reg])
            match_reg = []
    iters += 1
    if iters > 6000:
        break
len_arr = np.array(len_list)
ali.close()

match_len = 20
regs = list(filter(lambda x: len(x)==match_len, regs_match))
x_arr, y_arr = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
for i in range(match_len):
    x_arr = np.append(x_arr, np.full(len(regs), i))
    y_arr = np.append(y_arr, list(map(lambda x: x[i], regs)))

hist_bins = [range(0, 40, 1), range(0, 21, 1)]
def fit_autoreg(pars):
    reg_list = []
    n_iters = 1000
    l1, c1, r1, l2, c2, r2 = pars[1], pars[2], pars[3], pars[4], pars[5], pars[6]
    vals = np.append(np.random.triangular(l1, c1, r1, size=round(n_iters*19*pars[7])),
                 np.random.triangular(l2, c2, r2, size=round(n_iters*19*(1-pars[7]))))
    np.random.shuffle(vals)
    noise = list(vals)
    for j in range(n_iters):
        sim_reg = [random.randint(5, 8)]
        for i in range(1, 20):
            sim_reg.append(round(pars[0]*sim_reg[-1] + noise.pop() + pars[8]))
        reg_list.append(sim_reg)
        sim_x_arr, sim_y_arr = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    for z in range(match_len):
        sim_x_arr = np.append(sim_x_arr, np.full(len(reg_list), z))
        sim_y_arr = np.append(sim_y_arr, list(map(lambda x: x[z], reg_list)))
    real_hist = np.histogram2d(y_arr, x_arr, density=True, bins=hist_bins)[0][:, 2:-2]
    sim_hist = np.histogram2d(sim_y_arr, sim_x_arr, density=True, bins=hist_bins)[0][:, 2:-2]
    loss = np.sum(np.abs(real_hist-sim_hist))
    return loss

match_q_res = minimize(fit_autoreg, x0=[0.569, -10, -10, -8, -6, 7, 10, 0.33,  9.80965],
                       bounds=((0.4, 0.9), (-20, -10), (-10, -7), (-15, 10), (-10, 20), (0, 30), (10, 40), (0, 1), (5, 30)),
                       method='Nelder-Mead', tol=0.1)
model_dict.update({'autoreg_specs': list(match_q_res.x)})


exon_df = exon_ano[exon_ano['ENSG'].isin(top_exp)].copy()

print('fitting homopolymer profile')

ref_genome_seq = pysam.FastaFile(args.genome)
iters = 0
gene_len = 0
bases = ['A', 'T', 'G', 'C']
#homopolymers = [[], [], [], []] #lenght of hp for A, T, G and C
homopolymers = []
for chrom in set(exon_df['chr']):
    chr_seq = ref_genome_seq.fetch(chrom)
    print(chrom)
    for start_exon, end_exon, strand in zip(exon_df[exon_df['chr']==chrom]['start'],
                            exon_df[exon_df['chr']==chrom]['end'], exon_df[exon_df['chr']==chrom]['strand']):
        exon_seq = chr_seq[start_exon:end_exon]
        dig_seq = ''.join(map(str, seq_to_dig(exon_seq)))
        if len(dig_seq) == 0:
            #print(chrom, start_exon, end_exon)
            continue
        cur_sym = dig_seq[0]
        hp_len = 1
        for base_n in range(len(dig_seq)):
            if  dig_seq[base_n] == cur_sym:
                hp_len += 1
            else:
                if hp_len > 4:
                    hp_sym = bases[int(cur_sym)]
                    homopolymers.append(str(start_exon + base_n-hp_len)
                        + ':' + str(start_exon + base_n) + ':' + chrom + ':' + strand + ':' + hp_sym)
                hp_len = 1
                cur_sym = str(dig_seq[base_n])
        iters += 1
        #if iters%1000 == 0 and iters > 1000:
            #print('processed {} exons'.format(iters))
#homopolymers = list(map(lambda x : list(set(x)), homopolymers))
homopolymers = list(set(homopolymers))
#print('number of iterations is {}'.fprmat(iters))
ref_genome_seq.close()
#print('length of hps is {}'.format(len(homopolymers)))

shortcut_hps = []
for base in ['A', 'T', 'G', 'C']:
    for i in range(30):
        filtred_hp = list(filter(lambda x: (int(x.split(':')[1])-int(x.split(':')[0]))==i and x.split(':')[4]==base,
                                 homopolymers))
        if len(filtred_hp) <= 900:
            shortcut_hps += filtred_hp
        else:
            shortcut_hps += filtred_hp[:900]

hp_blocks = []
hp_block = ''
for i in range(len(shortcut_hps)):
    hp_block += shortcut_hps[i] + ';'
    if i%100==0 and i!=0 or i==len(shortcut_hps)-1:
        hp_blocks.append(hp_block[:-1])
        hp_block = ''

start = perf_counter()
hp_seqs = []
stats = []
with Pool(processes=threads_num) as pool:
    res = pool.map(get_hp_from_block, hp_blocks)
stats = list(res)
end = perf_counter()
print('time for fitting hp specs is {} seconds'.format(round(end-start, 2)))
merged_stats = []
for i in list(map(lambda x: x.split(';'), stats)):
    merged_stats += i
merged_stats = list(filter(lambda x: len(x.split(':'))>2, merged_stats))

hp_distr_specs = [[], [], [], []]
for j in range(4):
    base = ['A', 'T', 'G', 'C'][j]
    merged_hist = np.zeros(25)
    for i in range(5, 29):
        #print(i, hp_in_reads[2][i])
        #print(i, i - np.array(hp_in_reads[2][i]))
        length_hp_list = []
        for part in list(map(lambda z: list(map(int, z)), map(lambda y: y.split(':')[2:],
                  filter(lambda x: x.split(':')[0]==str(i) and x.split(':')[1]==base, merged_stats)))):
            length_hp_list += part
        #print(len(list(filter(lambda x: x>9, length_hp_list)))/(len(length_hp_list)+1), i)
        if len(length_hp_list) < 300:
            hp_distr_specs[j].append(hp_distr_specs[j][-1])
            #print('low number of reads')
            continue
        hp_len_distr_arr = np.array(length_hp_list, dtype=np.int64)
        min_res = minimize(mix_hp_distr_fit, args=(hp_len_distr_arr),
        x0=[5, 1], bounds=((1.8*(i**0.5), 2.2*(i**0.5)), (0.05, i**0.3)), method='Powell',
               tol=0.000001, options={'maxiter':1000})
        hp_distr_specs[j].append(list(min_res.x)+[i])

model_format_hp_distr = {}
for i in range(4):
    base = ['A', 'T', 'G', 'C'][i]
    hp_spec = ''
    for spec in hp_distr_specs[i]:
        hp_spec += str(spec[0]) + ':' + str(spec[1]) + ':' + str(spec[2]) + ';'
    model_format_hp_distr.update({base+'_hp_spec':[hp_spec[:-1]]})

model_dict.update(model_format_hp_distr)

hp_qual_stats = []
bases = ['A', 'T', 'G', 'C']
specified_len = 7
for i in range(4):
    base = bases[i]
    spec_sym = base
    mono_len_hps = list(filter(lambda x: int(x.split(':')[1]) - int(x.split(':')[0]) == specified_len
                                         and x.split(':')[4] == spec_sym, homopolymers))
    # len(mono_len_hps)
    # bam_fname = open(bam_fname, 'rb')
    # ali = pysam.AlignmentFile(bam_fname, mode='rb', index_filename='bam_fname.bai')
    if len(mono_len_hps) > 400:
        mono_len_hps = mono_len_hps[:400]
    with Pool(processes=threads_num) as pool:
        res = pool.map(get_hp_qual_from_reads, mono_len_hps)
    hp_qual_stats += list(res)
    # ali.close()
    # bam_fname.close()
for i in range(4):
    base = bases[i]
    spec_sym = base
    hp_seq_len = specified_len
    iters = 0
    hp_qual_seqs = []
    for seg_quals in list(filter(lambda x: x.split(';')[0].split(':')[1] == spec_sym, hp_qual_stats)):
        # print(seg_quals.split(';')[0].split(':'))
        if len(seg_quals.split(';')) == 1:
            continue
        # print(list(filter(lambda x: len(x[:-1].split(':'))==5, seg_quals.split(';')[1:])))
        hp_qual_seqs += list(filter(lambda x: len(x[:-1].split(':')) == hp_seq_len, seg_quals.split(';')[1:]))
        iters += 1
        if iters > 10000:
            break


hp_seq_len = specified_len
iters = 0
hp_qual_seqs = []
hp_qual_distr_specs = [[], [], [], []]
for j in range(4):
    base = ['A', 'T', 'G', 'C'][j]
    for seg_quals in list(filter(lambda x: x.split(';')[0].split(':')[1]==base, hp_qual_stats)):
        #print(seg_quals.split(';')[0].split(':'))
        if len(seg_quals.split(';')) == 1:
            continue
        #print(list(filter(lambda x: len(x[:-1].split(':'))==5, seg_quals.split(';')[1:])))
        hp_qual_seqs += list(filter(lambda x: len(x[:-1].split(':'))==hp_seq_len, seg_quals.split(';')[1:]))
        iters += 1
        if iters > 10000:
            break
    for i in range(hp_seq_len):
        pos_q_arr = np.array(list(map(lambda x: int(x[:-1].split(':')[i]), hp_qual_seqs)), dtype=np.int64)
        q_res = minimize(hp_qual_distr_fit_skewed_normal, args=(pos_q_arr),
            x0=[pos_q_arr.mean(), 10, -3], bounds=((pos_q_arr.mean(), pos_q_arr.mean()+1), (2, 15), (-10, 10)), method='Powell', #Nelder-Mead, pos_q_arr.mean()
                   tol=0.01, options={'maxiter':500})
        hp_qual_distr_specs[j].append(list(q_res.x)+[i])

model_format_hp_qual = {}
for i in range(4):
    base = ['A', 'T', 'G', 'C'][i]
    hp_spec = ''
    for spec in hp_qual_distr_specs[i]:
        hp_spec += str(spec[0]) + ':' + str(spec[1]) + ':' + str(spec[2]) + ':' + str(spec[3]) + ';'
    model_format_hp_qual.update({base+'_hp_spec_qual': [hp_spec[:-1]]})

model_dict.update(model_format_hp_qual)

print('Characterization is completed, fragmentation probability is {}'.format(round(frag_p, 6)))

model_dict.update({'frag_prob': [frag_p]})



model = pd.DataFrame(model_dict.items(), columns=['params', 'values'])
model.to_csv('full_model.tsv', index=False, sep ='\t')
os.system('rm -r tmp_tnrsim_files')