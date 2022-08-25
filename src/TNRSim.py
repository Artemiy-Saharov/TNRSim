#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
import argparse
from scipy.stats import skewnorm
from scipy.stats import geom
from multiprocessing import Process
import os
import pysam


parser = argparse.ArgumentParser(description='Simulator script')
parser.add_argument('-f', '--fragmentation_probability',
                    type=float, help='probability to get read break')
parser.add_argument('-e', '--exp_prof', type=str,
                    help='path to tsv file with number of reads to be simulated for each transcript')
parser.add_argument('-m', '--model', type=str, help='path to model file')
parser.add_argument('-t', '--transcriptome', type=str, help='fasta file with transcript sequences')
parser.add_argument('-O', '--output', type=str, default='simulated_reads.fasta', help='Output file with reads')
parser.add_argument('--threads', type=int, default=1, help='Number of CPU cores for simulation')
#parser.add_argument('--fastq', type=bool, default=False, help='if this option is specified, '
#                                                             'simulator will simulate reads in fastq format')
args = parser.parse_args()

threads = args.threads
transcripts = pysam.FastaFile(args.transcriptome)
out_file = args.output
exp_prof = pd.read_csv(args.exp_prof, sep='\t')
seq_model = pd.read_csv(args.model, sep='\t')


def fragmentation1(length, N, p_t):
    return N*(np.cumprod(np.full(length, 1-p_t)))


def weibull_geom(a, pr, weight, samp_number):
    rand_arr = np.append(np.ceil(np.random.weibull(a, round(samp_number*weight))),
                                 np.random.geometric(pr, round(samp_number*((1-weight)))))
    np.random.shuffle(rand_arr)
    return rand_arr


def pois_geom(lam, pr, weight, samp_number):
    rand_arr = np.append(1 + np.random.poisson(lam, round(samp_number*weight)),
                                 np.random.geometric(pr, round(samp_number*((1-weight)))))
    np.random.shuffle(rand_arr)
    return rand_arr

def base_to_dig(letter):
    if letter=='A': return 0
    if letter=='T': return 1
    if letter=='G': return 2
    if letter=='C': return 3

def seq_to_dig(read_seq):
    return list(map(base_to_dig, list(read_seq)))

get_values = lambda param: list(map(float, seq_model[seq_model['params']==param]['values'].values[0][1:-1].split(', ')))

hist_match = np.array(get_values('hist_match'))
mis_params = np.array(get_values('mis_params'))
del_params = np.array(get_values('del_params'))
ins_params = np.array(get_values('ins_params'))
err_prob = np.array(get_values('err_prob'))
mis_after_del_prob = get_values('mis_after_del')[0]
mis_after_ins_prob = get_values('mis_after_ins')[0]
stab = np.array(get_values('base_stability')).reshape(4, 4)
match_autoreg_specs = get_values('autoreg_specs')
mis_q_distr = np.array(get_values('mis_q_distr'))
ins_q_distr = np.array(get_values('ins_q_distr'))
end_q_distr = np.array(get_values('end_q_distr'))
start_q_distr = np.array(get_values('start_q_distr'))

mis_probs = []
for i in range(4):
    mis_probs.append(1 - stab[i,i])
mis_probs = np.array(mis_probs)
mis_probs = mis_probs*(1/mis_probs.sum())
mis_probs = mis_probs*2

hp_qual_model_df = seq_model[seq_model['params'].str.contains('hp_spec_qual')].copy()
hp_qual_dict = {}
for spec, value in zip(hp_qual_model_df['params'], hp_qual_model_df['values']):
    spec_sym = spec[0]
    #print(spec_sym)
    hp_qual_values = list(map(lambda x: tuple((float(x.split(':')[0]),
                                                float(x.split(':')[1]), float(x.split(':')[2]))), value[2:-2].split(';')))
    #print(hp_distr_values)
    for i, val in zip(range(1, 8), hp_qual_values):
        hp_qual_dict.update({str(i)+spec[0]:val})

hp_model_df = seq_model[seq_model['params'].str[1:]=='_hp_spec'].copy()
hp_distr_dict = {}
for spec, value in zip(hp_model_df['params'], hp_model_df['values']):
    spec_sym = spec[0]
    #print(spec_sym)
    hp_distr_values = list(map(lambda x: tuple((float(x.split(':')[0]),
                                                float(x.split(':')[1]))), value[2:-2].split(';')))
    #print(hp_distr_values)
    for i, val in zip(range(5, 29), hp_distr_values):
        hp_distr_dict.update({str(i)+spec[0]:val})



if args.fragmentation_probability == None:
    frag_prob = get_values('frag_prob')[0]
else:
    frag_prob = args.fragmentation_probability



lens = np.arange(1, 199)

def simulate_fastq(transcriptome, expression_prof, fragmentation_rate, filename):
    file = open(filename, 'w')
    for transcript_id in expression_prof['transcript_id']:
        seq = str(transcriptome.fetch(transcript_id))
        len_distr = fragmentation_rate*fragmentation1(len(seq),
                                 expression_prof[expression_prof['transcript_id']==transcript_id]['counts'].values[0],
                                 fragmentation_rate)
        N_full = float(expression_prof[expression_prof['transcript_id']==transcript_id]['counts'].values[0]) - np.sum(len_distr)
        simulate_isoform_with_q(seq, len_distr, N_full, file, transcript_id, expression_prof)
    file.close()


def simulate_isoform_with_q(isoform_seq, len_distr, N_intact, fasta, t_id, expression_prof):
    step_distr = 40
    isoform_counts = expression_prof[expression_prof['transcript_id'] == t_id]['counts'].values[0]
    isoform_len = len(isoform_seq)
    trun_ratio = N_intact / isoform_counts

    unalig_ratio = random.gauss(1.3, 0.5)
    if unalig_ratio < 0.01:
        unalig_ratio = 0.01
    num_long = round(isoform_counts / (unalig_ratio + 1))
    num_short = isoform_counts - num_long

    tran_hps = []
    bases = ['A', 'T', 'G', 'C']
    dig_seq = ''.join(map(str, seq_to_dig(isoform_seq))) + '4'
    cur_sym = dig_seq[0]
    hp_len = 1
    for base_n in range(len(dig_seq)):
        if dig_seq[base_n] == cur_sym:
            hp_len += 1
        else:
            if hp_len > 4:
                hp_sym = bases[int(cur_sym)]
                tran_hps.append(str(base_n - hp_len - isoform_len)
                                + ':' + str(base_n - isoform_len))
            hp_len = 1
            cur_sym = str(dig_seq[base_n])
    tran_hps = np.array(list(map(lambda x: tuple((int(x.split(':')[0]), int(x.split(':')[1]))), tran_hps)))
    counts = 0
    for frag_len in range(30, len(len_distr) - step_distr, step_distr):
        for read_number in range(round(step_distr * len_distr[frag_len])):
            add_long_end = True if counts < num_long * (1 - trun_ratio) else False
            nucl_seq, qual_seq = make_errors2(isoform_seq[random.randint(-step_distr - frag_len, -frag_len):],
                                              tran_hps, add_long_end)
            fasta.write('@' + t_id + '_read_' + str(counts) + '\n' + nucl_seq + '\n' + '+' + '\n' + qual_seq + '\n')
            counts += 1
    num_truncated = int(counts)
    for read_number in range(round(N_intact)):
        add_long_end = True if counts - num_truncated < num_long * trun_ratio else False
        nucl_seq, qual_seq = make_errors2(isoform_seq, tran_hps, add_long_end)
        fasta.write('@' + t_id + '_read_' + str(counts) + '\n' + nucl_seq + '\n' + '+' + '\n' + qual_seq + '\n')
        counts += 1



def gen_qual(seg_type, inp_seg_len):
    out_q = []
    if seg_type == 'match':
        l1, c1, r1, l2, c2, r2, triag_weigth = match_autoreg_specs[1], match_autoreg_specs[2], match_autoreg_specs[3], match_autoreg_specs[4], match_autoreg_specs[5], match_autoreg_specs[6], match_autoreg_specs[7]
        noise = np.append(np.random.triangular(l1, c1, r1, size=round(inp_seg_len*triag_weigth)),
                 np.random.triangular(l2, c2, r2, size=round(inp_seg_len*(1-triag_weigth))))
        np.random.shuffle(noise)
        noise = list(noise)
        r_seg_len, l_seg_len = (inp_seg_len%2)+(inp_seg_len//2), inp_seg_len//2
        init_q = random.choices(range(1, 40), mis_q_distr, k=2)
        r_seg, l_seg = [init_q[0]], [init_q[1]]
        for i in range(l_seg_len):
            l_seg.append(round(match_autoreg_specs[0]*l_seg[-1] + noise.pop() + match_autoreg_specs[8]))
        for i in range(r_seg_len):
            r_seg.append(round(match_autoreg_specs[0]*r_seg[-1] + noise.pop() + match_autoreg_specs[8]))
        out_q = r_seg[1:] + l_seg[:-1][::-1]
    if seg_type == 'mis':
        out_q = random.choices(range(1, 40), mis_q_distr, k=inp_seg_len)
    if seg_type == 'ins':
        out_q = random.choices(range(1, 40), ins_q_distr, k=inp_seg_len)
    if seg_type == 'start_unalig':
        out_q = random.choices(range(1, 35), start_q_distr, k=inp_seg_len)
    if seg_type == 'end_unalig':
        out_q = random.choices(range(1, 30), end_q_distr, k=inp_seg_len)
    return ''.join(map(lambda x: chr(x+33), out_q))

def read_hp(hp_length, sym_hp):
    if hp_length > 28: hp_length = 28
    dist_vals = hp_distr_dict[str(hp_length)+sym_hp]
    hp_len1 = round(random.gauss(dist_vals[0], dist_vals[1]))
    if hp_len1 < 1: hp_len1 = 1
    return hp_len1


def gen_hp_qual(hp_len, sym_hp):
    hp_qual_str = r''
    for hp_pos in range(1, hp_len+1):
        if hp_pos > 7: hp_pos = 7
        qual_specs = hp_qual_dict[str(hp_pos)+sym_hp]
        base_qual = round(skewnorm.rvs(a=qual_specs[2], loc=qual_specs[0], scale=qual_specs[1], size=1)[0])
        if base_qual < 1: base_qual = 1
        hp_qual_str += chr(base_qual+33)
    return hp_qual_str







lens = np.arange(1, 199)
def make_errors2(tr_seq, hps, add_long_unalig):
    # posible states are match, mis, del and ins
    tr_seq = str(tr_seq)
    read = ''
    q_str = r''
    pos = 0
    state = 'match'
    st_num = 0
    tr_len = len(tr_seq)
    tr_hps = hps + tr_len
    hp_df = pd.DataFrame(tr_hps)
    tr_hps = hp_df[hp_df[0]>0].to_numpy()
    if tr_len < 500:
        states_number = 120
    else:
        states_number = round(tr_len / 4)
    match_lens = np.array(random.choices(lens, weights=hist_match, k=states_number * 2)).astype(int)
    mis_lens = pois_geom(mis_params[0], mis_params[1], mis_params[2], states_number).astype(int)
    del_lens = weibull_geom(del_params[0], del_params[1], del_params[2], states_number).astype(int)
    ins_lens = weibull_geom(ins_params[0], ins_params[1], ins_params[2], states_number).astype(int)
    while tr_len > pos:
        # print('start')
        seg_len = 0
        mis_after_indel = False
        if state == 'match':
            if tr_hps.size == 0 or tr_hps[:, 0][(tr_hps[:, 0] - pos < 199) & (tr_hps[:, 0] - pos > 0)].size == 0:
                seg_len = match_lens[st_num]
                if pos + seg_len > tr_len:
                    seg_len = tr_len - pos
                match_len = int(seg_len)
                read += tr_seq[pos:pos + seg_len]
                q_str += gen_qual(state, seg_len)
            else:
                hp_start = tr_hps[:, 0][(tr_hps[:, 0] - pos < 199) & (tr_hps[:, 0] - pos > 0)].min()
                seg_len = match_lens[st_num]
                if hp_start - pos > 15:
                    while not hp_start - (pos + seg_len) > 10:
                        seg_len = random.choice(match_lens)
                    match_len = int(seg_len)
                    read += tr_seq[pos:pos + seg_len]
                    q_str += gen_qual(state, seg_len)
                else:
                    try:
                        hp_end = tr_hps[np.where(tr_hps == hp_start)[0], np.where(tr_hps == hp_start)[1] + 1][0]
                        after_hp = 0
                        seg_len = hp_end + after_hp - pos
                        match_len = int(seg_len)
                        read += tr_seq[pos:hp_start]
                        q_str += gen_qual(state, hp_start - pos + 4)[:-4]
                        # print(hp_end)
                        hp_in_read_len = read_hp(hp_end - hp_start, tr_seq[hp_start])
                        # print(tr_seq[hp_start:hp_end])
                        read += tr_seq[hp_start] * hp_in_read_len
                        read += tr_seq[hp_end:hp_end + after_hp]
                        q_str += gen_hp_qual(hp_in_read_len, tr_seq[hp_start])  # qual for hp_seg
                        state = 'hp'
                        # q_str += gen_qual(state, after_hp+5)[:-5]
                    except:
                        print('error while simulating hp, read is skipped')
                        print(tr_hps)
                        print('trouble hp start in pos' + str(hp_start))
                        print(tr_seq[hp_start - 1:hp_start + 10])
                        state = 'hp'
        elif state == 'mis':
            mismatch = ''
            seg_len = mis_lens[st_num]
            # adj_reg = read[pos-match_len:]
            changed = False
            shift = 0
            while not changed:
                if shift == 0:
                    if random.random() < mis_probs[base_to_dig(tr_seq[pos])]:
                        changed = True
                        continue
                    else:
                        shift += 1
                        continue
                if pos + shift + seg_len <= tr_len and random.random() < mis_probs[base_to_dig(tr_seq[pos + shift])]:
                    changed = True
                elif match_len - shift > 0 and random.random() < mis_probs[base_to_dig(read[-shift])]:
                    changed = True
                    shift = -shift
                else:
                    shift += 1
                if pos + shift + seg_len >= tr_len and match_len - shift < 0 and not changed:
                    shift = 0
            # if shift == -1: shift = -2
            if shift >= 0:
                read += tr_seq[pos:pos + shift]
            else:
                read = str(read[:shift])
            q_str = q_str[:-match_len]
            pos += shift
            q_str += gen_qual('match', match_len + shift)
            if pos + seg_len > tr_len:
                seg_len = tr_len - pos
            for i in range(seg_len):
                alp = ['A', 'T', 'G', 'C']
                dig_base = base_to_dig(tr_seq[pos + i])
                subs_probs = list(stab[dig_base])
                alp.remove(tr_seq[pos + i])
                subs_probs.pop(dig_base)
                mismatch += random.choices(alp, subs_probs)[0]
            read += mismatch
            q_str += gen_qual(state, seg_len)
        elif state == 'del':
            seg_len = del_lens[st_num]
            if random.random() < mis_after_del_prob:
                mis_after_indel = True
        elif state == 'ins':
            seg_len = ins_lens[st_num]
            insertion = ''
            for i in range(seg_len):
                insertion += random.choice(['A', 'T', 'G', 'C'])
            read += insertion
            q_str += gen_qual(state, seg_len)
            seg_len = 0
            if random.random() < mis_after_del_prob:
                mis_after_indel = True

        elif state == 'mis_after_indel':
            mismatch = ''
            seg_len = 1
            if pos + seg_len > tr_len:
                seg_len = tr_len - pos
            for i in range(seg_len):
                alp = ['A', 'T', 'G', 'C']
                dig_base = base_to_dig(tr_seq[pos + i])
                subs_probs = list(stab[dig_base])
                alp.remove(tr_seq[pos + i])
                subs_probs.pop(dig_base)
                mismatch += random.choices(alp, subs_probs)[0]
            read += mismatch
            q_str += gen_qual('mis', seg_len)

        if state == 'match':
            state = random.choices(['mis', 'ins', 'del'], weights=err_prob)[0]
        elif mis_after_indel:
            state = 'mis_after_indel'
        else:
            state = 'match'
        pos += seg_len
        st_num += 1

    polyA_len = geom.rvs(0.18, 4)
    read += ''.join(random.choices(['A', 'T', 'G', 'C'], weights=[0.91, 0.03, 0.03, 0.03], k=polyA_len))  # add polyA
    q_str += gen_qual('end_unalig', polyA_len)

    start_ualig_len = geom.rvs(0.3, -1)
    read = ''.join(random.choices(['A', 'T', 'G', 'C'], k=start_ualig_len)) + read
    q_str = gen_qual('start_unalig', start_ualig_len) + q_str

    if add_long_unalig:
        long_seg_len = random.randint(70, 120)
        read += ''.join(random.choices(['A', 'T', 'G', 'C'], weights=[31.6, 26.4, 1.7, 40.4],
                                       k=long_seg_len))  # add long unalig region
        q_str += gen_qual('end_unalig', long_seg_len)

        if len(read) != len(q_str):
            print(len(read) - len(q_str), len(read), len(q_str), match_len, pos)
            return read, q_str, len(read) - len(q_str)
    # print('{}'.format(state))
    return read, q_str


procs = []
prof_step = exp_prof.shape[0]//threads
for thread_num in range(threads):
    if thread_num + 1 == threads:
        part_prof = exp_prof[thread_num*prof_step:].copy()
    else:
        part_prof = exp_prof[thread_num*prof_step:(thread_num+1)*prof_step].copy()
    procs.append(Process(target=simulate_fastq,
                         args=(transcripts, part_prof, frag_prob, out_file + '.' + str(thread_num),)))
for i in range(threads):
    procs[i].start()
for i in range(threads):
    procs[i].join()

if threads == 1:
    os.system('mv {}.0 {}'.format(out_file, out_file))
else:
    for i in range(1, threads):
        os.system('cat {}.{} >> {}.0'.format(out_file, i, out_file))
        os.system('rm {}.{}'.format(out_file, i))
    os.system('mv {}.0 {}'.format(out_file, out_file))

print('Simulation successfully completed')
#simulate_fastq(transcriptome=transcripts, expression_prof=exp_prof, fragmentation_rate=frag_prob, filename=out_file)
