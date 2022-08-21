import numpy as np
import pandas as pd
import random
from Bio import SeqIO
import argparse


parser = argparse.ArgumentParser(description='Simulator script')
parser.add_argument('-f', '--fragmentation_probability',
                    type=float, help='probability to get read break')
parser.add_argument('-e', '--exp_prof', type=str,
                    help='path to tsv file with number of reads to be simulated for each transcript')
parser.add_argument('-m', '--model', type=str, help='path to model file')
parser.add_argument('-t', '--transcriptome', type=str, help='fasta file with transcript sequences')
parser.add_argument('-O', '--output', type=str, default='simulated_reads.fasta', help='Output file with reads')
parser.add_argument('--fastq', type=bool, default=False, help='if this option is specified, '
                                                              'simulator will simulate reads in fastq format')
args = parser.parse_args()

transcripts = SeqIO.index(args.transcriptome, 'fasta')
out_file = args.output
exp_prof = pd.read_csv(args.exp_prof, sep='\t')
seq_model = pd.read_csv(args.model, sep='\t')

get_values = lambda param: list(map(float, seq_model[seq_model['params']==param]['values'].values[0][1:-1].split(', ')))

hist_match = np.array(get_values('hist_match'))
mis_params = np.array(get_values('mis_params'))
del_params = np.array(get_values('del_params'))
ins_params = np.array(get_values('ins_params'))
err_prob = np.array(get_values('err_prob'))


if args.fragmentation_probability == None:
    frag_prob = get_values('frag_prob')[0]
else:
    frag_prob = args.fragmentation_probability



lens = np.arange(1, 199)

def simulate_fasta(transcriptome, expression_prof, fragmentation_rate, filename):
    file = open(filename, 'w')
    for transcript_id in expression_prof['transcript_id']:
        seq = str(transcriptome[transcript_id].seq)
        len_distr = fragmentation_rate*fragmentation1(len(seq),
                                 expression_prof[expression_prof['transcript_id']==transcript_id]['counts'].values[0],
                                 fragmentation_rate)
        N_full = float(expression_prof[expression_prof['transcript_id']==transcript_id]['counts'].values[0]) - np.sum(len_distr)
        simulate_isoform(seq, len_distr, N_full, file, transcript_id)
    file.close()


def simulate_isoform(isoform_seq, len_distr, N_intact, fasta, t_id):
    step_distr = 40
    isoform_len = len(isoform_seq)
    counts = 0
    for frag_len in range(30, len(len_distr)-step_distr, step_distr):
        for read_number in range(round(step_distr*len_distr[frag_len])):
            fasta.write('>' + t_id + '_read_' + str(counts) + '\n' +
                        make_errors(isoform_seq[random.randint(-step_distr-frag_len,-frag_len):isoform_len]) + '\n')
            counts += 1
    for read_number in range(round(N_intact)):
        fasta.write('>' + t_id + '_read_' + str(counts) + '\n' +
                        make_errors(isoform_seq) + '\n')
        counts += 1


def make_errors(tr_seq):
    # posible states are match, mis, del and ins
    tr_seq = str(tr_seq)
    read = ''
    pos = 0
    state = 'match'
    st_num = 0
    tr_len = len(tr_seq)
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
        if state == 'match':
            seg_len = match_lens[st_num]
            read += tr_seq[pos:pos + seg_len]
        elif state == 'mis':
            mismatch = ''
            seg_len = mis_lens[st_num]
            if pos + seg_len > tr_len:
                seg_len = tr_len - pos
            for i in range(seg_len):
                alp = ['A', 'T', 'G', 'C']
                alp.remove(tr_seq[pos + i])
                mismatch += random.choice(alp)
            read += mismatch
        elif state == 'del':
            seg_len = del_lens[st_num]
        elif state == 'ins':
            seg_len = ins_lens[st_num]
            insertion = ''
            for i in range(seg_len):
                insertion += random.choice(['A', 'T', 'G', 'C'])
            read += insertion
            seg_len = 0

        if state == 'match':
            state = random.choices(['mis', 'ins', 'del'], weights=err_prob)[0]
        else:
            state = 'match'
        pos += seg_len
        st_num += 1
        # print('{}'.format(state))
    return read



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


simulate_fasta(transcriptome=transcripts, expression_prof=exp_prof, fragmentation_rate=frag_prob, filename=out_file)
