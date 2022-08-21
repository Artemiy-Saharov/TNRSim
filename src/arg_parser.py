import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO

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

#transcripts = SeqIO.index(args.transcriptome, 'fasta')
out_file = args.output
exp_prof = pd.read_csv(args.exp_prof, sep='\t')
seq_model = pd.read_csv(args.model, sep='\t')

get_values = lambda param: list(map(float, seq_model[seq_model['params']==param]['values'].values[0][1:-1].split(', ')))

hist_match = np.array(get_values('hist_match'))
mis_params = np.array(get_values('mis_params'))
del_params = np.array(get_values('del_params'))
ins_params = np.array(get_values('ins_params'))
if args.fragmentation_probability == None:
    frag_prob = get_values('frag_prob')[0]
else:
    frag_prob = args.fragmentation_probability

print(seq_model)