#!/usr/bin/env python3
"""
TNRSim Characterizer — Complete Transcriptome-mode Characterization
Parallel fragmentation estimation + Error profile characterization
Stages 1-11 (excluding homopolymer stages 9-10, using defaults for transcriptome mode)
"""

"""
================================================================================
STAGE 1: IMPORTS
================================================================================
"""
import os
import sys
import numpy as np
import pandas as pd
import pysam
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, minimize
from scipy import cluster
from Bio import SeqIO
import argparse
import warnings
from multiprocessing import Pool, cpu_count
import pickle
import shutil
import random

warnings.filterwarnings('ignore')

# Проверка наличия fitting_functions
try:
    from fitting_functions import mix_weibul_distr_fit, mix_pois_distr_fit, digs_to_err
    FITTING_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("[WARNING] fitting_functions not found — using default error parameters")
    FITTING_FUNCTIONS_AVAILABLE = False
    # Заглушки для функций фитирования
    def mix_weibul_distr_fit(params, data):
        return np.sum((params[0] * np.exp(-params[1] * data) + params[2]) ** 2)
    def mix_pois_distr_fit(params, data):
        return np.sum((params[0] * np.exp(-params[1] * data) + params[2]) ** 2)
    def digs_to_err(d):
        mapping = {0: 'match', 1: 'mis', 2: 'ins', 3: 'del'}
        return mapping.get(int(d), 'match')


"""
================================================================================
STAGE 2: ARGUMENT PARSING
================================================================================
"""
parser = argparse.ArgumentParser(
    description='TNRSim Characterizer — Complete transcriptome-mode characterization',
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('-t', '--transcriptome', type=str, required=True,
                    help='Path to reference transcriptome FASTA')
parser.add_argument('-b', '--bam_file', type=str, required=True,
                    help='Path to BAM file aligned to transcriptome')
parser.add_argument('--threads', type=int, default=4,
                    help='Number of threads for parallel processing (default: 4)')
parser.add_argument('--min_reads', type=int, default=300,
                    help='Minimum reads per transcript (default: 300)')
parser.add_argument('--max_tx_error', type=int, default=100,
                    help='Max transcripts for error characterization (default: 100)')
parser.add_argument('-O', '--output', type=str, default='TNRSim_model.tsv',
                    help='Output model file (default: TNRSim_model.tsv)')
parser.add_argument('-f', '--fragmentation_only', action='store_true',
                    help='Estimate only fragmentation and exit')
parser.add_argument('--skip_error_char', action='store_true',
                    help='Skip error characterization stages (5,7,8)')

args = parser.parse_args()

threads_num = args.threads
bam_fname = args.bam_file
min_reads_threshold = args.min_reads
transcriptome_fasta = args.transcriptome
output_file = args.output
max_tx_error = args.max_tx_error

print(f"[INFO] TNRSim Characterizer — Complete Transcriptome Mode")
print(f"[INFO] BAM: {bam_fname}")
print(f"[INFO] Transcriptome: {transcriptome_fasta}")
print(f"[INFO] Threads: {threads_num}")
print(f"[INFO] Min reads/transcript: {min_reads_threshold}")
print(f"[INFO] Max transcripts for error char: {max_tx_error}")


"""
================================================================================
STAGE 3: FILE VALIDATION & TRANSCRIPT FILTERING
================================================================================
"""
# Проверка файлов
for fpath, fname in [(bam_fname, "BAM"), (transcriptome_fasta, "Transcriptome FASTA")]:
    if not os.path.exists(fpath):
        print(f"[ERROR] {fname} not found: {fpath}", file=sys.stderr)
        sys.exit(1)

# Проверка индекса BAM
bai_index = bam_fname + ".bai"
csi_index = bam_fname + ".csi"
if not os.path.exists(bai_index) and not os.path.exists(csi_index):
    print(f"[ERROR] BAM index not found. Run: samtools index {bam_fname}", file=sys.stderr)
    sys.exit(1)

# Временная директория
os.makedirs('tmp_tnrsim_files', exist_ok=True)

# Подсчёт ридов через samtools
print(f"[INFO] Counting reads per transcript...")
idxstats_cmd = f"samtools idxstats {bam_fname} | awk '$3 > 0 {{print $1 \"\\t\" $3}}' > ./tmp_tnrsim_files/transcript_counts.txt"
if os.system(idxstats_cmd) != 0:
    print("[ERROR] samtools idxstats failed", file=sys.stderr)
    sys.exit(1)

# Загрузка и фильтрация
tx_counts = pd.read_csv('./tmp_tnrsim_files/transcript_counts.txt', sep='\t',
                        names=['Transcript', 'Reads'])
high_cov_tx = tx_counts[tx_counts['Reads'] > min_reads_threshold]
print(f"[INFO] Filtered to {len(high_cov_tx)} transcripts with >{min_reads_threshold} reads")

if len(high_cov_tx) == 0:
    print(f"[ERROR] No transcripts pass threshold", file=sys.stderr)
    sys.exit(1)

# Загрузка последовательностей
print(f"[INFO] Loading transcript sequences...")
tx_sequences = {}
for record in SeqIO.parse(transcriptome_fasta, "fasta"):
    tx_sequences[record.id] = str(record.seq).upper().replace('U', 'T')

filtered_tx_ids = high_cov_tx['Transcript'].to_numpy()
tx_with_seq = [tx for tx in filtered_tx_ids if tx in tx_sequences]

if len(tx_with_seq) == 0:
    print(f"[ERROR] No filtered transcripts in FASTA", file=sys.stderr)
    sys.exit(1)

print(f"[INFO] Final set: {len(tx_with_seq)} transcripts")

# Сохранение для следующих этапов
np.save('./tmp_tnrsim_files/filtered_transcripts.npy', tx_with_seq)
np.save('./tmp_tnrsim_files/tx_sequences.npy', tx_sequences)


"""
================================================================================
STAGE 4: PARALLEL FRAGMENTATION ESTIMATION
================================================================================
"""
print(f"\n{'='*80}")
print("[STAGE 4] FRAGMENTATION PROBABILITY ESTIMATION (PARALLEL)")
print(f"{'='*80}")

# === Вспомогательные функции ===

def find_first_peak_geometric_p(lengths, n_bins=100, min_prominence=30):
    """Оценить p через фитирование первого пика"""
    if len(lengths) < 100:
        return None

    min_len = max(100, lengths.min())
    max_len = min(3000, lengths.max())
    hist, bin_edges = np.histogram(lengths[(lengths >= min_len) & (lengths <= max_len)], bins=n_bins)
    bin_width = np.diff(bin_edges)[0]

    if len(hist) == 0 or hist.sum() == 0:
        return None

    try:
        peaks, _ = find_peaks(hist, prominence=min_prominence, width=[1, 20], distance=5)
    except:
        return None

    if len(peaks) == 0:
        return None

    first_peak_idx = peaks[0]
    first_peak_pos = bin_edges[first_peak_idx]

    if bin_width * first_peak_idx > 1500:
        return None

    if len(peaks) > 1:
        valley_idx = first_peak_idx + np.argmin(hist[first_peak_idx:peaks[1]])
        fit_end = valley_idx
    else:
        half_max = hist[first_peak_idx] * 0.7
        fit_end = first_peak_idx
        for i in range(first_peak_idx, len(hist)):
            if hist[i] < half_max:
                fit_end = i
                break
        fit_end = min(fit_end, first_peak_idx + 50)

    fit_start = max(0, first_peak_idx - 10)
    if fit_end <= fit_start:
        return None

    x_data = np.arange(fit_start, fit_end) * bin_width + bin_edges[0]
    y_data = hist[fit_start:fit_end]

    if len(y_data) < 5 or y_data.max() < 10:
        return None

    local_mean = np.average(x_data, weights=y_data)
    p_init = 1 / (local_mean - bin_edges[0] + 1)
    p_init = np.clip(p_init, 0.0001, 0.01)

    def geo_decay_model(x, N, p):
        return N * np.exp(-p * x / bin_width)

    try:
        popt, _ = curve_fit(geo_decay_model, x_data - x_data[0], y_data,
                           p0=[y_data.max(), p_init],
                           bounds=([0, 0.0001], [np.inf, 0.05]),
                           maxfev=5000)
        return np.clip(popt[1], 0.0001, 0.05)
    except:
        short_reads = lengths[lengths <= first_peak_pos + bin_width * 20]
        if len(short_reads) > 50:
            return 1 / (np.mean(short_reads) - min_len + 1)
        return None


def extract_read_lengths(bam_path, transcript_id, min_length=100):
    """Извлечь длины прочтений для транскрипта"""
    bam = pysam.AlignmentFile(bam_path, 'rb')
    lengths = []
    try:
        for read in bam.fetch(transcript_id):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < 5 or read.query_length is None:
                continue
            if read.query_length >= min_length:
                lengths.append(read.query_length)
    except ValueError:
        pass
    finally:
        bam.close()
    return np.array(lengths) if lengths else np.array([])


def _process_single_transcript(args):
    """Обработка одного транскрипта в отдельном процессе"""
    bam_path, tid, min_reads, min_length, min_prom = args
    lengths = extract_read_lengths(bam_path, tid, min_length)
    if len(lengths) < min_reads:
        return None
    p_est = find_first_peak_geometric_p(lengths, n_bins=100, min_prominence=min_prom)
    if p_est is not None and 0.0001 <= p_est <= 0.05:
        return (tid, p_est)
    return None


def estimate_fragmentation_parallel(bam_path, tx_list, min_reads=300, n_workers=4):
    """Параллельная оценка фрагментации"""
    print(f"[INFO] Parallel processing with {n_workers} workers")

    worker_args = [(bam_path, tid, min_reads, 100, 30) for tid in tx_list]

    p_values = {}
    with Pool(processes=n_workers) as pool:
        results = pool.map(_process_single_transcript, worker_args, chunksize=20)

    for result in results:
        if result is not None:
            tid, p_est = result
            p_values[tid] = p_est

    print(f"[INFO] {len(p_values)} successful p estimates out of {len(tx_list)} transcripts")

    if len(p_values) < 10:
        print("[ERROR] Too few successful fits", file=sys.stderr)
        return None

    p_array = np.array(list(p_values.values()))
    frag_p_simple = np.mean(p_array)
    frag_p_final = np.clip(frag_p_simple, 0.00015, 0.0055)

    decay_rate = -np.log(1 - frag_p_final)
    half_life = np.log(2) / decay_rate if decay_rate > 0 else float('inf')

    print(f"[SUCCESS] Fragmentation p = {frag_p_final:.8f}")
    print(f"  Decay rate (λ) = {decay_rate:.8f}")
    print(f"  Half-life = {half_life:.1f} nt")

    return {
        'frag_p': frag_p_final,
        'frag_p_simple': frag_p_simple,
        'frag_p_median': np.median(p_array),
        'decay_rate': decay_rate,
        'half_life': half_life,
        'n_transcripts': len(p_values),
        'p_values': p_values
    }


# === Запуск Stage 4 ===

frag_result = estimate_fragmentation_parallel(
    bam_fname, tx_with_seq,
    min_reads=min_reads_threshold,
    n_workers=threads_num
)

if frag_result is None:
    sys.exit(1)

frag_p = frag_result['frag_p']

# Сохранение
np.save('./tmp_tnrsim_files/frag_p.npy', frag_p)
np.save('./tmp_tnrsim_files/frag_p_details.npy', frag_result)
pd.DataFrame([{'transcript_id': tid, 'p_estimate': p}
              for tid, p in frag_result['p_values'].items()
             ]).to_csv('./tmp_tnrsim_files/transcript_p_estimates.tsv', sep='\t', index=False)

if args.fragmentation_only:
    model_dict = {'frag_prob': [frag_p]}
    pd.DataFrame(model_dict.items(), columns=['params', 'values']).to_csv(
        output_file, sep='\t', index=False)
    print(f"[SUCCESS] Minimal model saved to {output_file}")
    shutil.rmtree('./tmp_tnrsim_files')
    sys.exit(0)

if args.skip_error_char:
    print("[INFO] Skipping error characterization (--skip_error_char)")
    shutil.rmtree('./tmp_tnrsim_files')
    sys.exit(0)


"""
================================================================================
STAGE 5: ERROR PROFILE CHARACTERIZATION
================================================================================
"""
print(f"\n{'='*80}")
print("[STAGE 5] ERROR PROFILE CHARACTERIZATION")
print(f"{'='*80}")

selected_tx = tx_with_seq[:max_tx_error]
print(f"[INFO] Processing {len(selected_tx)} transcripts for error characterization")

aggregated = {
    'sum_match_len_arr': [],
    'sum_err_arr': [],
    'ins1_arr': [],
    'del1_arr': [],
    'mis1_len_arr': [],
    'base_stability_counts': np.zeros((4, 4), dtype=np.int64)
}

total_reads = 0
total_bases = 0
base_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

bam = pysam.AlignmentFile(bam_fname, 'rb')

for tx_idx, tx_id in enumerate(selected_tx):
    if tx_id not in tx_sequences:
        continue
    tx_seq = tx_sequences[tx_id].upper()
    reads_in_tx = 0

    for read in bam.fetch(tx_id):
        if (read.is_secondary or read.is_supplementary or read.is_unmapped or
            read.mapping_quality < 10 or read.query_length is None or read.query_length < 50):
            continue

        cigartuples = read.cigartuples
        ref_start = read.reference_start
        if not cigartuples or ref_start is None or ref_start < 0:
            continue

        err_codes = []
        ins_lengths = []
        del_lengths = []
        rpos = 0
        refp = ref_start
        read_seq = read.query_sequence

        for op, length in cigartuples:
            if op == 0:  # MATCH/MISMATCH
                max_len = min(length, len(read_seq) - rpos, len(tx_seq) - refp)
                if max_len > 0:
                    for i in range(max_len):
                        rb = read_seq[rpos+i].upper()
                        rf = tx_seq[refp+i].upper() if refp+i < len(tx_seq) else 'N'
                        if rb in base_to_idx and rf in base_to_idx:
                            aggregated['base_stability_counts'][base_to_idx[rf], base_to_idx[rb]] += 1
                        if rb in 'ATGC' and rf in 'ATGC' and rb != rf:
                            err_codes.append(1)
                        else:
                            err_codes.append(0)
                    rpos += length
                    refp += length
            elif op == 1:  # INSERTION
                err_codes.extend([2] * length)
                ins_lengths.append(length)
                rpos += length
            elif op == 2:  # DELETION
                err_codes.extend([3] * length)
                del_lengths.append(length)
                refp += length
            elif op == 4:  # SOFT CLIP
                rpos += length

        if len(err_codes) == 0 or len(err_codes) / read.query_length < 0.80:
            continue

        err_arr = np.array(err_codes, dtype=np.int8)

        # Match segments
        error_positions = np.where(err_arr != 0)[0]
        if len(error_positions) > 1:
            match_lengths = np.diff(error_positions) - 1
            match_lengths = match_lengths[(match_lengths > 0) & (match_lengths < 200)]
            aggregated['sum_match_len_arr'].extend(match_lengths.tolist())

        aggregated['ins1_arr'].extend(ins_lengths)
        aggregated['del1_arr'].extend(del_lengths)

        # Mismatch runs
        mismatch_run = 0
        for et in err_arr:
            if et == 1:
                mismatch_run += 1
            elif mismatch_run > 0:
                aggregated['mis1_len_arr'].append(mismatch_run)
                mismatch_run = 0
        if mismatch_run > 0:
            aggregated['mis1_len_arr'].append(mismatch_run)

        aggregated['sum_err_arr'].extend(err_arr.tolist())
        reads_in_tx += 1
        total_bases += len(err_arr)

    if reads_in_tx > 0:
        total_reads += reads_in_tx

    if (tx_idx + 1) % 20 == 0:
        print(f"  → {tx_idx+1}/{len(selected_tx)} | reads: {total_reads:,} | bases: {total_bases/1e6:.1f}M")

bam.close()

print(f"[SUMMARY] Transcripts: {total_reads:,} reads | {total_bases:,} bases")
print(f"  Match segments: {len(aggregated['sum_match_len_arr']):,}")
print(f"  Insertions: {len(aggregated['ins1_arr']):,}")
print(f"  Deletions: {len(aggregated['del1_arr']):,}")
print(f"  Mismatch runs: {len(aggregated['mis1_len_arr']):,}")

# === Fitting distributions ===

print("[INFO] Fitting error distributions...")

# Match histogram
hist_match = np.full(198, 1/198)
if len(aggregated['sum_match_len_arr']) > 50:
    hist_match, _ = np.histogram(aggregated['sum_match_len_arr'], bins=198,
                                  range=(1,199), density=True)
    if hist_match.sum() > 0:
        hist_match = hist_match / hist_match.sum()

# Insertions
ins_params = [0.7, 0.5, 0.7]
if FITTING_FUNCTIONS_AVAILABLE and len(aggregated['ins1_arr']) > 50:
    try:
        from scipy.optimize import minimize
        ins_res = minimize(mix_weibul_distr_fit, args=(np.array(aggregated['ins1_arr']),),
                          x0=[0.7, 0.5, 0.7], bounds=((0.01,2.0),(0.1,0.95),(0.1,0.95)),
                          method='Powell', tol=1e-6, options={'maxiter':1000})
        ins_params = list(ins_res.x)
        print(f"  ✓ Insertion: α={ins_params[0]:.3f}, pr={ins_params[1]:.3f}, w={ins_params[2]:.3f}")
    except Exception as e:
        print(f"  ⚠ Insertion fit failed: {e}")

# Deletions
del_params = [0.7, 0.5, 0.7]
if FITTING_FUNCTIONS_AVAILABLE and len(aggregated['del1_arr']) > 50:
    try:
        del_res = minimize(mix_weibul_distr_fit, args=(np.array(aggregated['del1_arr']),),
                          x0=[0.7, 0.5, 0.7], bounds=((0.01,2.0),(0.1,0.95),(0.1,0.95)),
                          method='Powell', tol=1e-6, options={'maxiter':1000})
        del_params = list(del_res.x)
        print(f"  ✓ Deletion: α={del_params[0]:.3f}, pr={del_params[1]:.3f}, w={del_params[2]:.3f}")
    except Exception as e:
        print(f"  ⚠ Deletion fit failed: {e}")

# Mismatches
mis_params = [0.7, 0.65, 0.8]
if FITTING_FUNCTIONS_AVAILABLE and len(aggregated['mis1_len_arr']) > 30:
    try:
        mis_res = minimize(mix_pois_distr_fit, args=(np.array(aggregated['mis1_len_arr']),),
                          x0=[0.7, 0.65, 0.8], bounds=((0.01,2.0),(0.1,0.95),(0.1,0.95)),
                          method='Powell', tol=1e-6, options={'maxiter':1000})
        mis_params = list(mis_res.x)
        print(f"  ✓ Mismatch: λ={mis_params[0]:.3f}, pr={mis_params[1]:.3f}, w={mis_params[2]:.3f}")
    except Exception as e:
        print(f"  ⚠ Mismatch fit failed: {e}")

# === Markov chain ===

print("[INFO] Training Markov chain...")
err_prob = [0.015, 0.008, 0.007]
mis_after_ins = 0.12
mis_after_del = 0.18

if len(aggregated['sum_err_arr']) > 300:
    err_arr = np.array(aggregated['sum_err_arr'], dtype=np.int8)
    reduced_err = [err_arr[0]]
    for i in range(1, len(err_arr)):
        if err_arr[i] != err_arr[i-1]:
            reduced_err.append(err_arr[i])

    state_seq = [digs_to_err(int(d)) for d in reduced_err[:15000]
                 if digs_to_err(int(d)) in {'match','mis','ins','del'}]

    if len(state_seq) >= 200:
        states = ['match', 'mis', 'ins', 'del']
        state_to_idx = {s: i for i, s in enumerate(states)}
        trans_counts = np.zeros((4, 4), dtype=np.int64)
        for i in range(len(state_seq) - 1):
            f = state_to_idx.get(state_seq[i], -1)
            t = state_to_idx.get(state_seq[i+1], -1)
            if f != -1 and t != -1:
                trans_counts[f, t] += 1

        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_matrix = trans_counts.astype(float) / row_sums

        if np.any(trans_matrix[0, 1:] > 0.01):
            err_prob = trans_matrix[0, 1:].round(4).tolist()
            mis_after_ins = float(trans_matrix[2, 1].round(4))
            mis_after_del = float(trans_matrix[3, 1].round(4))
            print(f"  ✓ Trained: mis={err_prob[0]:.4f}, ins={err_prob[1]:.4f}, del={err_prob[2]:.4f}")

# === Base stability matrix ===

print("[INFO] Computing base stability matrix...")
base_stability_counts = aggregated['base_stability_counts']
row_sums = base_stability_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
base_stability = (base_stability_counts / row_sums).flatten().round(6).tolist()

# Save Stage 5
stage5_results = {
    'frag_prob': float(frag_p),
    'mis_params': mis_params,
    'del_params': del_params,
    'ins_params': ins_params,
    'hist_match': hist_match.tolist(),
    'err_prob': err_prob,
    'mis_after_ins': [mis_after_ins],
    'mis_after_del': [mis_after_del],
    'base_stability': base_stability
}
with open('./tmp_tnrsim_files/stage5_results.pkl', 'wb') as f:
    pickle.dump(stage5_results, f)
print("[SUCCESS] Stage 5 completed")


"""
================================================================================
STAGE 7: QUALITY PROFILES
================================================================================
"""
print(f"\n{'='*80}")
print("[STAGE 7] QUALITY PROFILE CHARACTERIZATION")
print(f"{'='*80}")

mis_q, ins_q, end_q, start_q = [], [], [], []
total_reads_q = 0

bam = pysam.AlignmentFile(bam_fname, 'rb')

for tx_idx, tx_id in enumerate(selected_tx[:50]):
    if tx_id not in tx_sequences:
        continue
    tx_seq = tx_sequences[tx_id].upper()

    for read in bam.fetch(tx_id):
        if (read.is_secondary or read.is_supplementary or read.is_unmapped or
            read.mapping_quality < 10 or read.query_length is None or read.query_length < 100):
            continue

        cigartuples = read.cigartuples
        ref_start = read.reference_start
        if not cigartuples or ref_start is None:
            continue

        rpos = 0
        refp = ref_start
        read_seq = read.query_sequence
        quals = read.query_qualities if read.query_qualities is not None else np.zeros(read.query_length, dtype=np.int32)

        # 5'-end soft clip
        first_op, first_len = cigartuples[0]
        if first_op == 4 and 3 <= first_len <= 20:
            start_q.extend(quals[:first_len].tolist())

        for op, length in cigartuples:
            if op == 0:
                max_len = min(length, len(read_seq) - rpos, len(tx_seq) - refp)
                if max_len > 0:
                    for i in range(max_len):
                        rb = read_seq[rpos+i].upper()
                        rf = tx_seq[refp+i].upper() if refp+i < len(tx_seq) else 'N'
                        q = quals[rpos+i] if rpos+i < len(quals) else 0
                        if 0 < q < 41 and rb in 'ATGC' and rf in 'ATGC' and rb != rf:
                            mis_q.append(q)
                    rpos += length
                    refp += length
            elif op == 1:
                if rpos + length <= len(quals):
                    ins_quals = quals[rpos:rpos+length]
                    ins_q.extend([q for q in ins_quals if 0 < q < 41])
                rpos += length
            elif op == 2:
                refp += length
            elif op == 4:
                rpos += length

        # 3'-end soft clip
        last_op, last_len = cigartuples[-1]
        if last_op == 4 and 3 <= last_len <= 30 and rpos == read.query_length - last_len:
            end_q.extend(quals[rpos:rpos+last_len].tolist())

        total_reads_q += 1

    if (tx_idx + 1) % 10 == 0:
        print(f"  → {tx_idx+1}/50 | reads: {total_reads_q:,}")

bam.close()

print(f"[SUMMARY] Mismatches: {len(mis_q):,} | Insertions: {len(ins_q):,}")
print(f"  5'-clips: {len(start_q):,} | 3'-clips: {len(end_q):,}")

def build_quality_hist(q_vals, bins, min_q, max_q, label):
    arr = np.array(q_vals, dtype=np.int32)
    arr = arr[(arr >= min_q) & (arr <= max_q)]
    if len(arr) == 0:
        hist = np.full(bins, 1/bins)
    else:
        hist, _ = np.histogram(arr, bins=bins, range=(min_q, max_q+1), density=True)
        if hist.sum() > 0:
            hist = hist / hist.sum()
        else:
            hist = np.full(bins, 1/bins)
    print(f"  {label}: {len(arr):,} → {len(hist)} bins")
    return hist.tolist()

mis_q_hist = build_quality_hist(mis_q, 39, 1, 39, "Mismatch")
ins_q_hist = build_quality_hist(ins_q, 39, 1, 39, "Insertion")
end_q_hist = build_quality_hist(end_q, 29, 1, 29, "3'-end")
start_q_hist = build_quality_hist(start_q, 34, 1, 34, "5'-start")

stage7_results = {
    'mis_q_distr': mis_q_hist,
    'ins_q_distr': ins_q_hist,
    'end_q_distr': end_q_hist,
    'start_q_distr': start_q_hist
}
with open('./tmp_tnrsim_files/stage7_results.pkl', 'wb') as f:
    pickle.dump(stage7_results, f)
print("[SUCCESS] Stage 7 completed")


"""
================================================================================
STAGE 8: AUTOREGRESSIVE QUALITY MODEL
================================================================================
"""
print(f"\n{'='*80}")
print("[STAGE 8] AUTOREGRESSIVE QUALITY MODEL")
print(f"{'='*80}")

match_regions = []
total_regions = 0

bam = pysam.AlignmentFile(bam_fname, 'rb')

for tx_idx, tx_id in enumerate(selected_tx[:30]):
    if tx_id not in tx_sequences:
        continue
    tx_seq = tx_sequences[tx_id].upper()

    for read in bam.fetch(tx_id):
        if (read.is_secondary or read.is_supplementary or read.is_unmapped or
            read.mapping_quality < 10 or read.query_length is None or read.query_length < 200):
            continue

        cigartuples = read.cigartuples
        ref_start = read.reference_start
        if not cigartuples or ref_start is None:
            continue

        rpos = 0
        refp = ref_start
        quals = read.query_qualities if read.query_qualities is not None else np.zeros(read.query_length, dtype=np.int32)
        current_match = []

        for op, length in cigartuples:
            if op == 0:
                max_len = min(length, len(tx_seq) - refp)
                if max_len > 0:
                    for i in range(max_len):
                        rb = read.query_sequence[rpos+i].upper() if rpos+i < read.query_length else 'N'
                        rf = tx_seq[refp+i].upper() if refp+i < len(tx_seq) else 'N'
                        if rb in 'ATGC' and rf in 'ATGC' and rb == rf:
                            q = quals[rpos+i] if rpos+i < len(quals) else 7
                            if 0 <= q <= 40:
                                current_match.append(q)
                            else:
                                if len(current_match) == 20:
                                    match_regions.append(np.array(current_match, dtype=np.int32))
                                    total_regions += 1
                                current_match = []
                        else:
                            if len(current_match) == 20:
                                match_regions.append(np.array(current_match, dtype=np.int32))
                                total_regions += 1
                            current_match = []
                    rpos += length
                    refp += length
                else:
                    rpos += length
                    refp += length
            else:
                if len(current_match) == 20:
                    match_regions.append(np.array(current_match, dtype=np.int32))
                    total_regions += 1
                current_match = []
                if op == 1 or op == 4:
                    rpos += length
                elif op == 2:
                    refp += length

        if len(current_match) == 20:
            match_regions.append(np.array(current_match, dtype=np.int32))
            total_regions += 1

    if (tx_idx + 1) % 10 == 0:
        print(f"  → {tx_idx+1}/30 | regions: {total_regions:,}")

bam.close()

print(f"[SUMMARY] Match regions of length 20: {total_regions:,}")

autoreg_specs = [0.569, -10, -10, -8, -6, 7, 10, 0.33, 9.80965]

if total_regions >= 100:
    print("[INFO] Optimizing autoregressive parameters...")
    n_regions = min(5000, len(match_regions))
    x_arr = np.tile(np.arange(20), n_regions)
    y_arr = np.concatenate([reg for reg in match_regions[:n_regions]])

    def fit_autoreg(pars):
        try:
            alpha = np.clip(pars[0], 0.4, 0.9)
            l1 = np.clip(pars[1], -20, -5)
            c1 = np.clip(pars[2], max(l1+0.1, -12), -5)
            r1 = np.clip(pars[3], max(c1+0.1, -15), 5)
            l2 = np.clip(pars[4], -10, 15)
            c2 = np.clip(pars[5], max(l2+0.1, 0), 25)
            r2 = np.clip(pars[6], max(c2+0.1, 8), 40)
            w1 = np.clip(pars[7], 0.01, 0.99)
            offset = np.clip(pars[8], 5, 30)

            noise1 = np.random.triangular(l1, c1, r1, size=round(n_regions*19*w1))
            noise2 = np.random.triangular(l2, c2, r2, size=round(n_regions*19*(1-w1)))
            noise = np.concatenate([noise1, noise2])
            np.random.shuffle(noise)

            sim_regs = []
            noise_idx = 0
            for _ in range(n_regions):
                sim_reg = [np.random.randint(5, 9)]
                for i in range(1, 20):
                    val = alpha * sim_reg[-1] + (noise[noise_idx] if noise_idx < len(noise) else 0) + offset
                    noise_idx += 1
                    sim_reg.append(int(np.clip(round(val), 0, 40)))
                sim_regs.append(sim_reg)

            sim_y = np.concatenate(sim_regs)
            real_hist = np.histogram2d(y_arr, x_arr, bins=[np.arange(0,41), np.arange(0,21)], density=True)[0]
            sim_hist = np.histogram2d(sim_y, np.tile(np.arange(20), n_regions),
                                      bins=[np.arange(0,41), np.arange(0,21)], density=True)[0]

            return np.sum(np.abs(real_hist[:, 2:18] - sim_hist[:, 2:18]))
        except:
            return 1e6

    try:
        res = minimize(fit_autoreg, x0=[0.569, -10, -10, -8, -6, 7, 10, 0.33, 9.80965],
                      bounds=((0.4, 0.9), (-20, -5), (-12, -5), (-15, 5), (-10, 20),
                              (0, 30), (8, 40), (0.01, 0.99), (5, 30)),
                      method='Powell', tol=0.1, options={'maxiter': 300})
        if res.success and res.fun < 10.0:
            autoreg_specs = res.x.tolist()
            print(f"  ✓ Optimization successful (loss={res.fun:.2f})")
    except Exception as e:
        print(f"  ⚠ Optimization failed: {e}")

stage8_results = {'autoreg_specs': autoreg_specs}
with open('./tmp_tnrsim_files/stage8_results.pkl', 'wb') as f:
    pickle.dump(stage8_results, f)
print("[SUCCESS] Stage 8 completed")


"""
================================================================================
STAGE 11: MODEL FINALIZATION
================================================================================
"""
print(f"\n{'='*80}")
print("[STAGE 11] MODEL FINALIZATION")
print(f"{'='*80}")

# Load all stages
with open('./tmp_tnrsim_files/stage5_results.pkl', 'rb') as f:
    stage5_results = pickle.load(f)
with open('./tmp_tnrsim_files/stage7_results.pkl', 'rb') as f:
    stage7_results = pickle.load(f)
with open('./tmp_tnrsim_files/stage8_results.pkl', 'rb') as f:
    stage8_results = pickle.load(f)

# Default homopolymer parameters (transcriptome mode)
hp_defaults = {
    'A_hp_spec': ['5.0:0.8:5;5.5:0.9:6;6.0:1.0:7;6.5:1.1:8;7.0:1.2:9;7.5:1.3:10;8.0:1.4:11;8.5:1.5:12;9.0:1.6:13;9.5:1.7:14;10.0:1.8:15;10.5:1.9:16;11.0:2.0:17;11.5:2.1:18;12.0:2.2:19;12.5:2.3:20;13.0:2.4:21;13.5:2.5:22;14.0:2.6:23;14.5:2.7:24;15.0:2.8:25;15.5:2.9:26;16.0:3.0:27;16.5:3.1:28'],
    'T_hp_spec': ['5.0:0.8:5;5.5:0.9:6;6.0:1.0:7;6.5:1.1:8;7.0:1.2:9;7.5:1.3:10;8.0:1.4:11;8.5:1.5:12;9.0:1.6:13;9.5:1.7:14;10.0:1.8:15;10.5:1.9:16;11.0:2.0:17;11.5:2.1:18;12.0:2.2:19;12.5:2.3:20;13.0:2.4:21;13.5:2.5:22;14.0:2.6:23;14.5:2.7:24;15.0:2.8:25;15.5:2.9:26;16.0:3.0:27;16.5:3.1:28'],
    'G_hp_spec': ['5.0:0.7:5;5.5:0.8:6;6.0:0.9:7;6.5:1.0:8;7.0:1.1:9;7.5:1.2:10;8.0:1.3:11;8.5:1.4:12;9.0:1.5:13;9.5:1.6:14;10.0:1.7:15;10.5:1.8:16;11.0:1.9:17;11.5:2.0:18;12.0:2.1:19;12.5:2.2:20;13.0:2.3:21;13.5:2.4:22;14.0:2.5:23;14.5:2.6:24;15.0:2.7:25;15.5:2.8:26;16.0:2.9:27;16.5:3.0:28'],
    'C_hp_spec': ['5.0:0.7:5;5.5:0.8:6;6.0:0.9:7;6.5:1.0:8;7.0:1.1:9;7.5:1.2:10;8.0:1.3:11;8.5:1.4:12;9.0:1.5:13;9.5:1.6:14;10.0:1.7:15;10.5:1.8:16;11.0:1.9:17;11.5:2.0:18;12.0:2.1:19;12.5:2.2:20;13.0:2.3:21;13.5:2.4:22;14.0:2.5:23;14.5:2.6:24;15.0:2.7:25;15.5:2.8:26;16.0:2.9:27;16.5:3.0:28'],
    'A_hp_spec_qual': ['7.0:10.0:-3.0:0;7.1:10.1:-2.9:1;7.2:10.2:-2.8:2;7.3:10.3:-2.7:3;7.4:10.4:-2.6:4;7.5:10.5:-2.5:5;7.6:10.6:-2.4:6'],
    'T_hp_spec_qual': ['7.0:10.0:-3.0:0;7.1:10.1:-2.9:1;7.2:10.2:-2.8:2;7.3:10.3:-2.7:3;7.4:10.4:-2.6:4;7.5:10.5:-2.5:5;7.6:10.6:-2.4:6'],
    'G_hp_spec_qual': ['6.5:9.5:-3.5:0;6.6:9.6:-3.4:1;6.7:9.7:-3.3:2;6.8:9.8:-3.2:3;6.9:9.9:-3.1:4;7.0:10.0:-3.0:5;7.1:10.1:-2.9:6'],
    'C_hp_spec_qual': ['6.5:9.5:-3.5:0;6.6:9.6:-3.4:1;6.7:9.7:-3.3:2;6.8:9.8:-3.2:3;6.9:9.9:-3.1:4;7.0:10.0:-3.0:5;7.1:10.1:-2.9:6']
}

# Build final model
model_dict = {}
model_dict.update(stage5_results)
model_dict.update(stage7_results)
model_dict.update(stage8_results)
model_dict.update(hp_defaults)

print("[SUMMARY] Final model parameters:")
print(f"  Fragmentation p = {model_dict['frag_prob']:.6f}")
print(f"  Mismatch params: λ={model_dict['mis_params'][0]:.3f}, pr={model_dict['mis_params'][1]:.3f}, w={model_dict['mis_params'][2]:.3f}")
print(f"  Insertion params: α={model_dict['ins_params'][0]:.3f}, pr={model_dict['ins_params'][1]:.3f}, w={model_dict['ins_params'][2]:.3f}")
print(f"  Deletion params: α={model_dict['del_params'][0]:.3f}, pr={model_dict['del_params'][1]:.3f}, w={model_dict['del_params'][2]:.3f}")
print(f"  Markov transitions: mis={model_dict['err_prob'][0]:.4f}, ins={model_dict['err_prob'][1]:.4f}, del={model_dict['err_prob'][2]:.4f}")

# Save TSV
model_df = pd.DataFrame(model_dict.items(), columns=['params', 'values'])
model_df.to_csv(output_file, sep='\t', index=False)
print(f"[SUCCESS] Model saved to: {output_file}")
print(f"  Total parameters: {len(model_df)}")

# Cleanup
print("[INFO] Cleaning up...")
try:
    shutil.rmtree('./tmp_tnrsim_files')
    print("  ✓ Temporary directory removed")
except Exception as e:
    print(f"  ⚠ Cleanup failed: {e}")

print(f"\n{'='*80}")
print("[CHARACTERIZATION COMPLETE]")
print(f"  Fragmentation p = {model_dict['frag_prob']:.6f}")
print(f"  Model: {output_file}")
print(f"{'='*80}")
