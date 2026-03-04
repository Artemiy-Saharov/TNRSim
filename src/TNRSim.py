#!/usr/bin/env python3
"""
TNRSim Simulator — Full pipeline with robust error simulation and length synchronization
"""

import os
import sys
import numpy as np
import pandas as pd
import pysam
import argparse
import random
from scipy.stats import skewnorm, geom
from multiprocessing import Pool, cpu_count
import time

_triangular_warnings_issued = set()

def normalize_scalar(value):
    """Convert single-element lists/arrays to scalars; preserve multi-element arrays."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 1:
            return float(value[0])
        return [float(v) for v in value]
    return float(value)


def parse_model_value(value_str):
    """Universal parser for TNRSim model values — handles both formats."""
    value_str = str(value_str).strip()

    # Special case: homopolymer specs are semicolon-delimited strings
    if ';' in value_str and ':' in value_str and not value_str.startswith('['):
        return value_str

    # Remove brackets if present
    if value_str.startswith('[') and value_str.endswith(']'):
        value_str = value_str[1:-1]

    # Split and convert to floats
    try:
        parts = [x.strip() for x in value_str.split(',') if x.strip()]
        return [float(p) for p in parts]
    except ValueError:
        return value_str


def load_homopolymer_params(model_df, param_suffix):
    """Correctly extract homopolymer parameters using ENDWITHS."""
    hp_dict = {}
    hp_rows = model_df[model_df['params'].str.endswith(param_suffix)]

    for _, row in hp_rows.iterrows():
        param_name = row['params']
        base = param_name[0]
        value_str = row['values']
        specs = str(value_str).strip()

        if specs.startswith('[') and specs.endswith(']'):
            specs = specs[1:-1]

        for spec_entry in specs.split(';'):
            spec_entry = spec_entry.strip()
            if not spec_entry:
                continue

            parts = spec_entry.split(':')

            if param_suffix == '_hp_spec' and len(parts) >= 3:
                try:
                    mu = float(parts[0])
                    sigma = float(parts[1])
                    ref_len = int(float(parts[2]))
                    hp_dict[f"{ref_len}{base}"] = (mu, sigma)
                except (ValueError, IndexError):
                    continue

            elif param_suffix == '_hp_spec_qual' and len(parts) >= 4:
                try:
                    loc = float(parts[0])
                    scale = float(parts[1])
                    shape = float(parts[2])
                    pos = int(float(parts[3]))
                    hp_dict[f"{pos}{base}"] = (loc, scale, shape)
                except (ValueError, IndexError):
                    continue

    return hp_dict


def validate_model(model_dict):
    """Validate critical model parameters with type normalization."""
    errors = []

    # Normalize scalar parameters
    for param in ['frag_prob', 'mis_after_ins', 'mis_after_del']:
        try:
            model_dict[param] = normalize_scalar(model_dict[param])
        except Exception as e:
            errors.append(f"Failed to normalize {param}: {e}")

    # Check fragmentation probability
    p = model_dict['frag_prob']
    if not (0.0001 <= p <= 0.01):
        errors.append(f"frag_prob={p:.6f} outside biologically plausible range [0.0001, 0.01]")

    # Check distributions
    for param in ['hist_match', 'mis_params', 'ins_params', 'del_params']:
        if param not in model_dict:
            errors.append(f"Missing '{param}' parameter")
        elif not isinstance(model_dict[param], (list, tuple, np.ndarray)) or len(model_dict[param]) == 0:
            errors.append(f"'{param}' must be a non-empty list/array")

    # Check base stability matrix
    base_stab = model_dict.get('base_stability')
    if base_stab is None or not isinstance(base_stab, (list, tuple, np.ndarray)) or len(base_stab) != 16:
        errors.append("base_stability must contain exactly 16 values (4x4 matrix)")

    # Check autoregressive specs
    autoreg = model_dict.get('autoreg_specs')
    if autoreg is None or not isinstance(autoreg, (list, tuple, np.ndarray)) or len(autoreg) != 9:
        errors.append("autoreg_specs must contain exactly 9 parameters")

    # Check homopolymer dictionaries
    for hp_dict_name in ['hp_distr_dict', 'hp_qual_dict']:
        hp_dict = model_dict.get(hp_dict_name)
        if hp_dict is None or not isinstance(hp_dict, dict):
            errors.append(f"{hp_dict_name} must be a dictionary")

    if errors:
        print("\n[CRITICAL] Model validation FAILED:")
        for err in errors:
            print(f"  ✗ {err}")
        sys.exit(1)

    print("\n[SUCCESS] Model validation PASSED")
    print(f"  Fragmentation probability: p = {model_dict['frag_prob']:.6f}")
    print(f"  Match histogram bins: {len(model_dict['hist_match'])}")
    base_stab_matrix = np.array(model_dict['base_stability']).reshape(4, 4)
    print(f"  Base stability matrix diagonal: {[round(x,3) for x in np.diag(base_stab_matrix)]}")
    print(f"  Homopolymer length specs: {len(model_dict['hp_distr_dict'])} entries")
    print(f"  Homopolymer quality specs: {len(model_dict['hp_qual_dict'])} entries")


def sanitize_triangular_params(l, c, r, min_spread=0.5):
    """
    Ensure l < c < r for triangular distribution with minimum spread.

    triangular() requires: left < mode < right (strict inequalities)
    This function enforces these constraints with a minimum spread.

    Parameters:
    -----------
    l, c, r : float
        Left, mode, right parameters
    min_spread : float
        Minimum distance between l-c and c-r (default: 0.5)

    Returns:
    --------
    tuple : (l, c, r) sanitized
    """
    l, c, r = float(l), float(c), float(r)

    # Ensure strict ordering: l < c < r
    if c <= l:
        c = l + min_spread

    if r <= c:
        r = c + min_spread

    # Ensure minimum spread on both sides
    if c - l < min_spread:
        l = c - min_spread

    if r - c < min_spread:
        r = c + min_spread

    # Final safety check
    assert l < c < r, f"Triangular params invalid after sanitization: l={l}, c={c}, r={r}"

    return l, c, r


def base_to_dig(letter):
    """Convert nucleotide to digit (0=A,1=T,2=G,3=C)."""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'U': 1}
    return mapping.get(letter.upper(), -1)


def identify_homopolymers(seq, min_length=5):
    """Identify homopolymers in sequence. Returns [(start, end, base), ...]."""
    hps = []
    if len(seq) < min_length:
        return hps

    cur_base = seq[0].upper()
    start = 0

    for i in range(1, len(seq) + 1):
        if i < len(seq) and seq[i].upper() == cur_base:
            continue
        else:
            hp_len = i - start
            if hp_len >= min_length:
                hps.append((start, i, cur_base))
            if i < len(seq):
                cur_base = seq[i].upper()
                start = i

    return hps


def gen_qual(seg_type, inp_seg_len, model_dict, warn_once=True):
    """
    Generate quality string of EXACT length inp_seg_len.
    FIX: Added strict triangular parameter sanitization (l < c < r).

    Parameters:
    -----------
    warn_once : bool
        If True, only warn once per unique parameter set to reduce spam
    """
    global _triangular_warnings_issued

    if inp_seg_len <= 0:
        return ''

    match_autoreg_specs = model_dict['autoreg_specs']
    mis_q_distr = np.array(model_dict['mis_q_distr'])
    ins_q_distr = np.array(model_dict['ins_q_distr'])
    end_q_distr = np.array(model_dict['end_q_distr'])
    start_q_distr = np.array(model_dict['start_q_distr'])

    out_q = np.zeros(inp_seg_len, dtype=np.int32)

    if seg_type == 'match' and inp_seg_len >= 2:
        # Извлекаем параметры авторегрессионной модели
        l1, c1, r1 = match_autoreg_specs[1], match_autoreg_specs[2], match_autoreg_specs[3]
        l2, c2, r2 = match_autoreg_specs[4], match_autoreg_specs[5], match_autoreg_specs[6]
        triag_weight = match_autoreg_specs[7]
        alpha = match_autoreg_specs[0]
        offset = match_autoreg_specs[8]

        # === FIX: Строгая санитизация с min_spread ===
        l1, c1, r1 = sanitize_triangular_params(l1, c1, r1, min_spread=0.5)
        l2, c2, r2 = sanitize_triangular_params(l2, c2, r2, min_spread=0.5)

        n_noise = inp_seg_len - 1

        # Ключ для отслеживания предупреждений
        param_key = f"({l1:.2f},{c1:.2f},{r1:.2f})_({l2:.2f},{c2:.2f},{r2:.2f})"

        # Генерация шума с защитой от ошибок
        try:
            noise1 = np.random.triangular(l1, c1, r1, size=round(n_noise * triag_weight))
            noise2 = np.random.triangular(l2, c2, r2, size=round(n_noise * (1 - triag_weight)))
        except ValueError as e:
            # Предупреждаем только один раз на уникальный набор параметров
            if warn_once and param_key not in _triangular_warnings_issued:
                print(f"[WARNING] Triangular distribution failed: {e}. Using normal distribution fallback.")
                print(f"  Parameters: set1=({l1:.2f},{c1:.2f},{r1:.2f}), set2=({l2:.2f},{c2:.2f},{r2:.2f})")
                _triangular_warnings_issued.add(param_key)

            # Fallback: нормальное распределение
            noise1 = np.random.normal(c1, max(0.5, abs(r1-l1)/4), size=round(n_noise * triag_weight))
            noise2 = np.random.normal(c2, max(0.5, abs(r2-l2)/4), size=round(n_noise * (1 - triag_weight)))

        noise = np.concatenate([noise1, noise2])
        np.random.shuffle(noise)
        noise = noise[:n_noise]

        center = inp_seg_len // 2
        out_q[center] = np.random.randint(7, 12)

        for i in range(center - 1, -1, -1):
            val = alpha * out_q[i + 1] + noise[i] + offset
            out_q[i] = int(np.clip(round(val), 3, 38))

        for i in range(center + 1, inp_seg_len):
            idx = i - center - 1 + center
            val = alpha * out_q[i - 1] + noise[idx] + offset
            out_q[i] = int(np.clip(round(val), 3, 38))

    elif seg_type == 'mis':
        bins = np.arange(1, 40)
        probs = mis_q_distr / mis_q_distr.sum() if mis_q_distr.sum() > 0 else np.full(39, 1/39)
        out_q = np.random.choice(bins, size=inp_seg_len, p=probs)

    elif seg_type == 'ins':
        bins = np.arange(1, 40)
        probs = ins_q_distr / ins_q_distr.sum() if ins_q_distr.sum() > 0 else np.full(39, 1/39)
        out_q = np.random.choice(bins, size=inp_seg_len, p=probs)

    elif seg_type == 'start_unalig':
        bins = np.arange(1, 35)
        probs = start_q_distr / start_q_distr.sum() if start_q_distr.sum() > 0 else np.full(34, 1/34)
        out_q = np.random.choice(bins, size=inp_seg_len, p=probs)

    elif seg_type == 'end_unalig':
        bins = np.arange(1, 30)
        probs = end_q_distr / end_q_distr.sum() if end_q_distr.sum() > 0 else np.full(29, 1/29)
        out_q = np.random.choice(bins, size=inp_seg_len, p=probs)

    return ''.join(chr(min(40, max(3, int(q))) + 33) for q in out_q)


def simulate_homopolymer(ref_len, base, hp_distr_dict):
    """Simulate observed homopolymer length based on reference length."""
    base = base.upper()
    if ref_len > 28:
        ref_len = 28

    key = f"{ref_len}{base}"
    if key not in hp_distr_dict:
        mu = 0.8 * np.log1p(ref_len)
        sigma = 0.9 * np.sqrt(ref_len)
    else:
        mu, sigma = hp_distr_dict[key]

    shift = int(round(np.random.normal(mu, sigma)))
    obs_len = max(1, ref_len + shift)
    return obs_len


def gen_hp_qual(hp_len, base, hp_qual_dict):
    """Generate position-specific quality for homopolymer."""
    base = base.upper()
    qual_str = ''

    for pos in range(hp_len):
        pos_key = f"{min(pos + 1, 7)}{base}"
        if pos_key in hp_qual_dict:
            loc, scale, shape = hp_qual_dict[pos_key]
            q_val = int(round(skewnorm.rvs(a=shape, loc=loc, scale=scale)))
        else:
            q_val = np.random.randint(5, 12)

        q_val = min(40, max(3, q_val))
        qual_str += chr(q_val + 33)

    return qual_str


def fragmentation_distribution(transcript_len, n_reads, frag_prob):
    """Generate fragment lengths according to geometric fragmentation model."""
    positions = np.arange(1, transcript_len + 1)
    survival = (1 - frag_prob) ** (positions - 1)
    density = survival * frag_prob
    density[-1] += (1 - frag_prob) ** transcript_len
    density = density / density.sum()
    sampled_lengths = np.random.choice(positions, size=n_reads, p=density)
    return sampled_lengths


def make_errors_with_sync(seq, homopolymers, model_dict):
    """Introduce errors with guaranteed sequence/quality length synchronization."""
    if len(seq) == 0:
        return '', ''

    hist_match = np.array(model_dict['hist_match'])
    mis_params = model_dict['mis_params']
    ins_params = model_dict['ins_params']
    del_params = model_dict['del_params']
    err_prob = model_dict['err_prob']
    mis_after_ins = model_dict['mis_after_ins']
    mis_after_del = model_dict['mis_after_del']
    base_stability = np.array(model_dict['base_stability']).reshape(4, 4)
    hp_distr_dict = model_dict['hp_distr_dict']
    hp_qual_dict = model_dict['hp_qual_dict']

    # Pre-generate distributions
    match_lens = np.random.choice(np.arange(1, 199), size=500, p=hist_match / hist_match.sum())
    mis_lens = np.random.poisson(mis_params[0], size=200) + 1
    ins_lens = np.ceil(np.random.weibull(ins_params[0], size=200)).astype(int)
    del_lens = np.ceil(np.random.weibull(del_params[0], size=200)).astype(int)

    read_bases = []
    read_quals = []
    pos = 0
    seq_len = len(seq)
    hp_idx = 0

    while pos < seq_len:
        in_hp = False
        current_hp = None

        if hp_idx < len(homopolymers):
            hp_start, hp_end, hp_base = homopolymers[hp_idx]
            if hp_start <= pos < hp_end:
                in_hp = True
                current_hp = (hp_start, hp_end, hp_base)
            elif pos >= hp_end:
                hp_idx += 1

        if in_hp:
            hp_start, hp_end, hp_base = current_hp
            ref_hp_len = hp_end - hp_start
            obs_hp_len = simulate_homopolymer(ref_hp_len, hp_base, hp_distr_dict)
            hp_seq = hp_base * obs_hp_len
            read_bases.append(hp_seq)
            read_quals.append(gen_hp_qual(obs_hp_len, hp_base, hp_qual_dict))
            pos = hp_end
            continue

        match_len = int(match_lens[np.random.randint(len(match_lens))])
        match_len = min(match_len, seq_len - pos)
        if match_len <= 0:
            pos += 1
            continue

        seg_seq = seq[pos:pos + match_len]
        seg_qual = gen_qual('match', match_len, model_dict)
        read_bases.append(seg_seq)
        read_quals.append(seg_qual)
        pos += match_len

        if pos < seq_len and np.random.random() < sum(err_prob):
            error_type = np.random.choice(['mis', 'ins', 'del'], p=np.array(err_prob) / sum(err_prob))

            if error_type == 'mis' and pos < seq_len:
                ref_base = seq[pos].upper()
                ref_idx = base_to_dig(ref_base)
                if ref_idx != -1:
                    subs_probs = base_stability[ref_idx].copy()
                    subs_probs[ref_idx] = 0
                    if subs_probs.sum() > 0:
                        subs_probs = subs_probs / subs_probs.sum()
                        obs_base = np.random.choice(['A', 'T', 'G', 'C'], p=subs_probs)
                    else:
                        obs_base = ref_base
                else:
                    obs_base = np.random.choice(['A', 'T', 'G', 'C'])
                read_bases.append(obs_base)
                read_quals.append(gen_qual('mis', 1, model_dict))
                pos += 1

            elif error_type == 'ins':
                ins_len = int(ins_lens[np.random.randint(len(ins_lens))])
                ins_seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=ins_len))
                read_bases.append(ins_seq)
                read_quals.append(gen_qual('ins', ins_len, model_dict))

            elif error_type == 'del':
                del_len = int(del_lens[np.random.randint(len(del_lens))])
                pos += del_len

    final_seq = ''.join(read_bases)
    final_qual = ''.join(read_quals)
    min_len = min(len(final_seq), len(final_qual))
    return final_seq[:min_len], final_qual[:min_len]


def add_biological_artifacts(read_seq, qual_str, model_dict, add_long_unalig=False):
    """Add poly-A tail and unaligned terminal regions."""
    # 5' unaligned region
    start_unalig_len = geom.rvs(0.3) - 1
    if start_unalig_len > 0:
        start_seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=start_unalig_len))
        start_qual = gen_qual('start_unalig', start_unalig_len, model_dict)
        read_seq = start_seq + read_seq
        qual_str = start_qual + qual_str

    # Poly-A tail
    polyA_len = geom.rvs(0.18) + 3
    polyA_seq = 'A' * polyA_len
    polyA_qual = gen_qual('end_unalig', polyA_len, model_dict)
    read_seq += polyA_seq
    qual_str += polyA_qual

    # Long 3' unaligned region
    if add_long_unalig:
        long_unalig_len = np.random.randint(70, 120)
        long_seq = ''.join(np.random.choice(
            ['A', 'T', 'G', 'C'],
            size=long_unalig_len,
            p=[0.316, 0.264, 0.017, 0.403]
        ))
        long_qual = gen_qual('end_unalig', long_unalig_len, model_dict)
        read_seq += long_seq
        qual_str += long_qual

    min_len = min(len(read_seq), len(qual_str))
    return read_seq[:min_len], qual_str[:min_len]


def simulate_transcript_chunk(args):
    """Simulate reads for a chunk of transcripts (multiprocessing worker)."""
    transcript_ids, counts_list, transcriptome_path, model_dict, seed_offset = args

    # Set thread-specific seed
    if seed_offset is not None:
        random.seed(seed_offset)
        np.random.seed(seed_offset)

    # Open transcriptome (each process has its own handle)
    transcripts = pysam.FastaFile(transcriptome_path)
    results = []
    total_reads = 0

    for tx_id, n_reads in zip(transcript_ids, counts_list):
        try:
            tx_seq = transcripts.fetch(tx_id).upper().replace('U', 'T')
        except KeyError:
            continue

        if len(tx_seq) < 50:
            continue

        homopolymers = identify_homopolymers(tx_seq, min_length=5)
        frag_lengths = fragmentation_distribution(len(tx_seq), n_reads, model_dict['frag_prob'])
        n_with_long_unalig = int(n_reads * 0.25)

        for i, target_len in enumerate(frag_lengths):
            max_start = max(0, len(tx_seq) - target_len)
            start_pos = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            fragment_seq = tx_seq[start_pos:start_pos + target_len]
            add_long_unalig = (i < n_with_long_unalig)

            read_seq, qual_str = make_errors_with_sync(fragment_seq, homopolymers, model_dict)
            read_seq, qual_str = add_biological_artifacts(read_seq, qual_str, model_dict, add_long_unalig)

            # Final length synchronization
            min_len = min(len(read_seq), len(qual_str))
            if min_len > 0:
                results.append(f"@{tx_id}_read_{total_reads}\n{read_seq[:min_len]}\n+\n{qual_str[:min_len]}\n")
                total_reads += 1

    transcripts.close()
    return results


def main():
    global _triangular_warnings_issued
    _triangular_warnings_issued = set()

    # ============================================================================
    # STAGE 1: Argument parsing (ORIGINAL INTERFACE PRESERVED)
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='TNRSim Simulator — Nanopore direct RNA sequencing simulator',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to TNRSim model file (TSV format)')
    parser.add_argument('-t', '--transcriptome', type=str, required=True,
                        help='FASTA file with transcript sequences')
    parser.add_argument('-e', '--exp_prof', type=str, required=True,
                        help='TSV file with expression profile (columns: transcript_id, counts)')
    parser.add_argument('-O', '--output', type=str, default='simulated_reads.fastq',
                        help='Output FASTQ file (default: simulated_reads.fastq)')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of CPU cores for simulation (default: 1)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('-f', '--fragmentation_probability', type=float, default=None,
                        help='Override fragmentation probability from model')

    args = parser.parse_args()

    print("="*80)
    print("TNRSim Simulator — Stage 1: Initialization")
    print("="*80)
    print(f"[INFO] Model file:          {args.model}")
    print(f"[INFO] Transcriptome:       {args.transcriptome}")
    print(f"[INFO] Expression profile:  {args.exp_prof}")
    print(f"[INFO] Output file:         {args.output}")
    print(f"[INFO] Threads:             {args.threads}")
    print(f"[INFO] Random seed:         {args.seed if args.seed is not None else 'None (non-deterministic)'}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[INFO] Random seed set to {args.seed} for reproducibility")

    # ============================================================================
    # STAGE 2: Model loading and parsing
    # ============================================================================
    print("\n" + "="*80)
    print("TNRSim Simulator — Stage 2: Model Loading & Parsing")
    print("="*80)

    for fname, label in [(args.model, 'model'), (args.transcriptome, 'transcriptome'), (args.exp_prof, 'expression profile')]:
        if not os.path.exists(fname):
            print(f"[ERROR] {label.capitalize()} file not found: {fname}", file=sys.stderr)
            sys.exit(1)

    try:
        seq_model = pd.read_csv(args.model, sep='\t')
        print(f"[INFO] Loaded model with {len(seq_model)} parameters")
    except Exception as e:
        print(f"[ERROR] Failed to load model file: {e}", file=sys.stderr)
        sys.exit(1)

    model_dict = {}
    for _, row in seq_model.iterrows():
        param_name = row['params']
        param_value = parse_model_value(row['values'])
        model_dict[param_name] = param_value

    print("[INFO] Extracting homopolymer parameters...")
    model_dict['hp_distr_dict'] = load_homopolymer_params(seq_model, '_hp_spec')
    model_dict['hp_qual_dict'] = load_homopolymer_params(seq_model, '_hp_spec_qual')

    if args.fragmentation_probability is not None:
        print(f"[INFO] Overriding fragmentation probability: {model_dict['frag_prob']} → {args.fragmentation_probability}")
        model_dict['frag_prob'] = args.fragmentation_probability

    # ============================================================================
    # STAGE 3: Data validation
    # ============================================================================
    print("\n" + "="*80)
    print("TNRSim Simulator — Stage 3: Data Validation")
    print("="*80)

    try:
        exp_prof = pd.read_csv(args.exp_prof, sep='\t')
        if 'transcript_id' not in exp_prof.columns or 'counts' not in exp_prof.columns:
            if 'Geneid' in exp_prof.columns:
                exp_prof.rename(columns={'Geneid': 'transcript_id'}, inplace=True)
            if exp_prof.shape[1] >= 2:
                exp_prof.rename(columns={exp_prof.columns[0]: 'transcript_id', exp_prof.columns[1]: 'counts'}, inplace=True)
        if 'transcript_id' not in exp_prof.columns or 'counts' not in exp_prof.columns:
            raise ValueError(f"Expression profile must contain 'transcript_id' and 'counts' columns")
        exp_prof = exp_prof[exp_prof['counts'] > 0].copy()
        exp_prof['counts'] = exp_prof['counts'].astype(int)
        print(f"[INFO] Loaded expression profile: {len(exp_prof)} transcripts with >0 reads")
    except Exception as e:
        print(f"[ERROR] Failed to load expression profile: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        transcripts = pysam.FastaFile(args.transcriptome)
        tx_count = len(transcripts.references)
        print(f"[INFO] Loaded transcriptome: {tx_count} transcripts")
        tx_in_fasta = set(transcripts.references)
        tx_in_exp = set(exp_prof['transcript_id'])
        tx_overlap = tx_in_fasta & tx_in_exp
        if len(tx_overlap) == 0:
            print("[ERROR] No overlap between expression profile and transcriptome!", file=sys.stderr)
            sys.exit(1)
        exp_prof = exp_prof[exp_prof['transcript_id'].isin(tx_overlap)].copy()
        print(f"[INFO] Filtered to {len(exp_prof)} transcripts present in both datasets")
        transcripts.close()
    except Exception as e:
        print(f"[ERROR] Failed to load transcriptome: {e}", file=sys.stderr)
        sys.exit(1)

    validate_model(model_dict)

    # ============================================================================
    # STAGES 4-8: Error simulation with multiprocessing
    # ============================================================================
    print("\n" + "="*80)
    print("TNRSim Simulator — Stages 4-8: Error Simulation & Parallel Execution")
    print("="*80)

    n_threads = min(args.threads, cpu_count(), len(exp_prof))
    print(f"[INFO] Using {n_threads} threads for simulation")

    # Split transcripts into chunks
    transcript_ids = exp_prof['transcript_id'].tolist()
    counts_list = exp_prof['counts'].tolist()
    chunk_size = max(1, len(transcript_ids) // n_threads)
    chunks = []
    for i in range(n_threads):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < n_threads - 1 else len(transcript_ids)
        chunk_tx = transcript_ids[start_idx:end_idx]
        chunk_counts = counts_list[start_idx:end_idx]
        seed_offset = args.seed + i if args.seed is not None else None
        chunks.append((chunk_tx, chunk_counts, args.transcriptome, model_dict.copy(), seed_offset))

    # Run simulation in parallel
    start_time = time.time()
    total_reads = 0

    with Pool(processes=n_threads) as pool:
        results = pool.map(simulate_transcript_chunk, chunks)

    # Write results to output file
    with open(args.output, 'w') as out_f:
        for chunk_results in results:
            for record in chunk_results:
                out_f.write(record)
                total_reads += 1

    elapsed = time.time() - start_time

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"[SUMMARY]")
    print(f"  Total transcripts simulated: {len(exp_prof)}")
    print(f"  Total reads generated:       {total_reads:,}")
    print(f"  Mean read length:            ~{int(1.0/model_dict['frag_prob'])} nt (theoretical)")
    print(f"  Execution time:              {elapsed:.1f} seconds")
    print(f"  Throughput:                  {total_reads/elapsed:.0f} reads/sec")
    print(f"\n[OUTPUT] FASTQ file saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
