# TNRSim
TNRSim (Truncated Nanopore Reads Simulator aka Transcriptomic Nanopore Reads simulator)
![TNRSim](https://github.com/Artemiy-Saharov/TNRSim/blob/main/Logo.png)
## Requirements:
- numpy
- pandas
- scipy
- pysam
- biopython

## Installation
Download TNRSim from Github:

    git clone https://github.com/Artemiy-Saharov/TNRSim.git
    
Create conda environment with all dependencies

    cd TNRSim
    conda env create -f tnrsim_env.yml # env name is tnrsim
Environment name is `tnrsim`

Now you can execute `TNRSim.py` and `characterization.py` in this environment

Optionally you could add src dir to `PATH`
## Usage
#### Simulation

    TNRSim.py -e TNRSim_basic_exp_prof.tsv -m dRNA_model.tsv -t transcriptome.fasta -O simulated_reads.fastq
  
Key feature of TNRSim is imitation of reads fragmentation (simulation of truncated reads). Fragmentation probability setting is responsible for read lenght distribution (you can see example of this distribution in TNRSim logo). Fragmentation probability is estimated during library characterization and specified in model file, but also could be provided with flag `-f/--fragmentation_probability` (0.0004 is low fragmentation probability (good library) and 0.0015 is high (bad library)).

    python3 TNRSim.py -f 0.001 -e /path/to/exp_prof.tsv -m /path/to/model.tsv -t /path/to/transcriptome.fasta -O /path/to/simulated_reads.fastq
    
Number of threads to use could be provided with flag `--threads`
    
#### Characterization

Library characterization

    python3 characterization.py -t /path/to/transcriptome.fa -b /path/to/your.bam --min_reads 300 -O /path/to/model.tsv
    
In general case it is don't needed to fit all model, just estimate fragmentation probability. It could be performed with flag `-f/--fragmentation_only`

    python3 characterization.py -f -t /path/to/transcriptome.fa -b /path/to/your.bam --min_reads 300
    
Number of threads to use could be provided with flag `--threads`

## Citation
Manuscript in preparation
