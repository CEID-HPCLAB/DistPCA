# 🧬 DistPCA: Tera-Scale Genomic PCA via Out-of-Core Distributed Parallelism

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20392866.svg)](https://doi.org/10.5281/zenodo.20392866) <br>
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white)
![HPC](https://img.shields.io/badge/HPC-71797E?style=flat&logo=dna&logoColor=white)
![Distributed Computing](https://img.shields.io/badge/Distributed%20Computing-E91E8C?style=flat&logo=apachehadoop&logoColor=white)
![Bioinformatics](https://img.shields.io/badge/Bioinformatics-228B22?style=flat&logo=dna&logoColor=white)


**DistPCA** is a distributed out-of-core C++ framework for tera-scale genomic Principal Component Analysis (PCA), designed to scale efficiently across both single- and multi-node computing systems. Built on top of **Message Passing Interface (MPI)**, it employs a hybrid multi-level parallelism scheme combining **multiprocessing**, **OpenMP multithreading**, **SIMD vectorization**, and **double buffering** across all three stages of the PCA pipeline (I/O, data preprocessing, numerical method). Evaluated on datasets reaching up to 11 TB, DistPCA achieves speedups of up to **58.2×** and over **98% reduction in wall-clock time**, while maintaining parallel efficiency above **82%** and preserving the accuracy of the recovered principal components (PCs). For a detailed description of the framework and experimental evaluation, please refer to our [preprint](https://www.biorxiv.org/content/10.64898/2026.05.15.725487v1).

## Table of Contents
- [Prerequisites & Installation](#prerequisites--installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Performance Evaluation](#performance-evaluation)
- [Reproducibility](#reproducibility)
- [File Structure](#file-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Prerequisites & Installation

Clone the repository:
```bash
git clone https://github.com/CEID-HPCLAB/DistPCA.git
cd DistPCA
```

Install **Intel MKL** (Base Toolkit) and **Intel MPI + OpenMP** (HPC Toolkit), which provides the `mpicxx` and `mpicc` wrappers:
```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit intel-hpckit
```

Then initialize the environment and build:
```bash
source /opt/intel/oneapi/setvars.sh
make        # compile
make clean  # remove build artifacts (if you want to re-build)
```

The executable will be available at `build/DistPCA.exe`.

## Usage

> [!IMPORTANT]
> Before running `DistPCA`, make sure the Intel oneAPI environment is initialized with `source /opt/intel/oneapi/setvars.sh`

Set the number of **OpenMP threads** to be used per MPI process:
```bash
export OMP_NUM_THREADS=<num_threads>
```

Once compiled, run `DistPCA` from the build directory:
```bash
mpirun -np <num_processes> ./build/DistPCA.exe \
  -bfile <file_path> \
  -nsv <nsv> \
  -nrhs <nrhs> \
  -power <num_power_iterations> \
  -crit <convergence_criterion> \
  -toll <convergence_tolerance> \
  -bsize <block_size> \
  -miter <max_iterations> \
  -verbose <verbose> \
  -fwrite <save_output> \
  -fullSVD <full_svd>
```

| Parameter   | Description |
|------------|-------------|
| `-np` | Number of MPI processes (**Mandatory**) |
| `-bfile` | Path to the input `.bed` dataset file (e.g., `../example/ToyHapmap`) (**Mandatory**) |
| `-nsv` | Number of sought principal components (Default: `10`) |
| `-nrhs` | Dimension of the target subspace (Default: `2 * nsv`) |
| `-power` | Number of power iterations to perform (Default: `1`) |
| `-crit` | Convergence criterion (Default: `2`) |
| `-toll` | Convergence tolerance (Default: `1e-3`) |
| `-bsize` | Total number of SNPs per block (Default: `100`) |
| `-miter` | Maximum iterations to run if convergence criterion is taking longer to achieve (Default: `100`) |
| `-verbose` | Logging level. If set to `2`, detailed convergence info is printed (Default: `1`) |
| `-fwrite` | Boolean flag. If set to `1`, stores the singular values and singular vectors (Default: `0`) |
| `-fullSVD` | Boolean flag. If set to `1`, computes full SVD using `LAPACKE` (only if dataset fits in RAM) (Default: `0`) |

> [!NOTE]
> `DistPCA` supports three convergence criteria. The first is the trace-based criterion, which monitors the relative change of the total explained variance (trace) between successive iterations. The second is the individual eigenvalue criterion, which checks the relative change of each singular value and requires all components to satisfy the specified tolerance. The third is the Mean Explained Variance (MEV) criterion, which assesses subspace convergence by measuring the average squared cosine similarity between successive eigenvector estimates. By default, the MEV criterion is used.

> [!NOTE]
> `DistPCA` supports three MPI-based parallelism schemes for computing the sought PCs. The first scheme, implemented in [SubspaceIteration_MPI](https://github.com/CEID-HPCLAB/DistPCA/blob/main/src/methods.cpp#L26), is an in-core method used when each MPI process can fully load its assigned portion of the dataset into RAM. The second scheme supports out-of-core computation of PCs and uses three levels of parallelism (multiprocessing, OpenMP multithreading, and SIMD vectorization). It is accessible through [BlockSubspaceIter_MPI_OOC](https://github.com/CEID-HPCLAB/DistPCA/blob/main/src/methods.cpp#L331). The third scheme is the full DistPCA implementation presented in the paper and extends the second scheme by additionally supporting compute–transfer overlap using a double-buffering strategy. It is implemented in [BlockSubspaceIter_MPI_OOC_double_buffering](https://github.com/CEID-HPCLAB/DistPCA/blob/main/src/methods.cpp#L760).

## Datasets

The datasets used in this research consist of three real-world and three synthetic datasets. Real-world datasets require preprocessing with [PLINK](https://www.cog-genomics.org/plink/), which can be installed as follows:

```bash
wget https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip
unzip plink_linux_x86_64_20231211.zip
sudo mv plink /usr/local/bin/
```

---

### Real-World Datasets

The three real-world datasets used in this work are the [1000 Genomes Project](https://www.nature.com/articles/nature15393), the [Simons Genome Diversity Project (SGDP)](https://www.nature.com/articles/nature18964), and the [Human Genome Diversity Project (HGDP)](https://pubmed.ncbi.nlm.nih.gov/15803201/). Their dimensions after preprocessing are summarized below, together with the commands required for downloading and preprocessing each dataset.


| Dataset | Individuals | SNPs |
|:-------:|:-----------:|:-----:|
| 1000 Genomes | 2,490 | 1,664,505 |
| SGDP | 345 | 694,659 |
| HGDP | 942 | 133,594 |

<br>


| **1000 Genomes** |
```bash
# Download from figshare
wget https://figshare.com/articles/dataset/1000_genomes_phase_3_files_with_SNPs_in_common_with_HapMap3/9208979
unzip 1000G_phase3_common_SNPs.zip
unzip 1000G_phase3_common_norel.zip

# Population ancestry panel (for coloring)
wget https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel

# Preprocessing
plink --bfile 1000G_phase3_common_norel --maf 0.01 --make-bed --out 1000G.qc
plink --bfile 1000G.qc --indep-pairwise 1000 50 0.2 --out 1000G.qc.prune
plink --bfile 1000G.qc --extract 1000G.qc.prune.prune.in --make-bed --out 1000G.qc.pruned
```

| **Simons Genome Diversity Project (SGDP)** |
```bash
# Download from Reich Lab
wget https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/variant_set/cteam_extended.v4.maf0.1perc.bed
wget https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/variant_set/cteam_extended.v4.maf0.1perc.bim.zip
unzip cteam_extended.v4.maf0.1perc.bim.zip
wget https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/variant_set/cteam_extended.v4.maf0.1perc.fam

# Preprocessing
plink --bfile sgdp --maf 0.01 --make-bed --out sgdp.qc
plink --bfile sgdp.qc --indep-pairwise 1000 50 0.2 --out sgdp.qc.prune
plink --bfile sgdp.qc --extract sgdp.qc.prune.prune.in --make-bed --out sgdp.qc.pruned
```

| **Human Genome Diversity Project (HGDP)** |
```bash
# Download from Reich Lab
wget -c https://reichdata.hms.harvard.edu/pub/datasets/humanOrigins/Harvard_HGDP-CEPH.tgz
tar -xvzf Harvard_HGDP-CEPH.tgz

# Preprocessing
plink --file all_snp --make-bed --out hgdp
plink --bfile hgdp --maf 0.01 --make-bed --out hgdp.qc
plink --bfile hgdp.qc --indep-pairwise 1000 50 0.2 --out hgdp.qc.prune
plink --bfile hgdp.qc --extract hgdp.qc.prune.prune.in --make-bed --out hgdp.qc.pruned
```

---

### Synthetic Datasets

The three synthetic datasets used in this work are:

| Dataset | Individuals | SNPs |
|:-------:|:-----------:|:-----:|
| 50K Genomes | 50,000 | 6,000,000 |
| 500K Genomes | 500,000 | 3,000,000 |
| 1M Genomes | 1,000,000 | 1,000,000 |

<br>

Synthetic datasets are generated using [DataSimulator](https://github.com/eugeniamaria/DataSimulator). Install dependencies and build:

```bash
sudo apt-get install libboost-all-dev libgsl-dev
git clone https://github.com/eugeniamaria/DataSimulator.git
cd DataSimulator && make
```

Run:
```bash
./GeneticDataSimulator -npop [int] -nregions [int] -nindividuals [int] -nSNP [int] -minfreq [double] -txtoutput [int] -filename [char]
```

Two output files are generated: `output_file.map` (SNP information) and `output_file.ped` (individual genotypes).

> [!WARNING]
> Generating large datasets (e.g. 1M Individuals × 1M SNPs) requires significant disk space. It is recommended to generate data in parts and merge them into `.bed` format via PLINK, rather than producing a single large `.ped` file.

## Performance Evaluation

### Experimental Setup

All experiments were conducted on the [ARIS supercomputer](https://www.hpc.grnet.gr/en/), a national Greek HPC cluster facility, using four thin compute nodes. Each thin node is partitioned into eight Non-Uniform Memory Access (NUMA) domains and is configured as follows:

| Component| Details |
|:---:|:---:|
| **CPU** | Dual-socket AMD EPYC 7742 (128 cores, 2.25 GHz) |
| **RAM** | 512 GB (restricted to 64 GB per node for all experiments) |
| **Filesystem** | GPFS |

A detailed overview of the ARIS infrastructure is available [here](https://doc.aris.grnet.gr/system/hardware/).

> [!NOTE]
> MPI ranks were distributed across NUMA domains, with OpenMP threads pinned to cores within each domain and fixed to **8** per rank throughout all experiments. Hyperthreading was disabled and MKL routines were accessed via `Intel oneAPI (v2025.0.1)`.

### Results

DistPCA demonstrates near-linear scalability, achieving speedups of up to **58.2×** and over **98% reduction in wall-clock time**, while maintaining parallel efficiency above **82%** across all evaluated scenarios. As shown in the figures, the *SGDP* and *HGDP* datasets are omitted, as they complete in under 5 seconds even with a single MPI rank.

<br>
<p align="center">
  <picture>
    <source srcset = "docs/figures/Fig1_light.png" media = "(prefers-color-scheme: dark)">
    <source srcset = "docs/figures/Fig1.png" media = "(prefers-color-scheme: light)">
    <img src = "docs/figures/Fig1_light.png" width = "95%" alt = "Runtime Performance of DistPCA across four distinct datasets">
  </picture>
  <br>
  <em>Figure 1: Runtime performance</em>
</p>

<br>
<p align="center">
  <picture>
    <source srcset = "docs/figures/Fig2_light.png" media = "(prefers-color-scheme: dark)">
    <source srcset = "docs/figures/Fig2.png" media = "(prefers-color-scheme: light)">
    <img src = "docs/figures/Fig2_light.png" width = "90%" alt = "Strong scaling speedup (left) and parallel efficiency (right)">
  </picture>
  <br>
  <em>Figure 2: Strong scaling speedup (left) and parallel efficiency (right)</em>
</p>


These performance gains are achieved while preserving the accuracy of the recovered PCs, as shown in the following figures.

<p align="center">
  <picture>
    <source srcset = "docs/figures/Fig3_light.png" media = "(prefers-color-scheme: dark)">
    <source srcset = "docs/figures/Fig3.png" media = "(prefers-color-scheme: light)">
    <img src = "docs/figures/Fig3_light.png" width = "90%" alt = "Strong scaling speedup (left) and parallel efficiency (right)">
  </picture>
  <br>
  <em>Figure 3<br><b>Left</b>: Entry-wise relative error of the 10 leading eigenvectors computed by DistPCA for the <b>1000 Genomes</b> dataset, compared to the eigenvectors returned by the full-rank SVD<br><b>Right</b>: Projection of the samples of the <b>1000 Genomes</b> dataset on the top two left singular vectors, as computed by DistPCA. Samples are grouped into five populations: AFR, AMR, EAS, EUR, and SAS</em>
</p>

<br>

As shown in the following table, *DistPCA* consistently outperforms *PCAone* [[1](https://genome.cshlp.org/content/early/2023/10/05/gr277525122), [2](https://github.com/Zilong-Li/PCAone)], the current state-of-the-art method for large-scale genomic PCA, across all datasets.

| Dataset        | PCAone | DistPCA | Speedup | Reduction % |
|----------------|--------|----------|----------|--------------|
| 1000 Genomes   | 173s   | **47s**  | 3.68x    | 72.8%        |
| 50K Genomes    | 9.1h   | **7.8h** | 1.17x    | 14.1%        |
| 500K Genomes   | 12.1h  | **2.3h** | 5.26x    | 78.5%        |
| 1M Genomes     | 7.9h   | **2.6h** | 3.04x    | 67.9%        |

## Reproducibility

### Regenerating the figures

All precomputed results from the conducted experiments are available [here](./docs/results/). To regenerate the figures directly from these outputs, run:

```bash
cd scripts/plots/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 runtime.py          # Runtime performance (Figure 3 in the paper)
python3 speedup.py          # Strong scaling speedup (Figure 4 in the paper)
python3 par_efficiency.py   # Parallel efficiency
python3 rel_error.py        # Entry-wise relative error of eigenvectors (Figure 5 in the paper)
python3 pop_structure.py    # Population structure (PC1 vs PC2) (Figure 6 in the paper)
``` 

### Reproducing the reported results

To reproduce the reported runtime results from scratch, first follow the [Datasets](#datasets) section to download and preprocess the real-world datasets and generate the synthetic ones. Once ready, move the `.bed`, `.bim`, and `.fam` files for each dataset under `scripts/experiments/` and run:

```bash
# Move dataset files to scripts/experiments/
mv <dataset>.bed <dataset>.bim <dataset>.fam scripts/experiments/

cd scripts/experiments/

bash run_1000G.sh
bash run_50K.sh
bash run_500K.sh
bash run_1M.sh
```
> [!NOTE]
> After executing the above scripts, the wall-clock time results for each worker configuration will be stored in the `/docs/results/runtime` directory in separate `.txt` files, one per dataset. 
> 
> By running the corresponding Python plot scripts located under `scripts/plots/` ([runtime.py](./scripts/plots/runtime.py), [speedup.py](./scripts/plots/speedup.py)), Figures 3 and 4 of the paper can be generated.

> [!IMPORTANT]
> A key parameter of the proposed hybrid multi-level parallelism scheme is the block size used to partition the dataset, as this determines the number of SNPs processed by each MPI rank at a time and, thus, the parallel I/O performance. The optimal size depends on several factors, including dataset size, RAM, I/O bandwidth, Last-Level Cache (LLC), and the number of compute nodes. In practice, the block size parameter should be selected such that the I/O workload generated by the workers matches the storage system’s capabilities (e.g., bandwidth and latency) and maximizes the LLC hit ratio.

> [!WARNING]
> Experiments were conducted on the [ARIS supercomputer](https://www.hpc.grnet.gr/en/) using four thin compute nodes. Wall-clock time results may exhibit slight variations depending on cluster infrastructure, node availability, and storage system.

To reproduce the reported accuracy results from scratch, after downloading the *1000 Genomes* dataset and moving it to `scripts/experiments/`, run:

```bash
cd scripts/experiments/

bash run_1000G_accuracy.sh
```

> [!NOTE]
> After executing the above script, the eigenvalues and corresponding eigenvectors computed by *DistPCA* will be stored in the `/docs/results/accuracy` directory in separate `.txt` files. Eigenvalues and eigenvectors computed via full SVD using *LAPACKE* are also stored in the same directory in a separate file.
>
> By running the corresponding Python plotting scripts located under `scripts/plots/` ([rel_error.py](./scripts/plots/rel_error.py), [pop_structure.py](./scripts/plots/pop_structure.py)), Figures 5 and 6 of the paper can be generated.

> [!CAUTION]
> The execution of `run_1000G_accuracy.sh` includes the computation of PCs via full SVD, which requires the entire dataset to be loaded into main memory in uncompressed form.
>
> The *1000 Genomes* dataset in uncompressed format exceeds 32 GB; therefore, reproducing the accuracy results requires a compute node with at least 40 GB of available RAM.


## File Structure
```
DistPCA/
├── docs/
│   ├── figures/            # Generated figures for the paper
│   └── results/            # Precomputed experimental results
│       ├── runtime/        # Runtime performance results (Figures 3 and 4 of the paper)
│       └── accuracy/       # Evaluation results for computed PCs (Figures 5 and 6 of the paper)
│
├── scripts/
│   ├── plots/              # Scripts to reproduce all figures
│   └── experiments/        # Scripts for running all experiments
│
├── src/                    # Core implementation of DistPCA
├── example/                # Toy dataset for testing and demonstration
│
├── Makefile
```

## Citation

If you find DistPCA useful for your research, please cite:

```bibtex
@article{mermigkis2026distpca,
  title     = {DistPCA: Tera-Scale Genomic PCA via Out-of-Core Distributed Parallelism},
  author    = {Mermigkis, Georgios and Sofotasios, Argiris and Kontopoulou, Eugenia-Maria and Gallopoulos, Efstratios and Hadjidoukas, Panagiotis},
  journal   = {bioRxiv},
  year      = {2026},
  doi       = {10.64898/2026.05.15.725487},
  url       = {https://www.biorxiv.org/content/10.64898/2026.05.15.725487v1}
}
```

## Acknowledgments
This work was supported by computational time granted from the National Infrastructures for Research and Technology S.A. (GRNET S.A.) in the National HPC facility - ARIS - under project ID pa260203distpca.
