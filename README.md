# 🧬 DistPCA: Tera-Scale Genomic PCA via Out-of-Core Distributed Parallelism

![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white)
![Distributed Computing](https://img.shields.io/badge/Distributed%20Computing-E91E8C?style=flat&logo=apachehadoop&logoColor=white)

**DistPCA** is a distributed C++ framework for tera-scale genomic Principal Component Analysis (PCA), designed to scale across both single- and multi-node HPC clusters. It employs a hybrid multi-level parallelism scheme combining **MPI**, **OpenMP**, **SIMD** vectorization, and **double buffering** across all three stages of the PCA pipeline (I/O, Preprocessing, Numerical Method). Evaluated on datasets reaching up to 11 TB, DistPCA achieves speedups of up to **58.2×** and over **98% reduction in wall-clock time**, while maintaining parallel efficiency above 82% and preserving the accuracy of the recovered principal components. For a detailed description of the framework and experimental evaluation, please refer to our [preprint](https://www.biorxiv.org/content/10.64898/2026.05.15.725487v1).

## Table of Contents
- [Prerequisites & Installation](#️-prerequisites--installation)
- [Usage](#-usage)
- [Datasets](#️-datasets)
- [Results](#-results)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)

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

The executable will be available at `build/TeraPCA_MPI.exe`.

## Usage

> [!NOTE]
> Before running, make sure the Intel oneAPI environment is initialized with `source /opt/intel/oneapi/setvars.sh`

Set the number of OpenMP threads:
```bash
export OMP_NUM_THREADS=8
```

Run DistPCA from the build directory:
```bash
mpirun -np <num_processes> ./build/TeraPCA_MPI.exe \
  -bfile <file_path> \
  -nsv 10 \
  -nrhs 20 \
  -blockPower_conv_crit 2 \
  -toll 1e-3 \
  -rfetched 5
```

| Parameter | Description |
|---|---|
| `num_processes` | Number of MPI processes |
| `file_path` | Path to the `.bed` file (e.g., `../example/ToyHapmap`) |
| `-nsv` | Number of principal components |
| `-nrhs` | Size of the subspace (default: `2 * nsv`) |
| `-blockPower_conv_crit` | Convergence criterion (`2` = MEV) |
| `-toll` | Convergence tolerance |
| `-rfetched` | Block size (number of SNPs per block) |

## Datasets

The datasets used in this research consist of three real-world and three synthetic datasets. Real-world datasets require preprocessing with [PLINK](https://www.cog-genomics.org/plink/), which can be installed as follows:

```bash
wget https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip
unzip plink_linux_x86_64_20231211.zip
sudo mv plink /usr/local/bin/
```

---

### Real-World Datasets

**1000 Genomes**
```bash
# Download from figshare
wget https://figshare.com/articles/dataset/1000_genomes_phase_3_files_with_SNPs_in_common_with_HapMap3/9208979
unzip 1000G_phase3_common_SNPs.zip
unzip 1000G_phase3_common_norel.zip

# Population ancestry panel (for coloring)
wget https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel

# Preprocess
plink --bfile 1000G_phase3_common_norel --maf 0.01 --make-bed --out 1000G.qc
plink --bfile 1000G.qc --indep-pairwise 1000 50 0.2 --out 1000G.qc.prune
plink --bfile 1000G.qc --extract 1000G.qc.prune.prune.in --make-bed --out 1000G.qc.pruned
```

**Simons Genome Diversity Project (SGDP)**
```bash
# Download from Reich Lab
wget https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/variant_set/cteam_extended.v4.maf0.1perc.bed
wget https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/variant_set/cteam_extended.v4.maf0.1perc.bim.zip
unzip cteam_extended.v4.maf0.1perc.bim.zip
wget https://sharehost.hms.harvard.edu/genetics/reich_lab/sgdp/variant_set/cteam_extended.v4.maf0.1perc.fam

# Preprocess
plink --bfile sgdp --maf 0.01 --make-bed --out sgdp.qc
plink --bfile sgdp.qc --indep-pairwise 1000 50 0.2 --out sgdp.qc.prune
plink --bfile sgdp.qc --extract sgdp.qc.prune.prune.in --make-bed --out sgdp.qc.pruned
```

**Human Genome Diversity Project (HGDP)**
```bash
# Download from Reich Lab
wget -c https://reichdata.hms.harvard.edu/pub/datasets/humanOrigins/Harvard_HGDP-CEPH.tgz
tar -xvzf Harvard_HGDP-CEPH.tgz

# Preprocess
plink --file all_snp --make-bed --out hgdp
plink --bfile hgdp --maf 0.01 --make-bed --out hgdp.qc
plink --bfile hgdp.qc --indep-pairwise 1000 50 0.2 --out hgdp.qc.prune
plink --bfile hgdp.qc --extract hgdp.qc.prune.prune.in --make-bed --out hgdp.qc.pruned
```

---

### Synthetic Datasets

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

## Results

All experiments were conducted on the **ARIS supercomputer**, a national Greek HPC cluster facility, using four thin compute nodes. Each thin node is configured as follows:

| Spec | Details |
|---|---|
| **CPU** | Dual-socket AMD EPYC 7742 (128 cores, 2.25 GHz) |
| **RAM** | 512 GB (restricted to 64 GB per node for all experiments) |
| **Storage** | IBM GPFS (high-performance parallel filesystem) |

MPI ranks were distributed across NUMA domains, with OpenMP threads pinned to cores within each domain and fixed to 8 per rank throughout all experiments. Hyperthreading was disabled and MKL routines were accessed via Intel oneAPI (v2025.0.1).

DistPCA achieves speedups of up to **58.2×** and over **98% reduction in wall-clock time**. SGDP and HGDP are omitted as both datasets complete in under 5 seconds even with a single MPI rank.

**Wall-Clock Time**

<!-- <img src="docs/figures/Fig1.png"/> -->
<p align="center">
  <picture>
    <source srcset = "docs/figures/Fig1.png" media = "(prefers-color-scheme: dark)">
    <source srcset = "docs/figures/Fig1_light.png" media = "(prefers-color-scheme: light)">
    <img src = "docs/figures/Fig1.png" width = "100%" alt = "Runtime performance across datasets">
  </picture>
</p>

**Strong Scaling Speedup**

<!-- <img src="docs/figures/Fig2.png" width="480" height="288"/> -->
<p align="center">
  <picture>
    <source srcset = "docs/figures/Fig2.png" media = "(prefers-color-scheme: dark)">
    <source srcset = "docs/figures/Fig2_light.png" media = "(prefers-color-scheme: light)">
    <img src = "docs/figures/Fig2.png" width = "100%" alt = "Speedup across datasets">
  </picture>
</p>

These speedups are achieved while preserving the accuracy in the recovered principal components as depicted in the following plots — **left**: entry-wise relative error of the 10 leading eigenvectors against full-rank SVD on the 1000 Genomes dataset, **right**: projection of individuals on the top two principal components (PC1, PC2) colored by population (AFR, AMR, EAS, EUR, SAS), demonstrating clear population clustering consistent with known stratification patterns.

<p align="center">
  <img src="docs/figures/Fig3.png" width="45%"/>
  &nbsp;&nbsp;
  <img src="docs/figures/Fig4.png" width="45%"/>
</p>

## Reproducibility

TBD

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