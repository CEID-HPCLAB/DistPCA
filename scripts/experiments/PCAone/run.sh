#!/bin/bash

source /opt/intel/oneapi/setvars.sh --force

export OMP_NUM_THREADS=64
export OMP_PROC_BIND=true
export OMP_PLACES=cores

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd -- "${SCRIPT_DIR}/../../../" && pwd)"
DATA_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

EXEC="${BASE_DIR}/PCAone/PCAone"

if [ $# -eq 0 ]; then
    DATASETS=("1000" "50K" "500K" "1M")
elif [ $# -eq 1 ]; then
    DATASETS=("$1")
else
    echo "Usage: $0 [dataset]"
    echo "Example: $0 1000"
    echo "         $0 50K"
    echo "         $0 500K"
    echo "         $0 1M"
    echo "         $0       # run all datasets"
    exit 1
fi

PC=20
METHOD=2
TOL=1e-3
VERBOSE=0
BLOCKSIZE=100

FIRST=true

for DATASET in "${DATASETS[@]}"
do

    case "$DATASET" in
        1000)
            DATASET_NAME="1000G.qc.pruned"
            ;;
        50K)
            DATASET_NAME="50K_Genomes"
            ;;
        500K)
            DATASET_NAME="500K_Genomes"
            ;;
        1M)
            DATASET_NAME="1M_Genomes"
            ;;
        *)
            echo "ERROR: Unknown dataset '${DATASET}'"
            echo "Available datasets: 1000, 50K, 500K, 1M"
            exit 1
            ;;
    esac

    BFILE="${DATA_DIR}/${DATASET_NAME}"

    if [ ! -e "${BFILE}.bed" ]; then
        echo "ERROR: Dataset '${DATASET_NAME}' not found"
        echo "Please retrieve or generate the dataset and place it in scripts/experiments/ before running this script"
        exit 1
    fi

    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo
    fi

    NINDIV=$(awk 'END {print NR}' "${BFILE}.fam")
    NSNPS=$(awk 'END {print NR}' "${BFILE}.bim")

    L=$((PC + PC))

    MEM=$(LC_NUMERIC=C awk \
        -v n="$NINDIV" \
        -v s="$NSNPS" \
        -v l="$L" \
        -v b="$BLOCKSIZE" \
        'BEGIN {
            memory = (b*n + 3*n*l + 2*s*l + 5*s) / 134217728
            printf "%.6f", memory
        }')

    echo "Running PCAone for dataset ${DATASET} Genomes with ${OMP_NUM_THREADS} OpenMP threads"

    "${EXEC}" \
        --bfile "${BFILE}" \
        --threads "${OMP_NUM_THREADS}" \
        --pc "${PC}" \
        --svd "${METHOD}" \
        --memory "${MEM}" \
        --tol-rsvd "${TOL}" \
        --verbose "${VERBOSE}" \
        -o "pcaone_${DATASET}_genomes"
    
    LOGFILE="pcaone_${DATASET}_genomes.log"

    TIME=$(grep "total elapsed wall time:" "${LOGFILE}" | head -n 1 | awk '{print $6}')

    echo "Time: ${TIME} seconds | OpenMP Threads: ${OMP_NUM_THREADS}"

    rm -f "pcaone_${DATASET}_genomes".*    

done