#!/bin/bash

source /opt/intel/oneapi/setvars.sh
export OMP_NUM_THREADS=8

BFILE="1000G.qc.pruned"
NSV=20
NRHS=40
CONV_CRIT=2
TOLL=1e-3
BLOCKSIZE=100

NUM_MPI_RANKS=(1 2 4 8 12 16 24 32 48 64)

count=0
total=${#NUM_MPI_RANKS[@]}

LOGFILE="1000_genomes.log"
RESULTS_FILE="../../docs/results/runtime/1000_genomes.txt"

> ${RESULTS_FILE}

for NP in "${NUM_MPI_RANKS[@]}"
do
    count=$((count + 1))

    echo "-----------------------------------------"
    echo "Running DistPCA with ${NP} MPI processes"
    echo "-----------------------------------------"
    echo ""

    mpirun -np ${NP} ../../build/DistPCA.exe \
      -bfile ${BFILE} \
      -nsv ${NSV} \
      -nrhs ${NRHS} \
      -crit ${CONV_CRIT} \
      -toll ${TOLL} \
      -bsize ${BLOCKSIZE} | tee ${LOGFILE}

    TIME=$(grep "Total wall-clock time elapsed:" ${LOGFILE} | head -n 1 | awk '{print $5}')

    echo "MPI processes: ${NP} | Time: ${TIME} seconds" | tee -a ${RESULTS_FILE}

    if [ $count -lt $total ]; then
        echo ""
    fi

    rm -f ${LOGFILE}
done