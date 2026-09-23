#!/bin/bash

source /opt/intel/oneapi/setvars.sh --force

export OMP_NUM_THREADS=2

BFILE="hgdp.qc.pruned"
NSV=20
NRHS=40
CONV_CRIT=2
TOL=1e-3
BLOCKSIZE=100

NUM_MPI_RANKS=(1 2 4)

count=0
total=${#NUM_MPI_RANKS[@]}

LOGFILE="hgdp.qc.pruned.log"

PHYSICAL_CORES=$(lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l)

for NP in "${NUM_MPI_RANKS[@]}"
do
    count=$((count + 1))

    REQUIRED_CORES=$((NP * OMP_NUM_THREADS))

    if (( REQUIRED_CORES > PHYSICAL_CORES )); then
        echo "-----------------------------------------"
        echo "MPI processes=${NP}, OpenMP threads/MPI process=${OMP_NUM_THREADS}: skipped (requires ${REQUIRED_CORES} cores, ${PHYSICAL_CORES} available)"
        echo "-----------------------------------------"
        echo ""
        continue
    fi

    echo "-----------------------------------------"
    echo -e "Running DistPCA with ${NP} MPI processes and ${OMP_NUM_THREADS} OpenMP threads per process\n"

    TEMP_LOG=$(mktemp)

    mpirun -np ${NP} ../build/DistPCA.exe \
    -bfile ${BFILE} \
    -nsv ${NSV} \
    -nrhs ${NRHS} \
    -crit ${CONV_CRIT} \
    -tol ${TOL} \
    -bsize ${BLOCKSIZE} > ${TEMP_LOG} 2>&1

    cat ${TEMP_LOG} >> ${LOGFILE}

    TIME=$(awk '
        /\[The following times are listed in seconds\]/ {
            in_seconds=1
        }
        in_seconds && /Total wall-clock time elapsed:/ {
            print $5
            exit
        }
    ' ${TEMP_LOG})

    echo "MPI processes: ${NP} | OpenMP threads per MPI process: ${OMP_NUM_THREADS} | Time: ${TIME} seconds"
    echo "-----------------------------------------"

    rm -f ${TEMP_LOG}

    if [ $count -lt $total ]; then
        echo ""
    fi

done