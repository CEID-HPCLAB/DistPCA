#!/bin/bash

source /opt/intel/oneapi/setvars.sh
export OMP_NUM_THREADS=8

NP=8
BFILE="1000G.qc.pruned"
NSV=10
NRHS=20
CONV_CRIT=2
TOL=1e-3
BLOCKSIZE=100
SVD=1
LOG=1
PREFIX="1000_genomes"

TARGET_DIR="../../docs/results/accuracy"

mpirun -np ${NP} ../../build/DistPCA.exe \
  -bfile ${BFILE} \
  -nsv ${NSV} \
  -nrhs ${NRHS} \
  -crit ${CONV_CRIT} \
  -tol ${TOL} \
  -bsize ${BLOCKSIZE} \
  -fullSVD ${SVD} \
  -fwrite ${LOG} \
  -prefix ${PREFIX}

mv ${PREFIX}_leftSingularVectors.txt ${TARGET_DIR}/
mv ${PREFIX}_singularValues.txt ${TARGET_DIR}/
mv ${PREFIX}_realLeftSingularVectors.txt ${TARGET_DIR}/
mv ${PREFIX}_realSingularValues.txt ${TARGET_DIR}/
rm ${PREFIX}_singvecs_accuracy.txt
rm ${PREFIX}_cosineValues.txt