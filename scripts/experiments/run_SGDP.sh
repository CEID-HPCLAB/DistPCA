#!/bin/bash

source /opt/intel/oneapi/setvars.sh
export OMP_NUM_THREADS=8

NP=8
BFILE="sgdp.qc.pruned"
NSV=20
NRHS=40
CONV_CRIT=2
TOL=1e-3
BLOCKSIZE=100

mpirun -np ${NP} ../../build/DistPCA.exe \
  -bfile ${BFILE} \
  -nsv ${NSV} \
  -nrhs ${NRHS} \
  -crit ${CONV_CRIT} \
  -tol ${TOL} \
  -bsize ${BLOCKSIZE}