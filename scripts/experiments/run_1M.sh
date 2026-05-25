#!/bin/bash

source /opt/intel/oneapi/setvars.sh
export OMP_NUM_THREADS=8

NP=8
BFILE="1M_Genomes"
NSV=20
NRHS=40
CONV_CRIT=2
TOLL=1e-3
RFETCHED=100

mpirun -np ${NP} ../../build/DistPCA.exe \
  -bfile ${BFILE} \
  -nsv ${NSV} \
  -nrhs ${NRHS} \
  -blockPower_conv_crit ${CONV_CRIT} \
  -toll ${TOLL} \
  -rfetched ${RFETCHED}