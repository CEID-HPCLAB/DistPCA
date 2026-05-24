#!/bin/bash
source /opt/intel/oneapi/setvars.sh
export OMP_NUM_THREADS=8

mpirun -np 8 ../build/TeraPCA_MPI.exe \
  -bfile 500K_Genomes \
  -nsv 20 \
  -nrhs 40 \
  -blockPower_conv_crit 2 \
  -toll 1e-3 \
  -rfetched 100