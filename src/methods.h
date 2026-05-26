/*
 * =============================================================================
 * DistPCA — CEID-HPCLAB Extended Fork of TeraPCA
 * =============================================================================
 * This file is part of DistPCA, a high-performance extension of the official
 * TeraPCA codebase, developed by the High Performance Computing Laboratory
 * at the Computer Engineering & Informatics Department (CEID), University
 * of Patras, Greece.
 *
 * Original TeraPCA: https://github.com/aritra90/TeraPCA
 * CEID-HPCLAB Repo: https://github.com/CEID-HPCLAB/DistPCA
 *
 * Authors (CEID-HPCLAB):  Georgios Mermigkis, Argiris Sofotasios, Eugenia-Maria
 *                         Kontopoulou, Efstratios Gallopoulos, Panagiotis
 *                         Hadjidoukas.
 * License:                MIT - See LICENSE file in the project root
 * =============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "structures.h"

#if defined(__x86_64__) || defined(_M_X64)
    #include "mkl.h"
    #include "mkl_lapacke.h"
#elif defined(__aarch64__) || defined(__arm__) || defined(__ARM_ARCH) || defined(arm64)
    #include <cblas.h>
    #include <lapacke.h>
    #include <chrono>

    static double dsecnd() {
        static auto start_time = std::chrono::high_resolution_clock::now();
        auto current_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(current_time - start_time).count();
    }

#else
    #error "Unsupported architecture: please define BLAS/LAPACK backend for this platform."
#endif


#define min(a,b) (a<=b?a:b)
#define max(a,b) (a>=b?a:b)

void SubspaceIteration_MPI(double *MAT, double *RHS2, logistics *logg);
void BlockSubspaceIter_MPI_OOC_double_buffering(const char* bedfile, double *RHS2, logistics *logg);
void BlockSubspaceIter_MPI_OOC(const char* bedfile, double *RHS2, logistics *logg);