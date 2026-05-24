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
#include "methods.h"
#include "utilities.h"
#include <mpi.h>
#include <vector>

int save_eigenval_tracking = 0;

void subspaceIteration_MPI(double *MAT, double *RHS2, logistics *logg) {
  
  int M = logg->M, N = logg->N, local_N = logg->local_N;
  int NRHS = logg->NRHS;
  int rank = logg->mpi_rank, size = logg->mpi_size;
  int max_iter = logg->blockPower_maxiter;
  int powers = logg->power;
  int ii, jj, converged = 0, kk, ii2, jj2;
  double tt1, tt2;
  double fone = 1.0, fzero = 0.0;
  double w[NRHS];
  
  double* B    = (double*) malloc(NRHS*NRHS*sizeof(double));
  double* B2   = (double*) malloc(NRHS*NRHS*sizeof(double));
  double* RHS_local  = (double*) malloc(NRHS*local_N*sizeof(double)); // Local portion of A'*RHS2
  double* ARHS = (double*) malloc(NRHS*M*sizeof(double));
  double* ARHS_local = (double*) malloc(NRHS*M*sizeof(double));
  double* SING_VALUES     = (double*) malloc(logg->NSV*sizeof(double));
  double* SING_VALUES_OLD = (double*) malloc(logg->NSV*sizeof(double));
  double* tau = (double*) malloc(NRHS*sizeof(double));
  double* RHS2_old = (double*) malloc(M*NRHS*sizeof(double));  //needed for MVE Mode2
  
  int info_sgeqrf_lapacke, info_sorgqr_lapacke;
  logg->delta_iter.resize(max_iter);
  logg->sing_values.resize(logg->NSV);
  logg->left_sing_vecs.resize(M*logg->NSV);
  
  //needed for MVE - initialize RHS2_old
  if (logg->blockPower_conv_crit == 2) {
    memcpy(RHS2_old, RHS2, M*NRHS*sizeof(double));
  }
  //============================================================

  for (ii = 0; ii < max_iter; ii++) {
    //============================================================
    // Multiply local A' by RHS2 from the right
    // Each rank computes: RHS_local = MAT_local' * RHS2
    //============================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                local_N, NRHS, M, fone, MAT, M, RHS2, NRHS, fzero, RHS_local, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
    //============================================================

    //============================================================           
    // Multiply local A by RHS_local from the right
    // Each rank computes: ARHS_local = MAT_local * RHS_local
    // Then sum across all ranks with MPI_Allreduce
    //============================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                M, NRHS, local_N, fone, MAT, M, RHS_local, NRHS, fzero, ARHS_local, NRHS);
    
    // Sum contributions from all ranks
    MPI_Allreduce(ARHS_local, ARHS, M*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
    //============================================================
    
    //============================================================
    //                       (AA')^{powers-1}
    //============================================================
    for (kk = 0; kk < powers-1; kk++) {
      // Orthogonalize (all ranks have same ARHS after Allreduce)
      tt1 = dsecnd();
      info_sgeqrf_lapacke = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);
      info_sorgqr_lapacke = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
      tt2 = dsecnd() - tt1;
      logg->TIME_2_GS = logg->TIME_2_GS + tt2;
      
      memcpy(RHS2, ARHS, M*NRHS*sizeof(double));
      
      // Local A' * RHS2
      tt1 = dsecnd();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                  local_N, NRHS, M, fone, MAT, M, RHS2, NRHS, fzero, RHS_local, NRHS);
      tt2 = dsecnd() - tt1;
      logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;

      // Local A * RHS_local and Allreduce
      tt1 = dsecnd();
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                  M, NRHS, local_N, fone, MAT, M, RHS_local, NRHS, fzero, ARHS_local, NRHS);
      MPI_Allreduce(ARHS_local, ARHS, M*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      tt2 = dsecnd() - tt1;
      logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
    }
    //============================================================

    //============================================================                               
    // Perform orthogonalization                                                                           
    //============================================================
    tt1 = dsecnd();
    info_sgeqrf_lapacke = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);
    info_sorgqr_lapacke = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_GS = logg->TIME_2_GS + tt2;
    //============================================================
    
    //============================================================
    // Multiply local A' by ARHS
    //============================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                local_N, NRHS, M, fone, MAT, M, ARHS, NRHS, fzero, RHS_local, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
    //============================================================

    //============================================================
    // Compute RHS_local' * RHS_local locally, then Allreduce
    //============================================================
    tt1 = dsecnd();
    double* B_local = (double*) malloc(NRHS*NRHS*sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                NRHS, NRHS, local_N, fone, RHS_local, NRHS, RHS_local, NRHS, fzero, B_local, NRHS);
    
    // Sum contributions from all ranks
    MPI_Allreduce(B_local, B, NRHS*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    free(B_local);
    
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    //============================================================

    //============================================================
    // Solve projected eigenvalue problem (all ranks do this)
    //============================================================
    tt1 = dsecnd();
    for (ii2 = 0; ii2 < NRHS; ii2++) {
      for (jj2 = 0; jj2 < NRHS; jj2++) {
        if (jj2 < ii2) {
          B[ii2*NRHS + jj2] = 0.0;
        }
      }
    }
 
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NRHS, B, NRHS, w);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_PROJECTED_SVD = logg->TIME_2_PROJECTED_SVD + tt2;
    //===========================================================

    //===========================================================
    // Reverse order of eigenvectors
    //===========================================================
    memcpy(B2, B, sizeof(double)*NRHS*NRHS);
    for (ii2 = 0; ii2 < NRHS; ii2++) {
      for (jj2 = 0; jj2 < NRHS; jj2++) {
        B[ii2*NRHS+jj2] = B2[ii2*NRHS+(NRHS-1-jj2)];
      }
    }
    //===========================================================

    //===========================================================
    // Multiply ARHS by eigenvectors
    //===========================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, NRHS, NRHS, fone, ARHS, NRHS, B, NRHS, fzero, RHS2, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    //==========================================================

    //==========================================================
    // Copy singular values and monitor convergence
    //==========================================================
    logg->delta_iter[ii] = 0.0;
    for (jj2 = 0; jj2 < logg->NSV; jj2++) {
      SING_VALUES[jj2] = sqrt(w[NRHS-1-jj2]);
      logg->sing_values[jj2] = SING_VALUES[jj2];
      logg->delta_iter[ii] = logg->delta_iter[ii] + logg->sing_values[jj2];
    }
    //==========================================================

    //==========================================================
    // Check convergence with three modes
    //==========================================================
    if (logg->blockPower_conv_crit == 0) {
      // MODE 0: Trace criterion
      if (ii > 0) {
        logg->blockPower_trace_error = fabs(logg->delta_iter[ii-1]-logg->delta_iter[ii])/logg->delta_iter[ii];
        if (rank == 0 && logg->PRINT_INFO > 1) 
          printf("Iteration %d: rel. error: %02.13f\n", ii, logg->blockPower_trace_error);
        if (logg->blockPower_trace_error <= logg->toll) {
          break;
        }
      }
      
    } else if (logg->blockPower_conv_crit == 1) {
      // MODE 1: Individual eigenvalue criterion
      if (ii > 0) {
        converged = 0;
        for (jj = 0; jj < logg->NSV; jj++) {
          if (fabs((SING_VALUES[jj]-SING_VALUES_OLD[jj])/SING_VALUES[jj]) <= logg->toll) {
            converged++;
          }
        }
        if (rank == 0 && logg->PRINT_INFO > 1)
          printf("Iteration %d: %d converged\n", ii, converged);
        if (converged == logg->NSV) {
          break;
        }
      }
      
    } else if (logg->blockPower_conv_crit == 2) {
      // MODE 2: MEV (Mean Explained Variance) criterion
      if (ii > 0) {
        double MEV = 0.0;
        
        // Compute mean squared cosine similarity between old and new eigenvectors
        for (jj = 0; jj < logg->NSV; jj++) {
          double dot_product = 0.0;
          for (ii2 = 0; ii2 < M; ii2++) {
            dot_product += RHS2_old[ii2*NRHS + jj] * RHS2[ii2*NRHS + jj];
          }
          MEV += dot_product * dot_product;  // Square of cosine similarity
        }
        MEV /= logg->NSV;  // Average over all principal components
        
        double mev_error = 1.0 - MEV;
        
        if (rank == 0 && logg->PRINT_INFO > 1) {
          printf("Iteration %d: MEV error: %02.13f\n", ii, mev_error);
        }
        
        if (mev_error <= logg->toll) {
          break;
        }
      }
    }

    // Store values for next iteration
    converged = 0;
    for (jj = 0; jj < logg->NSV; jj++) {
      SING_VALUES_OLD[jj] = SING_VALUES[jj];
    }
    
    //needed for MVE - store current eigenvectors for Mode2
    if (logg->blockPower_conv_crit == 2) {
      memcpy(RHS2_old, RHS2, M*NRHS*sizeof(double));
    }
    //========================================================== 
  }
  //============================================================

  logg->blockPower_total_its = (ii < max_iter) ? ii+1 : ii;
  
  // Copy left singular vectors
  for (ii = 0; ii < M; ii++) {
    for (jj = 0; jj < NRHS; jj++) {
      if (jj < logg->NSV) {
        logg->left_sing_vecs[ii*logg->NSV+jj] = RHS2[ii*NRHS+jj];
      }
    }
  }

  //==========================================================
  // Write results (rank 0 only)
  //==========================================================
  if (rank == 0 && logg->filewrite == 1) {
    string tempname;
    if (logg->prefixname.empty())
      tempname = ConstructFilename(*logg, "singularValues");
    else
      tempname = logg->prefixname + "_singularValues.txt";

    FILE *fwrite_singvalues = fopen(tempname.c_str(), "w");
    if (fwrite_singvalues == NULL) {
      printf("Unable to write to file. Aborting...");
      MPI_Finalize();
      exit(1);
    }

    std::vector<double> singularvals = logg->sing_values;
    std::transform(singularvals.begin(), singularvals.end(), singularvals.begin(), computeSquare);
    std::vector<string> individs = logg->indiv_ids;

    fprintf(fwrite_singvalues, "EIGENVALUES\n\n");
    for (ii = 0; ii < logg->NSV; ii++)
      fprintf(fwrite_singvalues, "%2.13lf\n", singularvals[ii]);
    fclose(fwrite_singvalues);

    if (logg->prefixname.empty())
      tempname = ConstructFilename(*logg, "singularVectors");
    else
      tempname = logg->prefixname + "_singularVectors.txt";
      
    FILE *fwrite_singvecs = fopen(tempname.c_str(), "w");
    if (fwrite_singvecs == NULL) {
      printf("Unable to write to file. Aborting...");
      MPI_Finalize();
      exit(1);
    }
    
    fprintf(fwrite_singvecs, "FID");
    for (jj = 0; jj < logg->NSV; jj++)
      fprintf(fwrite_singvecs, "\tPC%d", jj);
    fprintf(fwrite_singvecs, "\n");
    
    for (ii = 0; ii < M; ii++) {
      fprintf(fwrite_singvecs, "%8s", individs[ii].c_str());
      for (jj = 0; jj < logg->NSV; jj++)
        fprintf(fwrite_singvecs, "\t%2.13f", RHS2[ii*NRHS+jj]);
      fprintf(fwrite_singvecs, "\n");
    }
    fclose(fwrite_singvecs);
  }
  //==========================================================

  // Cleanup
  free(ARHS);
  free(ARHS_local);
  free(RHS_local);
  free(B2);
  free(B);
  free(SING_VALUES);
  free(SING_VALUES_OLD);
  free(RHS2_old);  // NEW: Free Mode 2 memory
  free(tau);
}

void BlockSubspaceIter_MPI_IO(const char* bedfile, double *RHS2, logistics *logg) {
    int M = logg->M, N = logg->N, NRHS = logg->NRHS;
    int local_N = logg->local_N;
    int local_N_start = logg->local_N_start;
    int max_iter = logg->blockPower_maxiter, min_dim = min(N,NRHS), powers = logg->power;
    double tt1, tt2;
    int converged = 0, ii, jj, kk, ii2, jj2;
    double fone = 1.0, fzero = 0.0;
    
    int rank = logg->mpi_rank;

    MPI_File fh;
    int mpi_err = MPI_File_open(MPI_COMM_WORLD, bedfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (mpi_err != MPI_SUCCESS) {
        if (rank == 0) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length;
            MPI_Error_string(mpi_err, error_string, &length);
            fprintf(stderr, "Error opening file %s: %s\n", bedfile, error_string);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double* ARHS            = (double*) malloc(M * NRHS * sizeof(double));
    double* ARHS_local      = (double*) malloc(M * NRHS * sizeof(double));
    double* SING_VALUES     = (double*) malloc(min_dim * sizeof(double));
    double* SING_VALUES_OLD = (double*) malloc(min_dim * sizeof(double));
    double* tau             = (double*) malloc(NRHS * sizeof(double));
    double* RHS2_old        = (double*) malloc(M * NRHS * sizeof(double));  //needed for MVE Mode2
    
    logg->sing_values.resize(logg->NSV);
    logg->delta_iter.resize(max_iter);
    logg->left_sing_vecs.resize(M * logg->NSV);

    int rows_fetched = logg->rows_fetched;
    int local_loops = local_N / rows_fetched;
    int remaining_rows = local_N - rows_fetched * local_loops;

    double inv_sqrtN = 1.0 / sqrt((double)N);
    double inv_sqrtM = 1.0 / sqrt((double)M);
    
    if (remaining_rows > 0)
      local_loops++;
    else
      remaining_rows = rows_fetched;

    std::vector<int> start(local_loops);
    std::vector<int> stop(local_loops);
    
    for(int ik = 0; ik < local_loops; ik++){
        start[ik] = local_N_start + ik * rows_fetched;
        stop[ik] = start[ik] + rows_fetched - 1;
        if (stop[ik] >= local_N_start + local_N) 
            stop[ik] = local_N_start + local_N - 1;
    }

    uint64_t np = (uint64_t)ceil((double)M / PACK_DENSITY);

    double* RHS = (double*) malloc(rows_fetched * NRHS * sizeof(double));
    double* B2 = (double*) malloc(NRHS * NRHS * sizeof(double));
    double* B2_duplicate = (double*) malloc(NRHS * NRHS * sizeof(double));
    
    uint64_t lmsize = (uint64_t)rows_fetched* (uint64_t)M;
    double* LOC_MAT = (double*)malloc(lmsize * sizeof(double));
    
    unsigned char *decbin = (unsigned char*)malloc(np * PACK_DENSITY * sizeof(unsigned char));
    unsigned char *readbin = (unsigned char*)malloc(np * sizeof(unsigned char));
    double *norm_tmp = (double*)malloc(M * sizeof(double));
    double *norm_precomp = (double*)calloc(4 * N, sizeof(double));
    bool *seen_snp = new bool[N]();


    auto _compute_matvec = [&]() {
        memset(ARHS_local, 0, M * NRHS * sizeof(double));
        
        for (jj = 0; jj < local_loops; jj++) {
            uint64_t actual_block_size = (jj == local_loops - 1) ? remaining_rows : (stop[jj] - start[jj] + 1);
            uint64_t startval = start[jj];
            uint64_t file_offset = 3 + np * startval;

            std::vector<unsigned char> block_buf(np * actual_block_size);
            
            tt1 = dsecnd();
            MPI_Status status;
            MPI_File_read_at(fh, file_offset, block_buf.data(), 
                           np * actual_block_size, MPI_UNSIGNED_CHAR, &status);
            tt2 = dsecnd() - tt1;
            logg->TIME_2_LOAD_MATRIX += tt2;


            tt1 = dsecnd();
  
            for (uint64_t block_row = 0; block_row < actual_block_size; block_row++) {
                unsigned char* snp_bin = block_buf.data() + block_row * np;
                uint64_t global_idx = startval + block_row;
                
                if (!seen_snp[global_idx]) {
                    decode_plink(decbin, snp_bin, np);
                    double avg, sd;
                    standardize(norm_tmp, decbin, M, avg, sd, inv_sqrtN);
            
                    if(sd > 1e-9){
                        norm_precomp[3+(global_idx*4)] = (0 - avg)/sd;
                        norm_precomp[2+(global_idx*4)] = (1 - avg)/sd;
                        norm_precomp[0+(global_idx*4)] = (2 - avg)/sd;
                        norm_precomp[1+(global_idx*4)] = 0;
                    }
                    seen_snp[global_idx] = true;
                    
                    for(int k = 0; k < M; k++) 
                        LOC_MAT[block_row * M + k] = norm_tmp[k];
          
                } else {
                    decode_plink_precomp(decbin, snp_bin, np);
                    for(int k = 0; k < M; k++) {
                        int b = (int)decbin[k];
                        double s = norm_precomp[b+(global_idx*4)];
                        LOC_MAT[block_row * M + k] = s * inv_sqrtN;
                    }
                }
            }
            
            tt2 = dsecnd() - tt1;
            logg->TIME_2_LOAD_MATRIX += tt2;

            
            // Matrix multiplication A × RHS2
            tt1 = dsecnd();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        actual_block_size, NRHS, M, 
                        fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);
            tt2 = dsecnd() - tt1;
            logg->TIME_2_MM += tt2;
            logg->TIME_2_MM_A += tt2;
            
            // Matrix multiplication A' × RHS  
            tt1 = dsecnd();
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        M, NRHS, actual_block_size, 
                        fone, LOC_MAT, M, RHS, NRHS, fone, ARHS_local, NRHS);
            tt2 = dsecnd() - tt1;
            logg->TIME_2_MM += tt2;
            logg->TIME_2_MM_A_TRANSPOSED += tt2;
        }
        
        MPI_Allreduce(ARHS_local, ARHS, M * NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    };

    // NEW: Initialize RHS2_old for Mode 2
    if (logg->blockPower_conv_crit == 2) {
        memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));
    }

    //================================================================
    for (ii = 0; ii < max_iter; ii++) {
         _compute_matvec();
        
        for (kk = 0; kk < powers - 1; kk++) {
            // Orthogonalization
            tt1 = dsecnd();
            LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);
            LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
            tt2 = dsecnd() - tt1;
            logg->TIME_2_GS += tt2;
            
            memcpy(RHS2, ARHS, M * NRHS * sizeof(double));
            
            _compute_matvec();
        }

        // Final orthogonalization
        tt1 = dsecnd();
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_GS += tt2;

        memcpy(RHS2, ARHS, M * NRHS * sizeof(double));

        _compute_matvec();

        tt1 = dsecnd();
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    NRHS, NRHS, M, 
                    fone, RHS2, NRHS, ARHS, NRHS, fzero, B2, NRHS);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;

        tt1 = dsecnd();
        for (ii2 = 0; ii2 < NRHS; ii2++) {
            for (jj2 = 0; jj2 < ii2; jj2++) 
                B2[ii2 * NRHS + jj2] = 0.0;
        }
        double* w = (double*) malloc(NRHS * sizeof(double));
        LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NRHS, B2, NRHS, w);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_PROJECTED_SVD += tt2;

        memcpy(B2_duplicate, B2, NRHS * NRHS * sizeof(double));
        for (ii2 = 0; ii2 < NRHS; ii2++) {
            for (jj2 = 0; jj2 < NRHS; jj2++) {
                B2[ii2 * NRHS + jj2] = B2_duplicate[ii2 * NRHS + (NRHS - 1 - jj2)];
            }
        }
        
        memcpy(ARHS, RHS2, M * NRHS * sizeof(double));
        tt1 = dsecnd();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, NRHS, NRHS, 
                    fone, ARHS, NRHS, B2, NRHS, fzero, RHS2, NRHS);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;
        
        logg->delta_iter[ii] = 0.0;
        for (jj = 0; jj < logg->NSV; jj++) {
            SING_VALUES[jj] = sqrt(w[NRHS - 1 - jj]);
            logg->sing_values[jj] = SING_VALUES[jj];
            logg->delta_iter[ii] += logg->sing_values[jj];
        }
        
        // Convergence check with three modes
        if (ii > 0) {
            if (logg->blockPower_conv_crit == 0) {
                // MODE 0: Trace-based criterion
                logg->blockPower_trace_error = fabs(logg->delta_iter[ii-1] - logg->delta_iter[ii]) / logg->delta_iter[ii];
                if (rank == 0 && logg->PRINT_INFO > 1) {
                    printf("Iteration %d: rel. error: %02.13f\n", ii, logg->blockPower_trace_error);
                }
                if (logg->blockPower_trace_error <= logg->toll) break;
                
            } else if (logg->blockPower_conv_crit == 1) {
                // MODE 1: Individual eigenvalue criterion
                converged = 0;
                for (jj = 0; jj < logg->NSV; jj++) 
                    if (fabs((SING_VALUES[jj] - SING_VALUES_OLD[jj]) / SING_VALUES[jj]) <= logg->toll) {
                        converged++;
                    }
                if (rank == 0 && logg->PRINT_INFO > 1) 
                    printf("Iteration %d: %d converged\n", ii, converged);

                if (converged == logg->NSV) break;
                
            } else if (logg->blockPower_conv_crit == 2) {
                // MODE 2: MEV (Mean Explained Variance) criterion - PCAone's approach
                // Measures subspace similarity between successive iterations
                double MEV = 0.0;
                
                // Compute mean squared cosine similarity between old and new eigenvectors
                for (jj = 0; jj < logg->NSV; jj++) {
                    double dot_product = 0.0;
                    for (ii2 = 0; ii2 < M; ii2++) {
                        dot_product += RHS2_old[ii2 * NRHS + jj] * RHS2[ii2 * NRHS + jj];
                    }
                    MEV += dot_product * dot_product;  // Square of cosine similarity
                }
                MEV /= logg->NSV;  // Average over all principal components
                
                double mev_error = 1.0 - MEV;
                
                if (rank == 0 && logg->PRINT_INFO > 1) {
                    printf("Iteration %d: MEV error: %02.13f\n", ii, mev_error);
                }
                
                if (mev_error <= logg->toll) break;
            }
        }
        
        // Store values for next iteration
        for (jj = 0; jj < logg->NSV; jj++) 
            SING_VALUES_OLD[jj] = SING_VALUES[jj];

        //needed for MVE - store current eigenvectors for Mode2
        if (logg->blockPower_conv_crit == 2) {
            memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));
        }

        free(w);
    }

    MPI_File_close(&fh);

    logg->blockPower_total_its = (ii < max_iter) ? ii + 1 : ii;
    
    for (ii = 0; ii < M; ii++) {
        for (jj = 0; jj < logg->NSV; jj++) 
            logg->left_sing_vecs[ii * logg->NSV + jj] = RHS2[ii * NRHS + jj];
    }

    if (rank == 0 && logg->filewrite == 1) {
        std::string tempname = logg->prefixname.empty() ? 
                               ConstructFilename(*logg, "singularValues") : 
                               logg->prefixname + "_singularValues.txt";
        
        FILE *fwrite_singvalues = fopen(tempname.c_str(), "w");
        if (fwrite_singvalues == NULL) {
            perror("fopen singularValues");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<double> singularvals = logg->sing_values;
        std::transform(singularvals.begin(), singularvals.end(), singularvals.begin(), computeSquare);
        std::vector<std::string> individ = logg->indiv_ids;

        fprintf(fwrite_singvalues, "EIGENVALUES\n\n");
        for (ii = 0; ii < logg->NSV; ii++)
            fprintf(fwrite_singvalues, "%2.13lf\n", singularvals[ii]);
        fclose(fwrite_singvalues);

        tempname = logg->prefixname.empty() ? 
                   ConstructFilename(*logg, "singularVectors") : 
                   logg->prefixname + "_singularVectors.txt";
        
        FILE *fwrite_singvecs = fopen(tempname.c_str(), "w");
        if (fwrite_singvecs == NULL) {
            perror("fopen singularVectors");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        fprintf(fwrite_singvecs, "FID");
        for (jj = 0; jj < logg->NSV; jj++)
            fprintf(fwrite_singvecs, "\tPC%d", jj);
        fprintf(fwrite_singvecs, "\n");
        
        for (ii = 0; ii < M; ii++) {
            fprintf(fwrite_singvecs, "%8s", individ[ii].c_str());
            for (jj = 0; jj < logg->NSV; jj++)
                fprintf(fwrite_singvecs, "\t%2.13f", RHS2[ii * NRHS + jj]);
            fprintf(fwrite_singvecs, "\n");
        }
        fclose(fwrite_singvecs);
    }

    // Cleanup
    free(LOC_MAT);
    free(ARHS);
    free(ARHS_local);
    free(RHS);
    free(SING_VALUES);
    free(SING_VALUES_OLD);
    free(RHS2_old);      //needed for MVE Mode2
    free(B2);
    free(B2_duplicate);
    free(tau);
    free(decbin);
    free(readbin);
    free(norm_tmp);
    free(norm_precomp);
    delete[] seen_snp;
}

void BlockSubspaceIter_MPI_IO_2ble_buffering(const char* bedfile, double *RHS2, logistics *logg) {
    int M = logg->M, N = logg->N, NRHS = logg->NRHS;
    int local_N = logg->local_N;
    int local_N_start = logg->local_N_start;
    int max_iter = logg->blockPower_maxiter, min_dim = min(N,NRHS), powers = logg->power;
    double tt1, tt2;
    int converged = 0, ii, jj, kk, ii2, jj2;
    double fone = 1.0, fzero = 0.0;
    
    int rank = logg->mpi_rank;

    MPI_File fh;
    int mpi_err = MPI_File_open(MPI_COMM_WORLD, bedfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (mpi_err != MPI_SUCCESS) {
        if (rank == 0) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length;
            MPI_Error_string(mpi_err, error_string, &length);
            fprintf(stderr, "Error opening file %s: %s\n", bedfile, error_string);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_fetched = logg->rows_fetched;
    int local_loops = local_N / rows_fetched;
    int remaining_rows = local_N - rows_fetched * local_loops;

    double inv_sqrtN = 1.0 / sqrt((double)N);
    double inv_sqrtM = 1.0 / sqrt((double)M);

    double* ARHS            = (double*) malloc(M * NRHS * sizeof(double));
    double* ARHS_local      = (double*) malloc(M * NRHS * sizeof(double));
    double* SING_VALUES     = (double*) malloc(min_dim * sizeof(double));
    double* SING_VALUES_OLD = (double*) malloc(min_dim * sizeof(double));
    double* tau             = (double*) malloc(NRHS * sizeof(double));
    double* RHS2_old        = (double*) malloc(M * NRHS * sizeof(double));
    double* RHS_buf         = (double*) malloc(rows_fetched * NRHS * sizeof(double));

    logg->sing_values.resize(logg->NSV);
    logg->delta_iter.resize(max_iter);
    logg->left_sing_vecs.resize(M * logg->NSV);
    
    
    if (remaining_rows > 0)
      local_loops++;
    else
      remaining_rows = rows_fetched;

    std::vector<int> start(local_loops);
    std::vector<int> stop(local_loops);
    
    for(int ik = 0; ik < local_loops; ik++){
        start[ik] = local_N_start + ik * rows_fetched;
        stop[ik] = start[ik] + rows_fetched - 1;
        if (stop[ik] >= local_N_start + local_N) 
            stop[ik] = local_N_start + local_N - 1;
    }

    uint64_t np = (uint64_t)ceil((double)M / PACK_DENSITY);

    double* B2 = (double*) malloc(NRHS * NRHS * sizeof(double));
    double* B2_duplicate = (double*) malloc(NRHS * NRHS * sizeof(double));
    
    uint64_t lmsize = (uint64_t)rows_fetched * (uint64_t)M;
    double* LOC_MAT[2]; // 2ble buffering 
    LOC_MAT[0] = (double*)malloc(lmsize * sizeof(double));
    LOC_MAT[1] = (double*)malloc(lmsize * sizeof(double));
    
    std::vector<std::vector<unsigned char>> block_buf(2);
    MPI_Request reqs[2];
    
    unsigned char *decbin = (unsigned char*)malloc(np * PACK_DENSITY * sizeof(unsigned char));
    double *norm_tmp = (double*)malloc(M * sizeof(double));
    double *norm_precomp = (double*)calloc(4 * N, sizeof(double));
    bool *seen_snp = new bool[N]();

    auto _compute_matvec_buff = [&]() {
        memset(ARHS_local, 0, M*NRHS*sizeof(double));
        
        // Process ALL blocks including the first one
        for(int jj = 0; jj < local_loops; jj++) {
            int actual_block_size = (jj == local_loops-1) ? remaining_rows : (stop[jj]-start[jj]+1);
            uint64_t file_offset = 3 + np * start[jj];
            
            // Read current block synchronously
            std::vector<unsigned char> block_buf(np * actual_block_size);
            
            tt1 = dsecnd();
            MPI_Status status;
            MPI_File_read_at(fh, file_offset, block_buf.data(),
                          np * actual_block_size, MPI_UNSIGNED_CHAR, &status);
            tt2 = dsecnd() - tt1;
            logg->TIME_2_LOAD_MATRIX += tt2;
            
            // Process the block
            tt1 = dsecnd();
            for(int row = 0; row < actual_block_size; row++) {
                unsigned char* snp_bin = block_buf.data() + row * np;
                int global_idx = start[jj] + row;
                
                if(!seen_snp[global_idx]) {
                    decode_plink_sse2(decbin, snp_bin, np);
                    double avg, sd;
                    standardize(norm_tmp, decbin, M, avg, sd, inv_sqrtN);
                    if(sd > 1e-9) {
                        norm_precomp[3+(global_idx*4)] = (0-avg)/sd;
                        norm_precomp[2+(global_idx*4)] = (1-avg)/sd;
                        norm_precomp[0+(global_idx*4)] = (2-avg)/sd;
                        norm_precomp[1+(global_idx*4)] = 0;
                    }
                    seen_snp[global_idx] = true;
                    for(int k = 0; k < M; k++) 
                        LOC_MAT[0][row*M+k] = norm_tmp[k];
                } else {
                    decode_plink_precomp_sse2(decbin, snp_bin, np);
                    for(int k = 0; k < M; k++) {
                        int b = (int)decbin[k];
                        LOC_MAT[0][row*M+k] = norm_precomp[b+(global_idx*4)] * inv_sqrtN;
                    }
                }
            }
            tt2 = dsecnd() - tt1;
            logg->TIME_2_LOAD_MATRIX += tt2;
            
            // Matrix multiplications
            tt1 = dsecnd();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      actual_block_size, NRHS, M,
                      1.0, LOC_MAT[0], M, RHS2, NRHS, 0.0, RHS_buf, NRHS);
            tt2 = dsecnd() - tt1;
            logg->TIME_2_MM += tt2;
            logg->TIME_2_MM_A += tt2;
            
            tt1 = dsecnd();
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                      M, NRHS, actual_block_size,
                      1.0, LOC_MAT[0], M, RHS_buf, NRHS, 1.0, ARHS_local, NRHS);
            tt2 = dsecnd() - tt1;
            logg->TIME_2_MM += tt2;
            logg->TIME_2_MM_A_TRANSPOSED += tt2;
        }
        
        MPI_Allreduce(ARHS_local, ARHS, M*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    };

    // Initialize RHS2_old for Mode 2
    if (logg->blockPower_conv_crit == 2) {
        memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));
    }


    // Added on 14/12/25: before getting in the main loop, open the .csv file to log the eigenvalues per iteration
    // Get current date and time for filename prefix
    time_t now = time(0);
    struct tm* timeinfo = localtime(&now);
    char datetime_prefix[32];
    strftime(datetime_prefix, sizeof(datetime_prefix), "%d_%m_%Y_%H:%M:%S", timeinfo);
    
    FILE *eigenval_tracking_file = NULL;
    if (rank == 0 && save_eigenval_tracking == 1) {
      std::string tracking_filename = std::string(datetime_prefix) + "_eigenvalue_tracking.csv";
      eigenval_tracking_file = fopen(tracking_filename.c_str(), "w");
      if (eigenval_tracking_file == NULL) {
        perror("fopen eigenvalue_tracking");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      
      // write header
      fprintf(eigenval_tracking_file, "iteration");
      for (jj = 0; jj < logg->NSV; jj++) {
        fprintf(eigenval_tracking_file, ",eigenvalue_%d,rel_change_%d", jj+1, jj+1);
      }
      fprintf(eigenval_tracking_file, "\n");
    }

    // Main iteration loop
    for (ii = 0; ii < max_iter; ii++) {
        _compute_matvec_buff();
        
        for (kk = 0; kk < powers - 1; kk++) {
            tt1 = dsecnd();
            LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);
            LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
            tt2 = dsecnd() - tt1;
            logg->TIME_2_GS += tt2;
            
            memcpy(RHS2, ARHS, M * NRHS * sizeof(double));
            _compute_matvec_buff();
        }

        tt1 = dsecnd();
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_GS += tt2;

        memcpy(RHS2, ARHS, M * NRHS * sizeof(double));
        _compute_matvec_buff();

        tt1 = dsecnd();
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    NRHS, NRHS, M, 
                    fone, RHS2, NRHS, ARHS, NRHS, fzero, B2, NRHS);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;

        tt1 = dsecnd();
        for (ii2 = 0; ii2 < NRHS; ii2++) {
            for (jj2 = 0; jj2 < ii2; jj2++) 
                B2[ii2 * NRHS + jj2] = 0.0;
        }
        double* w = (double*) malloc(NRHS * sizeof(double));
        LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NRHS, B2, NRHS, w);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_PROJECTED_SVD += tt2;

        memcpy(B2_duplicate, B2, NRHS * NRHS * sizeof(double));
        for (ii2 = 0; ii2 < NRHS; ii2++) {
            for (jj2 = 0; jj2 < NRHS; jj2++) {
                B2[ii2 * NRHS + jj2] = B2_duplicate[ii2 * NRHS + (NRHS - 1 - jj2)];
            }
        }
        
        memcpy(ARHS, RHS2, M * NRHS * sizeof(double));
        tt1 = dsecnd();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, NRHS, NRHS, 
                    fone, ARHS, NRHS, B2, NRHS, fzero, RHS2, NRHS);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;
        
        logg->delta_iter[ii] = 0.0;
        for (jj = 0; jj < logg->NSV; jj++) {
            SING_VALUES[jj] = sqrt(w[NRHS - 1 - jj]);
            logg->sing_values[jj] = SING_VALUES[jj];
            logg->delta_iter[ii] += logg->sing_values[jj];
        }
        
        // Eigenvalue tracking
        if (rank == 0 && eigenval_tracking_file != NULL) {
            fprintf(eigenval_tracking_file, "%d", ii);
            for (jj = 0; jj < logg->NSV; jj++) {
                double eigenval = SING_VALUES[jj] * SING_VALUES[jj];
                if (ii > 0) {
                    double eigenval_old = SING_VALUES_OLD[jj] * SING_VALUES_OLD[jj];
                    double rel_change = fabs(eigenval - eigenval_old) / eigenval;
                    fprintf(eigenval_tracking_file, ",%.13e,%.6e", eigenval, rel_change);
                } else {
                    fprintf(eigenval_tracking_file, ",%.13e,NA", eigenval);
                }
            }
            fprintf(eigenval_tracking_file, "\n");
            fflush(eigenval_tracking_file);
        }
        
        // Convergence check - ONLY after iteration 1
        if (ii > 0) {
            if (logg->blockPower_conv_crit == 0) {
                logg->blockPower_trace_error = fabs(logg->delta_iter[ii-1] - logg->delta_iter[ii]) / logg->delta_iter[ii];
                if (rank == 0 && logg->PRINT_INFO > 1) {
                    printf("Iteration %d: rel. error: %02.13f\n", ii, logg->blockPower_trace_error);
                }
                if (logg->blockPower_trace_error <= logg->toll) break;
                
            } else if (logg->blockPower_conv_crit == 1) {
                converged = 0;
                for (jj = 0; jj < logg->NSV; jj++) 
                    if (fabs((SING_VALUES[jj] - SING_VALUES_OLD[jj]) / SING_VALUES[jj]) <= logg->toll) {
                        converged++;
                    }
                if (rank == 0 && logg->PRINT_INFO > 1) 
                    printf("Iteration %d: %d converged\n", ii, converged);

                if (converged == logg->NSV) break;
                
            } else if (logg->blockPower_conv_crit == 2) {
                double MEV = 0.0;
                for (jj = 0; jj < logg->NSV; jj++) {
                    double dot_product = 0.0;
                    for (ii2 = 0; ii2 < M; ii2++) {
                        dot_product += RHS2_old[ii2 * NRHS + jj] * RHS2[ii2 * NRHS + jj];
                    }
                    MEV += dot_product * dot_product;
                }
                MEV /= logg->NSV;
                double mev_error = 1.0 - MEV;
                
                if (rank == 0 && logg->PRINT_INFO > 1) {
                    printf("Iteration %d: MEV error: %02.13f\n", ii, mev_error);
                }
                
                if (mev_error <= logg->toll) break;
            }
        }
        
        // Store current iteration for next comparison - ALWAYS do this
        for (jj = 0; jj < logg->NSV; jj++) 
            SING_VALUES_OLD[jj] = SING_VALUES[jj];

        if (logg->blockPower_conv_crit == 2) {
            memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));
        }

        free(w);
    }
    
    // Added on 14/12/25: close the eigenvalue tracking file + final eigenvalues
    if (rank == 0 && eigenval_tracking_file != NULL) {
        // empty line for gap
        fprintf(eigenval_tracking_file, "\n");
        
        // final approximate eigenvalues section
        fprintf(eigenval_tracking_file, "# Final Approximate Eigenvalues\n");
        fprintf(eigenval_tracking_file, "eigenvalue_index,final_value\n");
        
        for (jj = 0; jj < logg->NSV; jj++) {
            double final_eigenval = logg->sing_values[jj] * logg->sing_values[jj];
            fprintf(eigenval_tracking_file, "%d,%.13e\n", jj+1, final_eigenval);
        }
        
        fclose(eigenval_tracking_file);
    }

    MPI_File_close(&fh);

    logg->blockPower_total_its = (ii < max_iter) ? ii + 1 : ii;
    
    for (ii = 0; ii < M; ii++) {
        for (jj = 0; jj < logg->NSV; jj++) 
            logg->left_sing_vecs[ii * logg->NSV + jj] = RHS2[ii * NRHS + jj];
    }

    if (rank == 0 && logg->filewrite == 1) {
        std::string tempname = logg->prefixname.empty() ? 
                               ConstructFilename(*logg, "singularValues") : 
                               logg->prefixname + "_singularValues.txt";
        
        FILE *fwrite_singvalues = fopen(tempname.c_str(), "w");
        if (fwrite_singvalues == NULL) {
            perror("fopen singularValues");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<double> singularvals = logg->sing_values;
        std::transform(singularvals.begin(), singularvals.end(), singularvals.begin(), computeSquare);
        std::vector<std::string> individ = logg->indiv_ids;

        fprintf(fwrite_singvalues, "EIGENVALUES\n\n");
        for (ii = 0; ii < logg->NSV; ii++)
            fprintf(fwrite_singvalues, "%2.13lf\n", singularvals[ii]);
        fclose(fwrite_singvalues);

        tempname = logg->prefixname.empty() ? 
                   ConstructFilename(*logg, "singularVectors") : 
                   logg->prefixname + "_singularVectors.txt";
        
        FILE *fwrite_singvecs = fopen(tempname.c_str(), "w");
        if (fwrite_singvecs == NULL) {
            perror("fopen singularVectors");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        fprintf(fwrite_singvecs, "FID");
        for (jj = 0; jj < logg->NSV; jj++)
            fprintf(fwrite_singvecs, "\tPC%d", jj);
        fprintf(fwrite_singvecs, "\n");
        
        for (ii = 0; ii < M; ii++) {
            fprintf(fwrite_singvecs, "%8s", individ[ii].c_str());
            for (jj = 0; jj < logg->NSV; jj++)
                fprintf(fwrite_singvecs, "\t%2.13f", RHS2[ii * NRHS + jj]);
            fprintf(fwrite_singvecs, "\n");
        }
        fclose(fwrite_singvecs);
    }

    // Cleanup
    free(LOC_MAT[0]);
    free(LOC_MAT[1]);
    free(ARHS);
    free(ARHS_local);
    free(RHS_buf);
    free(SING_VALUES);
    free(SING_VALUES_OLD);
    free(RHS2_old);
    free(B2);
    free(B2_duplicate);
    free(tau);
    free(decbin);
    free(norm_tmp);
    free(norm_precomp);
    delete[] seen_snp;
}

void BlockSubspaceIter_MPI_IO_LoadOnly(const char* bedfile, double *RHS2, logistics *logg) {
    int M = logg->M, N = logg->N, NRHS = logg->NRHS;
    int local_N = logg->local_N;
    int local_N_start = logg->local_N_start;
    double tt1, tt2, tt_total_load = 0.0;
    
    int rank = logg->mpi_rank;
    int nprocs = logg->mpi_size;

    MPI_File fh;
    int mpi_err = MPI_File_open(MPI_COMM_WORLD, bedfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (mpi_err != MPI_SUCCESS) {
        if (rank == 0) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length;
            MPI_Error_string(mpi_err, error_string, &length);
            fprintf(stderr, "Error opening file %s: %s\n", bedfile, error_string);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_fetched = logg->rows_fetched;  // This is your blocksize
    int local_loops = local_N / rows_fetched;
    int remaining_rows = local_N - rows_fetched * local_loops;

    if (remaining_rows > 0)
        local_loops++;
    else
        remaining_rows = rows_fetched;

    std::vector<int> start(local_loops);
    std::vector<int> stop(local_loops);
    
    for(int ik = 0; ik < local_loops; ik++){
        start[ik] = local_N_start + ik * rows_fetched;
        stop[ik] = start[ik] + rows_fetched - 1;
        if (stop[ik] >= local_N_start + local_N) 
            stop[ik] = local_N_start + local_N - 1;
    }

    uint64_t np = (uint64_t)ceil((double)M / PACK_DENSITY);

    // Single buffer for loading (no computation, so no double buffering needed)
    std::vector<unsigned char> block_buf;
    unsigned char *decbin = (unsigned char*)malloc(np * PACK_DENSITY * sizeof(unsigned char));
    double *norm_tmp = (double*)malloc(M * sizeof(double));
    double *norm_precomp = (double*)calloc(4 * N, sizeof(double));
    bool *seen_snp = new bool[N]();

    if (rank == 0) {
        printf("\n=== LOAD-ONLY MODE ===\n");
        printf("Dataset: %s\n", bedfile);
        printf("M=%d, N=%d, np=%d\n", M, N, nprocs);
        printf("Block size (rows_fetched): %d\n", rows_fetched);
        printf("Rank %d: local_N=%d, local_loops=%d, remaining_rows=%d\n", 
               rank, local_N, local_loops, remaining_rows);
        printf("=====================\n\n");
    }

    double inv_sqrtN = 1.0 / sqrt((double)N);

    // === LOADING LOOP (NO COMPUTATION) ===
    tt1 = dsecnd();
    
    for(int jj = 0; jj < local_loops; jj++) {
        int actual_block_size = (jj == local_loops-1) ? remaining_rows : (stop[jj]-start[jj]+1);
        uint64_t file_offset = 3 + np * start[jj];
        
        // Resize buffer for this block
        block_buf.resize(np * actual_block_size);
        
        // Synchronous read (no async for simplicity in load-only mode)
        double tt_read_start = dsecnd();
        MPI_File_read_at(fh, file_offset, block_buf.data(),
                        np * actual_block_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
        double tt_read = dsecnd() - tt_read_start;
        
        // Decode and normalize (minimal processing to simulate realistic load)
        double tt_decode_start = dsecnd();
        for(int row = 0; row < actual_block_size; row++) {
            unsigned char* snp_bin = block_buf.data() + row * np;
            int global_idx = start[jj] + row;
            
            if(!seen_snp[global_idx]) {
                decode_plink(decbin, snp_bin, np);
                double avg, sd;
                standardize(norm_tmp, decbin, M, avg, sd, inv_sqrtN);
                
                if(sd > 1e-9) {
                    norm_precomp[3+(global_idx*4)] = (0-avg)/sd;
                    norm_precomp[2+(global_idx*4)] = (1-avg)/sd;
                    norm_precomp[0+(global_idx*4)] = (2-avg)/sd;
                    norm_precomp[1+(global_idx*4)] = 0;
                }
                seen_snp[global_idx] = true;
            } else {
                decode_plink_precomp(decbin, snp_bin, np);
                // Note: We decode but don't store - just simulating the work
            }
        }
        double tt_decode = dsecnd() - tt_decode_start;
        
        if (rank == 0 && jj % 100 == 0) {
            printf("Rank %d: Block %d/%d - Read: %.4fs, Decode: %.4fs\n", 
                   rank, jj+1, local_loops, tt_read, tt_decode);
        }
    }
    
    tt2 = dsecnd() - tt1;
    tt_total_load = tt2;

    MPI_File_close(&fh);

    // Gather timing statistics across all ranks
    double tt_load_max, tt_load_min, tt_load_avg;
    MPI_Reduce(&tt_total_load, &tt_load_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tt_total_load, &tt_load_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tt_total_load, &tt_load_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        tt_load_avg /= nprocs;
        printf("\n=== LOAD-ONLY RESULTS ===\n");
        printf("Total SNPs loaded: %d\n", N);
        printf("Block size: %d SNPs\n", rows_fetched);
        printf("Number of MPI processes: %d\n", nprocs);
        printf("\nTiming Statistics:\n");
        printf("  Max time: %.4f seconds\n", tt_load_max);
        printf("  Min time: %.4f seconds\n", tt_load_min);
        printf("  Avg time: %.4f seconds\n", tt_load_avg);
        printf("  Load imbalance: %.2f%%\n", 
               100.0 * (tt_load_max - tt_load_min) / tt_load_avg);
        printf("\nThroughput:\n");
        printf("  SNPs/second (per rank): %.2f\n", local_N / tt_load_max);
        printf("  SNPs/second (aggregate): %.2f\n", N / tt_load_max);
        printf("  MB/second (per rank): %.2f\n", 
               (local_N * np / 1024.0 / 1024.0) / tt_load_max);
        printf("========================\n");
    }

    // Store timing in logg structure for later analysis
    logg->TIME_2_LOAD_MATRIX = tt_total_load;
    logg->TIME_2_MM = 0.0;  // No computation
    logg->TIME_2_GS = 0.0;  // No orthogonalization
    logg->TIME_2_PROJECTED_SVD = 0.0;  // No SVD

    // Cleanup
    free(decbin);
    free(norm_tmp);
    free(norm_precomp);
    delete[] seen_snp;
}

void subspaceIteration(double *MAT, double *RHS2, logistics *logg) {
  //===========================================================
  int M = logg->M, N = logg->N, NRHS = logg->NRHS;
  int max_iter = logg->blockPower_maxiter;
  int powers = logg->power;
  int ii, jj, converged = 0, kk, ii2, jj2;
  double tt1, tt2;
  double fone = 1.0, fzero = 0.0;
  double w[NRHS];
  double* B    = (double*) malloc(NRHS*NRHS*sizeof(double));
  double* B2   = (double*) malloc(NRHS*NRHS*sizeof(double));
  double* RHS  = (double*) malloc(NRHS*N*sizeof(double));
  double* ARHS = (double*) malloc(NRHS*M*sizeof(double));
  double* SING_VALUES     = (double*) malloc(logg->NSV*sizeof(double));
  double* SING_VALUES_OLD = (double*) malloc(logg->NSV*sizeof(double));
  double* tau = (double*) malloc(NRHS*sizeof(double));
  int info, info_sgeqrf_lapacke, info_sorgqr_lapacke;
  double tr1[NRHS],tr2[NRHS];
  logg->delta_iter.resize(max_iter);
  logg->sing_values.resize(logg->NSV);
  logg->left_sing_vecs.resize(M*logg->NSV);
  //============================================================

  for (ii = 0; ii < max_iter; ii++) {
    //============================================================
    // Multiply A' by RHS2 from the right
    //============================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, NRHS, M, fone, MAT, M, RHS2, NRHS, fzero, RHS, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
    //============================================================

    //============================================================           
    // Multiply A by RHS from the right
    //============================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, N, fone, MAT, M, RHS, NRHS, fzero, ARHS, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
    //============================================================
    
    //============================================================
    //                       (AA')^{powers-1}
    //============================================================
    for ( kk = 0; kk < powers-1; kk++) {
      tt1 = dsecnd();
      info_sgeqrf_lapacke = LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau );
      info_sorgqr_lapacke = LAPACKE_dorgqr( LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau );
      tt2 = dsecnd() - tt1;
      logg->TIME_2_GS = logg->TIME_2_GS + tt2;
      
      memcpy(RHS2,ARHS,M*NRHS*sizeof(double));
      
      tt1 = dsecnd();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, NRHS, M, fone, MAT, M, RHS2, NRHS, fzero, RHS, NRHS);
      tt2 = dsecnd() - tt1;
      logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;

      tt1 = dsecnd();
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, N, fone, MAT, M, RHS, NRHS, fzero, ARHS, NRHS);
      tt2 = dsecnd() - tt1;
      logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
    }
    //============================================================

    //============================================================                               
    // Perform orthogonalization                                                                           
    //============================================================
    tt1 = dsecnd();
    info_sgeqrf_lapacke = LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau );
    info_sorgqr_lapacke = LAPACKE_dorgqr( LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau );
    tt2 = dsecnd() - tt1;
    logg->TIME_2_GS = logg->TIME_2_GS + tt2;
    //============================================================
    
    //============================================================
    // Multiply A' by RHS2 from the right
    //============================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, NRHS, M, fone, MAT, M, ARHS, NRHS, fzero, RHS, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
    //============================================================

    //============================================================
    // Now compute RHS'RHS
    //============================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, NRHS, NRHS, N, fone, RHS, NRHS, RHS, NRHS, fzero, B, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    //============================================================

    //============================================================
    // Solve projected eigenvalue problem
    //============================================================
    tt1 = dsecnd();
    for (ii2 = 0; ii2 < NRHS; ii2++) {
      for (jj2 = 0; jj2 < NRHS; jj2++) {
        if ( jj2 < ii2 ) {
	  B[ii2*NRHS + jj2] = 0.0;
        }
      }
    }
 
    info = LAPACKE_dsyev( LAPACK_ROW_MAJOR, 'V', 'U', NRHS, B, NRHS, w );
    tt2 = dsecnd() - tt1;
    logg->TIME_2_PROJECTED_SVD = logg->TIME_2_PROJECTED_SVD + tt2;
    //===========================================================

    //===========================================================
    // Copy back to RHS2 -- be careful, singular vectors 
    // come from smallest to largest
    //===========================================================
    memcpy(B2,B,sizeof(double)*NRHS*NRHS);
    for (ii2 = 0; ii2 < NRHS; ii2++) {
      for (jj2 = 0; jj2 < NRHS; jj2++) {
        B[ii2*NRHS+jj2] = B2[ii2*NRHS+(NRHS-1-jj2)];
      }
    }
    //===========================================================

    //===========================================================
    // Now multiply RHS2 by the eigenvectors from the right
    //===========================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, NRHS, NRHS, fone, ARHS, NRHS, B, NRHS, fzero, RHS2, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    //==========================================================

    //==========================================================
    // Copy singular values and monitor trace
    //==========================================================
    for (jj2 = 0; jj2 < logg->NSV; jj2++ ) {
      SING_VALUES[jj2] = sqrt(w[NRHS-1-jj2]);
      logg->sing_values[jj2] = SING_VALUES[jj2];
      logg->delta_iter[ii] = logg->delta_iter[ii] + logg->sing_values[jj2];
      if (logg-> PRINT_INFO >1)
	printf("At iteration:%d -->sing.val: %d is %02.13f\n", ii, jj2, SING_VALUES[jj2]);
    }
    //==========================================================

    //==========================================================
    //          Check convergence
    //==========================================================
    if ( logg->blockPower_conv_crit == 0) { // checking convergence based on the partial sum of singular values
      if (ii==0) {
        if (logg->PRINT_INFO > 1) 
	  printf("Partial sum at iteration: %d --> %02.13f\n", ii, logg->delta_iter[ii]);
      } else {
        logg->blockPower_trace_error = fabs(logg->delta_iter[ii-1]-logg->delta_iter[ii])/logg->delta_iter[ii];
        if (logg->PRINT_INFO > 1) 
	  printf("Partial sum at iteration: %d --> %02.13f. Rel. error: %02.13f\n", ii, logg->delta_iter[ii], logg->blockPower_trace_error);
        if (logg->blockPower_trace_error <= logg->toll) {
	  break;
        }
      }
    } else { // checking convergence based on the relative convergence of each individual singular value
      if (ii > 0) {
        for (jj = 0; jj < logg->NSV; jj++ ) {
          if (fabs((SING_VALUES[jj]-SING_VALUES_OLD[jj])/SING_VALUES[jj]) <= logg->toll ) {
            converged++;
          }
        }
	if (logg-> PRINT_INFO >1)
	  printf("At iteration: %d, %d singular values converged\n", ii, converged);
        if (converged == logg->NSV) {
          break;
        } 
      }
    }

    // prepare logistics for next iteration
    converged = 0;
    for (jj = 0; jj < logg->NSV; jj++ ) {
      SING_VALUES_OLD[jj] = SING_VALUES[jj];
    }
    //========================================================== 
  }
  //============================================================

  if ( ii < max_iter ) {
    logg->blockPower_total_its = ii+1;
  } else {
    logg->blockPower_total_its = ii;
  }
  
  for(ii = 0; ii < M; ii++ ) {
    for(jj = 0; jj < NRHS; jj++ ) {
      if ( jj < logg->NSV ) {
	logg->left_sing_vecs[ii*logg->NSV+jj] = RHS2[ii*NRHS+jj];
      }
    }
  }

  //==========================================================
  // write approx. singular values, left vectors to file
  //==========================================================
  if (logg->filewrite == 1) {
       string tempname;

       if (logg->prefixname.empty())
          tempname = ConstructFilename(*logg,"singularValues");
       else
          tempname = logg->prefixname + "_singularValues.txt";

       FILE *fwrite_singvalues = fopen(tempname.c_str(), "w");
       if (fwrite_singvalues==NULL) {
         printf("Unable to write to file. Aborting...");
         exit(1);
       }  

       std::vector<double> singularvals = logg->sing_values;
       std::transform(singularvals.begin(), singularvals.end(), singularvals.begin(), computeSquare);// square singular values
       std::vector<string> individs = logg->indiv_ids;

       fprintf(fwrite_singvalues, "EIGENVALUES\n\n");
       for(ii = 0; ii < logg->NSV; ii++ )
         fprintf(fwrite_singvalues, "%2.13lf\n", singularvals[ii]);  
       fclose(fwrite_singvalues);

       if (logg->prefixname.empty())
         tempname = ConstructFilename(*logg,"singularVectors");
       else
         tempname = logg->prefixname + "_singularVectors.txt";
       FILE *fwrite_singvecs = fopen(tempname.c_str(), "w");
       if (fwrite_singvecs==NULL) {
         printf("Unable to write to file. Aborting...");
         exit(1);
       }
       fprintf(fwrite_singvecs, "FID");
       for(jj = 0; jj < logg->NSV; jj++)
	  fprintf(fwrite_singvecs, "\tPC%d",jj);
       fprintf(fwrite_singvecs,"\n");
       for(ii = 0; ii < M; ii++ ){
	  fprintf(fwrite_singvecs, "%8s", individs[ii].c_str());	
          for(jj = 0; jj < logg->NSV; jj++ ) 
	    fprintf(fwrite_singvecs, "\t%2.13f", RHS2[ii*NRHS+jj]);
          fprintf(fwrite_singvecs, "\n");
       }
       fclose(fwrite_singvecs);
  }
  //==========================================================

  //==========================================================
  // Finalize program and deallocate resources                                                                             
  //==========================================================
  free(ARHS);
  free(RHS); 
  free(B2);
  free(B);
  free(SING_VALUES);                                                                                                    
  free(SING_VALUES_OLD);                                                                                                    
  free(tau);
  //==========================================================
  
}

void BlockSubspaceIter(std::ifstream& infile, double *RHS2, logistics *logg) {

  //==========================================================                            
  int    M = logg->M, N = logg->N, NRHS = logg->NRHS;
  int    max_iter = logg->blockPower_maxiter, min_dim = min(N,NRHS), powers = logg->power;
  double tt1, tt2;
  int    ione = 1,   converged = 0, ii, jj, kk, ii2, jj2;
  double fone = 1.0, minusfone = -1.0, fzero = 0.0;
  double* ARHS            = (double*) malloc(M*NRHS*sizeof(double));
  double* SING_VALUES     = (double*) malloc(min_dim*sizeof(double));
  double* SING_VALUES_OLD = (double*) malloc(min_dim*sizeof(double));
  double* tau             = (double*) malloc(NRHS*sizeof(double));
  int info_svd_lapacke, info_sgeqrf_lapacke, info_sorgqr_lapacke;
  logg->sing_values.resize(logg->NSV);
  logg->delta_iter.resize(max_iter);
  logg->left_sing_vecs.resize(M*logg->NSV);
  //================================================================                                          

  //================================================================
  int rows_fetched      = logg->rows_fetched;
  int loops             = N / rows_fetched;
  int remaining_rows    = N - rows_fetched*loops;
  unsigned int ik;
  int      colss        = logg->M;
  double* RHS           = (double*) malloc(max(remaining_rows,rows_fetched)*NRHS*sizeof(double));
  double* B2            = (double*) malloc(NRHS*NRHS*sizeof(double));
  double* B2_duplicate  = (double*) malloc(NRHS*NRHS*sizeof(double));
  double tr1[NRHS],tr2[NRHS];
  uint64_t np       = (unsigned long long)ceil((double)M/PACK_DENSITY);  //size of the packed data, in bytes, per SNP
  uint64_t actual_block_size=0, startval; 
  double *LOC_MAT;
  unsigned char *decbin  = (unsigned char*)malloc(np*PACK_DENSITY*sizeof(unsigned char*));
  unsigned char *readbin = (unsigned char*)malloc(np*sizeof(unsigned char*));
  double *norm_tmp = (double*)malloc(M*sizeof(double)); 
  //================================================================
  if (remaining_rows > 0) {
    loops = loops + 1;
  } else {
    remaining_rows = rows_fetched;
  }

  vector<int>start(loops);
  vector<int>stop(loops); 

  for(ik = 0 ; ik < loops ; ik++){
    start[ik]= ik * rows_fetched;
    stop[ik] = start[ik] + rows_fetched - 1;
    stop[ik] = stop[ik] >= N ? N - 1 : stop[ik];
  }

  uint64_t lmsize = max(remaining_rows,rows_fetched)*M;
  LOC_MAT = (double*)malloc(lmsize*sizeof(double));
  //================================================================
  double *norm_precomp = (double*)malloc((4*N)*sizeof(double));
  memset(norm_precomp,0,(4*N)*sizeof(double));
  bool *seen_snp = new bool[N]();
  //================================================================
  
  //================================================================
  for (ii = 0; ii < max_iter; ii++) {
    //====================================
    // Multiply AA' by RHS2 from the right
    //====================================
    infile.seekg(3, std::ifstream::beg);
    memset(ARHS, 0, M*NRHS*sizeof(double));
    for (jj = 0; jj < loops; jj++) {
      // prepare to fetch data from memory
      // pack more bytes than required because it needs to read all of the byte
      if(jj<loops-1){
        actual_block_size = stop[jj] - start[jj] + 1;
      }else{
	actual_block_size = remaining_rows;
      }
      startval = start[jj];
      infile.seekg(3 + np * startval);

      if ( jj == 0 || jj < loops-1 ) {
	// Load chunk of SNPs       
        tt1 = dsecnd();
	Read_Bed_Blocks(infile, np, actual_block_size, LOC_MAT, startval, logg, decbin, readbin, norm_tmp, seen_snp, norm_precomp);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_LOAD_MATRIX = logg->TIME_2_LOAD_MATRIX + tt2;
	
        // Multiply with A'
	tt1 = dsecnd();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_fetched, NRHS, M, fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);	
	tt2 = dsecnd() - tt1;
	logg->TIME_2_MM   = logg->TIME_2_MM + tt2;
	logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
	
        // Multiply with A
	tt1 = dsecnd();
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, rows_fetched, fone, LOC_MAT, M, RHS, NRHS, fone, ARHS, NRHS);
	tt2 = dsecnd() - tt1;
	logg->TIME_2_MM = logg->TIME_2_MM + tt2;
	logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;} 
      else{
        // Load chunk of SNPs
      	tt1 = dsecnd();
      	Read_Bed_Blocks(infile, np, actual_block_size, LOC_MAT, startval, logg, decbin, readbin, norm_tmp, seen_snp, norm_precomp);
        tt2 = dsecnd() - tt1;
      	logg->TIME_2_LOAD_MATRIX = logg->TIME_2_LOAD_MATRIX + tt2;
      	
        // Multiply with A'
      	tt1 = dsecnd();
      	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, remaining_rows, NRHS, M, fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);
      	tt2 = dsecnd() - tt1;
      	logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      	logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
      	
        // Multiply with A
      	tt1 = dsecnd();
      	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, remaining_rows, fone, LOC_MAT, M, RHS, NRHS, fone, ARHS, NRHS);
      	tt2 = dsecnd() - tt1;
      	logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      	logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
      }
    }

    //======================================================================
    // Trick to reduce the number of times that A is fetched from the memory
    //======================================================================
    for ( kk = 0; kk < powers-1; kk++) {
	//==========================
	// Perform orthogonalization
	//==========================
	tt1 = dsecnd();
	info_sgeqrf_lapacke = LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau );
	info_sorgqr_lapacke = LAPACKE_dorgqr( LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau );
	tt2 = dsecnd() - tt1;
	logg->TIME_2_GS = logg->TIME_2_GS + tt2;
	//
	memcpy(RHS2,ARHS,M*NRHS*sizeof(double));
        memset(ARHS, 0,  M*NRHS*sizeof(double));
	//==========================
	
	//====================================
	// Multiply AA' by RHS2 from the right
	//====================================
	infile.seekg(3, std::ifstream::beg);
	for (jj = 0; jj < loops; jj++) {
	  //pack more bytes than required because it needs to read all of the byte
          if(jj<loops-1){
            actual_block_size = stop[jj] - start[jj] + 1;
          }else{
            actual_block_size = remaining_rows;
          }
	  startval = start[jj];
	  infile.seekg(3 + np * startval);
	  if ( jj == 0 || jj < loops-1 ) {
	     
	    // Load chunk of SNPs
	    tt1 = dsecnd();
	    Read_Bed_Blocks(infile, np, actual_block_size, LOC_MAT, startval, logg, decbin, readbin, norm_tmp, seen_snp, norm_precomp);
	    tt2 = dsecnd() - tt1;
	    logg->TIME_2_LOAD_MATRIX = logg->TIME_2_LOAD_MATRIX + tt2;
	    
	    // Multiply with A'
	    tt1 = dsecnd();
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_fetched, NRHS, M, fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);
	    tt2 = dsecnd() - tt1;
	    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
	    logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
	 
	    // Multiply with A
	    tt1 = dsecnd();
	    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, rows_fetched, fone, LOC_MAT, M, RHS, NRHS, fone, ARHS, NRHS);
	    tt2 = dsecnd() - tt1;
	    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
	    logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
	  } else {
	 
	    // Load chunk of SNPs
	    tt1 = dsecnd();
	    Read_Bed_Blocks(infile, np, actual_block_size, LOC_MAT, startval, logg, decbin, readbin, norm_tmp, seen_snp, norm_precomp);
	    tt2 = dsecnd() - tt1;
	    logg-> TIME_2_LOAD_MATRIX = logg->TIME_2_LOAD_MATRIX + tt2;
	    
	    // Multiply with A'
	    tt1 = dsecnd();
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, remaining_rows, NRHS, M, fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);
	    tt2 = dsecnd() - tt1;
	    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
	    logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;

	    // Multiply with A
    	    tt1 = dsecnd();
    	    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, remaining_rows, fone, LOC_MAT, M, RHS, NRHS, fone, ARHS, NRHS);
    	    tt2 = dsecnd() - tt1;
    	    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    	    logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
	  }
	}
    }
    //======================================================================

    //==========================
    // Perform orthogonalization
    //==========================
    tt1 = dsecnd();
    info_sgeqrf_lapacke = LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau );
    info_sorgqr_lapacke = LAPACKE_dorgqr( LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau );
    tt2 = dsecnd() - tt1;
    logg->TIME_2_GS = logg->TIME_2_GS + tt2;
    //==========================

    //====================================
    // Multiply AA' by RHS2 from the right
    //====================================
    infile.seekg(3, std::ifstream::beg);
    memcpy(RHS2,ARHS,M*NRHS*sizeof(double));
    memset(ARHS, 0, M*NRHS*sizeof(double));
    for (jj = 0; jj < loops; jj++) {
        // prepare to fetch data from memory
        // pack more bytes than required because it needs to read all of the byte
        if(jj<loops-1){
          actual_block_size = stop[jj] - start[jj] + 1;
        }else{
          actual_block_size = remaining_rows;
        }
        startval = start[jj];
        infile.seekg(3 + np * startval);

        if ( jj == 0 || jj < loops-1 ) {
          // Load chunk of SNPs       
          tt1 = dsecnd();
  	  Read_Bed_Blocks(infile, np, actual_block_size, LOC_MAT, startval, logg, decbin, readbin, norm_tmp, seen_snp, norm_precomp);
          tt2 = dsecnd() - tt1;
          logg->TIME_2_LOAD_MATRIX = logg->TIME_2_LOAD_MATRIX + tt2;
	
          // Multiply with A'
	  tt1 = dsecnd();
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_fetched, NRHS, M, fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);	
	  tt2 = dsecnd() - tt1;
	  logg->TIME_2_MM   = logg->TIME_2_MM + tt2;
	  logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
	
          // Multiply with A
	  tt1 = dsecnd();
	  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, rows_fetched, fone, LOC_MAT, M, RHS, NRHS, fone, ARHS, NRHS);
	  tt2 = dsecnd() - tt1;
	  logg->TIME_2_MM = logg->TIME_2_MM + tt2;
	  logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;} 
        else{
          // Load chunk of SNPs
      	  tt1 = dsecnd();
      	  Read_Bed_Blocks(infile, np, actual_block_size, LOC_MAT, startval, logg, decbin, readbin, norm_tmp, seen_snp, norm_precomp);
          tt2 = dsecnd() - tt1;
      	  logg->TIME_2_LOAD_MATRIX = logg->TIME_2_LOAD_MATRIX + tt2;
      	
          // Multiply with A'
      	  tt1 = dsecnd();
      	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, remaining_rows, NRHS, M, fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);
      	  tt2 = dsecnd() - tt1;
      	  logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      	  logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
      	
          // Multiply with A
      	  tt1 = dsecnd();
      	  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, remaining_rows, fone, LOC_MAT, M, RHS, NRHS, fone, ARHS, NRHS);
      	  tt2 = dsecnd() - tt1;
      	  logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      	  logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
        }
    }
    //====================================

    //===================================================
    //                Now compute B'B
    //===================================================
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, NRHS, NRHS, M, fone, RHS2, NRHS, ARHS, NRHS, fzero, B2, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    //===================================================

    //=======================================
    // Solve the projected eigenvalue problem
    //=======================================
    tt1 = dsecnd();
    for (ii2 = 0; ii2 < NRHS; ii2++) {
      for (jj2 = 0; jj2 < NRHS; jj2++) {
        if ( jj2 < ii2 ) {
          B2[ii2*NRHS + jj2] = 0.0;
        }
      }
    }
    //
    double w[NRHS];
    int info = LAPACKE_dsyev( LAPACK_ROW_MAJOR, 'V', 'U', NRHS, B2, NRHS, w );
    tt2 = dsecnd() - tt1;
    logg->TIME_2_PROJECTED_SVD = logg->TIME_2_PROJECTED_SVD + tt2;
    //=======================================

    //==================================================================================
    // Be careful, singular vectors come from smallest to largest -- reverse their order
    //==================================================================================
    memcpy(B2_duplicate,B2,NRHS*NRHS*sizeof(double));
    for (ii2 = 0; ii2 < NRHS; ii2++) {
        for (jj2 = 0; jj2 < NRHS; jj2++) {
          B2[ii2*NRHS+jj2] = B2_duplicate[ii2*NRHS+(NRHS-1-jj2)];
        }
    }
    //==================================================================================
      
    //=====================================================
    // Now multiply RHS2 by the eigenvectors from the right
    //=====================================================
    memcpy(ARHS,RHS2,M*NRHS*sizeof(double));
    tt1 = dsecnd();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, NRHS, NRHS, fone, ARHS, NRHS, B2, NRHS, fzero, RHS2, NRHS);
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM = logg->TIME_2_MM + tt2;
    //=====================================================
      
    //=======================================
    // Copy singular values and monitor trace
    //=======================================
    for (jj = 0; jj < logg->NSV; jj++ ) {
	SING_VALUES[jj] = sqrt(w[NRHS-1-jj]);
	logg->sing_values[jj] = SING_VALUES[jj];
	logg->delta_iter[ii] = logg->delta_iter[ii] + logg->sing_values[jj];
	if (logg->PRINT_INFO >1){
	  printf("At iteration:%d -->sing.val: %d is %02.13f\n", ii, jj, SING_VALUES[jj]);}
    }
    //=======================================
      
    //=======================================
    //          Check convergence
    //=======================================
    if (logg->blockPower_conv_crit == 0) { // if checking convergence based on the sum
	if (ii==0) {
	  if (logg->PRINT_INFO > 1) {
	    printf("Partial sum at iteration: %d --> %02.13f\n", ii, logg->delta_iter[ii]);
	  }
	} else {
	  logg->blockPower_trace_error = fabs(logg->delta_iter[ii-1]-logg->delta_iter[ii])/logg->delta_iter[ii];
	  if (logg->PRINT_INFO > 1) {
	    printf("Partial sum at iteration: %d --> %02.13f. Rel. error: %02.13f\n", ii, logg->delta_iter[ii], logg->blockPower_trace_error);
	  }
	  if (logg->blockPower_trace_error <= logg->toll) {
	    break;
	  }
	}
      } else { // if checking convergence of each individual singular value
	if (ii > 0) {
	  for (jj = 0; jj < logg->NSV; jj++ ) {
	    if ( fabs((SING_VALUES[jj]-SING_VALUES_OLD[jj])/SING_VALUES[jj]) <= logg->toll ) {
	      converged++;
	    }
	  }
	  if (logg->PRINT_INFO >1){
	    printf("At iteration: %d, %d sing vals converged\n", ii, converged);}
	  if (converged == logg->NSV) {
	    break;
	  } 
	}
      }
      
      //=====================================
      // prepare logistics for next iteration
      //=====================================
      converged = 0;
      for (jj = 0; jj < logg->NSV; jj++ ) {
	SING_VALUES_OLD[jj] = SING_VALUES[jj];
      }
      //=====================================
  }
  //=======================================

  //=========================================================
  // If convergence achieved, increase iteration counter by 1
  //=========================================================
  if ( ii < max_iter ) {
    logg->blockPower_total_its = ii+1;
  } else {
    logg->blockPower_total_its = ii;
  }
  //=========================================================
    
  //=========================================================
  // Copy approx left singular vectors to RHS2
  //=========================================================
  for (ii = 0; ii < M; ii++ ) {
    for (jj = 0; jj < logg->NSV; jj++ ) {
      logg->left_sing_vecs[ii*logg->NSV+jj] = RHS2[ii*NRHS+jj];
    }
  }
  //=========================================================

  //============================================================================
  // If filewrite==1, write approximate singular values, left vectors to file
  //============================================================================
  if (logg->filewrite == 1) {
    string tempname;
    if (logg->prefixname.empty())
      tempname = ConstructFilename(*logg,"singularValues");
    else
      tempname = logg->prefixname + "_singularValues.txt";
    FILE *fwrite_singvalues = fopen(tempname.c_str(), "a");
    if (fwrite_singvalues==NULL) {
      printf("Unable to write to file. Aborting...");
      exit(1);
    }
   
    std::vector<double> singularvals = logg->sing_values;
    std::transform(singularvals.begin(), singularvals.end(), singularvals.begin(), computeSquare); 
    std::vector<string> individ = logg->indiv_ids; 
    
    fprintf(fwrite_singvalues, "EIGENVALUES\n\n"); 
    for(ii = 0; ii < logg->NSV; ii++ ) { 
      fprintf(fwrite_singvalues, "%2.13lf\n", singularvals[ii]);  
    }
    fclose(fwrite_singvalues);
      
    if (logg->prefixname.empty())
	tempname = ConstructFilename(*logg,"singularVectors");
    else
	tempname = logg->prefixname + "_singularVectors.txt";
    FILE *fwrite_singvecs = fopen(tempname.c_str(), "w");
    if (fwrite_singvecs==NULL) {
	printf("Unable to write to file. Aborting...");
	exit(1);
    }
	fprintf(fwrite_singvecs, "FID");
        for(jj = 0; jj < logg->NSV; jj++)
                fprintf(fwrite_singvecs, "\tPC%d",jj);
        fprintf(fwrite_singvecs,"\n");

    for (ii = 0; ii < M; ii++ ) {
      fprintf(fwrite_singvecs, "%8s",individ[ii].c_str());
      for (jj = 0; jj < logg->NSV; jj++ ) {
	  fprintf(fwrite_singvecs, "\t%2.13f", RHS2[ii*NRHS+jj]);
      }
      fprintf(fwrite_singvecs, "\n");
    }
    fclose(fwrite_singvecs);
  }   
  //============================================================================
   
  //==========================================                                 
  // Finalize program and deallocate resources 
  //==========================================                   
 
  free(LOC_MAT);
  free(ARHS);
  free(RHS); 
  free(SING_VALUES); 
  free(SING_VALUES_OLD); 
  free(B2);
  free(B2_duplicate);
  free(tau);
  free(readbin);
  free(decbin);
  free(norm_tmp);
  free(norm_precomp);
  delete[] seen_snp;
  start.clear();
  stop.clear(); 
  //==========================================   
}

void benchmarking(std::ifstream& infile, double *RHS2, logistics *logg) {

  //==========================================================                            
  int    M = logg->M, N = logg->N, NRHS = logg->NRHS;
  int    max_iter = logg->blockPower_maxiter, min_dim = min(N,NRHS), powers = logg->power;
  double tt1, tt2;
  int    ione = 1,   converged = 0, ii, jj, kk, ii2, jj2;
  double fone = 1.0, minusfone = -1.0, fzero = 0.0;
  double* ARHS            = (double*) malloc(M*NRHS*sizeof(double));
  double* tau             = (double*) malloc(NRHS*sizeof(double));
  int info_svd_lapacke, info_sgeqrf_lapacke, info_sorgqr_lapacke;
  //================================================================                                          

  //================================================================
  int rows_fetched      = logg->rows_fetched;
  int loops             = N / rows_fetched;
  int remaining_rows    = N - rows_fetched*loops;
  unsigned int ik;
  int      colss        = logg->M;
  double* RHS           = (double*) malloc(max(remaining_rows,rows_fetched)*NRHS*sizeof(double));
  unsigned int np       = (unsigned long long)ceil((double)M/PACK_DENSITY);  //size of the packed data, in bytes, per SNP
  unsigned int actual_block_size=0, startval; 
  double *LOC_MAT;
  unsigned char *decbin  = (unsigned char*)malloc(np*PACK_DENSITY*sizeof(unsigned char*));
  unsigned char *readbin = (unsigned char*)malloc(np*sizeof(unsigned char*));
  double *norm_tmp = (double*)malloc(M*sizeof(double)); 
  //================================================================
  if (remaining_rows > 0) {
    loops = loops + 1;
  } else {
    remaining_rows = rows_fetched;
  }

  vector<int>start(loops);
  vector<int>stop(loops); 

  for(ik = 0 ; ik < loops ; ik++){
    start[ik]= ik * rows_fetched;
    stop[ik] = start[ik] + rows_fetched - 1;
    stop[ik] = stop[ik] >= N ? N - 1 : stop[ik];
  }

  uint64_t lmsize = max(remaining_rows,rows_fetched)*M;
  LOC_MAT = (double*)malloc(lmsize*sizeof(double));
  //================================================================
  double *norm_precomp = (double*)malloc((4*N)*sizeof(double));
  memset(norm_precomp,0,(4*N)*sizeof(double));
  bool *seen_snp = new bool[N]();
  //================================================================

    //====================================
    // Multiply AA' by RHS2 from the right
    //====================================
    infile.seekg(3, std::ifstream::beg);
    memset(ARHS, 0, M*NRHS*sizeof(double));
    for (jj = 0; jj < loops; jj++) {
      // prepare to fetch data from memory
      // pack more bytes than required because it needs to read all of the byte
      if(jj<loops-1){
        actual_block_size = stop[jj] - start[jj] + 1;
      }else{
	actual_block_size = remaining_rows;
      }
      startval = start[jj];
      infile.seekg(3 + np * startval);

      if ( jj == 0 || jj < loops-1 ) {
	// Load chunk of SNPs       
        tt1 = dsecnd();
	Read_Bed_Blocks(infile, np, actual_block_size, LOC_MAT, startval, logg, decbin, readbin, norm_tmp, seen_snp, norm_precomp);
        tt2 = dsecnd() - tt1;
        logg->TIME_2_LOAD_MATRIX = logg->TIME_2_LOAD_MATRIX + tt2;
	
        // Multiply with A'
	tt1 = dsecnd();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_fetched, NRHS, M, fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);	
	tt2 = dsecnd() - tt1;
	logg->TIME_2_MM   = logg->TIME_2_MM + tt2;
	logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
	
        // Multiply with A
	tt1 = dsecnd();
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, rows_fetched, fone, LOC_MAT, M, RHS, NRHS, fone, ARHS, NRHS);
	tt2 = dsecnd() - tt1;
	logg->TIME_2_MM = logg->TIME_2_MM + tt2;
	logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;} 
      else{
        // Load chunk of SNPs
      	tt1 = dsecnd();
      	Read_Bed_Blocks(infile, np, actual_block_size, LOC_MAT, startval, logg, decbin, readbin, norm_tmp, seen_snp, norm_precomp);
        tt2 = dsecnd() - tt1;
      	logg->TIME_2_LOAD_MATRIX = logg->TIME_2_LOAD_MATRIX + tt2;
      	
        // Multiply with A'
      	tt1 = dsecnd();
      	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, remaining_rows, NRHS, M, fone, LOC_MAT, M, RHS2, NRHS, fzero, RHS, NRHS);
      	tt2 = dsecnd() - tt1;
      	logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      	logg->TIME_2_MM_A = logg->TIME_2_MM_A + tt2;
      	
        // Multiply with A
      	tt1 = dsecnd();
      	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, NRHS, remaining_rows, fone, LOC_MAT, M, RHS, NRHS, fone, ARHS, NRHS);
      	tt2 = dsecnd() - tt1;
      	logg->TIME_2_MM = logg->TIME_2_MM + tt2;
      	logg->TIME_2_MM_A_TRANSPOSED = logg->TIME_2_MM_A_TRANSPOSED + tt2;
      }
    }

    //==========================
    // Perform orthogonalization
    //==========================
    tt1 = dsecnd();
    info_sgeqrf_lapacke = LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau );
    info_sorgqr_lapacke = LAPACKE_dorgqr( LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau );
    tt2 = dsecnd() - tt1;
    logg->TIME_2_GS = logg->TIME_2_GS + tt2;
    //==========================

    cout<< "Benchmarking mode" << endl;
    cout<< "-----------------" << endl;
    cout<< "Number of threads used: " << logg->threads << endl;
    cout<< "Number of RHS: " << NRHS << endl;
    cout<< "Number of rows fetched: " << rows_fetched << endl;
    cout<< "Time to load A': " << logg->TIME_2_LOAD_MATRIX <<endl;
    cout<< "Time to perform MM: " << logg->TIME_2_MM <<endl;
    cout<< "Time to perform ORTH: " << logg->TIME_2_GS <<endl;

    //===========
    // Deallocate
    //===========
    free(LOC_MAT);
    free(ARHS);
    free(RHS);
    //===========

}