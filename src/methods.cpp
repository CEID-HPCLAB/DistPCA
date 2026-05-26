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

void SubspaceIteration_MPI(double *MAT, double *RHS2, logistics *logg) {
  
  int M = logg->M, N = logg->N, local_N = logg->local_N; int NRHS = logg->NRHS;
  int rank = logg->mpi_rank, size = logg->mpi_size;
  
  int max_iter = logg->blockPower_maxiter; int powers = logg->power;
  
  int ii, jj, converged = 0, kk, ii2, jj2; double tt1, tt2;
  double fone = 1.0, fzero = 0.0;
  
  double w[NRHS];
  
  double* B    = (double*) malloc(NRHS*NRHS*sizeof(double));
  double* B2   = (double*) malloc(NRHS*NRHS*sizeof(double));
  double* RHS_local  = (double*) malloc(NRHS*local_N*sizeof(double)); 
  double* ARHS = (double*) malloc(NRHS*M*sizeof(double));
  double* ARHS_local = (double*) malloc(NRHS*M*sizeof(double));
  double* SING_VALUES     = (double*) malloc(logg->NSV*sizeof(double));
  double* SING_VALUES_OLD = (double*) malloc(logg->NSV*sizeof(double));
  double* RHS2_old = (double*) malloc(M*NRHS*sizeof(double)); 
  double* tau = (double*) malloc(NRHS*sizeof(double));
  
  if (logg->blockPower_conv_crit == 2)
    memcpy(RHS2_old, RHS2, M*NRHS*sizeof(double));
  
  int info_sgeqrf_lapacke, info_sorgqr_lapacke;
  logg->delta_iter.resize(max_iter); logg->sing_values.resize(logg->NSV); logg->left_sing_vecs.resize(M*logg->NSV);

  for (ii = 0; ii < max_iter; ii++) {
    tt1 = dsecnd();
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                local_N, NRHS, M, fone, MAT, M, RHS2, NRHS, fzero, RHS_local, NRHS);
    
    tt2 = dsecnd() - tt1;
    
    logg->TIME_2_MM += tt2;
    logg->TIME_2_MM_A += tt2;
    
    tt1 = dsecnd();
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                M, NRHS, local_N, fone, MAT, M, RHS_local, NRHS, fzero, ARHS_local, NRHS);
    
    tt2 = dsecnd() - tt1;
    
    logg->TIME_2_MM += tt2;
    logg->TIME_2_MM_A_TRANSPOSED += tt2;

    tt1 = dsecnd();
    
    // Reduce partial results across all MPI processes to obtain the final C
    MPI_Allreduce(ARHS_local, ARHS, M*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    tt2 = dsecnd() - tt1;
    
    logg->TIME_2_MM += tt2;
  
    for (kk = 0; kk < powers - 1; kk++) {
      tt1 = dsecnd();

      info_sgeqrf_lapacke = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau); // QR factorization of C
      info_sorgqr_lapacke = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau); // Obtain Q 
      
      tt2 = dsecnd() - tt1;
      logg->TIME_2_GS += tt2;
      
      memcpy(RHS2, ARHS, M * NRHS * sizeof(double));
      
      tt1 = dsecnd();
      
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                  local_N, NRHS, M, fone, MAT, M, RHS2, NRHS, fzero, RHS_local, NRHS);
      
      tt2 = dsecnd() - tt1;
      
      logg->TIME_2_MM += tt2;
      logg->TIME_2_MM_A += tt2;

      tt1 = dsecnd();

      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                  M, NRHS, local_N, fone, MAT, M, RHS_local, NRHS, fzero, ARHS_local, NRHS);

      tt2 = dsecnd() - tt1;
      logg->TIME_2_MM += tt2;

      tt1 = dsecnd();
      
      // Reduce partial results across all MPI processes to obtain the final C
      MPI_Allreduce(ARHS_local, ARHS, M*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      
      tt2 = dsecnd() - tt1;
      logg->TIME_2_MM += tt2;

    }

    tt1 = dsecnd();
    
    info_sgeqrf_lapacke = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau); // QR factorization of C
    info_sorgqr_lapacke = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau); // Obtain Q
    
    tt2 = dsecnd() - tt1;
    logg->TIME_2_GS += tt2;
    
    tt1 = dsecnd();
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                local_N, NRHS, M, fone, MAT, M, ARHS, NRHS, fzero, RHS_local, NRHS);
                
    tt2 = dsecnd() - tt1;
    
    logg->TIME_2_MM += tt2;
    logg->TIME_2_MM_A += tt2;
    
    tt1 = dsecnd();
    
    double* B_local = (double*) malloc(NRHS * NRHS * sizeof(double));

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                NRHS, NRHS, local_N, fone, RHS_local, NRHS, RHS_local, NRHS, fzero, B_local, NRHS);
    
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM += tt2;
    logg->TIME_2_MM_A_TRANSPOSED += tt2;

    tt1 = dsecnd();
    
    MPI_Allreduce(B_local, B, NRHS*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM += tt2;

    free(B_local);
   
    tt1 = dsecnd();
    
    for (ii2 = 0; ii2 < NRHS; ii2++) {
      for (jj2 = 0; jj2 < NRHS; jj2++) {
        if (jj2 < ii2)
          B[ii2*NRHS + jj2] = 0.0;
      }
    }
 
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NRHS, B, NRHS, w);
    
    tt2 = dsecnd() - tt1;
    logg->TIME_2_PROJECTED_SVD += tt2;

    memcpy(B2, B, sizeof(double)*NRHS*NRHS);
    for (ii2 = 0; ii2 < NRHS; ii2++) {
      for (jj2 = 0; jj2 < NRHS; jj2++)
        B[ii2 * NRHS + jj2] = B2[ii2 * NRHS + (NRHS - 1 - jj2)];
    }
  
    tt1 = dsecnd();

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, NRHS, NRHS, fone, ARHS, NRHS, B, NRHS, fzero, RHS2, NRHS);
    
    tt2 = dsecnd() - tt1;
    logg->TIME_2_MM += tt2;

    logg->delta_iter[ii] = 0.0;
    
    for (jj2 = 0; jj2 < logg->NSV; jj2++) {
      SING_VALUES[jj2] = sqrt(w[NRHS - 1 - jj2]);
      logg->sing_values[jj2] = SING_VALUES[jj2];
      
      logg->delta_iter[ii] = logg->delta_iter[ii] + logg->sing_values[jj2];
    }
    
    if (logg->blockPower_conv_crit == 0) {
      // MODE 0: Trace-based criterion
      if (ii > 0) {
        logg->blockPower_trace_error = fabs(logg->delta_iter[ii - 1] - logg->delta_iter[ii]) / logg->delta_iter[ii];
        
        if (rank == 0 && logg->PRINT_INFO > 1) 
          cout << "Iteration " << ii << ": rel. error: "
                << fixed << setprecision(13)
                << logg->blockPower_trace_error << endl;
        
        if (logg->blockPower_trace_error <= logg->toll)
          break;

      }
      
    } else if (logg->blockPower_conv_crit == 1) {
      // MODE 1: Individual eigenvalue criterion
      if (ii > 0) {
        converged = 0;
        
        for (jj = 0; jj < logg->NSV; jj++) {
          if (fabs((SING_VALUES[jj]-SING_VALUES_OLD[jj])/SING_VALUES[jj]) <= logg->toll)
            converged++;
        }

        if (rank == 0 && logg->PRINT_INFO > 1)
          cout << "Iteration " << ii << ": " << converged << " converged" << endl;
        
        if (converged == logg->NSV)
          break;
      }
      
    } else if (logg->blockPower_conv_crit == 2) {
      // MODE 2: MEV (Mean Explained Variance) criterion
      if (ii > 0) {
        double MEV = 0.0;
        
        for (jj = 0; jj < logg->NSV; jj++) {
          double dot_product = 0.0;
          for (ii2 = 0; ii2 < M; ii2++)
            dot_product += RHS2_old[ii2*NRHS + jj] * RHS2[ii2*NRHS + jj];
        
          MEV += dot_product * dot_product; 
        }
        
        MEV /= logg->NSV; double mev_error = 1.0 - MEV;
        
        if (rank == 0 && logg->PRINT_INFO > 1) 
          cout << "Iteration " << ii << ": MEV error: "
                << fixed << setprecision(13)
                << mev_error << endl;
      
        if (mev_error <= logg->toll)
          break;
      }
    }

    converged = 0;
    for (jj = 0; jj < logg->NSV; jj++)
      SING_VALUES_OLD[jj] = SING_VALUES[jj];
    
    if (logg->blockPower_conv_crit == 2)
      memcpy(RHS2_old, RHS2, M*NRHS*sizeof(double));

  }

  logg->blockPower_total_its = (ii < max_iter) ? ii+1 : ii;
  
  for (ii = 0; ii < M; ii++) {
    for (jj = 0; jj < NRHS; jj++) {
      if (jj < logg->NSV)
        logg->left_sing_vecs[ii*logg->NSV+jj] = RHS2[ii*NRHS+jj];
    }
  }

  if (rank == 0 && logg->filewrite == 1) {
    string tempname;
    if (logg->prefixname.empty())
      tempname = ConstructFilename(*logg, "singularValues");
    
    else
      tempname = logg->prefixname + "_singularValues.txt";

    FILE *fwrite_singvalues = fopen(tempname.c_str(), "w");

    std::vector<double> singularvals = logg->sing_values;
    // std::transform(singularvals.begin(), singularvals.end(), singularvals.begin(), computeSquare);
    std::vector<string> individs = logg->indiv_ids;

    for (ii = 0; ii < logg->NSV; ii++)
      fprintf(fwrite_singvalues, "%2.13lf\n", singularvals[ii]);
    
    fclose(fwrite_singvalues);

    if (logg->prefixname.empty())
      tempname = ConstructFilename(*logg, "singularVectors");
    else
      tempname = logg->prefixname + "_singularVectors.txt";
      
    FILE *fwrite_singvecs = fopen(tempname.c_str(), "w");
    if (fwrite_singvecs == NULL) {
      cout << "Unable to write to file. Aborting..." << endl;
      
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

  free(ARHS); free(ARHS_local); free(RHS_local);
  free(B2); free(B);
  free(SING_VALUES); free(SING_VALUES_OLD);
  free(tau);

  if (logg->blockPower_conv_crit == 2)
    free(RHS2_old);
}

void BlockSubspaceIter_MPI_OOC(const char* bedfile, double *RHS2, logistics *logg) {
    int M = logg->M, N = logg->N, NRHS = logg->NRHS;
    int local_N = logg->local_N; int local_N_start = logg->local_N_start;
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

    int rows_fetched = logg->rows_fetched; int local_loops = local_N / rows_fetched;
    int remaining_rows = local_N - rows_fetched * local_loops;

    double inv_sqrtN = 1.0 / sqrt((double)N); double inv_sqrtM = 1.0 / sqrt((double)M);

    double* ARHS            = (double*) malloc(M * NRHS * sizeof(double));
    double* ARHS_local      = (double*) malloc(M * NRHS * sizeof(double));
    double* SING_VALUES     = (double*) malloc(min_dim * sizeof(double));
    double* SING_VALUES_OLD = (double*) malloc(min_dim * sizeof(double));
    double* tau             = (double*) malloc(NRHS * sizeof(double));
    double* RHS2_old        = (double*) malloc(M * NRHS * sizeof(double));
    double* RHS_buf         = (double*) malloc(rows_fetched * NRHS * sizeof(double));

    logg->sing_values.resize(logg->NSV); logg->delta_iter.resize(max_iter); logg->left_sing_vecs.resize(M * logg->NSV);
    
    if (remaining_rows > 0)
      local_loops++;

    else
      remaining_rows = rows_fetched;

    std::vector<int> start(local_loops); std::vector<int> stop(local_loops);
    
    for(int ik = 0; ik < local_loops; ik++){
        start[ik] = local_N_start + ik * rows_fetched;
        stop[ik] = start[ik] + rows_fetched - 1;
        
        if (stop[ik] >= local_N_start + local_N) 
            stop[ik] = local_N_start + local_N - 1;
    }

    uint64_t np = (uint64_t)ceil((double)M / PACK_DENSITY); // total number of bytes per row in the .bed file

    double* B2 = (double*) malloc(NRHS * NRHS * sizeof(double)); double* B2_duplicate = (double*) malloc(NRHS * NRHS * sizeof(double));
    
    uint64_t lmsize = (uint64_t)rows_fetched * (uint64_t)M;
    double* LOC_MAT = (double*)malloc(lmsize * sizeof(double)); // buffer to store the uncompressed and standardized genotypes for each fetched block

    unsigned char *decbin = (unsigned char*)malloc(np * PACK_DENSITY * sizeof(unsigned char)); // store uncompressed genotypes of a block
    
    // store precomputed standardized values for each possible genotype (0, 1, 2, missing) for each SNP to avoid redundant computations across iterations
    double *norm_precomp = (double*)calloc(4 * N, sizeof(double)); 
    double *norm_tmp = (double*)malloc(M * sizeof(double));
    
    bool *seen_snp = new bool[N](); // indicate whether standardization values for a SNP have been computed in previous iter

    std::vector<unsigned char> block_buf;
    block_buf.resize(rows_fetched * np); // buffer to store the compressed genotypes for each fetched block

    auto _compute_matvec_buff = [&]() {
        memset(ARHS_local, 0, M*NRHS*sizeof(double));

        uint64_t file_offset; int actual_block_size; 
        int global_idx;

        double avg, sd; // for standardization

        unsigned char* snp_bin; // pointer to the current SNP's compressed genotype data in the block buffer
      
        for(int jj = 0; jj < local_loops; jj++) {
            actual_block_size = (jj == local_loops - 1) ? remaining_rows : (stop[jj] - start[jj] + 1);
            
            file_offset = 3 + np * start[jj]; // byte offset in the .bed file for the starting byte of the current block
            
            tt1 = dsecnd();
            
            MPI_Status status;
            MPI_File_read_at(fh, file_offset, block_buf.data(), np * actual_block_size, MPI_UNSIGNED_CHAR, &status);
            
            tt2 = dsecnd() - tt1;
            logg->TIME_2_LOAD_MATRIX += tt2;
            
            tt1 = dsecnd();
            
            for(int row = 0; row < actual_block_size; row++) {
                snp_bin = block_buf.data() + row * np;
                
                global_idx = start[jj] + row;
                
                if(!seen_snp[global_idx]) {
                    decode_plink_sse2(decbin, snp_bin, np);
                    
                    standardize(norm_tmp, decbin, M, avg, sd, inv_sqrtN);
                    
                    if(sd > 1e-9) {
                        norm_precomp[3+(global_idx*4)] = (0 - avg) / sd;
                        norm_precomp[2+(global_idx*4)] = (1 - avg) / sd;
                        norm_precomp[0+(global_idx*4)] = (2 - avg) / sd;
                        norm_precomp[1+(global_idx*4)] = 0;
                    }
                    seen_snp[global_idx] = true;
                    
                    for(int k = 0; k < M; k++) 
                        LOC_MAT[row*M+k] = norm_tmp[k];
                
                } else {
                    decode_plink_precomp_sse2(decbin, snp_bin, np);
                    
                    for(int k = 0; k < M; k++) {
                        int b = (int)decbin[k];
                        LOC_MAT[row*M+k] = norm_precomp[b+(global_idx*4)] * inv_sqrtN;
                    }
                }
            }
            
            tt2 = dsecnd() - tt1;
            logg->TIME_2_LOAD_MATRIX += tt2;
            
            tt1 = dsecnd();
            
            // A^T @ C for the current block
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, actual_block_size, NRHS, M,
                      1.0, LOC_MAT, M, RHS2, NRHS, 0.0, RHS_buf, NRHS);
            
            tt2 = dsecnd() - tt1;
            
            logg->TIME_2_MM += tt2;
            logg->TIME_2_MM_A += tt2;
            
            tt1 = dsecnd();
            
            // A @ (A^T @ C) for the current block
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                      M, NRHS, actual_block_size,
                      1.0, LOC_MAT, M, RHS_buf, NRHS, 1.0, ARHS_local, NRHS);
            
            tt2 = dsecnd() - tt1;
            
            logg->TIME_2_MM += tt2;
            logg->TIME_2_MM_A_TRANSPOSED += tt2;
        }
        
        tt1 = dsecnd();
        
        // Reduce partial results across all MPI processes to obtain the final MMV result
        MPI_Allreduce(ARHS_local, ARHS, M*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;
    };

    if (logg->blockPower_conv_crit == 2)
        memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));

    // Added on 14/12/25: before getting in the main loop, open the .csv file to log the eigenvalues per iteration
    // Get current date and time for filename prefix
    // time_t now = time(0);
    // struct tm* timeinfo = localtime(&now);
    // char datetime_prefix[32];
    // strftime(datetime_prefix, sizeof(datetime_prefix), "%d_%m_%Y_%H:%M:%S", timeinfo);
    
    // FILE *eigenval_tracking_file = NULL;
    // if (rank == 0 && save_eigenval_tracking == 1) {
    //   std::string tracking_filename = std::string(datetime_prefix) + "_eigenvalue_tracking.csv";
    //   eigenval_tracking_file = fopen(tracking_filename.c_str(), "w");
    //   if (eigenval_tracking_file == NULL) {
    //     perror("fopen eigenvalue_tracking");
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    //   }
      
    //   // write header
    //   fprintf(eigenval_tracking_file, "iteration");
    //   for (jj = 0; jj < logg->NSV; jj++) {
    //     fprintf(eigenval_tracking_file, ",eigenvalue_%d,rel_change_%d", jj+1, jj+1);
    //   }
    //   fprintf(eigenval_tracking_file, "\n");
    // }

    // Main power iteration loop
    for (ii = 0; ii < max_iter; ii++) {
        _compute_matvec_buff();
        
        for (kk = 0; kk < powers - 1; kk++) {
            tt1 = dsecnd();
            
            // QR factorization of C to get the orthonormal basis Q for the subspace iter
            LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);

            // Generate the orthonormal matrix Q from the output of dgeqrf
            LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
            
            tt2 = dsecnd() - tt1;
            logg->TIME_2_GS += tt2;
            
            memcpy(RHS2, ARHS, M * NRHS * sizeof(double));

            // A @ (A^T @ Q) to get the updated C for the next power iteration
            _compute_matvec_buff();
        }
        
        // At least one power iteration is performed 
        tt1 = dsecnd();
        
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
        
        tt2 = dsecnd() - tt1;
        logg->TIME_2_GS += tt2;

        memcpy(RHS2, ARHS, M * NRHS * sizeof(double));
        _compute_matvec_buff(); 

        tt1 = dsecnd();
        
        // M = Q^T @ (A @ (A^T @ Q))
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, NRHS, NRHS, M, 
                    fone, RHS2, NRHS, ARHS, NRHS, fzero, B2, NRHS);
        
        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;

        tt1 = dsecnd();
        
        // Extract the upper triangular part of the projected matrix M 
        for (ii2 = 0; ii2 < NRHS; ii2++) {
            for (jj2 = 0; jj2 < ii2; jj2++) 
                B2[ii2 * NRHS + jj2] = 0.0;
        }
        
        double* w = (double*) malloc(NRHS * sizeof(double));

        // Eigen decomposition of the projected matrix M
        LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NRHS, B2, NRHS, w);
        
        tt2 = dsecnd() - tt1;
        logg->TIME_2_PROJECTED_SVD += tt2;

        memcpy(B2_duplicate, B2, NRHS * NRHS * sizeof(double));
        
        // eigenvals and eigenvecs in descending order
        for (ii2 = 0; ii2 < NRHS; ii2++) {
            for (jj2 = 0; jj2 < NRHS; jj2++) {
                B2[ii2 * NRHS + jj2] = B2_duplicate[ii2 * NRHS + (NRHS - 1 - jj2)];
            }
        }
        
        memcpy(ARHS, RHS2, M * NRHS * sizeof(double));
        
        tt1 = dsecnd();
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, NRHS, NRHS, 
                    fone, ARHS, NRHS, B2, NRHS, fzero, RHS2, NRHS);
        
        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;
        
        logg->delta_iter[ii] = 0.0;
        
        for (jj = 0; jj < logg->NSV; jj++) {
            SING_VALUES[jj] = sqrt(w[NRHS - 1 - jj]);
            logg->sing_values[jj] = SING_VALUES[jj];
            
            logg->delta_iter[ii] += logg->sing_values[jj];
        }
        
        // // Eigenvalue tracking
        // if (rank == 0 && eigenval_tracking_file != NULL) {
        //     fprintf(eigenval_tracking_file, "%d", ii);
        //     for (jj = 0; jj < logg->NSV; jj++) {
        //         double eigenval = SING_VALUES[jj] * SING_VALUES[jj];
        //         if (ii > 0) {
        //             double eigenval_old = SING_VALUES_OLD[jj] * SING_VALUES_OLD[jj];
        //             double rel_change = fabs(eigenval - eigenval_old) / eigenval;
        //             fprintf(eigenval_tracking_file, ",%.13e,%.6e", eigenval, rel_change);
        //         } else {
        //             fprintf(eigenval_tracking_file, ",%.13e,NA", eigenval);
        //         }
        //     }
        //     fprintf(eigenval_tracking_file, "\n");
        //     fflush(eigenval_tracking_file);
        // }
        
        if (ii > 0) {
            // Trace-based criterion (Mode 0), Individual eigenvalue criterion (Mode 1), or MEV criterion (Mode 2) 
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
                
                if (rank == 0 && logg->PRINT_INFO > 1)
                    printf("Iteration %d: MEV error: %02.13f\n", ii, mev_error);
                
                if (mev_error <= logg->toll) break;
            }
        }
        
        for (jj = 0; jj < logg->NSV; jj++) 
            SING_VALUES_OLD[jj] = SING_VALUES[jj];

        if (logg->blockPower_conv_crit == 2) {
            memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));
        }

        free(w);
    }
    
    // // Added on 14/12/25: close the eigenvalue tracking file + final eigenvalues
    // if (rank == 0 && eigenval_tracking_file != NULL) {
    //     // empty line for gap
    //     fprintf(eigenval_tracking_file, "\n");
        
    //     // final approximate eigenvalues section
    //     fprintf(eigenval_tracking_file, "# Final Approximate Eigenvalues\n");
    //     fprintf(eigenval_tracking_file, "eigenvalue_index,final_value\n");
        
    //     for (jj = 0; jj < logg->NSV; jj++) {
    //         double final_eigenval = logg->sing_values[jj] * logg->sing_values[jj];
    //         fprintf(eigenval_tracking_file, "%d,%.13e\n", jj+1, final_eigenval);
    //     }
        
    //     fclose(eigenval_tracking_file);
    // }

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
        // std::transform(singularvals.begin(), singularvals.end(), singularvals.begin(), computeSquare);
        std::vector<std::string> individ = logg->indiv_ids;

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

    free(LOC_MAT);
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

void BlockSubspaceIter_MPI_OOC_double_buffering(const char* bedfile, double *RHS2, logistics *logg) {
    int M = logg->M, N = logg->N, NRHS = logg->NRHS;
    int local_N = logg->local_N; int local_N_start = logg->local_N_start;
    int max_iter = logg->blockPower_maxiter, min_dim = min(N,NRHS), powers = logg->power;
    
    double tt1, tt2; int converged = 0, ii, jj, kk, ii2, jj2;
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

    int rows_fetched = logg->rows_fetched; int local_loops = local_N / rows_fetched;
    int remaining_rows = local_N - rows_fetched * local_loops;

    double inv_sqrtN = 1.0 / sqrt((double)N); double inv_sqrtM = 1.0 / sqrt((double)M);

    double* ARHS            = (double*) malloc(M * NRHS * sizeof(double));
    double* ARHS_local      = (double*) malloc(M * NRHS * sizeof(double));
    double* SING_VALUES     = (double*) malloc(min_dim * sizeof(double));
    double* SING_VALUES_OLD = (double*) malloc(min_dim * sizeof(double));
    double* tau             = (double*) malloc(NRHS * sizeof(double));
    double* RHS2_old        = (double*) malloc(M * NRHS * sizeof(double));
    
    if (logg->blockPower_conv_crit == 2)
      memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));
    
    double* RHS_buf         = (double*) malloc(rows_fetched * NRHS * sizeof(double));

    logg->sing_values.resize(logg->NSV); logg->delta_iter.resize(max_iter); logg->left_sing_vecs.resize(M * logg->NSV);
    
    if (remaining_rows > 0)
      local_loops++;
    
    else
      remaining_rows = rows_fetched;

    std::vector<int> start(local_loops); std::vector<int> stop(local_loops);
    
    for(int ik = 0; ik < local_loops; ik++){
        start[ik] = local_N_start + ik * rows_fetched; stop[ik] = start[ik] + rows_fetched - 1;
        
        if (stop[ik] >= local_N_start + local_N) 
            stop[ik] = local_N_start + local_N - 1;
    }

    uint64_t np = (uint64_t)ceil((double)M / PACK_DENSITY); // total number of bytes per row in the .bed file

    double* B2 = (double*) malloc(NRHS * NRHS * sizeof(double)); double* B2_duplicate = (double*) malloc(NRHS * NRHS * sizeof(double));
    
    uint64_t lmsize = (uint64_t)rows_fetched * (uint64_t)M; // size of the local block when uncompressed
    
    double* LOC_MAT[2]; // double buffering: LOC_MAT[0] for current block, LOC_MAT[1] for next block
    LOC_MAT[0] = (double*)malloc(lmsize * sizeof(double)); LOC_MAT[1] = (double*)malloc(lmsize * sizeof(double));

    std::vector<std::vector<unsigned char>> block_buf(2); // buffers
    MPI_Request reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL }; // for double buffering of non-blocking I/O

    for(int i = 0; i < 2; i++) block_buf[i].resize(rows_fetched * np);
    
    unsigned char *decbin = (unsigned char*)malloc(np * PACK_DENSITY * sizeof(unsigned char)); // store uncompressed genotypes of a block
    
    // store precomputed standardized values for each possible genotype (0, 1, 2, missing) for each SNP to avoid redundant computations across iterations
    double *norm_precomp = (double*)calloc(4 * N, sizeof(double)); 
    double *norm_tmp = (double*)malloc(M * sizeof(double));
    
    bool *seen_snp = new bool[N](); // indicate whether standardization values for a SNP have been computed in previous iter

    double avg, sd; // for standardization

    // Lambda function to compute A @ RHS2 and A' @ (A @ RHS2) for the current block, with double buffering for the next block
    auto _compute_matvec_buff = [&]() {
        memset(ARHS_local, 0, M*NRHS*sizeof(double));

        int cur = 0; int prev;
        uint64_t file_offset; int actual_block_size; 
        int global_idx;

        unsigned char* snp_bin; // pointer to the current SNP's compressed genotype data in the block buffer
        
        for(int jj = 0; jj < local_loops; jj++) {
            actual_block_size = (jj == local_loops - 1) ? remaining_rows : (stop[jj] - start[jj] + 1);
            
            file_offset = 3 + np * start[jj]; // byte offset in the .bed file for the starting byte of the current block
      
            tt1 = dsecnd();
            
            // Async read for the current block 
            MPI_File_iread_at(fh, file_offset, block_buf[cur].data(), np * actual_block_size, MPI_UNSIGNED_CHAR, &reqs[cur]);
            
            tt2 = dsecnd() - tt1;
            logg->TIME_2_LOAD_MATRIX += tt2;

            if (jj > 0) {
              prev = 1 - cur;
              tt1 = dsecnd();
              MPI_Wait(&reqs[prev], MPI_STATUS_IGNORE); // Wait for the previous block to be loaded before processing
              
              // Preprocess the previous block
              for(int row = 0; row < rows_fetched; row++) {
                snp_bin = block_buf[prev].data() + row * np;
                
                global_idx = start[jj - 1] + row;
                
                if(!seen_snp[global_idx]) {
                    decode_plink_sse2(decbin, snp_bin, np);
                    
                    standardize(norm_tmp, decbin, M, avg, sd, inv_sqrtN);
                    
                    if(sd > 1e-9) {
                        // Store precomputed standardized values for genotypes 0, 1, 2, and missing (encoded as 3) for current SNP
                        norm_precomp[3 + (global_idx * 4)] = (0 - avg)/sd;
                        norm_precomp[2 + (global_idx * 4)] = (1 - avg)/sd;
                        norm_precomp[0 + (global_idx * 4)] = (2 - avg)/sd;
                        norm_precomp[1 + (global_idx * 4)] = 0;
                    }
                    
                    seen_snp[global_idx] = true;
                    
                    for(int k = 0; k < M; k++) 
                        LOC_MAT[prev][row*M+k] = norm_tmp[k];

                  } else {
                      decode_plink_precomp_sse2(decbin, snp_bin, np);
                      
                      for(int k = 0; k < M; k++) {
                          int b = (int)decbin[k];
                          LOC_MAT[prev][row*M + k] = norm_precomp[b + (global_idx * 4)] * inv_sqrtN;
                      }
                  }
              }
              
              tt2 = dsecnd() - tt1;
              logg->TIME_2_LOAD_MATRIX += tt2;
            
            
              tt1 = dsecnd();
              // A^T @ C for the previous block
              cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_fetched, NRHS, M,
                          1.0, LOC_MAT[prev], M, RHS2, NRHS, 0.0, RHS_buf, NRHS);
              tt2 = dsecnd() - tt1;
              
              logg->TIME_2_MM += tt2;
              logg->TIME_2_MM_A += tt2;
              
              tt1 = dsecnd();
              // A^T @ (A @ C) for the previous block
              cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,M, NRHS, rows_fetched,
                        1.0, LOC_MAT[prev], M, RHS_buf, NRHS, 1.0, ARHS_local, NRHS);
              tt2 = dsecnd() - tt1;
              
              logg->TIME_2_MM += tt2;
              logg->TIME_2_MM_A_TRANSPOSED += tt2;
          }

          cur = 1 - cur;
        }

        int last = 1 - cur;
        MPI_Wait(&reqs[last], MPI_STATUS_IGNORE); // Wait for the last block to be loaded

        for(int row = 0; row < remaining_rows; row++) {
            snp_bin = block_buf[last].data() + row * np;
            global_idx = start[local_loops-1] + row;

            if(!seen_snp[global_idx]) {
                decode_plink_sse2(decbin, snp_bin, np);
                
                standardize(norm_tmp, decbin, M, avg, sd, inv_sqrtN);
                
                if(sd > 1e-9) {
                    norm_precomp[3+(global_idx*4)] = (0-avg)/sd;
                    norm_precomp[2+(global_idx*4)] = (1-avg)/sd;
                    norm_precomp[0+(global_idx*4)] = (2-avg)/sd;
                    norm_precomp[1+(global_idx*4)] = 0;
                }
                
                seen_snp[global_idx] = true;
                
                for(int k = 0; k < M; k++) 
                    LOC_MAT[last][row*M+k] = norm_tmp[k];
            
            } else {
                decode_plink_precomp_sse2(decbin, snp_bin, np);
                
                for(int k = 0; k < M; k++) {
                    int b = (int)decbin[k];
                    LOC_MAT[last][row*M+k] = norm_precomp[b+(global_idx*4)] * inv_sqrtN;
                }
            }
        }
        
        tt1 = dsecnd();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, remaining_rows, NRHS, M,
                  1.0, LOC_MAT[last], M, RHS2, NRHS, 0.0, RHS_buf, NRHS);
        tt2 = dsecnd() - tt1;
        
        logg->TIME_2_MM += tt2;
        logg->TIME_2_MM_A += tt2;
        
        tt1 = dsecnd();
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,M, NRHS, remaining_rows,
                  1.0, LOC_MAT[last], M, RHS_buf, NRHS, 1.0, ARHS_local, NRHS);
        tt2 = dsecnd() - tt1;
        
        logg->TIME_2_MM += tt2;
        logg->TIME_2_MM_A_TRANSPOSED += tt2;
        
        tt1 = dsecnd();
        
        // Reduce partial results across all MPI processes to obtain the final MMV result
        MPI_Allreduce(ARHS_local, ARHS, M*NRHS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;
    };

    if (logg->blockPower_conv_crit == 2)
        memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));


    // // Added on 14/12/25: before getting in the main loop, open the .csv file to log the eigenvalues per iteration
    // // Get current date and time for filename prefix
    // time_t now = time(0);
    // struct tm* timeinfo = localtime(&now);
    // char datetime_prefix[32];
    // strftime(datetime_prefix, sizeof(datetime_prefix), "%d_%m_%Y_%H:%M:%S", timeinfo);
    
    // FILE *eigenval_tracking_file = NULL;
    // if (rank == 0 && save_eigenval_tracking == 1) {
    //   std::string tracking_filename = std::string(datetime_prefix) + "_eigenvalue_tracking.csv";
    //   eigenval_tracking_file = fopen(tracking_filename.c_str(), "w");
    //   if (eigenval_tracking_file == NULL) {
    //     perror("fopen eigenvalue_tracking");
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    //   }
      
    //   // write header
    //   fprintf(eigenval_tracking_file, "iteration");
    //   for (jj = 0; jj < logg->NSV; jj++) {
    //     fprintf(eigenval_tracking_file, ",eigenvalue_%d,rel_change_%d", jj+1, jj+1);
    //   }
    //   fprintf(eigenval_tracking_file, "\n");
    // }

    for (ii = 0; ii < max_iter; ii++) {
        _compute_matvec_buff();
        
        for (kk = 0; kk < powers - 1; kk++) {
            tt1 = dsecnd();

            LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau); // QR factorization of C
            LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau); // Gen orthonormal Q

            tt2 = dsecnd() - tt1;
            logg->TIME_2_GS += tt2;
            
            memcpy(RHS2, ARHS, M * NRHS * sizeof(double));
            _compute_matvec_buff(); // A @ (A^T @ Q) to get the updated C for the next power iteration
        }

        tt1 = dsecnd();
        
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M, NRHS, ARHS, NRHS, tau);
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M, NRHS, NRHS, ARHS, NRHS, tau);
        
        tt2 = dsecnd() - tt1;
        
        logg->TIME_2_GS += tt2;

        memcpy(RHS2, ARHS, M * NRHS * sizeof(double));
        _compute_matvec_buff();

        tt1 = dsecnd();
        // M = Q^T @ (A @ (A^T @ Q))
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    NRHS, NRHS, M, fone, RHS2, NRHS, ARHS, NRHS, fzero, B2, NRHS);
        tt2 = dsecnd() - tt1;
        
        logg->TIME_2_MM += tt2;

        tt1 = dsecnd();

        // Extract the upper triangular part of the projected matrix M
        for (ii2 = 0; ii2 < NRHS; ii2++) {
            for (jj2 = 0; jj2 < ii2; jj2++) 
                B2[ii2 * NRHS + jj2] = 0.0;
        }
        
        double* w = (double*) malloc(NRHS * sizeof(double));

        // Eigen decomp M
        LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NRHS, B2, NRHS, w);
        tt2 = dsecnd() - tt1;
        
        logg->TIME_2_PROJECTED_SVD += tt2;

        memcpy(B2_duplicate, B2, NRHS * NRHS * sizeof(double));

        // eigenvals && eigenvecs in descending order
        for (ii2 = 0; ii2 < NRHS; ii2++) {
            for (jj2 = 0; jj2 < NRHS; jj2++)
                B2[ii2 * NRHS + jj2] = B2_duplicate[ii2 * NRHS + (NRHS - 1 - jj2)];
        }
        
        memcpy(ARHS, RHS2, M * NRHS * sizeof(double));
        tt1 = dsecnd();
    
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, NRHS, NRHS, fone, ARHS, NRHS, B2, NRHS, fzero, RHS2, NRHS);
        
        tt2 = dsecnd() - tt1;
        logg->TIME_2_MM += tt2;
        
        logg->delta_iter[ii] = 0.0;
        
        for (jj = 0; jj < logg->NSV; jj++) {
            SING_VALUES[jj] = sqrt(w[NRHS - 1 - jj]);
            logg->sing_values[jj] = SING_VALUES[jj];
            
            logg->delta_iter[ii] += logg->sing_values[jj];
        }
        
        // Eigenvalue tracking
        // if (rank == 0 && eigenval_tracking_file != NULL) {
        //     fprintf(eigenval_tracking_file, "%d", ii);
        //     for (jj = 0; jj < logg->NSV; jj++) {
        //         double eigenval = SING_VALUES[jj] * SING_VALUES[jj];
        //         if (ii > 0) {
        //             double eigenval_old = SING_VALUES_OLD[jj] * SING_VALUES_OLD[jj];
        //             double rel_change = fabs(eigenval - eigenval_old) / eigenval;
        //             fprintf(eigenval_tracking_file, ",%.13e,%.6e", eigenval, rel_change);
        //         } else {
        //             fprintf(eigenval_tracking_file, ",%.13e,NA", eigenval);
        //         }
        //     }
        //     fprintf(eigenval_tracking_file, "\n");
        //     fflush(eigenval_tracking_file);
        // }
        
        if (ii > 0) {
          // Trace-based criterion (Mode 0), Individual eigenvalue criterion (Mode 1), or MEV criterion (Mode 2)
            if (logg->blockPower_conv_crit == 0) {
                logg->blockPower_trace_error = fabs(logg->delta_iter[ii-1] - logg->delta_iter[ii]) / logg->delta_iter[ii];
                if (rank == 0 && logg->PRINT_INFO > 1)
                    printf("Iteration %d: rel. error: %02.13f\n", ii, logg->blockPower_trace_error);

                if (logg->blockPower_trace_error <= logg->toll) break;
                
            } else if (logg->blockPower_conv_crit == 1) {
                converged = 0;
                for (jj = 0; jj < logg->NSV; jj++) 
                    if (fabs((SING_VALUES[jj] - SING_VALUES_OLD[jj]) / SING_VALUES[jj]) <= logg->toll)
                        converged++;
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
                
                if (rank == 0 && logg->PRINT_INFO > 1)
                    printf("Iteration %d: MEV error: %02.13f\n", ii, mev_error);
                
                if (mev_error <= logg->toll) break;
            }
        }
        
        for (jj = 0; jj < logg->NSV; jj++) 
            SING_VALUES_OLD[jj] = SING_VALUES[jj];

        if (logg->blockPower_conv_crit == 2)
            memcpy(RHS2_old, RHS2, M * NRHS * sizeof(double));

        free(w);
    }
    
    // Added on 14/12/25: close the eigenvalue tracking file + final eigenvalues
    // if (rank == 0 && eigenval_tracking_file != NULL) {
    //     // empty line for gap
    //     fprintf(eigenval_tracking_file, "\n");
        
    //     // final approximate eigenvalues section
    //     fprintf(eigenval_tracking_file, "# Final Approximate Eigenvalues\n");
    //     fprintf(eigenval_tracking_file, "eigenvalue_index,final_value\n");
        
    //     for (jj = 0; jj < logg->NSV; jj++) {
    //         double final_eigenval = logg->sing_values[jj] * logg->sing_values[jj];
    //         fprintf(eigenval_tracking_file, "%d,%.13e\n", jj+1, final_eigenval);
    //     }
        
    //     fclose(eigenval_tracking_file);
    // }

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
        // std::transform(singularvals.begin(), singularvals.end(), singularvals.begin(), computeSquare);
        std::vector<std::string> individ = logg->indiv_ids;

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