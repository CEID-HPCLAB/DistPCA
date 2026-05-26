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
#include <vector>
#include "structures.h"
#include "utilities.h"
#include "methods.h"
#include "gaussian.h"
#include "omp.h"
#include "gennorm.h"
#include "io.h"
#include <mpi.h>

#if defined(__x86_64__) || defined(_M_X64)
    #include "mkl.h"
    #include "mkl_lapacke.h"
#elif defined(__aarch64__) || defined(__arm__) || defined(__ARM_ARCH) || defined(arm64)
    #include <cblas.h>
    #include <lapacke.h>
#else
    #error "Unsupported architecture: please define BLAS/LAPACK backend for this platform."
#endif

#define max(a,b) (a>=b?a:b)

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    int threads_per_rank = omp_get_max_threads();         
    
    if (threads_per_rank < 1) threads_per_rank = 1;
    mkl_set_num_threads(threads_per_rank);

    double tt1, tt2; int ii, jj, kk;
    
    struct logistics logg;

    char fname[1024]; char pname[1024];
    double fone = 1.0, fzero = 0.0;

    initialize_structure(&logg);
    logg.mpi_rank = rank; logg.mpi_size = size;
    
    int flg = findarg("help", NA, NULL, argc, argv);
    flg = findarg("about", NA, NULL, argc, argv);
    if (flg) {
        if (rank == 0)
            cout << "\nDistPCA\nDistributed Out-of-Core Principal Component Analysis (PCA) for Tera-Scale Genomic Data\n\n";

        MPI_Finalize();
        return 0;
    }
    
    int pflag = findarg("prefix",STR, pname,argc,argv);
    findarg("bfile",STR, fname, argc, argv);
    findarg("nrhs", INT, &logg.NRHS, argc, argv); findarg("nsv", INT, &logg.NSV, argc, argv);
    findarg("memory", DOUBLE, &logg.mem, argc, argv);
    findarg("bsize", INT, &logg.rows_fetched, argc, argv);
    findarg("verbose", INT, &logg.PRINT_INFO, argc, argv);
    findarg("fwrite", INT, &logg.filewrite, argc, argv);
    findarg("toll", DOUBLE, &logg.toll, argc, argv);
    findarg("miter", INT, &logg.blockPower_maxiter, argc, argv);
    findarg("crit", INT, &logg.blockPower_conv_crit, argc, argv);
    findarg("power", INT, &logg.power, argc, argv);
    findarg("fullSVD", INT, &logg.trueSVD, argc, argv);

    std::string bfile(fname);
    if(pflag)
      logg.prefixname = pname;
    
    logg.filename = bfile.c_str(); logg.pure_name = ExtractFileName(logg.filename);
    
    std::string strb(".bed"); std::string bedfile = bfile+".bed";

    std::ifstream bedin(bedfile.c_str(), std::ios::in | std::ios::binary);
    bedin.seekg(3, std::ifstream::beg); // 3-byte header of the .bed file
    
    if(!bedin){
        if (rank == 0) {
            std::string err = std::string("[Data::read_bed] Error reading file ") + bedfile;
            cout << endl << err << endl;
        }

        MPI_Finalize();
        exit(1);
    }
    
    string famfile = bfile + ".fam"; string bimfile = bfile + ".bim";
   
    string strf(".fam"); string strbim(".bim");
     logg.show_timestamp = 1;

    unsigned int line_num = 0; int nrows = -1;
    ifstream famin(famfile.c_str(), ios::in);
    
    if(rank == 0)
      cout << timestamp(&logg) << "Reading .fam file: " << famfile << endl;
    if(!famin){
      if (rank == 0) {
        string err = string("Error reading file '") + famfile + "': " + strerror(errno);
        cout << endl << err << endl;
      }
      MPI_Finalize();
      exit(1);
    }
    vector<string> famlines;

    while(famin){
      string line;
      getline(famin, line);
      
      if(!famin.eof() && (nrows == -1 || line_num < nrows)){
        if(line_num >= 0)
          famlines.push_back(line);
        
        line_num++;
      }
    }

    GetFamInfo(famlines, &logg); logg.M = line_num;
    famin.close();
    
    if (rank == 0)
      cout << endl << timestamp(&logg) << "Number of individuals (M): " << logg.M << endl;

    line_num = 0; nrows = -1;
    ifstream bimin(bimfile.c_str(), ios::in);
    
    if (rank == 0)
      cout << endl << timestamp(&logg) << "Reading .bim file: " << bimfile << endl;
    
    if(!bimin){
      if (rank == 0) {
        string err = string("Error reading file '") + bimfile + "': " + strerror(errno);
        cout << endl << err << endl;
      }
      
      MPI_Finalize();
      exit(1);
    }

    vector<string> bimlines;
    
    while(bimin){
      string line1;
      getline(bimin, line1);
      
      if(!bimin.eof() && (nrows == -1 || line_num < nrows)){
        if(line_num >= 0)
          bimlines.push_back(line1);
        
        line_num++;
      }
    }
    
    GetBimInfo(bimlines, &logg); logg.N = line_num;
    bimin.close();
    
    if (rank == 0)
      cout << endl << timestamp(&logg) << "Number of markers (N): " << logg.N << endl;

    int snps_per_rank = logg.N / size; int remainder = logg.N % size;

    logg.local_N_start = rank * snps_per_rank + min(rank, remainder);
    logg.local_N = snps_per_rank + (rank < remainder ? 1 : 0);
    logg.local_N_end = logg.local_N_start + logg.local_N;

    if (rank == 0) {
        cout << endl << timestamp(&logg) << "Number of MPI ranks: " << size << endl;
        cout << endl << timestamp(&logg) << "SNPs per rank: ~" << snps_per_rank << endl;
        cout << endl << timestamp(&logg) << "Dataset partitioning across MPI ranks:" << endl << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    cout << "Rank " << rank << ": SNPs [" << logg.local_N_start 
         << ", " << logg.local_N_end << ") = " << logg.local_N << " SNPs" << endl;
    
    MPI_Barrier(MPI_COMM_WORLD);

    int ram_KB; double ram_GB; ram_KB = GetRamInKB();
    
    if (rank == 0)
      cout << endl << timestamp(&logg) << "Size of RAM (in KB): " << ram_KB << endl;
    
    ram_GB = (double) ram_KB; ram_GB = ram_GB / 1000000;
    logg.ram_KB = ram_KB; logg.ram_GB = ram_GB;

    if (logg.mem == 0.0)
      logg.mem = logg.ram_GB / (10.0 * size);

    else
      logg.mem = logg.mem / size;

    if (rank == 0)
      cout << endl << timestamp(&logg) << "Memory per rank: " << logg.mem << " GB" << endl;
    
    if (getenv("OMP_NUM_THREADS")) {
      logg.threads = atoi(getenv("OMP_NUM_THREADS"));
    }

    if (rank == 0) {
        int llc_kb = get_llc_size_kb();
        if (llc_kb > 0) {
            cout << endl << timestamp(&logg) << "Detected LLC size: " << llc_kb << " KB (" 
                << llc_kb / 1024 << " MB)" << endl;
        }
    }
    
    if (logg.rows_fetched <= 0) {
        
        const double BYTES_PER_DOUBLE = 8.0;
        const double GB_TO_BYTES = 1e9;
        const double SAFETY_MARGIN = 2048.0 * 100000.0;

        double membuff =
            (3.0 * logg.local_N * BYTES_PER_DOUBLE) +
            (logg.M * logg.NSV * BYTES_PER_DOUBLE) +
            (2.0 * logg.local_N * logg.NRHS) +
            (3.0 * logg.M * BYTES_PER_DOUBLE) +
            SAFETY_MARGIN;

        double available_mem = (logg.mem * GB_TO_BYTES) - membuff;

        double blksize = 0.0;

        if (available_mem > 0.0)
          blksize = available_mem / (BYTES_PER_DOUBLE * logg.M);
        
        else
          blksize = (membuff + (logg.M * BYTES_PER_DOUBLE)) / 1e7;

        logg.rows_fetched = (int)blksize;
        
        if (logg.rows_fetched <= 0)
          logg.rows_fetched = logg.NSV;

        if (logg.rows_fetched >= logg.local_N)
          logg.rows_fetched = logg.local_N;
        
        if (rank == 0)
          cout << endl << timestamp(&logg) << "Memory: " << logg.rows_fetched << " SNPs per block" << endl;
    } 
    
    else {
        
        if (logg.rows_fetched > logg.local_N) {
            if (rank == 0) 
              cout << endl << timestamp(&logg) << "Warning: Specified block size (" << logg.rows_fetched 
                  << ") exceeds local SNPs (" << logg.local_N 
                  << "). Adjusting to local_N." << endl;
            
            logg.rows_fetched = logg.local_N;
        }
        
        if (logg.rows_fetched < 1) {
          if (rank == 0)
            cout << timestamp(&logg) << "Error: Block size must be at least 1. Setting to NSV." << endl;
          
          logg.rows_fetched = logg.NSV;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << endl << timestamp(&logg) << "Block size (SNPs per block): " << logg.rows_fetched << endl;
        
        double block_mem_gb = (2.0 * logg.rows_fetched * logg.M * 8.0) / 1e9;  
        double aux_mem_gb = (logg.M * logg.NRHS * 8.0 * 3) / 1e9;
        double total_mem_gb = block_mem_gb + aux_mem_gb;
        
        cout << endl << timestamp(&logg) << "Estimated memory per rank: " << total_mem_gb << " GB";
        cout << " (Block buffers: " << block_mem_gb << " GB, ";
        cout << "Auxiliary arrays: " << aux_mem_gb << " GB)" << endl;

        double block_size_mb = ((double)logg.rows_fetched * (double)logg.M * 8.0 ) / (1024.0 * 1024.0);
        cout << endl << timestamp(&logg) << "Block size (in MB): " << block_size_mb;
        cout << " (typical LLC sizes: 16-64 MB per socket)" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (logg.M <= 0 || logg.N <= 0) {
      if (rank == 0)
        cout << endl << timestamp(&logg) << "M and/or N were either zero or negative. Aborting..." << endl;
      
      MPI_Finalize();
      exit(1);
    }
    
    if (logg.NSV <= 0) {
      if (rank == 0)
        cout << endl << timestamp(&logg) << "NSV was either zero or negative. Aborting..." << endl;
      
      MPI_Finalize();
      exit(1);
    }
    
    if (logg.NSV > min(logg.M,logg.N)) {
      logg.NSV = min(logg.M,logg.N);
      
      if (rank == 0)
        cout << endl << timestamp(&logg) << "The value of NSV given was larger than min(M, N). Adjusting to NSV = min(M,N)..." << endl;
    }
    
    if (logg.NRHS <= 0) {
      if (rank == 0)
        cout << endl << timestamp(&logg) << "NRHS was either zero or negative. Adjusting to NRHS = min(2xNSV, min(M,N))..." << endl;
      
      logg.NRHS = min(2*logg.NSV,min(logg.M, logg.N));
    }
    
    if (logg.NRHS > min(logg.M, logg.N)) {
      logg.NRHS = min(logg.M, logg.N);
      
      if (rank == 0)
        cout << endl << timestamp(&logg) << "The value of NRHS given was larger than min(M,N). Adjusting to NRHS = min(M,N)..." << endl;
    }
    
    if (logg.NRHS < logg.NSV) {
      if (rank == 0)
        cout << endl << timestamp(&logg) << "NRHS can not be smaller than NSV. Adjusting to NRHS = min(2xNSV,min(M,N))..." << endl;
      
        logg.NRHS = min(2*logg.NSV, min(logg.M,logg.N));
    }
    
    double *MAT = NULL;
    if (logg.rows_fetched == logg.local_N) {
      uint64_t malloc_size = (uint64_t) logg.local_N * logg.M * sizeof(double);
      printf("local N -> %d\n", logg.local_N);
      printf("M -> %d\n", logg.M);
      printf("malloc_size (bytes) -> %lu\n", malloc_size);

      MAT = (double*) malloc(malloc_size);

      if (MAT == NULL) {
        cout << endl << timestamp(&logg) << "Rank " << rank << ": MAT malloc failed (size: " << malloc_size << " bytes)" << endl;
        
        MPI_Finalize();
        exit(1);
      }

      tt1 = dsecnd();
      
      if (rank == 0)
        cout << endl << timestamp(&logg) << "Reading .bed file (distributed across ranks)..." << endl;
      
      Read_Bed_Local(bedin, MAT, &logg);

      tt2 = dsecnd() - tt1;
      logg.TIME_2_LOAD_MATRIX = tt2;
      
      if (rank == 0)
        cout << endl << timestamp(&logg) << "Finished reading local data. Time: " << tt2 << " sec" << endl;
      bedin.close();
    }
    
    double mean, std_dev, norm_rv;
    mean = 0; std_dev = 1;
    
    double *RHS = (double*)malloc(logg.M*logg.NRHS*sizeof(double));

    tt1 = dsecnd();
    
     // Generate initial random RHS matrix (M x NRHS) with Gaussian entries
    for (jj = 0; jj < logg.M; jj++) {
      for (kk = 0; kk < logg.NRHS; kk++) {
        rand_val(jj*logg.M+kk); norm_rv = norm2(mean, std_dev);
        RHS[jj*logg.NRHS+kk] = (double) norm_rv; 
      }
    }

    tt2 = dsecnd() - tt1;
    logg.TIME_2_GENERATE_RHS = tt2;
    
    // Block size is small enough to fit entire local matrix in memory
    if (logg.rows_fetched == logg.local_N)
      SubspaceIteration_MPI(MAT, RHS, &logg);
    
    else
      BlockSubspaceIter_MPI_OOC_double_buffering(bedfile.c_str(), RHS, &logg);
      // BlockSubspaceIter_MPI_OOC(bedfile.c_str(), RHS, &logg);
    
    free(RHS);
    
    if (rank == 0 && logg.PRINT_INFO > 1) {
      cout << endl << "Approximate eigenvalues:" << endl;
      
      for (jj = 0; jj < logg.NSV; jj++)
        cout << "λ_" << jj << " = " << fixed << setprecision(13)
            << (logg.sing_values[jj] * logg.sing_values[jj]) << endl;
      
      cout << "\nFirst 5 individuals' eigenvectors:\n\n";

      cout << left << setw(20) << "FID";

      for (jj = 0; jj < logg.NSV; jj++)
          cout << setw(15) << ("PC" + to_string(jj));

      cout << "\n";

      cout << string(20 + 15 * logg.NSV, '-') << "\n";

      for (ii = 0; ii < min(5, logg.M); ii++) {

          cout << left << setw(20) << logg.indiv_ids[ii];

          for (jj = 0; jj < logg.NSV; jj++) {
              cout << setw(15)
                  << fixed << setprecision(6)
                  << logg.left_sing_vecs[ii * logg.NSV + jj];
          }

          cout << "\n";
      }
    }

    if (logg.trueSVD == 1 && size == 1) {

      int min_dim = min(logg.M, logg.N);

      double *TRUE_SING_VALUES      = (double*)malloc(min_dim * sizeof(double));
      double *TRUE_LEFT_SING_VECS   = (double*)malloc(logg.M * min_dim * sizeof(double));
      double *TRUE_RIGHT_SING_VECS  = (double*)malloc(logg.N * min_dim * sizeof(double));
      double *TRUE_superb           = (double*)malloc((min_dim - 1) * sizeof(double));

      double *sing_vals_relerror    = (double*)malloc(logg.NSV * sizeof(double));
      double *sing_vecs_relerror    = (double*)malloc(logg.M * logg.NSV * sizeof(double));

      double *copy1 = (double*)malloc(logg.M * logg.NSV * sizeof(double));
      double *copy2 = (double*)malloc(logg.M * logg.NSV * sizeof(double));

      double *UhatU = (double*)malloc(logg.NSV * logg.NSV * sizeof(double));
      double *CosineValues = (double*)malloc(logg.NSV * sizeof(double));

      double fone = 1.0, fzero = 0.0;

      mkl_dimatcopy('R','T', logg.N, logg.M, fone, MAT, logg.M, logg.N);

      printf("komple\n");

      double tt1 = dsecnd();

      int info = LAPACKE_dgesvd(
          LAPACK_ROW_MAJOR, 'S', 'S',
          logg.M, logg.N,
          MAT, logg.N,
          TRUE_SING_VALUES,
          TRUE_LEFT_SING_VECS, min_dim,
          TRUE_RIGHT_SING_VECS, logg.N,
          TRUE_superb
      );

      printf("komple2\n");

      logg.TIME_2_TRUE_SVD = dsecnd() - tt1;

      if (logg.PRINT_INFO > 1)
          printf("\nFULL SVD computed in %lf sec, info = %d\n\n",
                logg.TIME_2_TRUE_SVD, info);

      for (int i = 0; i < logg.NSV; i++) {

          double denom = TRUE_SING_VALUES[i] ? TRUE_SING_VALUES[i] : 1.0;

          sing_vals_relerror[i] =
              fabs(TRUE_SING_VALUES[i] - logg.sing_values[i]) / denom;

          if (logg.PRINT_INFO > 1)
              printf("RelError Sigma(%d): %e\n", i, sing_vals_relerror[i]);
      }

      printf("\n");

      for (int i = 0; i < logg.M; i++) {
          for (int j = 0; j < logg.NSV; j++) {

              sing_vecs_relerror[i * logg.NSV + j] =
                  fabs(fabs(TRUE_LEFT_SING_VECS[i * min_dim + j]) -
                      fabs(logg.left_sing_vecs[i * logg.NSV + j])) /
                  (fabs(TRUE_LEFT_SING_VECS[i * min_dim + j]) + 1e-12);

              copy1[i * logg.NSV + j] = logg.left_sing_vecs[i * logg.NSV + j];
              copy2[i * logg.NSV + j] = TRUE_LEFT_SING_VECS[i * min_dim + j];
          }
      }

      cblas_dgemm(
          CblasRowMajor, CblasTrans, CblasNoTrans,
          logg.NSV, logg.NSV, logg.M,
          fone,
          copy1, logg.NSV,
          copy2, logg.NSV,
          fzero,
          UhatU, logg.NSV
      );

      double frob = 0.0;

      for (int i = 0; i < logg.NSV; i++)
          for (int j = 0; j < logg.NSV; j++)
              frob += (i == j)
                  ? (fabs(UhatU[i * logg.NSV + j]) - 1.0) *
                    (fabs(UhatU[i * logg.NSV + j]) - 1.0)
                  : UhatU[i * logg.NSV + j] *
                    UhatU[i * logg.NSV + j];

      logg.frob_norm_angle = sqrt(frob);

      logg.cos_values.resize(logg.NSV);

      computeCosineError(
          copy1,
          copy2,
          logg.M,
          logg.NSV,
          CosineValues,
          &logg.cos_error
      );

      for (int i = 0; i < logg.NSV; i++) {
          logg.cos_values[i] = CosineValues[i];

          if (logg.PRINT_INFO > 1)
              printf("Cosine(%d): %lf\n", i, CosineValues[i]);
      }

      printf("\n");

      if (logg.filewrite == 1) {

          string vecFile, valFile, errFile, cosFile;

          if (logg.prefixname.empty()) {
              vecFile = ConstructFilename(logg,"realLeftSingularVectors");
              valFile = ConstructFilename(logg,"realSingularValues");
              errFile = ConstructFilename(logg,"singvecs_accuracy");
              cosFile = ConstructFilename(logg,"cosineValues");
          }
          else {
              vecFile = logg.prefixname + "_realLeftSingularVectors.txt";
              valFile = logg.prefixname + "_realSingularValues.txt";
              errFile = logg.prefixname + "_singvecs_accuracy.txt";
              cosFile = logg.prefixname + "_cosineValues.txt";
          }

          FILE *fvec = fopen(vecFile.c_str(), "w");
          FILE *fval = fopen(valFile.c_str(), "w");
          FILE *ferr = fopen(errFile.c_str(), "w");
          FILE *fcos = fopen(cosFile.c_str(), "w");

          if (!fvec || !fval || !ferr || !fcos) {
              printf("File error\n");
              exit(1);
          }

          for (int i = 0; i < logg.M; i++) {
              for (int j = 0; j < logg.NSV; j++)
                  fprintf(fvec, "% 2.13f ",
                          TRUE_LEFT_SING_VECS[i * min_dim + j]);
              fprintf(fvec, "\n");
          }

          for (int i = 0; i < logg.NSV; i++)
              fprintf(fval, "% 2.13f\n", TRUE_SING_VALUES[i]);

          for (int i = 0; i < logg.M * logg.NSV; i++)
              fprintf(ferr, "% 2.13lf\n", sing_vecs_relerror[i]);

          for (int i = 0; i < logg.NSV; i++)
              fprintf(fcos, "% 2.13lf\n", logg.cos_values[i]);

          fclose(fvec);
          fclose(fval);
          fclose(ferr);
          fclose(fcos);
      }

      free(TRUE_SING_VALUES);
      free(TRUE_LEFT_SING_VECS);
      free(TRUE_RIGHT_SING_VECS);
      free(TRUE_superb);

      free(sing_vals_relerror);
      free(sing_vecs_relerror);

      free(copy1);
      free(copy2);

      free(UhatU);
      free(CosineValues);
  }

  time_t rawtime;
  time(&rawtime);
  logg.timeinfo = localtime(&rawtime);

  if (rank == 0 && logg.PRINT_INFO > 0)
    print_statistics(logg);

  if (MAT != NULL)
    free(MAT);

  MPI_Finalize();
  return 0;
}