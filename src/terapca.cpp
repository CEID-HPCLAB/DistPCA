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

// Detect architecture and include appropriate BLAS/LAPACK headers
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

    //init MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // int threads_per_rank = omp_get_max_threads() / size;     # split threads among ranks
    int threads_per_rank = omp_get_max_threads();               // each rank uses all threads given from OMP_NUM_THREADS
    if (threads_per_rank < 1) threads_per_rank = 1;
    mkl_set_num_threads(threads_per_rank);
    if (rank == 0) {
      cout << "MKL threads per rank: " << threads_per_rank << endl;
    }

    double tt1, tt2;
    int ii, jj, kk;
    struct logistics logg;

    char fname[1024];
    char pname[1024];
    double fone = 1.0, fzero = 0.0;

    //init log
    initialize_structure(&logg);
    logg.mpi_rank = rank;
    logg.mpi_size = size;
    
    //take arguments
    int flg = findarg("help", NA, NULL, argc, argv);
    flg = findarg("about", NA, NULL, argc, argv);
    if (flg) {
        if (rank == 0) {
            printf("\nTeraPCA - MPI Version\n");
            printf("Distributed Principal Component Analysis for Genomic Data\n\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    int pflag = findarg("prefix",STR, pname,argc,argv);
    findarg("bfile",STR, fname, argc, argv);
    findarg("nrhs", INT, &logg.NRHS, argc, argv);
    findarg("nsv", INT, &logg.NSV, argc, argv);
    findarg("memory", DOUBLE, &logg.mem, argc, argv);
    findarg("rfetched", INT, &logg.rows_fetched, argc, argv);
    findarg("print", INT, &logg.PRINT_INFO, argc, argv);
    findarg("filewrite", INT, &logg.filewrite, argc, argv);
    findarg("toll", DOUBLE, &logg.toll, argc, argv);
    findarg("blockPower_maxiter", INT, &logg.blockPower_maxiter, argc, argv);
    findarg("blockPower_conv_crit", INT, &logg.blockPower_conv_crit, argc, argv);
    findarg("power", INT, &logg.power, argc, argv);
    findarg("trueSVD", INT, &logg.trueSVD, argc, argv);
    // findarg("benchmarking", INT, &logg.benchmarking, argc, argv);

    std::string bfile(fname);
    if(pflag)
      logg.prefixname = pname;
    logg.filename = bfile.c_str();
    logg.pure_name = ExtractFileName(logg.filename);
    

    //read bed file only if RANK 0
    std::string strb(".bed");
    std::string bedfile = bfile+".bed";

    std::ifstream bedin(bedfile.c_str(), std::ios::in | std::ios::binary);
    bedin.seekg(3, std::ifstream::beg);
    if(!bedin){
        if (rank == 0) {
            std::string err = std::string("[Data::read_bed] Error reading file ") + bedfile;
            cout << endl << err << endl;
        }
        MPI_Finalize();
        exit(1);
    }
    
    //read fam and bim files
    string famfile = bfile + ".fam";
    string bimfile = bfile + ".bim";
    logg.show_timestamp = 1;
    string strf(".fam");
    string strbim(".bim");

    //read fam file only if RANK 0
    unsigned int line_num = 0;
    int nrows = -1;
    ifstream famin(famfile.c_str(), ios::in);
    if(rank == 0)
      cout << endl << timestamp(&logg) << "Reading .fam file: " << famfile << endl;
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
    GetFamInfo(famlines, &logg);
    logg.M = line_num;
    famin.close();
    
    if (rank == 0)
      cout << endl << timestamp(&logg) << "Number of individuals (M): " << logg.M << endl;

    //read bim file only if RANK 0
    line_num = 0;
    nrows = -1;
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
    GetBimInfo(bimlines, &logg);
    logg.N = line_num;
    bimin.close();
    
    if (rank == 0)
      cout << endl << timestamp(&logg) << "Number of markers (N): " << logg.N << endl;

    // Distribute SNPs across MPI ranks
    int snps_per_rank = logg.N / size;
    int remainder = logg.N % size;

    logg.local_N_start = rank * snps_per_rank + min(rank, remainder);
    logg.local_N = snps_per_rank + (rank < remainder ? 1 : 0);
    logg.local_N_end = logg.local_N_start + logg.local_N;

    if (rank == 0) {
        cout << endl << "========================================" << endl;
        cout << "MPI Configuration" << endl;
        cout << "========================================" << endl;
        cout << "Number of MPI ranks: " << size << endl;
        cout << "SNPs per rank: ~" << snps_per_rank << endl;
        cout << "========================================" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    cout << "Rank " << rank << ": SNPs [" << logg.local_N_start 
         << ", " << logg.local_N_end << ") = " << logg.local_N << " SNPs" << endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //determine ram size - we keep that even though we use blocksize
    int ram_KB;
    double ram_GB;
    if (rank == 0)
      cout << endl << timestamp(&logg) << "Determining size of RAM..." << endl;
    ram_KB = GetRamInKB();
    if (rank == 0) {
      cout << endl << timestamp(&logg) << "Size of RAM (in KB): " << ram_KB << endl;
    }
    ram_GB = (double) ram_KB;
    ram_GB = ram_GB / 1000000;
    logg.ram_KB = ram_KB;
    logg.ram_GB = ram_GB;

    if (logg.mem == 0.0)
      logg.mem = logg.ram_GB / (10.0 * size);  // user did not specify memory => each rank uses 1/10 of total RAM

    else
      logg.mem = logg.mem / size; // user specified memory => each rank gets fraction

    if (rank == 0)
      cout << endl << timestamp(&logg) << "Memory per rank: " << logg.mem << " GB" << endl;
    
    //extract number of threads from OMP_NUM_THREADS
    if (getenv("OMP_NUM_THREADS")) {
      logg.threads = atoi(getenv("OMP_NUM_THREADS"));
    }

    //report LLC size (rank 0 only)
    if (rank == 0) {
        int llc_kb = get_llc_size_kb();
        if (llc_kb > 0) {
            cout << "Detected LLC size: " << llc_kb << " KB (" 
                << llc_kb/1024 << " MB)" << endl;
        }
    }
    
    //compute the number of rows that can fit in the allocated memory (using local_N)
    if (logg.rows_fetched <= 0) {
        // User did not specify block size, compute based on memory
        if (rank == 0)
          cout << endl << timestamp(&logg) << "Computing block size based on available memory..." << endl;
        
        double membuff = (3*(logg.local_N*8.0)) + (logg.M*logg.NSV*8.0) + 
                        (2*(logg.local_N*logg.NRHS)) + (3*logg.M*8.0) + 2048*100000;
        double workmem = (logg.mem*1000000000) - membuff;
        double blksize;
        
        if (workmem > 0)
          blksize = workmem/(8.0*logg.M);
        else
          blksize = (membuff + (logg.M*8.0))/10000000;

        logg.rows_fetched = (int)blksize;
        
        if (logg.rows_fetched <= 0)
          logg.rows_fetched = logg.NSV;
        
        // Check if there is enough space to store the entire local matrix
        if (logg.rows_fetched >= logg.local_N) {
          logg.rows_fetched = logg.local_N;
        }
        
        if (rank == 0)
          cout << endl << "Memory: " << logg.rows_fetched << " SNPs per block (per rank)" << endl;
    } else {
        // User specified block size explicitly
        if (rank == 0)
          cout << endl << "Block size: " << logg.rows_fetched << " SNPs per block (per rank)" << endl;
        
        // Validate block size
        if (logg.rows_fetched > logg.local_N) {
          if (rank == 0)
            cout << "Warning: Specified block size (" << logg.rows_fetched 
                << ") exceeds local SNPs (" << logg.local_N 
                << "). Adjusting to local_N." << endl;
          logg.rows_fetched = logg.local_N;
        }
        
        if (logg.rows_fetched < 1) {
          if (rank == 0)
            cout << "Error: Block size must be at least 1. Setting to NSV." << endl;
          logg.rows_fetched = logg.NSV;
        }
    }

    // Print summary for all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        cout << endl << "========================================" << endl;
        cout << "Block Size Configuration" << endl;
        cout << "========================================" << endl;
        cout << "Block size (SNPs per block): " << logg.rows_fetched << endl;
        
        // Estimate memory usage with this block size
        double block_mem_gb = (2.0 * logg.rows_fetched * logg.M * 8.0) / 1e9;  // Two buffers
        double aux_mem_gb = (logg.M * logg.NRHS * 8.0 * 3) / 1e9;  // ARHS, ARHS_local, RHS_buf
        double total_mem_gb = block_mem_gb + aux_mem_gb;
        
        cout << "Estimated memory per rank: " << total_mem_gb << " GB" << endl;
        cout << "  - Block buffers: " << block_mem_gb << " GB" << endl;
        cout << "  - Auxiliary arrays: " << aux_mem_gb << " GB" << endl;
        
        // Cache analysis hint
        cout << endl << "Cache Considerations:" << endl;
        double block_size_mb = (logg.rows_fetched * logg.M * 8.0) / (1024.0 * 1024.0);
        cout << "  Single block size: " << block_size_mb << " MB" << endl;
        cout << "  (Typical LLC sizes: 16-64 MB per socket)" << endl;
        cout << "========================================" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    //check # of rows and columns of matrix A
    if (logg.M <= 0 || logg.N <= 0) {
      if (rank == 0)
        printf("M and/or N were either zero or negative. Aborting...\n");
      MPI_Finalize();
      exit(1);
    }
    
    //# of singular vectors to compute
    if (logg.NSV <= 0) {
      if (rank == 0)
        printf("NSV was either zero or negative. Aborting...\n");
      MPI_Finalize();
      exit(1);
    }
    if (logg.NSV > min(logg.M,logg.N)) {
      logg.NSV = min(logg.M,logg.N);
      if (rank == 0)
        printf("The value of NSV given was larger than min(M,N). Adjusting to NSV=min(M,N)...\n");
    }
    
    //# of columns in the initial subspace
    if (logg.NRHS <= 0) {
      if (rank == 0)
        printf("NRHS was either zero or negative. Adjusting to NRHS=min(2xNSV,min(M,N))...\n");
      logg.NRHS = min(2*logg.NSV,min(logg.M,logg.N));
    }
    if (logg.NRHS > min(logg.M,logg.N)) {
      logg.NRHS = min(logg.M,logg.N);
      if (rank == 0)
        printf("The value of NRHS given was larger than min(M,N). Adjusting to NRHS=min(M,N)...\n");
    }
    if (logg.NRHS < logg.NSV) {
      if (rank == 0)
        printf("NRHS can not be smaller than NSV. Adjusting to NRHS=min(2xNSV,min(M,N))...\n");
      logg.NRHS = min(2*logg.NSV,min(logg.M,logg.N));
    }
    
    // Load local portion of matrix (each rank loads its SNPs)
    double *MAT = NULL;
    if (logg.rows_fetched == logg.local_N) {
      uint64_t malloc_size = (uint64_t) logg.local_N * logg.M * sizeof(double);

      MAT = (double*) malloc(malloc_size);

      if (MAT == NULL) {
        printf("Rank %d: MAT malloc failed (size: %llu bytes)\n", rank, malloc_size);
        MPI_Finalize();
        exit(1);
      }

      tt1 = dsecnd();
      if (rank == 0)
        cout << endl << timestamp(&logg) << "Reading .bed file (distributed across ranks)..." << endl;
      
      // Each rank reads only its portion of SNPs
      Read_Bed_Local(bedin, MAT, &logg);

      tt2 = dsecnd() - tt1;
      logg.TIME_2_LOAD_MATRIX = tt2;
      
      if (rank == 0)
        cout << endl << timestamp(&logg) << "Finished reading local data. Time: " << tt2 << " sec" << endl;
    }
    
    //Fill the RHS matrix with normal random numbers (all ranks generate same RHS)
    double mean, std_dev, norm_rv;
    mean = 0;
    std_dev = 1;
    double *RHS = (double*)malloc(logg.M*logg.NRHS*sizeof(double));

    tt1 = dsecnd();
    for (jj = 0; jj < logg.M; jj++) {
      for (kk = 0; kk < logg.NRHS; kk++) {
        rand_val(jj*logg.M+kk);
        norm_rv = norm2(mean, std_dev);
        RHS[jj*logg.NRHS+kk] = (double) norm_rv;
      }
    }
    tt2 = dsecnd() - tt1;
    logg.TIME_2_GENERATE_RHS = tt2;
    
    //Compute leading left singular vectors
    // if (logg.benchmarking == 1) {
    //   if (rank == 0)
    //     printf("Benchmarking mode not yet implemented for MPI version.\n");
    //   free(RHS);
    //   if (MAT != NULL)
    //     free(MAT);
    //   bedin.close();
    //   MPI_Finalize();
    //   return 0;
    // }
    
    if (rank == 0)
      cout << endl << timestamp(&logg) << "Starting distributed subspace iteration..." << endl;
    
    if (logg.rows_fetched == logg.local_N) {
      subspaceIteration_MPI(MAT, RHS, &logg);
    } else {
      if (rank == 0)
        cout << endl << timestamp(&logg) << "Reading .bed file by blocks (MPI-IO mode)." << endl;
      bedin.close();  // Close ifstream, we'll use MPI_File instead
      BlockSubspaceIter_MPI_IO_2ble_buffering(bedfile.c_str(), RHS, &logg);
      //BlockSubspaceIter_MPI_IO(bedfile.c_str(), RHS, &logg);
    }
    
    free(RHS);
    bedin.close();
    
    //Print approximate singular values (rank 0 only)
    if (rank == 0 && logg.PRINT_INFO > 1) {
      cout << endl << "Approximate eigenvalues:" << endl;
      for (jj = 0; jj < logg.NSV; jj++) {
        printf("  λ_%d = %02.13f\n", jj, logg.sing_values[jj] * logg.sing_values[jj]);
      }
      
      cout << endl << "First 5 individuals' eigenvectors:" << endl;
      cout << "FID";
      for (jj = 0; jj < logg.NSV; jj++)
        cout << "\tPC" << jj;
      cout << endl;
      
      for (ii = 0; ii < min(5, logg.M); ii++) {
        printf("%s", logg.indiv_ids[ii].c_str());
        for (jj = 0; jj < logg.NSV; jj++)
          printf("\t%.6f", logg.left_sing_vecs[ii*logg.NSV + jj]);
        printf("\n");
      }
    }
    
    //Compute true SVD (disabled in MPI mode for now - would need to gather full matrix)
    if (logg.rows_fetched == logg.local_N && logg.trueSVD == 1) {
      if (rank == 0) {
        printf("\n========================================\n");
        printf("WARNING: trueSVD option not fully supported in MPI mode\n");
        printf("Would require gathering full matrix to rank 0\n");
        printf("Skipping full SVD computation.\n");
        printf("========================================\n");
      }
    }

    //Store date and time of current simulation
    time_t rawtime;
    time(&rawtime);
    logg.timeinfo = localtime(&rawtime);

    //Print program statistics (rank 0 only)
    if (rank == 0 && logg.PRINT_INFO > 0) {
      cout << endl << "========================================" << endl;
      cout << "MPI Run Complete" << endl;
      cout << "========================================" << endl;
      print_statistics(logg);
    }

    if (MAT != NULL)
      free(MAT);

    MPI_Finalize();
    return 0;
}