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
#include "omp.h"
#include "structures.h"
#include <string>
#include <sstream>
#include <immintrin.h>
#include "utilities.h"

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

// Determine ths OS, needed for memory size retrieval
#if defined(__APPLE__) || defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

using namespace std;

string toString(int value) {
  ostringstream s;
  s << value;
  return s.str();
}
 
string timestamp(struct logistics *logg);

int GetRamInKB(void)
{
    
#if defined(__linux__)
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (!meminfo) {
        cout << "Error: Cannot open /proc/meminfo" << endl;
        return -1;
    }
    
    char line[256];
    while(fgets(line, sizeof(line), meminfo)) {
        int ram;
        if(sscanf(line, "MemTotal: %d kB", &ram) == 1) {
            fclose(meminfo);
            return ram;
        }
    }
    fclose(meminfo);
    return -1;
    
#elif defined(__APPLE__) || defined(__MACH__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
    
    int64_t ram_bytes;
    size_t size = sizeof(ram_bytes);
    
    if (sysctlbyname("hw.memsize", &ram_bytes, &size, NULL, 0) == 0) {
        // Convert bytes to KB
        return (int)(ram_bytes / 1024);
    } else {
        cout << "Error: Cannot get memory size via sysctl" << endl;
        return -1;
    }
    
#else
    #error "Unsupported platform for memory detection. Please implement GetRamInKB() for this OS."
#endif
}

int get_llc_size_kb(void) {
    FILE *fp = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (!fp) return -1;
    
    int llc_kb;
    int ret = fscanf(fp, "%dK", &llc_kb);
    fclose(fp);
    
    if (ret != 1) return -1;
    return llc_kb;
}

std::string ExtractFileName(const std::string& fullPath){
  const size_t lastSlashIndex = fullPath.find_last_of("/\\");
  const size_t lastDotIndex   = fullPath.substr(lastSlashIndex + 1).find_last_of(".");
  return fullPath.substr(lastSlashIndex+1,lastDotIndex);
}

string ConstructFilename(struct logistics logg, string fileType){
  string year  = toString(1900 + logg.timeinfo->tm_year);
  string month;
  if (logg.timeinfo->tm_mon + 1 < 9) 
    month = "0" + toString(logg.timeinfo->tm_mon + 1);
  else 
    month =  toString(logg.timeinfo->tm_mon + 1);
  string day   = toString(logg.timeinfo->tm_mday); 
  string atime = toString(logg.timeinfo->tm_hour) + "-" + toString(logg.timeinfo->tm_min) + "-" + toString(logg.timeinfo->tm_sec);
  string cname = logg.pure_name + "_" + year + month + day + "_" + atime + "_" + fileType + ".txt";
  return cname;
}

void print_statistics(struct logistics logg) {

  printf("----------------------------------------------------------\n");
  printf("DistPCA software package, Version: 1.0.\n");
  printf( "Current local time and date: %s", asctime (logg.timeinfo) );
  printf("Total number of threads exploited: %d\n",logg.threads);
  printf("RAM size in KBs: %d\n",logg.ram_KB);
  printf("RAM size in GBs: %lf\n",logg.ram_GB);
  printf("Total number of matrix rows: %d\n", logg.M);
  printf("Total number of matrix columns: %d\n", logg.N);
  printf("Total number of singular pairs sought: %d\n", logg.NSV);
  printf("Total number of right-hand sides used: %d\n", logg.NRHS);
  printf("Value of power: %d\n", logg.power);
  printf("Total number of rows fetched from the disk per block: %d\n", logg.rows_fetched);
  if (logg.rows_fetched < logg.N) {
    printf("Total number of times which the data matrix was fetched from the memory: %d\n", logg.blockPower_total_its*(logg.power+1));
    printf("Average amount of time elapsed per matrix fetching: %lf\n", logg.TIME_2_LOAD_MATRIX / (logg.blockPower_total_its*(logg.power+1)));
  }
  printf("\n[The following times are listed in seconds]\n\n");
  printf("Time to generate the right-hand sides matrix: %02.13f\n",logg.TIME_2_GENERATE_RHS);
  printf("Time to load the data matrix: %02.13f\n",logg.TIME_2_LOAD_MATRIX);
  printf("Time to perform the MM products (overall): %02.13f\n",logg.TIME_2_MM);
  printf("Time to perform the MM products (with A): %02.13f\n",logg.TIME_2_MM_A);
  printf("Time to perform the MM products (with A^T): %02.13f\n",logg.TIME_2_MM_A_TRANSPOSED);
  printf("Time to perform the orthonormalization: %02.13f\n",logg.TIME_2_GS);
  printf("Time to solve the projection eigenvalue problem: %02.13f\n",logg.TIME_2_PROJECTED_SVD);
  printf("\nTotal wall-clock time elapsed: %02.13f\n",logg.TIME_2_PROJECTED_SVD + logg.TIME_2_GENERATE_RHS + logg.TIME_2_LOAD_MATRIX + logg.TIME_2_MM + logg.TIME_2_GS + logg.TIME_2_OTHER);
  printf("Total wall-clock time elapsed (without including the amount of time spent on loading the matrix): %02.13f\n",logg.TIME_2_PROJECTED_SVD + logg.TIME_2_GENERATE_RHS + logg.TIME_2_MM + logg.TIME_2_GS + logg.TIME_2_OTHER);

  printf("\n[The following times are listed in hours]\n\n");
  printf("Time to generate the right-hand sides matrix: %02.13f\n",logg.TIME_2_GENERATE_RHS/3600);
  printf("Time to load the data matrix: %02.13f\n",logg.TIME_2_LOAD_MATRIX/3600);
  printf("Time to perform the MM products (overall): %02.13f\n",logg.TIME_2_MM/3600);
  printf("Time to perform the MM products (with A): %02.13f\n",logg.TIME_2_MM_A/3600);
  printf("Time to perform the MM products (with A^T): %02.13f\n",logg.TIME_2_MM_A_TRANSPOSED/3600);
  printf("Time to perform the orthonormalization: %02.13f\n",logg.TIME_2_GS/3600);
  printf("Time to solve the projection eigenvalue problem: %02.13f\n",logg.TIME_2_PROJECTED_SVD/3600);
  printf("\nTotal wall-clock time elapsed: %02.13f\n",(logg.TIME_2_PROJECTED_SVD + logg.TIME_2_GENERATE_RHS + logg.TIME_2_LOAD_MATRIX + logg.TIME_2_MM + logg.TIME_2_GS + logg.TIME_2_OTHER)/3600);
  printf("Total wall-clock time elapsed (without including the time spent on loading the matrix): %02.13f\n",(logg.TIME_2_PROJECTED_SVD + logg.TIME_2_GENERATE_RHS + logg.TIME_2_MM + logg.TIME_2_GS + logg.TIME_2_OTHER)/3600);
 
  if (logg.rows_fetched == logg.N && logg.trueSVD == 1) {
    printf("\nTotal wall-clock time to compute the full ('econ') SVD: %02.13f\n",logg.TIME_2_TRUE_SVD);
    printf("||Uhat^TU - I||_F: %02.13f\n",logg.frob_norm_angle);
    printf("||Uhat^TU - I||_2: %02.13f\n",logg.cos_error);
  }

}

void initialize_structure(struct logistics *logg) {
  
  logg->filename = "noprefix";
  logg->prefixname = "noprefix";
  logg->pure_name = "noprefix";

  logg->M = 0; 
  logg->N = 0; 
  logg->NSV = 10; 
  logg->NRHS = 20; 
  logg->threads = 1; 
  logg->max_threads = 1;
  logg->rows_fetched = 0;
  logg->PRINT_INFO = 1;
  logg->filewrite = 0;
  logg->power = 1;
  logg->mem = 0;

  logg->blockPower_total_its = 0;
  logg->blockPower_trace_error = 0.0;
  logg->blockPower_maxiter = 100;
  logg->blockPower_conv_crit = 1;
  logg->toll = 1e-3;

  logg->TIME_2_GS = 0; 
  logg->TIME_2_LOAD_MATRIX=0;
  logg->TIME_2_MM = 0; 
  logg->TIME_2_MM_A_TRANSPOSED = 0;
  logg->TIME_2_MM_A = 0; 
  logg->TIME_2_PROJECTED_SVD = 0;
  logg->TIME_2_GENERATE_RHS = 0;
  logg->TIME_2_OTHER = 0;
  
  logg->frob_norm_angle = 0;
  logg->cos_error = 0;
  logg->trueSVD = 0;

}

void  computeCosineError(double *MatA, double *MatB, int M, int NSV, double *CosineValues, double *CosineError){
  /* 
     MatA         : M x NSV -- computed NSV right singular vectors
     MatB         : M x NSV -- real dominant NSV right  singular vectors
     M            : Number of rows of the matrices                                                                                                                                                                 
     NSV          : Number of columns of the matrices
     CosineValues : Computing the quantity diag(MatA'MatB)
     CosineError  : Computing the quantity ||MatA'MatB-I||_2
  */

  int ii, jj;
  double *SingularValues;
  double *MatC, *U, *V, *superb;

  MatC = (double*) malloc(NSV*NSV*sizeof(double));
  SingularValues   = (double*) malloc(NSV*sizeof(double));
  superb = (double*) malloc(NSV*sizeof(double));

  cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,NSV,NSV,M,1,MatA,NSV,MatB,NSV,0,MatC,NSV);

  for ( ii = 0; ii < M; ii++ ){
    for ( jj = 0; jj < NSV; jj++){
      if ( ii == jj){
	CosineValues[jj] = MatC[ii*NSV+jj];
	MatC[ii*NSV+jj] = fabs(MatC[ii*NSV+jj])-1;
      }
    }
  }

  LAPACKE_dgesvd(LAPACK_ROW_MAJOR,'N','N',NSV,NSV,MatC,NSV,SingularValues,U,NSV,V,NSV,superb);


  *CosineError = SingularValues[0];

  free(MatC);
  free(SingularValues);
}

void standardize(double *normX, unsigned char *nnX, int N, double &avg, double &sd, double &inv_sqrtM)
{
    double sum = 0;
    unsigned int ctr = 0;

    #pragma omp parallel for reduction(+:sum, ctr)
    for (unsigned int im = 0; im < N; im++)
    {
        double x = (double)nnX[im];
        if (nnX[im] != MISSING)
        {
            sum += x;
            ctr++;
        }
    }

    double all_freq = sum / ctr;
    double P = all_freq / 2.0;
    double stddev = sqrt(2.0 * P * (1 - P));
    avg = all_freq;
    sd = stddev;

    #pragma omp parallel for schedule(static)
    for (unsigned int jm = 0; jm < N; jm++)
    {
        double x = (double)nnX[jm];
        if (x == MISSING)
            normX[jm] = 0;
        else
            normX[jm] = (x - avg)/stddev * inv_sqrtM;

        if (isnan(normX[jm]))
            normX[jm] = 0;
    }
}

double computeSquare(double x) {return x*x;};

void GetBimInfo( vector< string> lines,struct logistics *logg){
	for(unsigned int i = 0 ; i < lines.size() ; i++)
   {
       stringstream ss(lines[i]);
       string s;
       vector< string> tokens;

      while(ss >> s)
	 tokens.push_back(s);
      logg->snp_ids.push_back(tokens[1]);
   }
}

void GetFamInfo(vector<string> lines,struct logistics *logg){
	for(unsigned int i = 0 ; i < lines.size() ; i++)
   {
       stringstream ss(lines[i]);
       string s;
       vector< string> tokens;
      while(ss >> s)
		tokens.push_back(s);
		logg->fam_ids.push_back(tokens[0]);
		logg->indiv_ids.push_back(tokens[1]);
   }
}
  
string timestamp(struct logistics *logg){
   if(logg->show_timestamp)
   {
      time_t t = time(NULL);
      char *s = asctime(localtime(&t));
      s[strlen(s) - 1] = '\0';
       string str(s);
      str =  string("[") + str +  string("] ");
      return str;
   }
   else
      return  string("");
}

void decode_plink_sse2(uint8_t* __restrict out, const uint8_t* __restrict in, unsigned n)
{
    unsigned i = 0;
    
    const __m128i lut = _mm_setr_epi8(0,3,1,2, 0,3,1,2, 0,3,1,2, 0,3,1,2);
    const __m128i mask02 = _mm_set1_epi8(0x03);

    for (; i + 16 <= n; i += 16)
    {
        __m128i x = _mm_loadu_si128((const __m128i*)(in + i));

        __m128i g0 = _mm_and_si128(x, mask02);
        __m128i g1 = _mm_and_si128(_mm_srli_epi16(x, 2), mask02);
        __m128i g2 = _mm_and_si128(_mm_srli_epi16(x, 4), mask02);
        __m128i g3 = _mm_and_si128(_mm_srli_epi16(x, 6), mask02);

        __m128i p0 = _mm_unpacklo_epi8(g0, g1);
        __m128i p1 = _mm_unpackhi_epi8(g0, g1);
        __m128i p2 = _mm_unpacklo_epi8(g2, g3);
        __m128i p3 = _mm_unpackhi_epi8(g2, g3);

        __m128i o0 = _mm_unpacklo_epi16(p0, p2);
        __m128i o1 = _mm_unpackhi_epi16(p0, p2);
        __m128i o2 = _mm_unpacklo_epi16(p1, p3);
        __m128i o3 = _mm_unpackhi_epi16(p1, p3);

        __m128i m0 = _mm_shuffle_epi8(lut, o0);
        __m128i m1 = _mm_shuffle_epi8(lut, o1);
        __m128i m2 = _mm_shuffle_epi8(lut, o2);
        __m128i m3 = _mm_shuffle_epi8(lut, o3);

        _mm_storeu_si128((__m128i*)(out + i * 4 +  0), m0);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 16), m1);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 32), m2);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 48), m3);
    }

    for (; i < n; i++)
    {
        uint8_t b = in[i];
        unsigned base = i * 4;
        out[base+0] = (uint8_t[4]){0,3,1,2}[(b >> 0) & 3];
        out[base+1] = (uint8_t[4]){0,3,1,2}[(b >> 2) & 3];
        out[base+2] = (uint8_t[4]){0,3,1,2}[(b >> 4) & 3];
        out[base+3] = (uint8_t[4]){0,3,1,2}[(b >> 6) & 3];
    }
}

void decode_plink_precomp_sse2(uint8_t* __restrict out, const uint8_t* __restrict in, unsigned n)
{
    unsigned i = 0;
    
    const __m128i lut = _mm_setr_epi8(3,1,2,0, 3,1,2,0, 3,1,2,0, 3,1,2,0);
    const __m128i mask02 = _mm_set1_epi8(0x03);

    for (; i + 16 <= n; i += 16)
    {
        __m128i x = _mm_loadu_si128((const __m128i*)(in + i));

        __m128i g0 = _mm_and_si128(x, mask02);
        __m128i g1 = _mm_and_si128(_mm_srli_epi16(x, 2), mask02);
        __m128i g2 = _mm_and_si128(_mm_srli_epi16(x, 4), mask02);
        __m128i g3 = _mm_and_si128(_mm_srli_epi16(x, 6), mask02);

        __m128i p0 = _mm_unpacklo_epi8(g0, g1);
        __m128i p1 = _mm_unpackhi_epi8(g0, g1);
        __m128i p2 = _mm_unpacklo_epi8(g2, g3);
        __m128i p3 = _mm_unpackhi_epi8(g2, g3);

        __m128i o0 = _mm_unpacklo_epi16(p0, p2);
        __m128i o1 = _mm_unpackhi_epi16(p0, p2);
        __m128i o2 = _mm_unpacklo_epi16(p1, p3);
        __m128i o3 = _mm_unpackhi_epi16(p1, p3);

        __m128i m0 = _mm_shuffle_epi8(lut, o0);
        __m128i m1 = _mm_shuffle_epi8(lut, o1);
        __m128i m2 = _mm_shuffle_epi8(lut, o2);
        __m128i m3 = _mm_shuffle_epi8(lut, o3);

        _mm_storeu_si128((__m128i*)(out + i * 4 +  0), m0);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 16), m1);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 32), m2);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 48), m3);
    }

    for (; i < n; i++)
    {
        uint8_t b = in[i];
        unsigned base = i * 4;
        out[base+0] = (uint8_t[4]){3,1,2,0}[(b >> 0) & 3];
        out[base+1] = (uint8_t[4]){3,1,2,0}[(b >> 2) & 3];
        out[base+2] = (uint8_t[4]){3,1,2,0}[(b >> 4) & 3];
        out[base+3] = (uint8_t[4]){3,1,2,0}[(b >> 6) & 3];
    }
}

void Read_Bed_Local(std::ifstream &in, double *temp3, struct logistics *logg) {

    const size_t M = (size_t)logg->M;
    const size_t local_N = (size_t)logg->local_N;
    const size_t local_N_start = (size_t)logg->local_N_start;

    const size_t np = (size_t)ceil((double)M / PACK_DENSITY);

    unsigned char *decbin   = new unsigned char[np * PACK_DENSITY];
    unsigned char *readbin  = new unsigned char[np];
    double *norm_tmp        = new double[M];

    double inv_sqrtN = 1.0 / sqrt((double)logg->N);

    size_t snp_offset = 3 + local_N_start * np;
    in.seekg((std::streamoff)snp_offset, std::ios::beg);

    double avg, sd;

    for (size_t j = 0; j < local_N; j++) {

        in.read((char*)readbin, np);

        decode_plink_sse2(decbin, readbin, np);
        standardize(norm_tmp, decbin, (int)M, avg, sd, inv_sqrtN);

        size_t base = j * M;

        for (size_t k = 0; k < M; k++) {
            temp3[base + k] = norm_tmp[k];
        }
    }

    delete[] norm_tmp;
    delete[] decbin;
    delete[] readbin;
}