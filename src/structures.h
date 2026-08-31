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
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <string.h>
#include <sstream>
#include <errno.h>
#include <stdexcept>
#include <iomanip>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ctype.h>

#define PACK_DENSITY 4
#define MISSING 3
#define MASK0 3	  /* 3 << 2 * 0 */
#define MASK1 12  /* 3 << 2 * 1 */
#define MASK2 48  /* 3 << 2 * 2 */
#define MASK3 192 /* 3 << 2 * 3 */

using namespace std;
using std::vector;

#ifndef LOGISTICS_STRUCT
#define LOGISTICS_STRUCT
struct logistics
{ 
  // Timing variable
  struct tm* timeinfo;

  string filename;
  string prefixname;
  string pure_name;

  // General variables
  int M, N, NSV, NRHS;
  int loops, rows_fetched;
  int PRINT_INFO;
  int power;
  double mem; 
  int    threads, max_threads;
  int    ram_KB;
  double ram_GB;

  // for blockPower
  int    blockPower_total_its;
  int    blockPower_maxiter;
  int    blockPower_conv_crit;
  double blockPower_trace_error; 
  double tol;                  
  vector<double> delta_iter;

  vector<double> left_sing_vecs;
  vector<double> sing_values;
  vector<double> cos_values;

  int filewrite;

  double TIME_2_GENERATE_RHS, TIME_2_LOAD_MATRIX, TIME_2_PROJECTED_SVD, TIME_2_TRUE_SVD, TIME_2_MM, TIME_2_GS, TIME_2_OTHER;
  double TIME_2_MM_A_TRANSPOSED, TIME_2_MM_A;

  double frob_norm_angle;
  double cos_error;
  int trueSVD;

	vector<string> indiv_ids;
	vector<string> fam_ids;
	vector<string> snp_ids;  
	bool show_timestamp ;

  // conf vars used for MPI exec
  int mpi_rank; int mpi_size;
  int local_N_start; int local_N_end; int local_N;
};
#endif