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
#include "structures.h"

string toString(int value);

int GetRamInKB(void);

int get_llc_size_kb(void);

string ExtractFileName(const std::string& fullPath);

string ConstructFilename(struct logistics logg, string fileType);

void print_statistics(struct logistics logg);

void initialize_structure(struct logistics *logg);

void computeCosineError(double *MatA, double *MatB, int M, int NSV, double *CosineValues, double *CosineError);

void decode_plink_sse2(unsigned char * __restrict__ out, const unsigned char * __restrict__ in, const unsigned int n);

void decode_plink_precomp_sse2(unsigned char * __restrict__ out, const unsigned char * __restrict__ in, const unsigned int n);
   
void standardize(double *normX, unsigned char *nnX, int N, double &avg, double &sd, double &inv_sqrtM);

double computeSquare (double x); 

void GetFamInfo(vector<string> famlines,struct logistics *logg);

void GetBimInfo(vector<string> bimlines,struct logistics *logg);

void Read_Bed_Local(std::ifstream &in, double *temp3, struct logistics *logg);

string timestamp(struct logistics *logg);