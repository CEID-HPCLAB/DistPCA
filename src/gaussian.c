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
#include <math.h>
#include "gaussian.h"

#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1 + (IM - 1) / NTAB)
#define EPS 1.2e-7
#define RNMX (1.0 - EPS)

float ran1(long *idum)
{
    int j;
    long k;
    static long iy = 0;
    static long iv[NTAB];
    float temp;

    if (*idum <= 0 || !iy) {
        if (-(*idum) < 1)
            *idum = 1;
        else
            *idum = -(*idum);

        for (j = NTAB + 7; j >= 0; j--) {
            k = (*idum) / IQ;
            *idum = IA * (*idum - k * IQ) - IR * k;

            if (*idum < 0)
                *idum += IM;

            if (j < NTAB)
                iv[j] = *idum;
        }

        iy = iv[0];
    }

    k = (*idum) / IQ;
    *idum = IA * (*idum - k * IQ) - IR * k;

    if (*idum < 0)
        *idum += IM;

    j = iy / NDIV;
    iy = iv[j];
    iv[j] = *idum;

    temp = AM * iy;

    if (temp > RNMX)
        return RNMX;

    return temp;
}

#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX


float gasdev(long *idum)
{
    float ran1(long *idum);

    static int iset = 0;
    static float gset;

    float fac, rsq, v1, v2;

    if (iset == 0) {
        do {
            v1 = 2.0f * ran1(idum) - 1.0f;
            v2 = 2.0f * ran1(idum) - 1.0f;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0f || rsq == 0.0f);

        fac = sqrt(-2.0f * log(rsq) / rsq);
        gset = v1 * fac;
        iset = 1;

        return v2 * fac;
    } else {
        iset = 0;
        return gset;
    }
}