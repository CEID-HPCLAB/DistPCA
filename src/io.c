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
#include <string.h>
#include "io.h"

int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv)
{
    int *outint;
    double *outdouble;
    char *outchar;

    for (int i = 0; i < argc; i++) {

        if (argv[i][0] != '-')
            continue;

        if (!strcmp(argname, argv[i] + 1)) {

            if (type == NA)
                return 1;

            if (i + 1 >= argc)
                return 0;

            switch (type) {

                case INT:
                    outint = (int *)val;
                    *outint = atoi(argv[i + 1]);
                    return 1;

                case DOUBLE:
                    outdouble = (double *)val;
                    *outdouble = atof(argv[i + 1]);
                    return 1;

                case STR:
                    outchar = (char *)val;
                    sprintf(outchar, "%s", argv[i + 1]);
                    return 1;

                default:
                    printf("unknown arg type\n");
                    return 0;
            }
        }
    }

    return 0;
}