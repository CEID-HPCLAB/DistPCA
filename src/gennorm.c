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
#include "gennorm.h"

double norm2(double mean, double std_dev)
{
  double   u, r, theta;
  double   x;
  double   norm_rv;

  u = 0.0;
  while (u == 0.0)
    u = rand_val(0);

  r = sqrt(-2.0 * log(u));

  theta = 0.0;
  while (theta == 0.0)
    theta = 2.0 * PI * rand_val(0);

  x = r * cos(theta);

  norm_rv = (x * std_dev) + mean;

  return(norm_rv);
}

double rand_val(int seed)
{
  const long  a =      16807; 
  const long  m = 2147483647;  
  const long  q =     127773;  
  const long  r =       2836;  
  static long x;              
  long        x_div_q;         
  long        x_mod_q;         
  long        x_new;           


  if (seed > 0)
    {
      x = seed;
      return(0.0);
    }

  x_div_q = x / q; x_mod_q = x % q;
  x_new = (a * x_mod_q) - (r * x_div_q);
  
  if (x_new > 0)
    x = x_new;
  
  else
    x = x_new + m;

  return((double) x / m);
}
