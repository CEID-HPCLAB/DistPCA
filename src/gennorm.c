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

  // Generate u
  u = 0.0;
  while (u == 0.0)
    u = rand_val(0);

  // Compute r
  r = sqrt(-2.0 * log(u));

  // Generate theta
  theta = 0.0;
  while (theta == 0.0)
    theta = 2.0 * PI * rand_val(0);

  // Generate x value
  x = r * cos(theta);

  // Adjust x value for specified mean and variance
  norm_rv = (x * std_dev) + mean;

  // Return the normally distributed RV value
  return(norm_rv);
}

// This is a linear congruential generator that returns a pseudorandom value between 0.0 and 1.0
double rand_val(int seed)
{
  const long  a =      16807;  // Multiplier
  const long  m = 2147483647;  // Modulus
  const long  q =     127773;  // m div a
  const long  r =       2836;  // m mod a
  static long x;               // Random int value
  long        x_div_q;         // x divided by q
  long        x_mod_q;         // x modulo q
  long        x_new;           // New x value

  // Set the seed if argument is non-zero and then return zero
  if (seed > 0)
    {
      x = seed;
      return(0.0);
    }

  // RNG using integer arithmetic
  x_div_q = x / q;
  x_mod_q = x % q;
  x_new = (a * x_mod_q) - (r * x_div_q);
  if (x_new > 0)
    x = x_new;
  else
    x = x_new + m;

  // Return a random value between 0.0 and 1.0
  return((double) x / m);
}
