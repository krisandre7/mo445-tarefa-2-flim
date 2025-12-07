#ifndef _MSPS_H_
#define _MSPS_H_

#include <assert.h>
#include <oldift.h>

/* Type for a fitness function. That is, an optimization problem. */
typedef double (*problem) ( double *x, void *context);

struct optContext
{
  problem f;                 /* The fitness function to be optimized. */
  void *fContext;            /* The problem's context. */
  size_t fDim;               /* Dimensionality of problem. */
  double *theta;             /* Theta to be initially considered and to be return
                                as the best one in the end of the optimization. */
  double * lowerInit;   /* Lower initialization boundary. */
  double * upperInit;   /* Upper initialization boundary. */
  double * lowerBound;  /* Lower search-space boundary. */
  double * upperBound;  /* Upper search-space boundary. */
  size_t numIterations;      /* Number of fitness-evaluations. */
  char verbose;              /* Whether it prints information from the optmiation or not. */
};

double MSPS( double *param, void* context);

#endif //#ifndef _MSPS_H_
