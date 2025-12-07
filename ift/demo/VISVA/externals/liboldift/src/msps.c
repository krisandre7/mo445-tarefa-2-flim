#include "msps.h"

/***************************************************************
 * Multiscale Parameter Search (MSPS)
 *
 * A sample application can be found in ift/util/src/imgreg.c
 **************************************************************/

/*Whether the method is for minimization or maximization. */
/* > is maximizatio, < is minimization */
#define MinMaxOp >

/***************************************************************
 * These functions are, up to now, of particular use of MSPS.
 * They can be made public if it is desirable.
 **************************************************************/
void InitVector(double *v,  double value, size_t n);
void CopyVector(double *dest,  double *src, size_t n);
void SumVector(double *dest,  double *a,  double *b, size_t n);

/***************************************************************
 * Of particular use of MSPS.
 **************************************************************/
char BoundTest(double *x, size_t i,  double *lower,  double *upper);
void PrintFunctionEval( char *label,  double *theta, size_t n, double v );

/***************************************************************
 * The optimization itself.
 **************************************************************/
double MSPS( double *param, void* context)
{
  /* Cast void-ptr context to correct struct-type. */
  struct optContext *c = (struct optContext*) context;

  /* Clone context to local variables for easier reference. */
  problem f = c->f;                         /* Fitness function. */
  void *fContext = c->fContext;             /* Context for fitness function. */
  size_t n = c->fDim;                       /* Dimensionality of problem. */
  double * lowerInit  = c->lowerInit;  /* Lower initialization boundary. */
  double * upperInit  = c->upperInit;  /* Upper initialization boundary. */
  double * lowerBound = c->lowerBound; /* Lower search-space boundary. */
  double * upperBound = c->upperBound; /* Upper search-space boundary. */
  size_t numIterations = c->numIterations;  /* Number of iterations to perform. */
  char verbose = c->verbose;

  /* Retrieve the parameters of the method. */
  double scales = param[0];
  double degree = param[1];
  double gamma  = param[2];

  /* Initialize search-range factor and decrease-factor. */
  double r = 1;                    /* Search-range. */
  double q = 1 / pow(2.0, gamma);  /* Decrease-factor. */

  /* Iteration variables. */
  size_t i, j, k=0, bigIter=1, bestP=0;

  /* Rounded scales*/
  size_t rScales = (size_t) (scales + 0.5);

  double scaleNorm;

  /* Allocate and set up the delta matrix. */
  double **delta = (double**) malloc(sizeof(double*)*n);
  for ( i=0; i < n; i++ )
  {
    delta[i] = AllocDoubleArray(rScales);
    scaleNorm = pow( rScales, degree ) / (( upperInit[i] - lowerInit[i] ) / 2 );
    for ( j=1; j <= rScales; j++ )
    {
      delta[i][j-1] = pow( j, degree ) / scaleNorm;
    }
  }

  /* Allocate thetas and deltas. */
  double *theta      = AllocDoubleArray(n);
  double *thetaTmp   = AllocDoubleArray(n);
  double *thetaPrime = AllocDoubleArray(n);

  double *deltaPrimeS = AllocDoubleArray(n);
  double *deltaPrimeP = AllocDoubleArray(n);

  double *VP          = AllocDoubleArray(n);

  /* Temporary store. */
  double pivot;
  char bounded;

  /* Auxiliary flags to avoid unnecessary function evaluations */
  char testIntra, testInter;

  /* Function value variables. */
  double V, V0, Vneg, Vpos, Vprime;

  /* Initial position. */
  CopyVector(theta, c->theta, n);
  CopyVector(thetaPrime, theta, n);

  /* Compute the function value at the initial position. */
  Vprime = f( theta, fContext );

  do
  {
    if ( verbose )
    {
      printf( "iter #%lu\n", bigIter++ );
      PrintFunctionEval( "  init:", thetaPrime, n, Vprime );
    }

    V0 = Vprime;
    CopyVector(theta, thetaPrime, n);
    InitVector(deltaPrimeP, 0.0, n);
    InitVector(VP, Vprime, n);
    testInter = 0;

    for (j=0; j < rScales && k < numIterations; j++) //Scales
    {
      if ( verbose ) printf( "\n  scale #%lu\n", j+1 );

      testIntra = 0;

      for (i=0; i < n && k < numIterations; i++) //Parameters
      {
        if ( verbose ) printf( "    par #%lu\n", i+1 );

        deltaPrimeS[i] = 0;
        V = V0;
        pivot = theta[i];
        theta[i] =  pivot + ( delta[i][j] * r );

        bounded = BoundTest(theta, i, lowerBound, upperBound);
        if (!bounded)
        {
          Vpos = f( theta, fContext ); k++;
          if ( Vpos MinMaxOp V )      { V = Vpos;     deltaPrimeS[i] = delta[i][j] * r; testIntra = 1; }
          if ( Vpos MinMaxOp VP[i] )  { VP[i] = Vpos; deltaPrimeP[i] = delta[i][j] * r; testInter = 1; }
          if ( Vpos MinMaxOp Vprime ) { Vprime = Vpos; }

          if (verbose ) PrintFunctionEval( "      pos:", theta, n, Vpos );
        }

        theta[i] =  pivot - ( delta[i][j] * r );
        bounded = BoundTest(theta, i, lowerBound, upperBound);
        if (!bounded)
        {
          Vneg = f( theta, fContext ); k++;
          if ( Vneg MinMaxOp V )      { V = Vneg;      deltaPrimeS[i] = -delta[i][j] * r; testIntra = 1; }
          if ( Vneg MinMaxOp VP[i] )  { VP[i] = Vneg;  deltaPrimeP[i] = -delta[i][j] * r; testInter = 1; }
          if ( Vneg MinMaxOp Vprime ) { Vprime = Vneg; }

          if (verbose ) PrintFunctionEval( "      neg:", theta, n, Vneg );
        }
        theta[i] = pivot;
      }
      //Test the scale's composed displacement
      if ( testIntra )
      {
        SumVector(thetaTmp, theta, deltaPrimeS, n);
        V = f( thetaTmp, fContext ); k++;
        if ( V MinMaxOp Vprime )  { Vprime = V;  CopyVector(thetaPrime, thetaTmp, n); }

        if (verbose ) PrintFunctionEval( "\n    comp:", thetaTmp, n, V );
      }
    }

    //Test the interscale composed displacement
    if ( rScales > 1 && testInter )
    {
      SumVector(thetaTmp, theta, deltaPrimeP, n);
      V = f(thetaTmp, fContext ); k++;
      if ( V MinMaxOp Vprime )  { Vprime = V; CopyVector(thetaPrime, thetaTmp, n); }

      if (verbose ) PrintFunctionEval( "\n  inter-comp.:", thetaTmp, n, V );
    }

    /***************************************************************
     * Test the best scale or interscale composed displacement
     * against the individual parameter best fitness.
     *
     * Here there is a difference between the pseudo-code and the
     * implementation due to efficiency.
     *
     * The initial sum is just to ensure that if V==Vprime after the
     * loop, then the best displacement is certainly based on only
     * one of the parameters.
     **************************************************************/
    if ( Vprime MinMaxOp Vprime + 1.0 )
      V = Vprime + 1.0; //minimization
    else
      V = Vprime - 1.0; //maximization

    for (i=0; i< n; i++) //parameters (dimensions)
    {
      if ( VP[i] MinMaxOp V ) { bestP = i; V = VP[i]; }
    }
    if ( V == Vprime ) //implies that the best fitness comes from the axis
    {
      CopyVector(thetaTmp, theta, n);
      thetaTmp[bestP] = thetaTmp[bestP] + deltaPrimeP[bestP];
      CopyVector(thetaPrime, thetaTmp, n);
    }

    if (verbose )
    {
      PrintFunctionEval( "best:", thetaPrime, n, Vprime ); printf("\n");
    }

    //If no improvement was obtained, decrease the sampling-range.
    if ( Vprime == V0)
      r *= q;

  } while ( k < numIterations );

  CopyVector( c->theta, thetaPrime, n );

  /* Delete thetas and deltas. */
  for (i=0; i < n; i++ )
    free(delta[i]);

  free(delta);

  free(theta);
  free(thetaTmp);
  free(thetaPrime);

  free(deltaPrimeS);
  free(deltaPrimeP);
  free(VP);

  /* Return best-found function value. */
  return Vprime;
}

/***************************************************************
 * These functions are, up to now, of particular use of MSPS.
 * They can be made public if it is desirable.
 **************************************************************/

/***************************************************************
 * Initializes vector v of size n with values value
 **************************************************************/
void InitVector(double *v,  double value, size_t n)
{
  size_t i;

  assert(v);

  for (i=0; i<n; i++)
  {
    v[i] = value;
  }
}
/***************************************************************
 * Copy vector src of size n to vector dest.
 **************************************************************/
void CopyVector(double *dest,  double *src, size_t n)
{
  size_t i;

  assert(dest);
  assert(src);

  for (i=0; i<n; i++)
  {
    dest[i] = src[i];
  }
}

/***************************************************************
 * Sum vectors a and b and store the result in vector dest.
 **************************************************************/
void SumVector(double *dest,  double *a,  double *b, size_t n)
{
   size_t i;

   assert(dest);
   assert(a);
   assert(b);

   for (i=0; i<n; i++)
   {
      dest[i] = a[i]+b[i];
   }
}
/***************************************************************
 * Test if position x is within the search-space.
 **************************************************************/
char BoundTest(double *x, size_t i,  double *lower,  double *upper)
{
   assert(upper[i] >= lower[i]);

   if (x[i] < lower[i])
   {
      x[i] = lower[i];
      return 1;
   }
   else if (x[i] > upper[i])
   {
      x[i] = upper[i];
      return 1;
   }
   return 0;
}

/***************************************************************
 * Print details of the function evaluation in the given theta.
 **************************************************************/
void PrintFunctionEval( char *label,  double *theta, size_t n, double v )
{
  size_t i;

  printf("%s ", label);
  for ( i=0; i<n; i++ )
  {
    printf( "%+02.2f ",theta[i] );
    fflush(stdout);
  }

  printf("- v: %f \n", v );
}
