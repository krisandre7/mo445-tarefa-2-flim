//IFT main header file
#include <ift.h>


/***************************************************************
 * Note: The three functions below are not strictly related to
 * the registration. They should be in another place, but are
 * still here until we decide the right place to put them.
 **************************************************************/
void NormalizeImage(Image *img);
void GetTransform(double T[3][3], double  *theta, int nrowsA, int ncolsA, int nrowsB, int ncolsB, bool inv);
Image *GetImgReged(double  *theta, Image  *fixed, Image  *deformable, char  superpose);
Image *GetMosaic(Image *img1, Image  *img2, int patchSize );

/*Whether it writes an output registered image or not. */
#define writeRegistration 1

/*Scalar above which the edges are considered in the registration */
#define edgeThreshold 30;

/***************************************************************
 * Context structure for image registration.
 **************************************************************/
struct ImgRegContext
{
  CImage *fixed;          /* Image to register to. */
  CImage *deformable;     /* Image to register from. */
};

/***************************************************************
 * Prepare color image to registration
 * Option: 1 - just replaces the luminance by the gradient.
 *         2 - replaces the luminance by the resulting Euclidean distance
 *             transform based on the gradient.
 **************************************************************/
CImage *Prepare2Registration( CImage  *cImg, char Option )
{
  CImage *cImgPre;
  Image  *img;
  AdjRel *A=Circular(1.5); //the radius is dependent on the image's size
  int    p;
  int threshold = edgeThreshold; // 0; Falcao

  cImgPre = CopyCImage(cImg);
  NormalizeImage( cImgPre->C[0] );     // Normalize luminance Images
  img = MorphGrad( cImgPre->C[0], A ); // EdgeDetection( cImgPre->C[0] ); // Compute gradient Falcao

  int x,y;

  if ( Option == 1)
  {
    for ( y=0; y < cImgPre->C[0]->nrows; y++)
    {
      for ( x=0; x < cImgPre->C[0]->ncols; x++)
      {
        p = x + cImgPre->C[0]->tbrow[y];
          //Edges up to a certain magnitude are set to zero.
        if (img->val[p] > threshold )
          cImgPre->C[0]->val[p]=img->val[p];
        else
          cImgPre->C[0]->val[p]=0;
      }
    }
    DestroyImage(&img);

  } else if ( Option == 2 )
  {
    
    for ( y=0; y < cImgPre->C[0]->nrows; y++)
    {
      for ( x=0; x < cImgPre->C[0]->ncols; x++)
      {
        p = x + cImgPre->C[0]->tbrow[y];
        //Edges from a certain magnitude are set as seed to the EDT .
        if (img->val[p] > threshold )
        {
          cImgPre->C[0]->val[p] = 0;
        }
        else
        {
          cImgPre->C[0]->val[p] = 1;
        }
      }
    }
    DestroyImage(&img);

    // Euclidean Distance Transform from the thresholded magnitude
    img = TSEDistTrans( cImgPre->C[0] );

    for ( y=0; y < img->nrows; y+=1)
      for ( x=0; x < img->ncols; x+=1)
      {
        p = x + img->tbrow[y];
        cImgPre->C[0]->val[p]=sqrt(img->val[p]);
      }
    
  }

  DestroyAdjRel(&A);
  DestroyImage(&img);
  return cImgPre;
}

/***************************************************************
 * Fitness function to be minimized in the optimization process
 **************************************************************/
double ImgReg(  double *x, void *context )
{
  struct ImgRegContext * c = (struct ImgRegContext *) context;

  double  fitness = 0.0;
  CImage  *fixed = c->fixed;
  CImage  *deformable = c->deformable;

  double T[3][3];
  double theta[4];
  theta[0] = x[0]; theta[1] = x[1];
  theta[2] = x[2]; theta[3] = x[3];

  GetTransform( T, theta, fixed->C[0]->nrows,
                          fixed->C[0]->ncols,
                          deformable->C[0]->nrows,
                          deformable->C[0]->ncols,
                          0); //Not inverse

  double x1, y1;
  int    x2, y2;
  int p, q, nelems=0;

  for ( y2 = 0; y2 < deformable->C[0]->nrows; y2++ )
  {
    for ( x2 = 0; x2 < deformable->C[0]->ncols; x2++ )
    {
      q  = x2 + deformable->C[0]->tbrow[y2];
      if ( deformable->C[0]->val[q] > 0 )
      {
        x1 = T[0][0]*x2 + T[0][1]*y2 + T[0][2];
        y1 = T[1][0]*x2 + T[1][1]*y2 + T[1][2];

        if ( ValidPixel( fixed->C[0], (int) x1, (int) y1 ) )
        {
          p  = (int) x1 + fixed->C[0]->tbrow[(int) y1];
          fitness = fitness + ( fixed->C[0]->val[p] * (deformable->C[0]->val[q]/255.0) );

        } else {
          /*
           * Extrapolates the fixed Euclidean Distance transformed image
           */
          double dx1 = 0;
          double dy1 = 0;
          int x1bound = (int) x1;
          int y1bound = (int) y1;
          if ( x1 < 0 )
          {
            dx1 = x1*-1;
            x1bound = 0;
          } else if ( x1 > fixed->C[0]->ncols )
          {
            dx1 = x1 - (double) fixed->C[0]->ncols;
            x1bound = fixed->C[0]->ncols-1;
          }
          if ( y1 < 0 )
          {
            dy1 = y1*-1;
            y1bound = 0;
          } else if ( y1 > fixed->C[0]->nrows )
          {
            dy1 = dy1 - (double) fixed->C[0]->nrows;
            y1bound = fixed->C[0]->nrows-1;
          }

          int pbound  = x1bound + fixed->C[0]->tbrow[y1bound];
          int ipbound = fixed->C[0]->val[pbound];

          fitness = fitness + ( ipbound * (deformable->C[0]->val[q]/255.0));
        }
        nelems++;
      }
    }
  }
  return ( fitness / nelems );
}

/************************************************************
 * Demo program for image registration
 ************************************************************/
int main(int argc, char **argv)
{
  /*--------------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if ( argc != 3 && argc != 5 )
  {
    fprintf(stderr,"usage: %s <fixed.ppm> <deformable.ppm> [fix-features.pgm] [def-features.pgm] \n",argv[0]);
    fprintf(stderr,"       output files imgreg-*.ppm from the registration. \n" );
    exit(-1);
  }

  struct ImgRegContext imgRegContext;

  /* Timing variables. */
  clock_t t1, t2;

  /* Timing start. */
  t1 = clock();

  char fixedFile[80];
  char deformableFile[80];

  /* Retrieves the arguments */
  sprintf(fixedFile,"%s",argv[1]);
  sprintf(deformableFile,"%s",argv[2]);

  /* Read and prepare images for registration. */
  CImage *fixedRGB,  *deformableRGB;
  CImage *fixed2Reg, *deformable2Reg, *tmpImage;

  fixedRGB  = ReadCImage( fixedFile );
  deformableRGB  = ReadCImage( deformableFile );

  /* If only two arguments were provided, proceed with the standard characterization. */
  if ( argc == 3 )
  {
    tmpImage  = CImageRGBtoYCbCr( fixedRGB );
    fixed2Reg = Prepare2Registration( tmpImage, 2 );
    DestroyCImage( &tmpImage );

    tmpImage       = CImageRGBtoYCbCr( deformableRGB );
    deformable2Reg = Prepare2Registration( tmpImage, 1 );
    DestroyCImage( &tmpImage );
  }
  else /* Otherwise, retrieve the optional parameters. */
  {
    fixed2Reg      = (CImage *) calloc(1, sizeof(CImage));
    deformable2Reg = (CImage *) calloc(1, sizeof(CImage));

    fixed2Reg->C[0] = ReadImage( argv[3] );
    deformable2Reg->C[0] = ReadImage( argv[4] );
  }

  /* MSPS control parameters: # of scales, its increasing
   * degree and the sampling-range decrease factor. */
  double settings[] = {2, 1.86, 0.66};

  size_t  numIterations = 500;   /* Number of iterations. */

  int nCols = deformable2Reg->C[0]->ncols;
  int nRows = deformable2Reg->C[0]->nrows;

  /*
   * Boundaries of the search space from which the optimization will be guided
   * Elements 0: Rotation in degrees
   *          1: x displacement in pixels
   *          2: y displacement in pixels
   *          3: scale factor
   */
  double lowerBound[] = {-20,-( nCols/2 ),-( nRows/2 ), 0.75};
  double upperBound[] = { 20, ( nCols/2 ), ( nRows/2 ), 1.25};
  /*
   * Boundaries for the initialization of the displacements.
   */
  double lowerInit[] = {-10,-( nCols/4 ),-( nRows/4 ), 0.875};
  double upperInit[] = { 10, ( nCols/4 ), ( nRows/4 ), 1.125};

  /* Define the context for the MSPS optimization */
  struct optContext mspsContext;

  mspsContext.f          = ImgReg;
  mspsContext.fContext   = (void*) &imgRegContext;
  mspsContext.fDim       = 4;
  mspsContext.theta      = AllocDoubleArray( mspsContext.fDim );
  mspsContext.lowerInit  = lowerInit;
  mspsContext.upperInit  = upperInit;
  mspsContext.lowerBound = lowerBound;
  mspsContext.upperBound = upperBound;
  mspsContext.numIterations = numIterations;
  mspsContext.verbose    = 1;

  double vPrime, *thetaPrime = AllocDoubleArray( mspsContext.fDim );

  /* Initialize theta as the identity transform. */
  mspsContext.theta[0] = 0.0; //Rotation
  mspsContext.theta[1] = 0.0; //X translation
  mspsContext.theta[2] = 0.0; //Y translation
  mspsContext.theta[3] = 1.0; //Scaling

  /* Set the image registration context. */
  imgRegContext.fixed = fixed2Reg;
  imgRegContext.deformable = deformable2Reg;

  vPrime = MSPS( settings, (void*) &mspsContext );

  /* Retrives thetaPrime. */
  size_t i;
  for ( i=0; i < mspsContext.fDim; i++ )
    thetaPrime[i] = mspsContext.theta[i];

  printf("Fitness value: %g \n", vPrime);
  printf("Theta in vector notation:\n[ %g, %g, %g, %g ]\n",
          thetaPrime[0], thetaPrime[1], thetaPrime[2], thetaPrime[3] );

  if (writeRegistration)
  {
    if ( argc ==3 )
    {
      WriteImage(imgRegContext.fixed->C[0], "imgreg-fixedEDT.ppm");
      WriteImage(imgRegContext.deformable->C[0], "imgreg-deformableGrad.ppm");
    }

    CImage reged1, reged2;
    Image *reged3;

    reged1.C[0] = GetImgReged( thetaPrime, fixedRGB->C[0], deformableRGB->C[0], 0 );
    reged1.C[1] = GetImgReged( thetaPrime, fixedRGB->C[1], deformableRGB->C[1], 0 );
    reged1.C[2] = GetImgReged( thetaPrime, fixedRGB->C[2], deformableRGB->C[2], 0 );

    reged2.C[0] = GetMosaic( fixedRGB->C[0], reged1.C[0], 8 );
    reged2.C[1] = GetMosaic( fixedRGB->C[1], reged1.C[1], 8 );
    reged2.C[2] = GetMosaic( fixedRGB->C[2], reged1.C[2], 8 );

    WriteCImage(&reged1, "imgreg-reged1.ppm");
    WriteCImage(&reged2, "imgreg-reged2.ppm");

    DestroyImage(&(reged1.C[0]));
    DestroyImage(&(reged1.C[1]));
    DestroyImage(&(reged1.C[2]));
    DestroyImage(&(reged2.C[0]));
    DestroyImage(&(reged2.C[1]));
    DestroyImage(&(reged2.C[2]));

    reged3 = GetImgReged(thetaPrime,
                        imgRegContext.fixed->C[0],
                        imgRegContext.deformable->C[0],
                        1);
    WriteImage(reged3, "imgreg-reged3.ppm");
    DestroyImage(&reged3);
  }

  /* Timing end. */
  t2 = clock();

  /* Display results of optimization along with time-usage. */
  printf("\nTime usage for the registration: %g seconds\n",
         (double)(t2 - t1) / CLOCKS_PER_SEC);

  DestroyCImage(&fixedRGB);

  DestroyCImage(&deformableRGB);


  if ( argc == 3 )
  {
    DestroyCImage(&fixed2Reg);
    DestroyCImage(&deformable2Reg);
  }
  else
  {
    DestroyImage( &(fixed2Reg->C[0]) );
    DestroyImage( &(deformable2Reg->C[0]) );
    free(fixed2Reg);
    free(deformable2Reg);
  }

  free(mspsContext.theta);
  free(thetaPrime);

  /* ------------------------------------------------------ */
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
     MemDinInicial,MemDinFinal);
  /* ------------------------------------------------------ */

  return 0;
}

/***************************************************************
 * The functions below are not strictly related to the
 * registration anymore. They should be in another place, but are
 * still here until we decide the right place to put them.
 **************************************************************/

/***************************************************************
 * Stretching from 0 to 255.
 **************************************************************/
void NormalizeImage(Image *img)
{
  int p,n;
  float a;
  int f1=MinimumValue(img);
  int f2=MaximumValue(img);
  int g1=0, g2=255;

  n = img->ncols*img->nrows;
  if (f1 != f2)
    a = (float)(g2-g1)/(float)(f2-f1);
  else
    a = INT_MAX;

  for (p=0; p < n; p++)
  {
    if (a != INT_MAX)
      img->val[p] = (int)(a*(img->val[p]-f1)+g1);
    else
      img->val[p] = g2;
  }
}

/***************************************************************
 * Linear interpolation.
 **************************************************************/
float Interpolate( float dqa, float dqb, int iqa, int iqb )
{
  return dqb*(float)iqa + dqa*(float)iqb;
}

/***************************************************************
 * Returns the matrix for image registration based on theta.
 **************************************************************/
void GetTransform(double T[3][3], double  *theta, int nrowsA, int ncolsA, int nrowsB, int ncolsB, bool inv)
{
  double trash, sin_alpha, cos_alpha;
  double rotation= 360.*modf(theta[0]/360.,&trash);;
  if (rotation < 0.0)
    rotation += 360.0;

  rotation = rotation*PI/180.0;
  cos_alpha = cos(rotation);
  sin_alpha = sin(rotation);

  double dx, dy;
  double scale;

  double xcA=ncolsA/2.0, xcB=ncolsB/2.0;
  double ycA=nrowsA/2.0, ycB=nrowsB/2.0;

  float R[3][3],S[3][3],Txy[3][3],TcA[3][3],TcB[3][3],M1[3][3],M2[3][3];
  int i,j,k;

  if ( !inv )
  {
    scale = theta[3];
    dx=theta[1];
    dy=theta[2];
    xcB=-xcB; ycB=-ycB;
  }
  else
  {
    sin_alpha = -sin_alpha;
    scale = 1/theta[3];
    dx=-theta[1];
    dy=-theta[2];
    xcA=-xcA; ycA=-ycA;
  }

  R[0][0]=cos_alpha;   R[0][1]=sin_alpha;   R[0][2]=0.0;
  R[1][0]=-sin_alpha;  R[1][1]=cos_alpha;   R[1][2]=0.0;
  R[2][0]=0.0;         R[2][1]=0.0;         R[2][2]=1.0;

  S[0][0]=scale;       S[0][1]=0.0;         S[0][2]=0.0;
  S[1][0]=0.0;         S[1][1]=scale;       S[1][2]=0.0;
  S[2][0]=0.0;         S[2][1]=0.0;         S[2][2]=1.0;

  Txy[0][0]=1.0;       Txy[0][1]=0.0;       Txy[0][2]=dx;
  Txy[1][0]=0.0;       Txy[1][1]=1.0;       Txy[1][2]=dy;
  Txy[2][0]=0.0;       Txy[2][1]=0.0;       Txy[2][2]=1.0;

  TcA[0][0]=1.0;       TcA[0][1]=0.0;       TcA[0][2]=xcA;
  TcA[1][0]=0.0;       TcA[1][1]=1.0;       TcA[1][2]=ycA;
  TcA[2][0]=0.0;       TcA[2][1]=0.0;       TcA[2][2]=1.0;

  TcB[0][0]=1.0;       TcB[0][1]=0.0;       TcB[0][2]=xcB;
  TcB[1][0]=0.0;       TcB[1][1]=1.0;       TcB[1][2]=ycB;
  TcB[2][0]=0.0;       TcB[2][1]=0.0;       TcB[2][2]=1.0;

  //Matrix chain multiplication
  if ( !inv )
  {
    /************************************************/
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        M1[k][j] = 0.0;
    // M1 = R*TcB -- center B at its origin and rotates
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        for (i=0; i < 3; i++)
          M1[k][j]+=R[k][i]*TcB[i][j];
    /************************************************/
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        M2[k][j] = 0.0;
    // M2 = Txy*M1 -- translates
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        for (i=0; i < 3; i++)
          M2[k][j]+=Txy[k][i]*M1[i][j];
    /************************************************/
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        M1[k][j] = 0.0;
    // M1 = S*M2 -- scales
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        for (i=0; i < 3; i++)
          M1[k][j]+=S[k][i]*M2[i][j];
    /************************************************/
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        T[k][j] = 0.0;
    // T = TcA*M1 -- return back to the center of A
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        for (i=0; i < 3; i++)
          T[k][j]+=TcA[k][i]*M1[i][j];
    /************************************************/
  }
  else //Inverse transform
  {
    /************************************************/
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        M1[k][j] = 0.0;
    // M1 = S*TcA -- center B at the origin of A and scales back
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        for (i=0; i < 3; i++)
          M1[k][j]+=S[k][i]*TcA[i][j];
    /************************************************/
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        M2[k][j] = 0.0;
    // M2 = Txy*M1 -- translates back
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        for (i=0; i < 3; i++)
          M2[k][j]+=Txy[k][i]*M1[i][j];
    /************************************************/
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        M1[k][j] = 0.0;
    // M1 = R*M2 -- rotates back
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        for (i=0; i < 3; i++)
          M1[k][j]+=R[k][i]*M2[i][j];
    /************************************************/
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        T[k][j] = 0.0;
    // T = TcB*M1 -- return back to the center of B
    for (k=0; k < 3; k++)
      for (j=0; j < 3; j++)
        for (i=0; i < 3; i++)
          T[k][j]+=TcB[k][i]*M1[i][j];
    /************************************************/
  }
}

/***************************************************************
 * Returns the deformable image in the fixed's domain,
 * superposing them or not.
 **************************************************************/
Image *GetImgReged(double  *theta, Image  *fixed, Image  *deformable, char superpose)
{
  double T[3][3];

  GetTransform( T, theta, fixed->nrows,
                          fixed->ncols,
                          deformable->nrows,
                          deformable->ncols,
                          1); //Inverse

  Image* reged = CreateImage(fixed->ncols, fixed->nrows );

  int   x1, y1;
  float x2, y2;
  int p, q1, q2, q3, q4;

  for ( y1 = 0; y1 < fixed->nrows; y1++ )
  {
    for ( x1 = 0; x1 < fixed->ncols; x1++ )
    {
      p  = x1 + fixed->tbrow[y1];

      if ( superpose )
        reged->val[p] = fixed->val[p];

      x2 = T[0][0]*x1 + T[0][1]*y1 + T[0][2];
      y2 = T[1][0]*x1 + T[1][1]*y1 + T[1][2];

      if ( ValidPixel( deformable, (int) x2  , (int) y2 ) &&
           ValidPixel( deformable, (int) x2+1, (int) y2+1 ))
      {
        /* Bilinear Interpolation */
        q1 = (int) x2   + deformable->tbrow[(int) y2];
        q2 = q1+1;
        q3 = (int) x2   + deformable->tbrow[(int) y2+1];
        q4 = (int) x2+1 + deformable->tbrow[(int) y2+1];

        float dqa = x2 - ((int) x2);
        float dqb = ((int) x2+1) - x2;
        int iqa = deformable->val[q1];
        int iqb = deformable->val[q2];
        float iq12 = Interpolate( dqa, dqb, iqa, iqb );

        iqa = deformable->val[q3];
        iqb = deformable->val[q4];
        float iq34 = Interpolate( dqa, dqb, iqa, iqb );

        dqa = y2 - ((int) y2);
        dqb = ((int) y2+1) - y2;

        float ip = Interpolate( dqa, dqb, iq12, iq34 );

        reged->val[p] = MAX(reged->val[p], ip );
      }
    }
  }
  return(reged);
}

/***************************************************************
 * Returns a mosaic from img1 and img2 with respect to a square
 * patch of size patchSize.
 **************************************************************/
Image *GetMosaic(Image *img1, Image  *img2, int patchSize )
{
  /* In this implementation, the images must have the same size. */
  assert( img1->ncols == img2->ncols );
  assert( img1->nrows == img2->nrows );

  Image* mosaic = CreateImage(img1->ncols, img1->nrows );

  int  x, y, p;
  double rowFlag, colFlag;

  for ( y = 0; y < img1->nrows; y++ )
  {
    for ( x = 0; x < img1->ncols; x++ )
    {
      p  = x + img1->tbrow[y];

      rowFlag = fmod( floor( (double) ( y / patchSize ) ), 2 );
      colFlag = fmod( floor( (double) ( x / patchSize ) ), 2 );

      if ( rowFlag == 0.0 )
      {
        if ( colFlag == 0.0 )
          mosaic->val[p] = img1->val[p];
        else
          mosaic->val[p] = img2->val[p];
      } else
      {
        if ( colFlag == 0.0 )
          mosaic->val[p] = img2->val[p];
        else
          mosaic->val[p] = img1->val[p];
      }
    }
  }

  return(mosaic);
}
