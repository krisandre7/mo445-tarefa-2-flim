//IFT main header file
#include <ift.h>

/***************************************************************
 * Note: The three functions below are not strictly related to
 * the registration. They should be in another place, but are
 * still here until we decide the right place to put them.
 **************************************************************/
Scene *NormalizeAccHist3( Scene *scn );
float Distance(float *f1, float *f2, int n);
Scene *TextGradient( Scene *scn );
Scene *getWaterGray3( Scene *scn, float di, float factor );
Scene *getWGBorder( Scene *label );
Set *getSetBorder(Scene *scn );
void calculateTransformation( double *theta, int n, Point center, float RT[4][4] );

/***************************************************************
 * Context structure for volume registration.
 **************************************************************/
struct BrainRegContext
{
  Scene  *fixedGrad;  // Gradient of the fixed volume.
  Scene  *movingVol;     // Moving volume.
  Set    *movingSet;  // Set of watershed line points from the moving image.
  Point  center;
};

/***************************************************************
 * Prepare brain volumes to registration
 **************************************************************/
void PrepBrain2Reg( Scene *fixed, Scene *moving, struct BrainRegContext *context )
{
  //Pre-processing for the fixed volume
  printf("[preparing fixed] "); fflush(stdout);

  Scene *fixedNorm, *fixedGrad;

  fixedNorm = NormalizeAccHist3( fixed );
  fixedGrad = TextGradient( fixedNorm );
  context->fixedGrad = NormalizeAccHist3( fixedGrad );

  DestroyScene( &fixedGrad );
  DestroyScene( &fixedNorm );

  //Pre-processing for the moving volume
  printf("[preparing moving] "); fflush(stdout);
  Scene *movingNorm, *movingWG, *movingWGLines;
  Set *S = NULL;
  Point center;

  movingNorm = NormalizeAccHist3( moving );
  movingWG = getWaterGray3( movingNorm, 1.8, 0.07 );
  movingWGLines = getWGBorder( movingWG );
  S = getSetBorder( movingWGLines );

  center.x = moving->xsize / 2;
  center.y = moving->ysize / 2;
  center.z = moving->zsize / 2;

  context->movingVol = moving;
  context->movingSet = S;
  context->center    = center;

  DestroyScene( &movingNorm );
  DestroyScene( &movingWG );
  DestroyScene( &movingWGLines );

}

/***************************************************************
 * Fitness function to be minimized in the optimization process
 **************************************************************/
double BrainMatch( double *x, void *context )
{
  struct BrainRegContext *c = (struct BrainRegContext *) context;

  float T[4][4];

  double *theta_aux=AllocDoubleArray(9); // 7 parameters (3 rot, 3 trans, 1 scale)
  theta_aux[0]=x[0];
  theta_aux[1]=x[1];
  theta_aux[2]=x[2];
  theta_aux[3]=x[3];
  theta_aux[4]=x[4];
  theta_aux[5]=x[5];
  theta_aux[6]=x[6]; // Sx=Sy=Sz
  theta_aux[7]=x[6];
  theta_aux[8]=x[6];

  calculateTransformation(theta_aux, 9, c->center, T);

  //Calculates T(Sj) and the distance at the same time
  Set *S = c->movingSet;
  Scene *fixed_grad = c->fixedGrad;
  Scene *moving = c->movingVol;
  int p;
  double D=0.0;
  Voxel v, v_transf;
  int counter=0;

  while (S != NULL)
  {
    counter++;
    p = S->elem;
    v.z = p / (moving->xsize * moving->ysize);
    v.y = (p - moving->tbz[v.z]) / (moving->xsize);
    v.x = (p - moving->tbz[v.z]) % (moving->xsize);
    v_transf = Transform_Voxel(T, v);
    if (ValidVoxel(fixed_grad, v_transf.x, v_transf.y, v_transf.z))
      D += fixed_grad->data[fixed_grad->tbz[v_transf.z] + fixed_grad->tby[v_transf.y] + v_transf.x];

    S = S->next;
  }

  free(theta_aux);

  if (counter!=0)
    D = ( D/counter ) / 4096;
  else D = 0;

  return 1-D;
}

/************************************************************
 * Demo program for volume registration.
 *
 * The characterization are here considered for the brain
 * registration problem. But the program can be extended
 * to any other volume registration by replacing
 * PrepBrain2Reg and BrainMatch.
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

  if( argc != 4 )
  {
    printf( "Usage must be: %s <fixed scene> <moving scene> <output registered scene> \n", argv[0] );
    return 0;
  }

  /* Timing variables. */
  clock_t t1, t2;

  /********************************************
   * Problem layer
   ********************************************/
  #define numParameters 7
  struct BrainRegContext brainRegContext;

  Scene *fixed  = ReadScene( argv[1] );
  Scene *moving = ReadScene( argv[2] );

  t1 = clock(); /* Timing start. */

  PrepBrain2Reg( fixed, moving, &brainRegContext );

  /*
  * Boundaries of the search space from which the optimization will be guided.
  * Elements 0: RX, 1: RY, 2: RZ, 3: TX, 4: TY, 5: TZ, 6: S
  */
  double lowerBound[numParameters];
  double upperBound[numParameters];

  lowerBound[0] = -45.0;             upperBound[0] = 45.0;
  lowerBound[1] = -45.0;             upperBound[1] = 45.0;
  lowerBound[2] = -45.0;             upperBound[2] = 45.0;
  lowerBound[3] = -fixed->xsize / 8; upperBound[3] = fixed->xsize / 8;
  lowerBound[4] = -fixed->ysize / 8; upperBound[4] = fixed->ysize / 8;
  lowerBound[5] = -fixed->zsize / 8; upperBound[5] = fixed->zsize / 8;
  lowerBound[6] =  0.85;             upperBound[6] =  1.15;

  /*
  * Boundaries for the initialization of the displacements.
  */
  double lowerInit[numParameters];
  double upperInit[numParameters];

  lowerInit[0] = -22.5;             upperInit[0] = 22.5;
  lowerInit[1] = -22.5;             upperInit[1] = 22.5;
  lowerInit[2] = -22.5;             upperInit[2] = 22.5;
  lowerInit[3] = -fixed->xsize /16; upperInit[3] = fixed->xsize /16;
  lowerInit[4] = -fixed->ysize /16; upperInit[4] = fixed->ysize /16;
  lowerInit[5] = -fixed->zsize /16; upperInit[5] = fixed->zsize /16;
  lowerInit[6] =  0.925;            upperInit[6] =  1.075;

  /*
  * Position from which the optimization starts. Here is the identity
  * transformation, but it could be a random point as well.
  */
  double theta0[]= { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

  /********************************************
   * Optimization layer
   ********************************************/
  /* MSPS control parameters:
   * 0: # of scales
   * 1: its increasing degree
   * 2: sampling-range decrease factor.
   */
  double settings[] = { 3, 2.4, 1.82 };

  size_t  numIterations = 1750;   /* Number of iterations. */

  /* Define the context for the MSPS optimization */
  struct optContext mspsContext;

  mspsContext.f          = BrainMatch;
  mspsContext.fContext   = (void*) &brainRegContext;
  mspsContext.fDim       = numParameters;
  mspsContext.theta      = theta0;
  mspsContext.lowerInit  = lowerInit;
  mspsContext.upperInit  = upperInit;
  mspsContext.lowerBound = lowerBound;
  mspsContext.upperBound = upperBound;
  mspsContext.numIterations = numIterations;
  mspsContext.verbose    = 1;

  printf("[optimizing]\n "); fflush(stdout);

  double Vprime;
  Vprime = MSPS( settings, (void*) &mspsContext );

  t2 = clock(); /* Timing end. */

  /* Display results of optimization along with time-usage. */
  printf("Fitness value: %g \n", Vprime);
  printf("Theta prime in vector notation:\n[ %g, %g, %g, %g, %g, %g, %g]\n",
          mspsContext.theta[0], mspsContext.theta[1], mspsContext.theta[2],
          mspsContext.theta[3], mspsContext.theta[4], mspsContext.theta[5],
          mspsContext.theta[6]);

  printf( "\nTime usage for brain registration: %g seconds\n",
          ( double ) ( t2 - t1 ) / CLOCKS_PER_SEC );

  float T[4][4];
  double thetaAux[] = { mspsContext.theta[0],
                        mspsContext.theta[1],
                        mspsContext.theta[2],
                        mspsContext.theta[3],
                        mspsContext.theta[4],
                        mspsContext.theta[5],
                        mspsContext.theta[6], // Sx=Sy=Sz
                        mspsContext.theta[6],
                        mspsContext.theta[6] };

  calculateTransformation( thetaAux, 9, brainRegContext.center, T );

  Scene *reged = CreateScene( fixed->xsize, fixed->ysize, fixed->zsize );
  transformScene( moving, T, reged );

  printf("[write] ");fflush(stdout);
  WriteScene( reged, argv[3] );
  printf("[done]\n");fflush(stdout);

  DestroyScene( &( brainRegContext.fixedGrad ) );
  DestroySet( &( brainRegContext.movingSet ) );

  DestroyScene( &fixed );
  DestroyScene( &moving );
  DestroyScene( &reged );

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
 * No description.
 **************************************************************/
Scene *NormalizeAccHist3( Scene *scn )
{
  Scene *out = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  Curve *acc = NormAccHistogram3(scn);
  int max, i, n, min;
  for (i = acc->n - 1; acc->Y[i] > 0.991; i--);
  max = i;
  for (i = 1; acc->Y[i] < 0.1; i++);
  min = i;

  n = scn->xsize * scn->ysize * scn->zsize;
  for (i = 0; i < n; i++) {
      if (scn->data[i] < min)
          out->data[i] = 0;

      else if (scn->data[i] <= max) {
          out->data[i] = ((scn->data[i] - min)*4095) / (max - min);
      } else
          out->data[i] = 4095;
  }
  DestroyCurve(&acc);
  return (out);
}

/***************************************************************
 * No description.
 **************************************************************/
float Distance( float *f1, float *f2, int n )
{
  int i;
  float dist;

  dist = 0;
  for (i=0; i < n; i++)
    dist += (f2[i]-f1[i]);//(f2[i]-f1[i])/2.0;
  //dist /= n;

  return(dist);//exp(-(dist-0.5)*(dist-0.5)/2.0));
}

/***************************************************************
 * No description.
 **************************************************************/
Scene *TextGradient( Scene *scn )
{
  float   dist,gx,gy,gz;
  int     i,p,q,n=scn->xsize*scn->ysize*scn->zsize;//,Imax=MaximumValue3(scn);
  Voxel   u,v;
  AdjRel3 *A=Spheric(1.0),*A6=Spheric(1.0);
  float   *mg=AllocFloatArray(A6->n);
  Scene   *grad=CreateScene(scn->xsize,scn->ysize,scn->zsize);


  typedef struct _features {
    float *f;
  } Features;

  Features *feat=(Features *)calloc(n,sizeof(Features));
  for (p=0; p < n; p++)
    feat[p].f = AllocFloatArray(A->n);

  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++) {
        p = u.x + scn->tby[u.y] + scn->tbz[u.z];
        for (i=0; i < A->n; i++) {
          v.x = u.x + A->dx[i];
          v.y = u.y + A->dy[i];
          v.z = u.z + A->dz[i];
          if (ValidVoxel(scn,v.x,v.y,v.z)){
            q = v.x + scn->tby[v.y] + scn->tbz[v.z];
            feat[p].f[i]=(float)scn->data[q];///(float)Imax;
          }
        }
      }

  for (i=0; i < A6->n; i++)
    mg[i]=sqrt(A6->dx[i]*A6->dx[i]+A6->dy[i]*A6->dy[i]+A6->dz[i]*A6->dz[i]);

  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++) {
        p = u.x + scn->tby[u.y] + scn->tbz[u.z];
        gx = gy = gz = 0.0;
        for (i=1; i < A6->n; i++) {
          v.x = u.x + A6->dx[i];
          v.y = u.y + A6->dy[i];
          v.z = u.z + A6->dz[i];
          if (ValidVoxel(scn,v.x,v.y,v.z)){
            q = v.x + scn->tby[v.y] + scn->tbz[v.z];
            dist = Distance(feat[p].f,feat[q].f,A->n);
            gx  += dist*A6->dx[i]/mg[i];
            gy  += dist*A6->dy[i]/mg[i];
            gz  += dist*A6->dz[i]/mg[i];
          }
        }
        grad->data[p]=(int)sqrt(gx*gx + gy*gy + gz*gz);//(100000.0*sqrt(gx*gx + gy*gy + gz*gz));
      }

  for (p=0; p < n; p++)
    free(feat[p].f);
  free(feat);

  free(mg);
  DestroyAdjRel3(&A);
  DestroyAdjRel3(&A6);
  return(grad);
}

/***************************************************************
 * No description.
 **************************************************************/
Scene *getWaterGray3( Scene *scn, float di, float factor )
{
  Scene *grad=NULL,*handicap=NULL;
  Scene    *label=NULL;

  AdjRel3   *A=NULL;

  grad  = TextGradient(scn);
  A = Spheric(di);
  handicap = Add3(grad,(int)(factor*MaximumValue3(grad)));
  label = WaterGray3(grad,handicap,A);

  DestroyAdjRel3(&A);
  DestroyScene(&grad);
  DestroyScene(&handicap);

  return(label);
}

/***************************************************************
 * No description.
 **************************************************************/
Scene *getWGBorder( Scene *label )
{
  Scene *hscn=CreateScene(label->xsize,label->ysize,label->zsize);
  int p,q,i;
  AdjRel3 *A=NULL;
  Voxel u,v;

  A    = Spheric(1.0);
  for (u.z=0; u.z < hscn->zsize; u.z++){
    for (u.y=0; u.y < hscn->ysize; u.y++){
      for (u.x=0; u.x < hscn->xsize; u.x++){
  p = u.x + hscn->tby[u.y] + hscn->tbz[u.z];
  for (i=1; i < A->n; i++){
    v.x = u.x + A->dx[i];
    v.y = u.y + A->dy[i];
    v.z = u.z + A->dz[i];
    if (ValidVoxel(hscn,v.x,v.y,v.z)){
      q = v.x + hscn->tby[v.y] + hscn->tbz[v.z];
      if (label->data[p] < label->data[q]){
        hscn->data[p] = 1;
        break;
      }
    }
  }
      }
    }
  }
  DestroyAdjRel3(&A);
  return(hscn);
}

/***************************************************************
 * No description.
 **************************************************************/
Set *getSetBorder(Scene *scn){

    AdjRel3 *adj=Spheric(1.0);
    Scene *borda=GetBorder3(scn, adj);
    Set *S=NULL;
    int i, n;
    n=scn->xsize*scn->ysize*scn->zsize;
    for(i=0;i<n;i++){
        if(borda->data[i]==1){
            InsertSet(&S, i);
        }
    }
    DestroyScene(&borda);
    DestroyAdjRel3(&adj);
    return S;
}

/***************************************************************
 * Parameters:
 * theta - array with the  parameters of the transform
 * n - # of elements in theta
 * center - the center of the rotation
 * RT - is the resulting transformation matrix
 **************************************************************/
void calculateTransformation( double *theta, int n, Point center, float RT[4][4] )
{
  // Compute transformation in this order: centered rotation (center is cx,cy,cz), translation, scaling and shear. Parameters theta are Rx, Ry, Rz, Tx, Ty, Tz, Sx, Sy, Sz, SHxy, SHxz, SHyz, SHyz, SHzx, SHzy.
  // The matrix multiplication is done backwards:
  // M = T SH S -Tc Rz Ry Rx Tc
  float Rx=0,Ry=0,Rz=0,Tx=0,Ty=0,Tz=0;
  float Sx=1,Sy=1,Sz=1,SHxy=0,SHxz=0,SHyx=0,SHyz=0,SHzx=0,SHzy=0;
  RealMatrix *trans1,*rot,*rot1,*rot2,*rot3,*trans2;
  RealMatrix *aux1,*aux2;
  RealMatrix *trans, *scale, *shear, *final;

  if (n>=3) {
    Rx = theta[0] * PI / 180.0;
    Ry = theta[1] * PI / 180.0;
    Rz = theta[2] * PI / 180.0;
  }
  if (n>=6) {
    Tx = theta[3];
    Ty = theta[4];
    Tz = theta[5];
  }
  if (n>=9) {
    Sx = theta[6];
    Sy = theta[7];
    Sz = theta[8];
  }
  if (n==15) {
    SHxy = theta[9];
    SHxz = theta[10];
    SHyx = theta[11];
    SHyz = theta[12];
    SHzx = theta[13];
    SHzy = theta[14];
  }

  // Rotation
  trans1 = TranslationMatrix3(-center.x,-center.y,-center.z);
  rot1 = RotationMatrix3(0,Rx);
  rot2 = RotationMatrix3(1,Ry);
  rot3 = RotationMatrix3(2,Rz);
  trans2 = TranslationMatrix3(center.x,center.y,center.z);
  // Compose transform rot=trans2 x rot3 x rot2 x rot1 x trans1
  aux1 = MultRealMatrix(trans2,rot3);
  aux2 = MultRealMatrix(aux1,rot2);
  DestroyRealMatrix(&aux1);
  aux1 = MultRealMatrix(aux2,rot1);
  DestroyRealMatrix(&aux2);
  rot = MultRealMatrix(aux1,trans1);
  DestroyRealMatrix(&aux1);
  DestroyRealMatrix(&rot1);
  DestroyRealMatrix(&rot2);
  DestroyRealMatrix(&rot3);
  DestroyRealMatrix(&trans1);
  DestroyRealMatrix(&trans2);

  // Translation
  trans = TranslationMatrix3(Tx,Ty,Tz);

  // Scale
  scale = ScaleMatrix3(Sx,Sy,Sz);

  // Shear
  shear = ShearMatrix3(SHxy,SHxz,SHyx,SHyz,SHzx,SHzy);

  // compose final final = trans x shear x scale x rot
  aux1 = MultRealMatrix(trans,shear);
  aux2 = MultRealMatrix(aux1,scale);
  DestroyRealMatrix(&aux1);
  final = MultRealMatrix(aux2,rot);
  DestroyRealMatrix(&aux2);
  DestroyRealMatrix(&rot);
  DestroyRealMatrix(&trans);
  DestroyRealMatrix(&scale);
  DestroyRealMatrix(&shear);

  RT[0][0]=final->val[0][0];
  RT[0][1]=final->val[0][1];
  RT[0][2]=final->val[0][2];
  RT[0][3]=final->val[0][3];
  RT[1][0]=final->val[1][0];
  RT[1][1]=final->val[1][1];
  RT[1][2]=final->val[1][2];
  RT[1][3]=final->val[1][3];
  RT[2][0]=final->val[2][0];
  RT[2][1]=final->val[2][1];
  RT[2][2]=final->val[2][2];
  RT[2][3]=final->val[2][3];
  RT[3][0]=final->val[3][0];
  RT[3][1]=final->val[3][1];
  RT[3][2]=final->val[3][2];
  RT[3][3]=final->val[3][3];
  DestroyRealMatrix(&final);

}
