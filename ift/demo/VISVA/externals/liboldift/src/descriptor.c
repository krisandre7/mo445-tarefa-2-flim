#include "descriptor.h"

/********************************************
Functions that deal with feature vectors
********************************************/

Ap_FeatureVector2D *Read2DFeatureVectors(char *rfilename, int nimages)
{
  Ap_FeatureVector2D *Feature_Vectors = NULL;
  int i, j, nbins, idDc, idIm, typeDc;
  FILE *fp;

  Feature_Vectors = (Ap_FeatureVector2D *) calloc(nimages,sizeof(Ap_FeatureVector2D));
  
  fp = fopen(rfilename,"r");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n", rfilename);
    exit(-1);
  }
  printf("Reading descriptors...\n");
  for (i=0; i<nimages; i++){
    fscanf(fp, "%d", &idDc);
    fscanf(fp, "%d", &idIm);
    fscanf(fp, "%d", &typeDc);
    fscanf(fp, "%d", &nbins);
    Feature_Vectors[i] = CreateFeatureVector2D(nbins);
    for (j=0;j<nbins; j++){
      fscanf(fp, "%lf ", &Feature_Vectors[i]->X[j]);
      fscanf(fp, "%lf ", &Feature_Vectors[i]->Y[j]);
    }
  }
  fclose(fp);
  printf("Done!\n");
  return Feature_Vectors;
}

Ap_FeatureVector1D *Read1DFeatureVectors(char *rfilename, int  nimages)
{
  Ap_FeatureVector1D *Feature_Vectors = NULL;
  int i, j, nbins, idDc, idIm, typeDc;
  FILE *fp;

  printf("Reading descriptors...\n");
  Feature_Vectors = (Ap_FeatureVector1D *) calloc(nimages,sizeof(Ap_FeatureVector1D));
  fp = fopen(rfilename,"r");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n", rfilename);
    exit(-1);
  }
  
  for (i=0; i<nimages; i++){
    fscanf(fp, "%d", &idDc);
    fscanf(fp, "%d", &idIm);
    fscanf(fp, "%d", &typeDc);
    fscanf(fp, "%d", &nbins);
    Feature_Vectors[i] = CreateFeatureVector1D(nbins);
    for (j=0;j<nbins; j++){
      fscanf(fp, "%lf ", &Feature_Vectors[i]->X[j]);
    }
  }
  fclose(fp);
  printf("Done!\n");
  return Feature_Vectors;
}

FeatureVector1D *CreateFeatureVector1D(int n)
{
  FeatureVector1D *desc=NULL;
  
  desc = (FeatureVector1D *) calloc(1,sizeof(FeatureVector1D));
  if (desc != NULL) {
    desc->X = AllocDoubleArray(n);
    desc->n = n;
  } else {
    Error(MSG1,"CreateFeatureVector");
  }
  return(desc);
}

void DestroyFeatureVector1D(FeatureVector1D **desc)
{
  FeatureVector1D *aux;
  
  aux = *desc;
  if (aux != NULL){
    if (aux->X != NULL) {
      free(aux->X);
    }
    free(aux);
    *desc = NULL;
  }
}

void WriteFeatureVector1D(FeatureVector1D *desc,char *filename)
{
  FILE *fp;
  int i;
  
  fp = fopen(filename,"w");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }
  for (i=0; i < desc->n; i++)
    fprintf(fp,"%f\n",desc->X[i]);
  
  fclose(fp);
}

FeatureVector2D *CreateFeatureVector2D(int n)
{
  FeatureVector2D *desc=NULL;
  
  desc = (FeatureVector2D *) calloc(1,sizeof(FeatureVector2D));
  if (desc != NULL) {
    desc->X = AllocDoubleArray(n);
    desc->Y = AllocDoubleArray(n);
    desc->n = n;
  } else {
    Error(MSG1,"CreateFeatureVector");
  }
  return(desc);
}

void DestroyFeatureVector2D(FeatureVector2D **desc)
{
  FeatureVector2D *aux;
  
  aux = *desc;
  if (aux != NULL){
    if (aux->X != NULL) free(aux->X);
    if (aux->Y != NULL) free(aux->Y);
    free(aux);
    *desc = NULL;
  }
}

void WriteFeatureVector2D(FeatureVector2D *desc,char *filename)
{
  FILE *fp;
  int i;
  
  fp = fopen(filename,"w");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }
  for (i=0; i < desc->n; i++)
    fprintf(fp,"%f\t%f\n",desc->X[i],desc->Y[i]);
  
  fclose(fp);
}

void SortFeatureVector1D(FeatureVector1D *desc, int left, int right, char order)
{
  int pivot;
  
  if (left < right) {
    pivot = PartFeatureVector1D(desc,left,right,order);
    SortFeatureVector1D(desc,left,pivot-1,order);
    SortFeatureVector1D(desc,pivot+1,right,order); 
  }
}

int PartFeatureVector1D (FeatureVector1D *desc, int left, int right, char order)
{
  double x;
  int i;
  double X;
  
  x = desc->X[left];
  i = left;
  
  do {
    if (order == INCREASING){
      while ((desc->X[left] <= x)&&(left <= right)) left++;
      while (desc->X[right]  > x) right--;
    } else { /* order = DECREASING */
      while ((desc->X[left] >= x)&&(left <= right)) left++;
      while (desc->X[right]  < x) right--;
    }
    if (left < right){
      X = desc->X[left];
      desc->X[left]  = desc->X[right];
      desc->X[right] = X;
      left++; right--;
    }
  } while (left <= right);
  
  left = i;
  
  if (left != right){
    X = desc->X[left];
    desc->X[left]  = desc->X[right];
    desc->X[right] = X;
  }
  
  return (right);
}

void SortFeatureVector2D(FeatureVector2D *desc, int left, int right, char order)
{
  int pivot;
  
  if (left < right) {
    pivot = PartFeatureVector2D(desc,left,right,order);
    SortFeatureVector2D(desc,left,pivot-1,order);
    SortFeatureVector2D(desc,pivot+1,right,order); 
  }
}

int PartFeatureVector2D (FeatureVector2D *desc, int left, int right, char order)
{
  double y;
  int i;
  double X,Y;
  
  y = desc->Y[left];
  i = left;
  
  do {
    if (order == INCREASING){
      while ((desc->Y[left] <= y)&&(left <= right)) left++;
      while (desc->Y[right]  > y) right--;
    } else { /* order = DECREASING */
      while ((desc->Y[left] >= y)&&(left <= right)) left++;
      while (desc->Y[right]  < y) right--;
    }
    if (left < right){
      X = desc->X[left];
      Y = desc->Y[left];
      desc->X[left]  = desc->X[right];
      desc->Y[left]  = desc->Y[right];
      desc->X[right] = X;
      desc->Y[right] = Y;
      left++; right--;
    }
  } while (left <= right);
  
  left = i;
  
  if (left != right){
    X = desc->X[left];
    Y = desc->Y[left];
    desc->X[left]  = desc->X[right];
    desc->Y[left]  = desc->Y[right];
    desc->X[right] = X;
    desc->Y[right] = Y;
  }
  
  return (right);
}

FeatureVector1D *CopyFeatureVector1D(FeatureVector1D *desc)
{
  FeatureVector1D *descc;
  
  descc = CreateFeatureVector1D(desc->n);
  memcpy(descc->X,desc->X,desc->n*sizeof(double));
   
  return(descc);
}


FeatureVector2D *CopyFeatureVector2D(FeatureVector2D *desc)
{
  FeatureVector2D *descc;
  
  descc = CreateFeatureVector2D(desc->n);
  memcpy(descc->X,desc->X,desc->n*sizeof(double));
  memcpy(descc->Y,desc->Y,desc->n*sizeof(double));
  
  return(descc);
}

void DescInvertXY(FeatureVector2D *desc)
{
  double tmp;
  int i;
  for (i=0; i<desc->n; i++){
    tmp = desc->X[i];
    desc->X[i] = desc->Y[i];
    desc->Y[i] = tmp;
  }
}


/********************************************
Contour Multiscale Fractal Dimension
********************************************/
Curve *PolynomToFractalCurve(Polynom *P, double lower, double higher, int nbins){
  int i;
  Curve *descriptor = NULL;

  descriptor = SamplePolynom(P, lower, higher, nbins);
  for (i=0; i < descriptor->n; i++) 
    descriptor->Y[i] = 2.0 - descriptor->Y[i];
  
  return descriptor;
}

Curve *ContourMSFractal(Image *in)
{
  Image *cont = NULL;
  Curve *descriptor = NULL;
  Polynom *P;
  double lower = 1.0;
  double higher = 5.0;
  int nbins = 100;
  int degree = 10;
  //timer  tic,toc;
  
  //gettimeofday(&tic,NULL);  
  
  cont = LabelContour(in);
  P = MSFractal(cont, 256, degree, lower, higher, 0, 0.0, 6.0);
  descriptor = PolynomToFractalCurve(P, lower, higher, nbins);
  //gettimeofday(&toc,NULL);
  //printf("Time in msecs %f\n",(toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001);
  
  DestroyPolynom(&P);
  DestroyImage(&cont);
  return (descriptor);
}

/**********************************************************/

/********************************************
Moments Invariants
********************************************/
double  MomentPQ(int p, int q, Image *img, int max){
  int i, x, y;
  double sum = 0.0;
  
  for (i = 0; i < img->ncols * img->nrows; i++)
    {
      if (img->val[i] != 0 )
	{
	  x = 1 + i%img->ncols; /* don't do 0^0 */
	  y = 1 + i/img->ncols;      
	  sum = sum + (pow(x, p) * pow(y,q) * ((double) img->val[i]/max));
	}
    }
  
  return (sum);
}

Curve *MomentInv(Image *img) {
  Curve * curve;
  int i;
  int max;
  
  /* mPQ momentes of order (P+Q) */
  double  m00, m01, m10, m11, m20, m02, m12, m21, m03, m30;
  
  /* center moments */
  double xc, yc;
  
  /* normalized central moments (Eta) */
  double n02, n20, n11, n12, n21, n03, n30; 
  
  /* invariant moments (Phi) */
  double f1, f2, f3, f4, f5, f6, f7;
  
  float g; /* gamma = (p+q)/2 + 1  */
  
  /* n = Upq/U00 */
 
  max = 0;
  
  for (i = 0; i < img->ncols *  img->nrows ; i++)
    {
      if (img->val[i] > max)
	max = img->val[i];
    }
  
  m00 = MomentPQ(0,0, img, max);
  m01 = MomentPQ(0,1, img, max);
  m10 = MomentPQ(1,0, img, max);
  m11 = MomentPQ(1,1, img, max); 
  m12 = MomentPQ(1,2, img, max); 
  m21 = MomentPQ(2,1, img, max); 
  m02 = MomentPQ(0,2, img, max);
  m20 = MomentPQ(2,0, img, max); 
  m03 = MomentPQ(0,3, img, max);
  m30 = MomentPQ(3,0, img, max);  
    
  xc = (double) m10 / m00;
  yc = (double) m01 / m00;
  
  /* n00 = 1.0;  */
  
  /* n10 = 0.0; */
  
  /* n01 = 0.0; */
  
  g = 2.0; 
  
  n20 = (double) (m20 - (xc * m10)) / (pow(m00, g));
  
  n02 = (double) (m02 - (yc * m01)) / (pow(m00, g));
  
  n11 = (double) (m11 - (yc * m10)) / (pow(m00, g));
  
  g = 2.5;
  
  n30 = (double) (m30 - (3 * xc * m20) + (2 * (pow(xc,2)) * m10) ) / (pow(m00, g));   
  
  n12 = (double) (m12 - (2 * yc * m11) - (xc * m02) + (2 * pow(yc,2) * m10) ) / (pow(m00, g)) ;
  
  n21 = (double) (m21 - (2 * xc * m11) - (yc * m20) + (2 * pow(xc,2) * m01) ) / (pow(m00, g));
  
  n03 = (double) (m03 - (3 * yc * m02) + (2 * (pow(yc,2)) * m01)) / (pow(m00, g)); 
  
  f1 = (double) n20 + n02;
  
  f2 = (double) (pow((n20 - n02),2)) + (4 * (pow(n11,2)));
  
  f3 = (double) (pow((n30 - (3 * n12)),2)) + (pow( ( (3 * n21) - n03),2));
  
  f4 = (double) (pow((n30 + n12),2)) + (pow((n21 + n03),2));
  
  f5 = (double) ((n30 - (3 * n12)) * (n30 + n12) * ((pow((n30 + n12),2)) - (3 * (pow((n21 + n03),2))))) + 
    (((3 * n21) - n03) * (n21 + n03) * ((3 * (pow((n30 + n12),2))) - (pow((n21 + n03),2))));
  
  f6 = (double) ((n20 + n02) * ((pow((n30 + n12),2)) - (pow((n21 + n03),2)))) +
    ((4 * n11) * (n30 + n12) * (n21 + n03)); 
  
  f7 = (double) (((3 * n21) - n03) * (n30 + n12) * ((pow((n30 + n12),2)) - (3 * (pow((n21 + n03),2))))) +
    (((3 * n12) - n30) * (n21 + n03) * ((3 * (pow((n30 + n12),2))) - (pow((n21 + n03),2))));
  
  curve = CreateCurve(7);
  
  curve->X[0] = 0.0;
  curve->Y[0] = f1;
  
  curve->X[1] = 1.0;
  curve->Y[1] = f2;
  
  curve->X[2] = 2.0; 
  curve->Y[2] = f3;
  
  curve->X[3] = 3.0; 
  curve->Y[3] = f4;
  
  curve->X[4] = 4.0; 
  curve->Y[4] = f5;
  
  curve->X[5] = 5.0; 
  curve->Y[5] = f6;
  
  curve->X[6] = 6.0; 
  curve->Y[6] = f7;

  return (curve);
}

Curve *MomentInvariant(Image *img) { // contorno e objeto inteiro
  Image *contour = NULL;
  Curve *c1 = NULL;
  Curve *c2 = NULL;
  Curve *curve = NULL;
  int i;
  //timer  tic,toc;

  //gettimeofday(&tic,NULL);  
  contour = LabelContour(img);
  c1 = MomentInv(contour);
  c2 = MomentInv(img);
  
  curve = CreateCurve(c1->n + c2->n);
  for (i=0; i<c1->n; i++){
    curve->X[i] = i;
    curve->Y[i] = c1->Y[i];
  }
  for (i=0; i<c2->n; i++){
    curve->X[i+c1->n] = i + c1->n;
    curve->Y[i+c1->n] = c2->Y[i];
  }
  
  DestroyCurve(&c1);
  DestroyCurve(&c2);
  DestroyImage(&contour);
  
  //gettimeofday(&toc,NULL);
  //printf("Time in msecs %f\n",(toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001);
  return (curve);    
}


/********************************************
Fourier Descriptor
********************************************/
double Cabs(double x, double y) {
  return (sqrt( (x*x) + (y*y)));
}

Curve *Image2Curve(Image *img) { /* img = binary image */
  Curve3D *curve; //ponto (x,y) e z = ordem no contorno
  Curve *curve2;
  Image *contour = NULL;
  int i;
  int npixels;
  int count = 0;
  int j = 0 ;
  
  contour = LabelContPixel(img);

  npixels = contour->ncols * contour->nrows;
  
  for (i = 0; i < npixels; i++){
    if (contour->val[i]!=0)
      count++;
  }
  
  curve = CreateCurve3D(count);
  
  for (i = 0; i < npixels; i++){
    if (contour->val[i]!=0){
      curve->X[j] = i % contour->ncols;
      curve->Y[j] = i / contour->ncols;
      curve->Z[j] = contour->val[i];
      j++;
    }
  }  
  
  SortCurve3D(curve, 0, (curve->n - 1), INCREASING);
  
  curve2 = CreateCurve(curve->n);
  for (i=0;i<curve->n;i++){
    curve2->X[i]=curve->X[i];
    curve2->Y[i]=curve->Y[i];
  }
  
  DestroyCurve3D(&curve);
  DestroyImage(&contour);
  return (curve2);
}

Curve *FourierDescriptor(Image *img) {
  
  Curve *c = NULL; 
  Curve *curve = NULL;
  Curve *mag = NULL;
  int i;
  int tam;
  int nn = 0;
  double normfactor = 0;
  //timer  tic,toc;
  
  //gettimeofday(&tic,NULL);  

  curve = Image2Curve(img);
   
  tam = curve->n;
  i = 1;
  nn = tam;
  while(nn != 1) {
    nn >>= 1;
    ++i;
  }
  
  for(; i; i--)
    nn <<= 1;
  
  printf("%d %d\n",nn,tam);
  
  if (nn < 128)
    nn = 128;
  
  c = CreateCurve(nn);
  
  for (i = 0; i < tam ; i++)
    {
      c->X[i] = (double) curve->X[i];	
      c->Y[i] = (double) curve->Y[i];
    }
  
  for (i = tam ; i < nn ; i++)
    {
      c->X[i] = 0;
      c->Y[i] = 0;
    }
  
  FFT(1, nn,c->X ,c->Y);

  mag = CreateCurve(126);
  normfactor = Cabs(c->X[0],c->Y[0]);
  
  for (i = 1; i <= 126; i++)
    {
      mag->X[i-1] = i - 1;
      mag->Y[i-1] = (double) (Cabs(c->X[i],c->Y[i])/normfactor);
    }
  
  DestroyCurve(&c);  
  DestroyCurve(&curve);  
  //gettimeofday(&toc,NULL);
  //printf("Time in msecs %f\n",(toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001);
  return (mag);
  
}

/*********************************************************
Curvature Scale Space
 *********************************************************/

/* Auxiliar functions to handle double vectors */
/*************************************************************************/
/* Shift *s* positions to LEFT (circular: 1 (circular) , (0: otherwise)) */
/*************************************************************************/
double *SHL(double *V, int n, int s, int circular) { 
  int i;                          
  double *shlV = AllocDoubleArray(n);
  
  for(i = 0; i < n; i++){
    shlV[i] = V[(i+s) % n];
  }
  
  if (!circular){
    for (i = n-1 ; i <= n - s; i--){
      shlV[i] = 0;
    }
  }
  
  return (shlV);
}


/*************************************************************************/
/* Shift *s* positions to RIGHT (circular: 1 (circular) , (0: otherwise)) */
/*************************************************************************/
double *SHR(double *V, int n, int s, int circular) { 
  int i;
  double *shrV = AllocDoubleArray(n);
  
  for(i = 0; i < n; i++){
    shrV[(i+s)%n] = V[i];
  }
  
  if (!circular){
    for (i = 0 ; i < s ; i++){
      shrV[i] = 0;
    }
  }
  
  return (shrV);
}


/*************************************************************************/
/* Compute the sum of the first n elements of array V */
/*************************************************************************/
double SumArray(double *V, int n) {   
  double sum = 0.0;
  int i;
  
  for(i = 0; i < n; i++){
    sum = sum + V[i];
  }
  
  return(sum);
}
/*************************************************************************/
/* Create a copy of array V */
/*************************************************************************/
double *CopyArray(double *V, int n) { 
  int i;
  double *cpyV = AllocDoubleArray(n);
  
  for(i = 0; i < n; i++){
    cpyV[i] = V[i];
  }
  return (cpyV);
}
/*************************************************************************/
/* IN-PLACE normalization of vector V*/
/*************************************************************************/
void NormalizeFVector(double *V, int n, double sum){ 
  int i;
  
  for (i = 0; i < n; i++){
    V[i] = (double) V[i] / sum;
  }
}

/********************************************************************************/
/* Functions for curve resampling */

/*************************************************************************/
/* Create a vector with the distances between curve->X[i] and curve->Y[i] */
/*************************************************************************/
double *CurveDistance(Curve *curve){ 
  Curve *sc = NULL; 
  double p1, p2, *dist = NULL;
  int i;
  
  sc = (Curve *) calloc(1,sizeof(Curve)); //shifted curve
  sc->n = curve->n;
  
  sc->X = SHL(curve->X, curve->n, 1, 1) ;
  sc->Y = SHL(curve->Y, curve->n, 1, 1) ;
  
  dist = AllocDoubleArray(curve->n);
  
  for (i = 0; i < curve->n; i++){
    p1 = pow((curve->X[i] - sc->X[i]), 2);
    p2 = pow((curve->Y[i] - sc->Y[i]), 2);
    dist[i] = sqrt(p1 + p2);
  }
  
  DestroyCurve(&sc);
  
  return (dist); 
}

/*************************************************************************/
/*  Compute a n equidistant-bin curve. Usually n = 200 */
/*************************************************************************/
Curve *ResampleCurve(Curve *curve, int n){ 
  Curve *vector = CreateCurve(n);   /* new resampled curve */
  int i, j, cnt, orilen;
  double newx, curx, nextx;
  double newy, cury, nexty;
  double *disarr, arclen, d, remaining;
  Curve *c = NULL;
  
  c = CopyCurve(curve);
  orilen = c->n; 
  disarr = CurveDistance(c) ;  
  arclen = SumArray(disarr, orilen);
  
  NormalizeFVector(c->X, c->n, arclen);
  NormalizeFVector(c->Y, c->n, arclen);
  
  free(disarr);
  disarr = CurveDistance(c); // recalculate the distance array
  
  vector->X[0] = c->X[0]; /* 1st point is always in the resampled curve */
  vector->Y[0] = c->Y[0];
  j = 1 ; /* next empty in vector */
  
  /* realloc memory for the original curve c */
  c->X = (double *) realloc(c->X, ((c->n + 1) * sizeof(double)));  
  c->Y = (double *) realloc(c->Y, ((c->n + 1) * sizeof(double))); 
  c->n++;
  orilen++;
  
  c->X[c->n - 1] = c->X[0];   //  closes the curve c:  ABC ->  ABCA
  c->Y[c->n - 1] = c->Y[0];
  
  // begin interpolation step
  cnt = 0;                   // find the points by tracing the boundary
  for (i = 2; i <= n; i++){
    remaining = (double) 1 / n;
    while (1){ 
      if (remaining > disarr[cnt]){
	remaining = remaining - disarr[cnt] ;
	cnt++;
      }
      else{
	nextx = c->X[cnt + 1];
	curx = c->X[cnt];
	nexty = c->Y[cnt + 1];
	cury = c->Y[cnt]; 
	
	d = disarr[cnt];
	
	newx = curx + (((nextx - curx) / d) * remaining);
	newy = cury + (((nexty - cury) / d) * remaining);
	
	disarr[cnt]= d - remaining;
	c->X[cnt] = newx;
	c->Y[cnt] = newy;  
	
	break;
      }
    }	
    
    vector->X[j] = newx; // the return values (resampled vector)
    vector->Y[j] = newy;
    j++;
  }
  
  DestroyCurve(&c);
  free(disarr); 
  return (vector); 
}


/********************************************************************************/
/* Functions for gaussian vector generation */

/*************************************************************************/
/* Compute a mirrored linear space (from -range/2 to +range/2) */   
/*************************************************************************/
double *Linspace(int range, int nsamples) { 
  double *l = NULL, delta = 0.0;
  int i;
  
  l = AllocDoubleArray(nsamples); 
  
  delta = (double) range / (nsamples/2);
  l[nsamples/2] = 0;
  
  for (i = (nsamples/2)+1 ; i < nsamples; i++){
    l[i] = l[i-1] + delta;
    l[nsamples-1-i] = l[i];
  }
  
  return (l);
} 

/*************************************************************************/
/*  Compute gaussian vector.
    function [result] = gaussian ( t , sigma )
    result = exp(-t .* t /(2* sigma *sigma ))/(sigma *sqrt (2*pi)) ;  */
/*************************************************************************/
double *Gaussian(double *V, int sigma, int nsamples){    
  double *g;
  int i;
  
  g = AllocDoubleArray(nsamples); 
  for (i = 0; i < nsamples; i++) {
    g[i] = (double) exp(-(V[i] * V[i]) / (2 * sigma * sigma)) / (sigma *sqrt(2*PI)) ;
  }
  
  return (g);
}


/*************************************************************************/
/*  Compute a normalized gaussian vector. */
/*************************************************************************/
double *GenGaussianVector(int nsamples, int sigma, int range){  
  double *g, *l, sum = 0.0;
  
  l = Linspace(range, nsamples) ;
  g = Gaussian(l, sigma, nsamples);
  
  sum = SumArray(g, nsamples);  
  NormalizeFVector(g, nsamples, sum); 
  
  free(l);
  return (g);		  
}

/********************************************************************************/
/* Functions to compute CSS image */

/*************************************************************************/
/* Digital differenciation */  
/*************************************************************************/
double *MyDev(double *c, int n) {  
  int i;
  double *a, *b, *d;
  
  a = SHL(c, n, 1, 1);
  b = SHR(c, n, 1, 1);
  
  d = AllocDoubleArray(n);  
  for (i = 0; i < n ; i++)
    d[i]= a[i] - b[i];
  
  free(a); 
  free(b); 
  
  return (d);
}

/*************************************************************************/
/* Compute a circular convolution for CSS */
/*************************************************************************/
double *WarpConv(double *V, int v_size, double *B, int B_size) { 
  int i, j, k, l;
  double sum, *conv = NULL;
  
  conv = AllocDoubleArray(v_size);   // convolution vector
  for (i=0, j=v_size/2; i<v_size; i++, j++){
    sum = 0.0;
    for (k=0, l=j; k<B_size; k++, l++)
      sum+=B[k]*V[l%v_size];
    conv[i] = sum;
  }
  
  return conv;
}

/*************************************************************************/
/* Compute curvature of smoothed curve */
/* k = ( (Xt .* Ytt) - (Xtt .* Yt) ) ./ ( ( (Xt .* Xt) + (Yt .* Yt) ) . ^ (1.5) );*/
/*************************************************************************/
double *CompCurvature(double *Xt, double *Yt, double *Xtt, double *Ytt, int n){   
  int i;
  double *k = AllocDoubleArray(n); 
  
  for (i = 0; i < n; i++){
    k[i] = (double) ((Xt[i]*Ytt[i]) - (Xtt[i]*Yt[i])) / 
      pow(((Xt[i] * Xt[i]) + (Yt[i] * Yt[i])), 1.5) ;
  }
  
  return (k);
}

/*************************************************************************/
/* Extract zeros from contour points in vector V */
/*************************************************************************/
double *CSSZeros(double *v, int n){ 
  double *z = NULL;
  int i;
  
  z = SHR(v, n, 1, 1); // circular shift of 1 to right    
  
  for (i = 0; i < n; i++){
    if ((z[i] * v[i]) <= 0) // zero crossing is the transition from + to - points
      z[i] = 1; // is a zero
    else 
      z[i] = 0; // otherwise
  }
  
  return (z);   // z[i] = 1, if i is a zero crossing point
}

/*************************************************************************/
/* Mark in the CSS image the zero points for sigma = s  */
/*************************************************************************/
void MarkCSSImage(Image *img, double *zeros, int s, int color) { 
  int i;
  
  for (i = 0; i < 200; i++){
    if (zeros[i] == 1){
      img->val[img->tbrow[img->nrows - s - 1] + (i%200)] = color; 
      // CSS image: sigmas are the lines, while cols are the image contour points
    }
  }
}

/*************************************************************************/
/* Compute zero position from binary information */
// array   1 2 3 4 5 6 7 8 
//         0 0 1 1 0 1 0 1
// becames curve  X,Y
//         3,4  6,8
/*************************************************************************/
Curve *Bin2Curve(double * a, int t) { 
  int i, j=0, k=0;
  int n = (SumArray(a, t)/2);
  Curve * curve = CreateCurve(n);
  
  for (i = 0; i < t; i++)
    if (a[i]==1.0){
      if (j==0){
	curve->X[k] = i;
	j++;
      }
      else{
	curve->Y[k] = i;
	j--;
	k++;
      }
    }
  return curve;
}

/*************************************************************************/
/* Compute diference between array bins. */
/*************************************************************************/
int BinArrayDiff(Curve *curve){
  int i, diff = 0;
  
  for (i=0; i<curve->n; i++){
    diff += curve->Y[i]-curve->X[i];
  }

  return diff;
}


/*************************************************************************/
/*  Write curve evolution */
/*************************************************************************/
void WriteCurveEvolution(double *X, double *Y, int sigma){
  char fname[10];
  FILE *fp = NULL;
  int i; 
  
  sprintf(fname,"conv%d.xy",sigma);
  if((fp = fopen(fname, "w")) == NULL){
    printf("arquivo nao pode ser aberto\n");
    exit(-1);
  }
  for (i = 0; i <  200; i++){
    fprintf(fp,"%f %f \n",X[i], Y[i]);
  }
  if (fclose (fp) == EOF) {
    perror ("Close failed");
  }
}

/*************************************************************************/
/*  Write zero evolution */
/*************************************************************************/
void WriteZerosEvolution(double *X, double *Y, double *zeros, int sigma){
  char fname[10];
  FILE *fp = NULL;
  int i; 
  
  sprintf(fname,"zeros%d.xy",sigma);
  if((fp = fopen(fname, "w")) == NULL){
    printf("arquivo nao pode ser aberto\n");
    exit(-1);
  }
  for (i = 0; i <  200; i++){
    if (zeros[i]==1)
      fprintf(fp,"%f %f \n",X[i], Y[i]);
  }
  if (fclose (fp) == EOF) {
    perror ("Close failed");
  }
}


/*************************************************************************/
/* Compute CSS image peaks. */
/*************************************************************************/
int FindPeaks(Curve * peaklist, int npeaks, double * aorig, double *borig, int sigma){
  double *a, *b, *a1, *a2;
  double *bs; // shifted
  double sum;
  int i, j, k, t=200;
  int diff1, diff2; 
  Curve *a1_curve, *a2_curve;

  if ((sigma == 80)&&(SumArray(borig, 200)!=0.0)){
    j = 0;
    sum = 0.0;
    for (i=0; i<200;i++)
      if (borig[i]==1){
	j++;
	sum+=i;
      }
    peaklist->Y[npeaks]= 105;
    peaklist->X[npeaks]= sum/j;
    if (peaklist->X[npeaks] > 200)
      peaklist->X[npeaks] = peaklist->X[npeaks] - 200;
    //printf("-->peak %d, sigma %d  = %f\n",npeaks, 105, peaklist->X[npeaks]);  
    npeaks++;
  }
  
  a = CopyArray(aorig,t);
  b = CopyArray(borig,t);
  
  for (j = 0; j <= 6; j++){
    bs = SHR(b, t, j, 1);
    for (i = 0; i < t; i++)
      if ((a[i] + bs[i])==2){
	a[i]=0;
	b[(i-j+t)%t]=0;
      }
    free(bs);
    //****************//
    bs = SHL(b, t, j, 1);
    for (i = 0 ; i < t; i++)
      if((a[i] + bs[i])==2){ 
	a[i]=0;
	b[(i+j+t)%t]=0;
      }
    free(bs);
  }

  a1 = CopyArray(a,t);
  a1_curve = Bin2Curve(a1, t); 
  diff1 = BinArrayDiff(a1_curve);
  
  for (i = 0; i < t; i++) 
    if (a[i]==1)
      break;
  
  a2 = CopyArray(a,t);
  a2 = realloc(a2, (i+201)*sizeof(double));
  for (k = t; k < (i+201) ; k++){
    a2[k] = 0;
  }
  a2[i] = 0;
  a2[i+200] = 1; 
  a2_curve = Bin2Curve(a2, i+201);
  diff2 = BinArrayDiff(a2_curve);
  
  /* error routine */
  if (a1_curve->n !=a2_curve->n)
    printf("a1 %d a2 %d\n",a1_curve->n,a2_curve->n);
  
  if  (diff1 < diff2){
    for (i = 0; i < a1_curve->n; i++){
      peaklist->Y[npeaks]= sigma;
      peaklist->X[npeaks]= (double) (a1_curve->X[i] +  a1_curve->Y[i]) / 2 ;
      //printf("peak %d, sigma %d  = %f\n",npeaks, sigma, peaklist->X[npeaks]);
      npeaks++;
    }
  }
  else{
    for (i = 0; i < a2_curve->n; i++){
      peaklist->Y[npeaks]= sigma;
      peaklist->X[npeaks]= ((double) (a2_curve->X[i] +  a2_curve->Y[i]) / 2);
      if (peaklist->X[npeaks] > 200)
	peaklist->X[npeaks] = peaklist->X[npeaks] - 200;
      //printf("peak %d, sigma %d  = %f\n",npeaks, sigma, peaklist->X[npeaks]);  
      npeaks++;
    }
  }
  
  free(a);
  free(b);
  free(a1);
  free(a2);
  DestroyCurve(&a1_curve);
  DestroyCurve(&a2_curve);
  return npeaks;
}

/*************************************************************************/
/*  Constructs the CSSImage and returns the peaks */
/*************************************************************************/
Curve *CSSImage(Curve *curve, int curve_evolution){  
  int startsigma = 1;  /* gaussian parameter */
  int endsigma = 150;  /* gaussian parameter */
  int range = 250;    /* gaussian parameter */
  int nsamples = 199; /* gaussian kernel size  */
  int npeaks = 0;
  int nzeros = -1;
  int nprev = -1;
  int i; 
  //int j;
  double *g, *z, *X, *Xt, *Xtt, *Y, *Yt, *Ytt, *zeros =NULL, *prev_zero_cross = NULL;
  Curve *peaklist = NULL;
  Image *img = NULL;
  
  if (curve->n != 200){
    printf("[x y] should be resampled to 200 points.\n");
    exit(-1);
  }
  
  zeros = AllocDoubleArray(200);
  prev_zero_cross = AllocDoubleArray(200);
  
  peaklist = CreateCurve(100);
  
  img = CreateImage(curve->n, endsigma);
  
  for (i = startsigma; ((i <= endsigma)&&(nzeros!=0)); i++){  
    // find the curvature zeros for each sigma
    g = GenGaussianVector(nsamples, i ,range);
    
    X = WarpConv(curve->X, curve->n, g, nsamples); 
    Y = WarpConv(curve->Y, curve->n, g, nsamples); 
    
    Xt = MyDev(X, 200);   // 1st dev
    Yt = MyDev(Y, 200);
    Xtt = MyDev(Xt, 200); // 2nd dev
    Ytt = MyDev(Yt, 200);
    z = CompCurvature(Xt, Yt, Xtt, Ytt, 200); 
  
    free(prev_zero_cross);
    prev_zero_cross = CopyArray(zeros, 200);
    nprev = nzeros;
    
    free(zeros);
    zeros = CSSZeros(z, 200);
    nzeros = SumArray(zeros, 200);
    
    /********************************************/
    if (curve_evolution){ // fprintf curve evolution
      WriteCurveEvolution(X, Y, i);
      WriteZerosEvolution(X, Y, zeros, i);
    }
    /********************************************/
    
    free(g);
    free(X);
    free(Xt);
    free(Xtt);
    free(Y);
    free(Yt);
    free(Ytt);
    free(z);
    
    MarkCSSImage(img, zeros, i - startsigma, 1);
    
    if (((nzeros % 2)!=0)&& (i > 8)){
      printf("Erro! Numero impar (nzeros = %d)\n", nzeros);
      exit(-1);
    }
    if ((nzeros > nprev) && (i!=startsigma)){
      printf("nzeros > nprev %d %d\n", nzeros, nprev);
      /*exit(-1);*/ //break; /* don't need to try more sigmas */
    }
    
    if ((i>8)&&(i<=80)){
      npeaks = FindPeaks(peaklist, npeaks, prev_zero_cross, zeros, i);
    }
  }
  
  free(prev_zero_cross);
  free(zeros);
  
  for (i = 0; i < npeaks; i++){
    PaintCircle(img, (int) peaklist->X[i] + img->tbrow[img->nrows - ((int) peaklist->Y[i])], 1, 2);
  }
  /*******************************/
  if (0)
    WriteImage(img, "cssimg.pgm") ;
  /*******************************/
  DestroyImage(&img);  
  
  peaklist->X = (double *) realloc(peaklist->X, (npeaks * sizeof(double))); 
  peaklist->Y = (double *) realloc(peaklist->Y, (npeaks * sizeof(double)));
  peaklist->n = npeaks;
  //WriteCurve(peaklist, "peaklist.txt");
  
  return (peaklist);
}

/*************************************************************************/
/*  Compute CSS descriptor */
/*************************************************************************/
Curve *CSS(Image *bin){
  
  Curve *contour = NULL;
  Curve *descriptor = NULL;
  Curve *res_contour = NULL;
  //timer  tic,toc;
  
  //gettimeofday(&tic,NULL);  
  
  if (MaximumValue(bin)!=1){
    printf("Erro! Maximum value should be 1\n");
    exit(-1);
  }
  
  contour = Image2Curve(bin);   
  res_contour = ResampleCurve(contour,200);
  descriptor = CSSImage(res_contour, 0);
  
  DestroyCurve(&contour);
  DestroyCurve(&res_contour);

  //gettimeofday(&toc,NULL);
  //printf("Time in msecs %f\n",(toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001);
  return (descriptor);
}


/*********************************************************
BIC
*********************************************************/

int *Quantize_colors(CImage *img, int color_dim){
  unsigned long i;
  unsigned long r, g, b;
  unsigned long fator_g, fator_b;
  int *color, n;
  
  n = img->C[0]->nrows * img->C[0]->ncols;  

  color = (int *) calloc(n, sizeof(int));
  if(color==NULL){
    printf("\nOut of memory \n");
    exit(-1);
  }
  
  fator_g = color_dim;
  fator_b = fator_g*color_dim;
  
  for(i=0; i<n; i++){
    r = color_dim*img->C[0]->val[i]/256;
    g = color_dim*img->C[1]->val[i]/256;
    b = color_dim*img->C[2]->val[i]/256;
    
    color[i] = (r + fator_g*g + fator_b*b);
  }
  return color;
}
/***************************************************************************/
unsigned char Compute_log(float value){
  unsigned char result;
  
  value = 255. * value;
  if(value==0.)       result=0;
  else if(value<1.)   result=1;
  else if(value<2.)   result=2;
  else if(value<4.)   result=3;
  else if(value<8.)   result=4;
  else if(value<16.)  result=5;
  else if(value<32.)  result=6;
  else if(value<64.)  result=7;
  else if(value<128.) result=8;
  else                result=9;
  
  return result;
}

/**************************************************************************/
void Compress_histogram(unsigned char *ch, unsigned long *h, 
			unsigned long max, int size){
  int i;
  unsigned char v;
  
  for(i=0; i<size; i++){
    v = Compute_log((float) h[i] / (float) max);
    ch[i] = (unsigned char)(48 + v);
  }
}


/****************************************************************************/
void Compute_frequency_property(Image *img, Property *ppt){
  
  unsigned long x, y, p, q;
  int i, border;
  AdjRel *A;
  Pixel v;
  
  A = Circular(1.0);
  
  for(y=0L; y<img->nrows; y++){
    for(x=0L; x<img->ncols; x++){
      p = x + img->tbrow[y];
      border=FALSE;
      for (i=1; i < A->n; i++){
	v.x = x + A->dx[i];
	v.y = y + A->dy[i];
	if (ValidPixel(img,v.x,v.y)){
	  q = v.x + img->tbrow[v.y];
	  if(ppt[p].color!=ppt[q].color){ 
	    border=TRUE;
	    break;
	  }
	}
      }
      if(border==FALSE) 
	ppt[p].frequency=LOW;
      else ppt[p].frequency=HIGH;
    }
  }
  DestroyAdjRel(&A);
}

/****************************************************************************/
Property *Compute_pixels_properties(CImage *img)
{
  Property *p;
  int *color, i, n;
  
  n = img->C[0]->nrows * img->C[0]->ncols;  
  
  p = (Property *) calloc(n, sizeof(Property));
  if(p==NULL){
    printf("\nOut of memory \n");
    exit(-1);
  }
  
  color = Quantize_colors(img, 4);
  for(i=0; i<n; i++) 
    p[i].color=color[i];
  Compute_frequency_property(img->C[0], p);
  
  free(color);
  return p;
}
/****************************************************************************/
VisualFeature *Compute_histograms(Property *p, int n){
  VisualFeature *vf = (VisualFeature *) calloc(1,sizeof(VisualFeature));
  unsigned long i;
  
  for(i=0; i<DESC_SIZE; i++){
    vf->colorH[i] = 0;
    vf->lowH[i] = 0;
    vf->highH[i] = 0;
  }
  
  for(i=0; i<n; i++){
    /*GCH*/
    vf->colorH[p[i].color]++;
    
    /*Frequency*/
    if(p[i].frequency==LOW) 
      vf->lowH[p[i].color]++;
    else 
      vf->highH[p[i].color]++;
  }
  return vf;
}

/**************************************************************************/

CompressedVisualFeature *Compress_histograms(VisualFeature *vf, int npixels)
{
  CompressedVisualFeature *cvf =  (CompressedVisualFeature *) calloc(1,sizeof(CompressedVisualFeature));
  
  Compress_histogram(cvf->colorH, vf->colorH, npixels, DESC_SIZE);
  Compress_histogram(cvf->lowH, vf->lowH, npixels, DESC_SIZE);
  Compress_histogram(cvf->highH, vf->highH, npixels, DESC_SIZE);
  
  return cvf;
}

/***************************************************************************/

void Write_visual_features(char *filename,char *dbname, CompressedVisualFeature *cvf)
{
  FILE *file;
  int i;
  
  if((file = fopen(dbname, "a+t")) == NULL)
    {
      fprintf(stderr, "Can't open %s \n", dbname);
      exit(-1);
    }
  
  fprintf(file, "%s\t", filename);
  for(i=0;i<DESC_SIZE;i++)
    {
      fprintf(file, "%c%c", cvf->lowH[i], cvf->highH[i]);
      /*fprintf(file, "%c%c%c", cvf->colorH[i], cvf->lowH[i], cvf->highH[i]);*/
       }
  fprintf(file, "\n");
  
  fclose(file);
}


/**************************************************************************/
CompressedVisualFeature *Extract_visual_features(CImage *img)
{
  Property *p;
  VisualFeature *vf;
  CompressedVisualFeature *cvf;
  int npixels;
  
  npixels = img->C[0]->nrows * img->C[0]->ncols;
  
  p = Compute_pixels_properties(img);
  vf = Compute_histograms(p, npixels);
  cvf = Compress_histograms(vf, npixels);
  
  free(p);
  return cvf;
}
/*************************************************************************/
Curve *BIC(CImage *img)
{
  Property *p;
  VisualFeature *vf;
  CompressedVisualFeature *cvf;
  int i, npixels;
  Curve *curve = CreateCurve(2*DESC_SIZE);
  
  npixels = img->C[0]->nrows * img->C[0]->ncols;
  
  p = Compute_pixels_properties(img);
  vf = Compute_histograms(p, npixels);
  cvf = Compress_histograms(vf, npixels);
  
  for (i=0; i<DESC_SIZE; i++){
    curve->X[i] = i;
    curve->Y[i] = cvf->lowH[i];
    curve->X[i+DESC_SIZE] = i+DESC_SIZE;
    curve->Y[i+DESC_SIZE] = cvf->highH[i];
  }

  free(p);
  free(vf);
  free(cvf);
  return curve;
}
/******************************************************************/
double gray_level_BIC(Image *img1, Image *img2){
  Property *p1, *p2;
  VisualFeature *vf1, *vf2;
  CompressedVisualFeature *cvf1, *cvf2;
  int i, n1, n2;
  double dist;

  /* computing properties*/
  n1 = img1->nrows * img1->ncols;  
  n2 = img2->nrows * img2->ncols;  

  p1 = (Property *) calloc(n1, sizeof(Property));
  p2 = (Property *) calloc(n2, sizeof(Property));

  if((p1==NULL)||(p2==NULL)){
    printf("\nOut of memory \n");
    exit(-1);
  }
  
  for(i=0; i<n1; i++){
    p1[i].color=img1->val[i];
  }
  for(i=0; i<n2; i++){
    p2[i].color=img2->val[i];
  }
  
  Compute_frequency_property(img1,p1);
  Compute_frequency_property(img2,p2);
  /***********************************/
  
  vf1 = Compute_histograms(p1, n1);
  vf2 = Compute_histograms(p2, n2);
  
  cvf1 = Compress_histograms(vf1, n1);
  cvf2 = Compress_histograms(vf2, n2);
  
  dist = 0.0;
  for (i=0; i<DESC_SIZE; i++){
    dist += fabs(cvf1->lowH[i] - cvf2->lowH[i]);
    dist += fabs(cvf1->highH[i] - cvf2->highH[i]);
  }
  
  free(p1);
  free(p2);
  free(vf1);
  free(vf2);
  free(cvf1);
  free(cvf2);
  return dist;
}

FeatureVector1D  *CurveTo1DFeatureVector(Curve *curve){
  FeatureVector1D *fv;

  fv = CreateFeatureVector1D(curve->n);
  memcpy(fv->X,curve->Y,curve->n*sizeof(double));

 return fv;
}

void Destroy1DFV(Ap_FeatureVector1D **fv, int nimages){
 int i;
 Ap_FeatureVector1D *aux;

 aux = *fv;
 for (i=0; i<nimages; i++){
   DestroyFeatureVector1D(&aux[i]);
 }
 free(aux);
 aux = NULL; ;

}

