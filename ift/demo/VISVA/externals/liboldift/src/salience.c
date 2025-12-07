#include "salience.h"

/********************************************
Contour Saliences
********************************************/
Curve3D *iftContourSaliences(Image *bin,int threshold_in,int threshold_out,int angle_in,int angle_out)
{

  Curve3D *saliences = NULL;
  Curve3D *convex_saliences = NULL;
  Curve3D *concave_saliences = NULL;
  int i;
  int maxdist = 10;
  
  convex_saliences  = SkelCont(bin,maxdist,threshold_in, angle_in, INTERIOR);
  concave_saliences = SkelCont(bin, maxdist, threshold_out, angle_out, EXTERIOR);
  saliences = CreateCurve3D(convex_saliences->n + concave_saliences->n);
  printf("convex->n=%d, concave->n=%d\n", convex_saliences->n, concave_saliences->n);
  for (i=0; i<convex_saliences->n; i++){
    saliences->X[i] = convex_saliences->X[i];
    saliences->Y[i] = convex_saliences->Y[i];
    saliences->Z[i] = convex_saliences->Z[i];
  }
  for (i=convex_saliences->n; i<saliences->n;i++){
    saliences->X[i] = concave_saliences->X[i-convex_saliences->n];
    saliences->Y[i] = concave_saliences->Y[i-convex_saliences->n];
    saliences->Z[i] = concave_saliences->Z[i-convex_saliences->n];
  }
  
  DestroyCurve3D(&convex_saliences);
  DestroyCurve3D(&concave_saliences);
  return saliences;
}

Image *PaintContourSaliences(Image *bin, int threshold_in,int threshold_out,int angle_in,int angle_out, int side)
{
  Image *points=NULL;
  Curve3D *curve=NULL;

  if (side == BOTH){
    curve = iftContourSaliences(bin,threshold_in,threshold_out,angle_in,angle_out);
  }
  else if (side == EXTERIOR){
    curve = iftContourConcaveSaliences(bin,threshold_out,angle_out);
  }
  else if (side == INTERIOR){
    curve = iftContourConvexSaliences(bin,threshold_in,angle_in);
  }
  
  points = PaintSaliences(bin, curve);
  
  DestroyCurve3D(&curve);
  return(points);
}


Curve3D *iftContourConvexSaliences(Image *bin,int threshold,int angle)
{
  Curve3D *saliences = NULL;
  int maxdist = 10;
  
  saliences = SkelCont(bin,maxdist,threshold, angle, INTERIOR);
  return(saliences);
}

Curve3D *iftContourConcaveSaliences(Image *bin,int threshold,int angle)
{
  Curve3D *saliences = NULL;
  int maxdist = 10;
  saliences = SkelCont(bin,maxdist,threshold, angle, EXTERIOR);
  return(saliences);
}

Curve *ContourSaliences(Image *in)
{
  Curve3D *saliences = NULL;
  Curve *descriptor = NULL;
  Image *contour = NULL;
  int i, p, max;
  //timer  tic,toc;
  
  //gettimeofday(&tic,NULL);
  
  contour   = LabelContPixel(in);
  saliences = iftContourSaliences(in, 5, 20, 50, 110);    
  
  descriptor = CreateCurve(saliences->n);
  max = MaximumValue(contour);
  for (i=0; i<saliences->n; i++){
    descriptor->X[i] = saliences->Z[i];
    p = (int)saliences->X[i]+contour->tbrow[(int)saliences->Y[i]];
    descriptor->Y[i] = (double)(((contour->val[p])-1))/max;
  }
  SortCurve(descriptor, 0, (descriptor->n - 1), INCREASING);
  InvertXY(descriptor);  
  
  DestroyCurve3D(&saliences);
  DestroyImage(&contour);
  //gettimeofday(&toc,NULL);
  //printf("Time in msecs %f\n",(toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001);
  return (descriptor);
}

double FourierSalienceDistance(FeatureVector2D *d1, FeatureVector2D *d2)
{
  double *r1 = NULL;
  double *imag1 = NULL;
  double *r2 = NULL;
  double *imag2 = NULL;
  double *r_cor = NULL;
  double *imag_cor = NULL;
  double sum1, sum2, c1, c2, max;
  int i, index,n=2048;
  FeatureVector2D *descriptor1;
  FeatureVector2D *descriptor2;
  
  descriptor1 = CopyFeatureVector2D(d1);
  descriptor2 = CopyFeatureVector2D(d2);
  
  sum1 = 0.0;
  for (i=0; i<descriptor1->n; i++) {
    sum1 += descriptor1->Y[i]*descriptor1->Y[i];
  }
  sum2 = 0.0;
  for (i=0; i<descriptor2->n; i++) {
    sum2 += descriptor2->Y[i]*descriptor2->Y[i];
  }
  
  c1 = sqrt(1.0/(sum1));
  c2 = sqrt(1.0/(sum2));
  
  for (i=0; i<descriptor1->n; i++) {
    descriptor1->Y[i]= c1*descriptor1->Y[i];
  }

  for (i=0; i<descriptor2->n; i++) {
    descriptor2->Y[i]= c2*descriptor2->Y[i];
  }
  
  r1 = AllocDoubleArray(n);
  imag1 = AllocDoubleArray(n);
  r2 = AllocDoubleArray(n);
  imag2 = AllocDoubleArray(n);
  r_cor = AllocDoubleArray(n);
  imag_cor = AllocDoubleArray(n);
  
  for (i=0; i<descriptor1->n; i++) {
    index = ROUND(descriptor1->X[i]*2048);
    r1[index] = 100*descriptor1->Y[i];
    //imag1[i] = 0.0;
  }
  
  for (i=0; i<descriptor2->n; i++) {
    index = ROUND(descriptor2->X[i]*2048);
    r2[index] = 100*descriptor2->Y[i];
    //imag2[i] = 0.0;
  }
  
  FFT(1, n, r1, imag1);
  FFT(1, n, r2, imag2);
  
  for (i=0; i<n; i++) {
    imag1[i] = -imag1[i];
  } 
  
  for (i=0; i<n; i++) {
    r_cor[i] = r1[i]*r2[i] - imag1[i]*imag2[i];
    imag_cor[i] = r1[i]*imag2[i] + imag1[i]*r2[i];
  } 
  
  FFT(-1, n, r_cor, imag_cor);
 
  max = INT_MIN;
  for (i=0; i<n; i++) {
    if (max < r_cor[i]){
      max = r_cor[i];
    }
  }
  //printf("MAX=%lf\n", max);
  
  /*fp = fopen("correlation.txt","w");   
  for (i=0; i<n; i++){
    fprintf(fp, "%lf %lf\n", (float) i, r_cor[i]);
  }
  fclose(fp);*/    

  free(r1);
  free(imag1);
  free(r2);
  free(imag2);
  free(r_cor);
  free(imag_cor);
  DestroyFeatureVector2D(&descriptor1);
  DestroyFeatureVector2D(&descriptor2);
  return max;  
}

double MatchDistance(FeatureVector1D *v1, FeatureVector1D *v2){
  int i;
  double dist, sum;
  FeatureVector1D *aux1 = CopyFeatureVector1D(v1);
  FeatureVector1D *aux2 = CopyFeatureVector1D(v2);
  
  FeatureVector1D *d1 = CreateFeatureVector1D(v1->n);
  FeatureVector1D *d2 = CreateFeatureVector1D(v2->n);

  sum = 0.0;
  for (i = 0; i < v1->n ; i++){
    sum += v1->X[i];
  }
  for (i = 0; i < v1->n ; i++){
    aux1->X[i]/=sum;
  }
  
  sum = 0.0;
  for (i = 0; i < v2->n ; i++){
    sum += v2->X[i];
  }
  for (i = 0; i < v2->n ; i++){
    aux2->X[i]/=sum;
  }
  
  d1->X[0] = aux1->X[0];
  d2->X[0] = aux2->X[0];
  
  for (i = 1; i < v2->n ; i++){
    d1->X[i] = d1->X[i-1]+ aux1->X[i];
    d2->X[i] = d2->X[i-1]+ aux2->X[i];
  }
  
  dist = L1Distance(d1, d2);
  
  DestroyFeatureVector1D(&aux1);
  DestroyFeatureVector1D(&aux2);
  DestroyFeatureVector1D(&d1);
  DestroyFeatureVector1D(&d2);
  
  return dist;
  
}


double ContSalieDistance(FeatureVector2D *D1, FeatureVector2D *D2){
  
  int i;
  int n = MIN(D1->n, D2->n);
  double deltaX, dist, eps = 0.2;
  
  dist = 0.0;
  for (i=0; i<n; i++){
    deltaX = fabs(D1->X[i]-D2->X[i]);
    if (deltaX<=eps){
      dist = dist + sqrt(pow(D1->X[i] - D2->X[i],2)+
			 pow(D1->Y[i] - D2->Y[i],2));
    }else
      dist = dist + fabs(D1->Y[i]) + fabs(D2->Y[i]);    
  }
  
  for (i=n; i<D1->n; i++){
    dist = dist + fabs(D1->Y[i]);
  }
  
  for (i=n; i<D2->n; i++){
    dist = dist + fabs(D2->Y[i]);
  }
  
  return dist;
}
/****************************************************************************/
FeatureVector2D *CircularRotation(FeatureVector2D *descriptor, double delta){
  
  FeatureVector2D *c = CreateFeatureVector2D(descriptor->n);
  int i;
  
  for (i=0; i<descriptor->n; i++){
    c->Y[i] = descriptor->X[i] + delta;
    if (c->Y[i]<0.0)
      c->Y[i] = 1.0 + c->Y[i];
    if (c->Y[i]>1.0)
      c->Y[i] = 1.0 - c->Y[i];
    c->X[i] = descriptor->Y[i];
  }
  SortFeatureVector2D(c, 0, (c->n-1), INCREASING);
  DescInvertXY(c);
  return c;
}
/**************************************************************************/
void WriteInstance(int i, int j, 
		   FeatureVector2D *descriptor1, FeatureVector2D *descriptor2, 
		   FeatureVector2D *d1, FeatureVector2D *d2, 
		   FeatureVector2D *D1, FeatureVector2D *D2, 
		   double delta1, double delta2, double distance){
  int k;
  
  printf("******************************\n");
  printf("i = %d j = %d\n", i, j);
  printf("delta1 = %lf delta2=%lf\n", delta1, delta2);
  printf("distance = %lf\n", distance);
  printf("********* descriptor1 *****************\n");
  for (k=0; k < descriptor1->n; k++)
    printf("%f\t%f\n",descriptor1->X[k],descriptor1->Y[k]);
  printf("********* descriptor2 *****************\n");
  for (k=0; k < descriptor2->n; k++)
    printf("%f\t%f\n",descriptor2->X[k],descriptor2->Y[k]);
  printf("********* d1 *****************\n");
  for (k=0; k < d1->n; k++)
    printf("%f\t%f\n",d1->X[k],d1->Y[k]);
  printf("********* d2 *****************\n");
  for (k=0; k < d2->n; k++)
    printf("%f\t%f\n",d2->X[k],d2->Y[k]);
  printf("********* D1 *****************\n");
  for (k=0; k < D1->n; k++)
    printf("%f\t%f\n",D1->X[k],D1->Y[k]);
  printf("********* D2 *****************\n");
  for (k=0; k < D2->n; k++)
    printf("%f\t%f\n",D2->X[k],D2->Y[k]);
  
}
/**************************************************************************/
double Matching(FeatureVector2D *descriptor1, FeatureVector2D *descriptor2, int order)
{
  FeatureVector2D *d1 = NULL;
  FeatureVector2D *d2 = NULL;
  FeatureVector2D *D1  = NULL;
  FeatureVector2D *D2  = NULL;
  double max1, max2;
  double dist, distance = INT_MAX;
  int i,j;
  
  d1 = CopyFeatureVector2D(descriptor1);
  d2 = CopyFeatureVector2D(descriptor2);
  SortFeatureVector2D(d1, 0, (d1->n - 1), order);
  SortFeatureVector2D(d2, 0, (d2->n - 1), order);
  
  max1 = fabs(d1->Y[0]);
  max2 = fabs(d2->Y[0]);
  
  i = 0;
  while ((i<d1->n)&&
	 ((fabs(d1->Y[i]) - max1)<=(fabs(0.2*  max1)))){
    j = 0;
    while((j<d2->n)&&
	  ((fabs(d2->Y[j]) - max2)<=(fabs(0.2 * max2)))){
      if (d1->Y[i]*d2->Y[j]>0.0){
	D1 = CircularRotation(descriptor1, -d1->X[i]);
	D2 = CircularRotation(descriptor2, -d2->X[j]);
	dist = ContSalieDistance(D1, D2);
	//WriteInstance(i, j, descriptor1, descriptor2, d1, d2, D1, D2, -d1->X[i], -d2->X[j], dist);
	if (dist < distance)
	  distance = dist;
	DestroyFeatureVector2D(&D1);
	DestroyFeatureVector2D(&D2);
      }
      j++;
    }
    i++;
  }
  
  DestroyFeatureVector2D(&d1);
  DestroyFeatureVector2D(&d2);
  return distance;
}

double ContourSalienceMatching(FeatureVector2D *descriptor1, FeatureVector2D *descriptor2){
  double convex_distance = INT_MIN;
  double concave_distance = INT_MIN;
  
  convex_distance = Matching(descriptor1, descriptor2, DECREASING);
  concave_distance = Matching(descriptor1, descriptor2, INCREASING);
  //printf("convex_distance = %lf  concave_distance = %lf\n", convex_distance, concave_distance);
  return(MIN(convex_distance, concave_distance));
}


