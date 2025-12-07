#include "curve.h"
#include "common.h"

Curve *CreateCurve(int n)
{
  Curve *curve=NULL;

  curve = (Curve *) calloc(1,sizeof(Curve));
  if (curve != NULL) {
    curve->X = AllocDoubleArray(n);
    curve->Y = AllocDoubleArray(n);
    curve->n = n;
  } else {
    Error(MSG1,"CreateCurve");
  }
  return(curve);
}

void DestroyCurve(Curve **curve)
{
  Curve *aux;

  aux = *curve;
  if (aux != NULL){
    if (aux->X != NULL) free(aux->X);
    if (aux->Y != NULL) free(aux->Y);
    free(aux);
    *curve = NULL;
  }
}

Curve *ReadCurve(char *filename)
{
  FILE *fp;
  int i,n;
  Curve *curve;
  float x,y;
  
  fp = fopen(filename,"r");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }
  fscanf(fp,"%d\n",&n);
  curve= CreateCurve(n);
  for (i=0; i < curve->n; i++) {
    fscanf(fp,"%f %f\n",&x,&y);
    curve->X[i] = (double)x;
    curve->Y[i] = (double)y;
  }
  fclose(fp);
  return (curve);
}

void WriteCurve(Curve *curve,char *filename)
{
  FILE *fp;
  int i;
  
  fp = fopen(filename,"w");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }
  fprintf(fp,"%d\n",curve->n);
  for (i=0; i < curve->n; i++)
    fprintf(fp,"%f %f\n",curve->X[i],curve->Y[i]);
  
  fclose(fp);
}


void WriteCurve2Gnuplot(Curve *curve,char *filename){
  FILE *fp;
  int i;
  
  fp = fopen(filename,"w");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }
  fprintf(fp,"#%d\n",curve->n);
  for (i=0; i < curve->n; i++)
    fprintf(fp,"%f %f\n",curve->X[i],curve->Y[i]);
  
  fclose(fp);
}


Curve *CopyCurve(Curve *curve)
{
  Curve *curvec;

  curvec = CreateCurve(curve->n);
  memcpy(curvec->X,curve->X,curve->n*sizeof(double));
  memcpy(curvec->Y,curve->Y,curve->n*sizeof(double));
  return(curvec);
}

Curve3D *CreateCurve3D(int n)
{
  Curve3D *curve=NULL;

  curve = (Curve3D *) calloc(1,sizeof(Curve3D));
  if (curve != NULL) {
    curve->X = AllocDoubleArray(n);
    curve->Y = AllocDoubleArray(n);
    curve->Z = AllocDoubleArray(n);
    curve->n = n;
  } else {
    Error(MSG1,"CreateCurve3D");
  }
  return(curve);
}

void DestroyCurve3D(Curve3D **curve)
{
  Curve3D *aux;

  aux = *curve;
  if (aux != NULL){
    if (aux->X != NULL) free(aux->X);
    if (aux->Y != NULL) free(aux->Y);
    if (aux->Z != NULL) free(aux->Z);
    free(aux);
    *curve = NULL;
  }
}

void WriteCurve3D(Curve3D *curve,char *filename)
{
  FILE *fp;
  int i;
  
  fp = fopen(filename,"w");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }
  for (i=0; i < curve->n; i++)
    fprintf(fp,"%f %f %f\n",curve->X[i],curve->Y[i],curve->Z[i]);

  fclose(fp);
}

void SortCurve3D(Curve3D *curve, int left, int right, char order)
{
  int pivot;
 
  if (left < right) {
    pivot = PartCurve3D(curve,left,right,order);
    SortCurve3D(curve,left,pivot-1,order);
    SortCurve3D(curve,pivot+1,right,order); 
  }
}

int PartCurve3D (Curve3D *curve, int left, int right, char order)
{
  double z;
  int i;
  double X,Y,Z;
 
  z = curve->Z[left];
  i = left;
 
  do {
    if (order == INCREASING){
      while ((left <= right) && (curve->Z[left] <= z)) left++;
      while (curve->Z[right] > z) right--;
    } else { /* order = DECREASING */
      while ((curve->Z[left] >= z)&&(left <= right)) left++;
      while (curve->Z[right]  < z) right--;
    }
    if (left < right){
      X = curve->X[left];
      Y = curve->Y[left];
      Z = curve->Z[left];
      curve->X[left]  = curve->X[right];
      curve->Y[left]  = curve->Y[right];
      curve->Z[left]  = curve->Z[right];
      curve->X[right] = X;
      curve->Y[right] = Y;
      curve->Z[right] = Z;
      left++; right--;
    }
  } while (left <= right);

  left = i;

  if (left != right){
    X = curve->X[left];
    Y = curve->Y[left];
    Z = curve->Z[left];
    curve->X[left]  = curve->X[right];
    curve->Y[left]  = curve->Y[right];
    curve->Z[left]  = curve->Z[right];
    curve->X[right] = X;
    curve->Y[right] = Y;
    curve->Z[right] = Z;
  }

  return (right);
}

void SortCurve(Curve *curve, int left, int right, char order)
{
  int pivot;
 
  if (left < right) {
    pivot = PartCurve(curve,left,right,order);
    SortCurve(curve,left,pivot-1,order);
    SortCurve(curve,pivot+1,right,order); 
  }
}

int PartCurve (Curve *curve, int left, int right, char order)
{
  double y;
  int i;
  double X,Y;
 
  y = curve->Y[left];
  i = left;
 
  do {
    if (order == INCREASING){
      while ((curve->Y[left] <= y)&&(left <= right)) left++;
      while (curve->Y[right]  > y) right--;
    } else { /* order = DECREASING */
      while ((curve->Y[left] >= y)&&(left <= right)) left++;
      while (curve->Y[right]  < y) right--;
    }
    if (left < right){
      X = curve->X[left];
      Y = curve->Y[left];
      curve->X[left]  = curve->X[right];
      curve->Y[left]  = curve->Y[right];
      curve->X[right] = X;
      curve->Y[right] = Y;
      left++; right--;
    }
  } while (left <= right);

  left = i;

  if (left != right){
    X = curve->X[left];
    Y = curve->Y[left];
    curve->X[left]  = curve->X[right];
    curve->Y[left]  = curve->Y[right];
    curve->X[right] = X;
    curve->Y[right] = Y;
  }

  return (right);
}


void InvertXY(Curve *curve)
{
  double tmp;
  int i;
  for (i=0; i<curve->n; i++){
    tmp = curve->X[i];
    curve->X[i] = curve->Y[i];
    curve->Y[i] = tmp;
  }
}


Curve *FillCurve(Curve *c1, double dx) {
  
  double dy,x,y;
  int i,j,n;
  Curve *c2;

  n = (int)((c1->X[c1->n-1] - c1->X[0])/dx) + 1;

  InvertXY(c1);
  SortCurve(c1, 0, c1->n-1,INCREASING);
  InvertXY(c1);

  c2 = CreateCurve(n);

  x = c1->X[0];
  y = c1->Y[0];
  j = 0;
  dy = 0.;

  for (i = 0; i < n; i++) {
    if (x >= c1->X[j]) {
      y = c1->Y[j];
      j++;
      if (j < c1->n)
	dy = (c1->Y[j] - c1->Y[j-1]) / (c1->X[j] - c1->X[j-1]) * dx;
    }
    c2->X[i] = x;
    c2->Y[i] = y;
    x += dx;
    y += dy;
  }
  return(c2);
}

Curve *JoinCurve(Curve *c1, Curve *c2) {
  
  int i,j,n;
  Curve *c3,*c4;

  n = c1->n+c2->n;
 
  c3 = CreateCurve(n);

  
  for (i=0, j=0; i < c1->n; i++) {
    c3->X[j] = c1->X[i];
    c3->Y[j] = c1->Y[i];
    j++;
  }
  for (i=0; i < c2->n; i++) {
    c3->X[j] = c2->X[i];
    c3->Y[j] = c2->Y[i];
    j++;
  }

  InvertXY(c3);
  SortCurve(c3, 0, c3->n-1,INCREASING);
  InvertXY(c3);

  for (i=0; i < c3->n-1; i++) {
    if (c3->X[i] == c3->X[i+1]) {
      c3->X[i+1] = INT_MIN;
      c3->Y[i] = MAX(c3->Y[i],c3->Y[i+1]);
      n--;
    }
  }

  c4 = CreateCurve(n);
  
  for (i=0,j=0; i < c3->n; i++) {
    if (c3->X[i] != INT_MIN) {
      c4->X[j] = c3->X[i];
      c4->Y[j] = c3->Y[i];
      j++;
    }
  }

  DestroyCurve(&c3);

  return(c4);
}

Curve3D *GetLinePoints (Voxel *p, Voxel *q) {

  Voxel d;
  double dist;
  int i,n;
  Curve3D *c;

  d.x = q->x - p->x;
  d.y = q->y - p->y;
  d.z = q->z - p->z;

  dist = sqrt((double)(d.x*d.x + d.y*d.y + d.z*d.z));
  n = ROUND(dist);
  c = CreateCurve3D(n);
  for (i=0;i<n;i++) {
     c->X[i]= ((double)p->x + (double)(d.x*i)/ dist);
     c->Y[i]= ((double)p->x + (double)(d.x*i)/ dist);
     c->Z[i]= ((double)p->x + (double)(d.x*i)/ dist);
  }
  return(c);
}
 
double CurveMaximum(Curve *c) {
  
  int i;
  double max = 0.0;
  
  for (i=0;i<c->n;i++)
    if (max < c->Y[i])
      max = c->Y[i];
  return max;

}

int CurveMaximumPosition(Curve *c) {
  
  int i, position = 0;
  double max = -FLT_MAX;
  
  for (i=0;i<c->n;i++)
    if (max < c->Y[i]) {
      max = c->Y[i];
      position = c->X[i];
    }
  return position;

}

double CurveMinimum(Curve *c) {
  
  int i;
  double min = FLT_MAX;
  
  for (i=0;i<c->n;i++)
    if (min > c->Y[i])
      min = c->Y[i];
  return min;

}

int CurveMinimumPosition(Curve *c) {
  
  int i, position = 0;
  double min = FLT_MAX;
  
  for (i=0;i<c->n;i++)
    if (min > c->Y[i]) {
      min = c->Y[i];
      position = c->X[i];
    }
  return position;

}

double CurveNonZeroMinimum(Curve *c) {
  
  int i;
  double min = FLT_MAX;
  
  for (i=0;i<c->n;i++)
    if ( (min > c->Y[i]) && (c->Y[i] != 0) )
      min = c->Y[i];
  return min;

}

int CurveNonZeroMinimumPosition(Curve *c) {
  
  int i, position = 0;
  double min = FLT_MAX;
  
  for (i=0;i<c->n;i++)
    if ((min > c->Y[i]) && (c->Y[i] != 0) ) {
      min = c->Y[i];
      position = c->X[i];
    }
  return position;
  
}

double CurveLocalMaximum(Curve *c, int pmin, int pmax) {
  
  int i;
  double max = -FLT_MAX;
  if( pmin < 0 ) pmin = 0;
  if( pmax > c->n ) pmax = c->n - 1;
  for (i=pmin;i<=pmax;i++)
    if (max < c->Y[i])
      max = c->Y[i];
  return max;

}

int CurveLocalMaximumPosition(Curve *c, int pmin, int pmax) {
  
  int i, position = 0;
  double max = -FLT_MAX;
  
  if( pmin < 0 ) pmin = 0;
  if( pmax > c->n ) pmax = c->n - 1;
  for (i=pmin;i<=pmax;i++)
    if (max < c->Y[i]) {
      max = c->Y[i];
      position = c->X[i];
    }
  return position;

}

double CurveLocalMinimum(Curve *c, int pmin, int pmax) {
  
  int i;
  double min = FLT_MAX;
  
  if( pmin < 0 ) pmin = 0;
  if( pmax > c->n ) pmax = c->n - 1;
  for (i=pmin;i<=pmax;i++)
    if (min > c->Y[i])
      min = c->Y[i];
  return min;

}

int CurveLocalMinimumPosition(Curve *c, int pmin, int pmax) {
  
  int i, position = 0;
  double min = FLT_MAX;
  
  if( pmin < 0 ) pmin = 0;
  if( pmax > c->n ) pmax = c->n - 1;
  for (i=pmin;i<=pmax;i++)
    if (min > c->Y[i]){
      min = c->Y[i];
      position = c->X[i];
    }
  return position;

}

double CurveNonZeroLocalMinimum(Curve *c, int pmin, int pmax) {
  
  int i;
  double min = FLT_MAX;
  
  if( pmin < 0 ) pmin = 0;
  if( pmax > c->n ) pmax = c->n - 1;
  for (i=pmin;i<=pmax;i++)
    if ((min > c->Y[i]) && (c->Y[i] != 0))
      min = c->Y[i];
  return min;

}

int CurveNonZeroLocalMinimumPosition(Curve *c, int pmin, int pmax) {
  
  int i, position = 0;
  double min = FLT_MAX;
  
  if( pmin < 0 ) pmin = 0;
  if( pmax > c->n ) pmax = c->n - 1;
  for (i=pmin;i<=pmax;i++)
    if ((min > c->Y[i]) && (c->Y[i] != 0)){
      min = c->Y[i];
      position = c->X[i];
    }
  return position;

}

int CurveMeanPosition(Curve *c) {
  
  int i;
  double mean = 0.0;
  int size = 0;
  
  for (i=0;i<=c->n - 1;i++)
    mean += c->Y[i];
  mean /= 2;
  size = 0;
  for ( i = 0; size < mean; i++ )
    size += c->Y[i];
  return i;
}

int CurveLocalMeanPosition(Curve *c, int pmin, int pmax) {
  
  int i;
  double mean = 0.0;
  double size;
  
  if( pmin < 0 ) pmin = 0;
  if( pmax > c->n ) pmax = c->n - 1;
  for (i=pmin;i<=pmax;i++)
    mean += c->Y[i];
  mean /= 2;
  size = 0.0;
  for (i=pmin;size < mean;i++)
    size += c->Y[i];
  return i;
}

