#include "metrics.h"

double ZNCC(FeatureVector1D *v1, FeatureVector1D *v2){
  double S1, S11, S2, S22, S12;
  double corr;
  int i;
  
  S1 = 0.0;
  S11 = 0.0;
  S2 = 0.0;
  S22 = 0.0;
  S12 = 0.0;
  for (i=0; i<v1->n; i++){
    S1 += v1->X[i];
    S11 += v1->X[i]*v1->X[i];
    S2 += v2->X[i];
    S22 += v2->X[i]*v2->X[i];
    S12 += v1->X[i]*v2->X[i];
  }
  
  corr = (v1->n*S12 - S1*S2)/(sqrt(v1->n*S11 - S1*S1)*(v1->n*S22 - S2*S2));
  return corr;
}

double FourierDistance(FeatureVector1D *v1, FeatureVector1D *v2){
  int i;
  double *r1 = NULL;
  double *imag1 = NULL;
  double *r2 = NULL;
  double *imag2 = NULL;
  double norm1, norm2, dist;
  FeatureVector1D *d1 = CreateFeatureVector1D(v1->n-1);
  FeatureVector1D *d2 = CreateFeatureVector1D(v2->n-1);
  

  r1 = AllocDoubleArray(v1->n);
  imag1 = AllocDoubleArray(v1->n);
  r2 = AllocDoubleArray(v2->n);
  imag2 = AllocDoubleArray(v2->n);
  
  for (i=0; i<v1->n; i++) {
    r1[i] = v1->X[i];
    imag1[i] = 0.0;
    r2[i] = v2->X[i];
    imag2[i] = 0.0;
  }
  
  FFT(1, v1->n, r1, imag1);
  FFT(1, v1->n, r2, imag2);
  
  norm1 = sqrt(r1[0]*r1[0]+imag1[0]*imag1[0]);
  norm2 = sqrt(r2[0]*r2[0]+imag2[0]*imag2[0]);
  for (i = 1; i < v1->n ; i++){
    d1->X[i-1] = sqrt(r1[i]*r1[i]+imag1[i]*imag1[i])/norm1;
    d2->X[i-1] = sqrt(r2[i]*r2[i]+imag2[i]*imag2[i])/norm2;
  }
  
  dist = L1Distance(d1, d2);
  
  free(r1);
  free(imag1);
  free(r2);
  free(imag2);
  DestroyFeatureVector1D(&d1);
  DestroyFeatureVector1D(&d2);
  
  return dist;  
}


double EuclideanDistance(FeatureVector1D *v1, FeatureVector1D *v2) { 
  int i;
  double sum = 0.0;
  double z = 0.0;
  
  for (i = 0; i < v1->n ; i++){
    z = v1->X[i] - v2->X[i]; 
    sum += z*z;
  }
  sum = sqrt(sum);
  return (sum);
}

double L1Distance(FeatureVector1D *v1, FeatureVector1D *v2) { 
  int i;
  double sum = 0.0;
  double z = 0.0;
  
  for (i = 0; i < v1->n ; i++){
    z = fabs(v1->X[i] - v2->X[i]); 
    sum = sum + z;
  }
  return (sum);
}

double Lifnt(FeatureVector1D *v1, FeatureVector1D *v2) { 
  int i;
  double max = -1.0;
  double z = 0.0;
  
  for (i = 0; i < v1->n ; i++){
    z = fabs(v1->X[i] - v2->X[i]); 
    if (max < z) max = z;
  }
  return (max);
}

double KSStatistics(FeatureVector1D *v1, FeatureVector1D *v2){
  int i;
  double dist, sum, z;
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
  
  dist= -1.0;
  for (i = 0; i < v1->n ; i++){
    z = fabs(d1->X[i] - d2->X[i]); 
    if (z > dist)
      dist = z;
  }
  
  DestroyFeatureVector1D(&aux1);
  DestroyFeatureVector1D(&aux2);
  DestroyFeatureVector1D(&d1);
  DestroyFeatureVector1D(&d2);
  
  return dist;
}
