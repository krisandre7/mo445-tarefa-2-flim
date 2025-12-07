
#include "realarray.h"


real MaxValInRealArray(real *A, int n){
  real max;
  int i;

  max = A[0];
  for(i=1; i<n; i++)
    if(A[i]>max)
      max = A[i];
  return max;
}


real MinValInRealArray(real *A, int n){
  real min;
  int i;

  min = A[0];
  for(i=1; i<n; i++)
    if(A[i]<min)
      min = A[i];
  return min;
}


int *RealArray2IntArray(real *A, int n,
			int Imin, int Imax){
  real Rmin,Rmax;
  int *vi,i;

  Rmin = MinValInRealArray(A, n);
  Rmax = MaxValInRealArray(A, n);

  vi = AllocIntArray(n);
  for(i=0; i<n; i++)
    vi[i] = Imin+ROUND((Imax-Imin)*(A[i]-Rmin)/(Rmax-Rmin));
  return vi;
}


void RealArrayStatistics(real *A, int n,
			 real *mean,
			 real *stdev){
  real sum=0.0,sum2=0.0,num=0.0,tmp;
  int i;
  
  for(i=0; i<n; i++){
    sum  += A[i];
    sum2 += A[i]*A[i];
    num++;
  }
  *mean = sum/num;
  tmp = sum2-(sum*sum)/num;
  *stdev = sqrtreal(tmp/num);
}


real *ReadTxt2RealArray(char *filename, int *n){
  FILE *fp=NULL;
  real num,*A=NULL;
  int ret,i,size=0;

  fp = fopen(filename,"r");
  if(fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }

  while(1){
    ret = fscanf(fp," %f", &num);
    if(ret==EOF)
      break;
    size++;
  }

  if(size>0){
    rewind(fp);
    A = AllocRealArray(size);

    i = 0;
    while(i<size){
      ret = fscanf(fp," %f", &A[i]);
      if(ret==EOF)
	break;
      i++;
    }
  }
  *n = size;
  fclose(fp);
  return A;
}

