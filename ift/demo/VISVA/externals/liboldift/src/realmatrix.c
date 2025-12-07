
#include "scene.h"
#include "realmatrix.h"


RealMatrix *CreateRealMatrix(int ncols,int nrows){
  RealMatrix *mat=NULL;
  real *aux;
  int i;

  mat = (RealMatrix *) calloc(1,sizeof(RealMatrix));
  if(mat == NULL)
    Error(MSG1,"CreateRealMatrix");

  aux = (real *)calloc(nrows*ncols, sizeof(real));
  mat->val = (real **) calloc(nrows, sizeof(real *));
  if(mat->val == NULL || aux == NULL)
    Error(MSG1,"CreateRealMatrix");

  mat->val[0] = aux;
  for(i=1; i<nrows; i++) 
    mat->val[i] = mat->val[i-1] + ncols;

  mat->ncols = ncols;
  mat->nrows = nrows;
 
  return(mat);
}


void        DestroyRealMatrix(RealMatrix **mat){
  RealMatrix *aux;

  aux = *mat;
  if(aux != NULL){
    if(aux->val != NULL){
      if(*(aux->val) != NULL)
	free(*(aux->val));
      free(aux->val); 
    }
    free(aux);
    *mat = NULL;
  }
}


RealMatrix *CloneRealMatrix(RealMatrix *mat){
  RealMatrix *matc;

  matc = CreateRealMatrix(mat->ncols, mat->nrows);
  memcpy(matc->val[0], mat->val[0],
	 mat->ncols*mat->nrows*sizeof(real));
  
  return(matc);
}


void        CopyRealMatrix(RealMatrix *dest, 
			   RealMatrix *src){
  if(dest->ncols!=src->ncols ||
     dest->nrows!=src->nrows)
    Error("Incompatible matrices","CopyRealMatrix");

  memcpy(dest->val[0], src->val[0], 
	 src->ncols*src->nrows*sizeof(real));
}


void   PrintRealMatrix(RealMatrix *M){
  int x, y;
  
  printf("\n");
  for(y=0; y<M->nrows; y++){
    for(x=0; x<M->ncols; x++)
      printf("%3.5lf ", M->val[y][x]);
    printf("\n");
  }
  printf("\n");
}


void   PrintRealMatrixDimension(RealMatrix *M){
  printf("\n%d x %d\n", M->nrows, M->ncols);
}


RealMatrix *InvertRealMatrix(RealMatrix *A){
  RealMatrix *B=NULL;
  RealMatrix *I=NULL;
  int i,j,k;
  double m;
  
  if(A->ncols!=A->nrows)
    Error("Matrix dimension error","InvertRealMatrix");
  
  I = CreateRealMatrix(A->ncols, A->nrows);
  B = CloneRealMatrix(A);
  
  for(i=0; i<A->nrows; i++)
    I->val[i][i] = 1.0;

  for(k=0; k<A->nrows; k++){
    m = B->val[k][k];
    //if(m < 0.0000000000001)
      // Error("Singular matrix","InvertRealMatrix");   THIS IS WRONG
    
    B->val[k][k] = 1.0;
    for(j=0; j<A->ncols; j++){
      if(j!=k)
	B->val[k][j] = B->val[k][j]/m;
      I->val[k][j] = I->val[k][j]/m;
    }
    
    for(i=0; i<A->nrows; i++){
      if(i!=k){
	m = B->val[i][k]/B->val[k][k];

	B->val[i][k] = 0.0;
	for(j=0; j<A->ncols; j++){
	  if(j!=k)
	    B->val[i][j] = B->val[i][j] - m*B->val[k][j];
	  I->val[i][j] = I->val[i][j] - m*I->val[k][j];
	}
      }
    }
  }
  
  DestroyRealMatrix(&B);
  return I;
}


real        CompRealMatrixTrace(RealMatrix *M){
  real sum;
  int i;
  
  if(M->ncols!=M->nrows)
    Error("Matrix dimension error","CompRealMatrixTrace");
  
  sum = 0.0;
  for (i=0; i<M->nrows; i++)
    sum += M->val[i][i];
  
  return sum;
}


RealMatrix *MultRealMatrix(RealMatrix *A, 
			   RealMatrix *B){
  RealMatrix *M = NULL;
  int i,j,k;
  
  if(A->ncols!=B->nrows)
    Error("Matrix dimension error","MultRealMatrix");
  
  M = CreateRealMatrix(B->ncols, A->nrows);
  for(i=0; i<M->nrows; i++){
    for(j=0; j<M->ncols; j++){
      M->val[i][j] = 0.0;
      for (k=0; k<A->ncols; k++)
	M->val[i][j] += A->val[i][k]*B->val[k][j];
    }
  }
  return(M);
}


RealMatrix *MultRealMatrixByScalar(RealMatrix *A, 
				   real k){
  RealMatrix *M = NULL;
  int i,j;
  
  M = CreateRealMatrix(A->ncols, A->nrows);
  for(i=0; i<M->nrows; i++){
    for(j=0; j<M->ncols; j++){
      M->val[i][j] = k*A->val[i][j];
    }
  }
  return(M);
}


RealMatrix *DiffRealMatrix(RealMatrix *A, 
			   RealMatrix *B){
  RealMatrix *M = NULL;
  int i,j;
  
  if((A->ncols!=B->ncols)||(A->nrows!=B->nrows))
    Error("Matrix dimension error","DiffRealMatrix");
  
  M = CreateRealMatrix(A->ncols, A->nrows);
  for(i=0; i<M->nrows; i++){
    for(j=0; j<M->ncols; j++){
      M->val[i][j] = A->val[i][j] - B->val[i][j];
    }
  }
  return(M);
}


RealMatrix *SumRealMatrix(RealMatrix *A, 
			  RealMatrix *B){
  RealMatrix *M = NULL;
  int i,j;
  
  if((A->ncols!=B->ncols)||(A->nrows!=B->nrows))
    Error("Matrix dimension error","SumRealMatrix");
  
  M = CreateRealMatrix(A->ncols, A->nrows);
  for(i=0; i<M->nrows; i++){
    for(j=0; j<M->ncols; j++){
      M->val[i][j] = A->val[i][j] + B->val[i][j];
    }
  }
  return(M);
}


RealMatrix *TransposeRealMatrix(RealMatrix *A){
  RealMatrix *M = NULL;
  int i,j;
    
  M = CreateRealMatrix(A->nrows, A->ncols);
  for(i=0; i<M->nrows; i++){
    for(j=0; j<M->ncols; j++){
      M->val[i][j] = A->val[j][i];
    }
  }
  return(M);
}


real RealMatrixDistanceL2(RealMatrix *Y, 
			  RealMatrix *X){
  RealMatrix *A,*B,*R;
  real d;
  
  A = DiffRealMatrix(X, Y);
  B = TransposeRealMatrix(A);
  R = MultRealMatrix(A, B);
  d = CompRealMatrixTrace(R);
  d = sqrtreal(d);
  DestroyRealMatrix(&A);
  DestroyRealMatrix(&B);
  DestroyRealMatrix(&R);

  return (d);
}


RealMatrix *ReadRealMatrix(char *filename){
  RealMatrix *M;
  char msg[512];
  int  ncols,nrows,size,n,p;
  float  *faux=NULL;
  double *daux=NULL;
  FILE *fp;

  fp = fopen(filename,"rb");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"ReadRealMatrix");
  }
  
  fread(&ncols, sizeof(int), 1, fp);
  fread(&nrows, sizeof(int), 1, fp);
  fread(&size,  sizeof(int), 1, fp);

  M = CreateRealMatrix(ncols, nrows);
  n = ncols*nrows;
  if(size==sizeof(real))
    fread(M->val[0], sizeof(real), n, fp);
  else if(size==sizeof(float)){
    faux = AllocFloatArray(n);
    fread(faux, sizeof(float), n, fp);
    for(p=0; p<n; p++)
      M->val[0][p] = (real)faux[p];
    free(faux);
  }
  else if(size==sizeof(double)){
    daux = AllocDoubleArray(n);
    fread(daux, sizeof(double), n, fp);
    for(p=0; p<n; p++)
      M->val[0][p] = (real)daux[p];
    free(daux);
  }
  else
    Error("Bad or corrupted file",
	  "ReadRealMatrix");
  fclose(fp);

  return M;
}


void    WriteRealMatrix(RealMatrix *M,
			char *filename){
  char msg[512];
  int n,ncols,nrows,size;
  FILE *fp;
  
  fp = fopen(filename,"wb");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"WriteRealMatrix");
  }

  size  = sizeof(real);
  ncols = M->ncols;
  nrows = M->nrows;
  n = ncols*nrows;
  fwrite(&ncols,    sizeof(int),  1, fp);
  fwrite(&nrows,    sizeof(int),  1, fp);
  fwrite(&size,     sizeof(int),  1, fp);
  fwrite(M->val[0], sizeof(real), n ,fp);
  fclose(fp);
}


bool   ValidRealMatrixEntry(RealMatrix *M,
			    int i, int j){
  if((j >= 0)&&(j < M->ncols)&&
     (i >= 0)&&(i < M->nrows))
    return(true);
  else
    return(false);
}


real   MaximumRealMatrixValue(RealMatrix *M){
  real max;
  int p,n;

  n = M->ncols*M->nrows;
  max = M->val[0][0];
  for(p=1; p<n; p++)
    if(M->val[0][p] > max)
      max = M->val[0][p];
  
  return(max);
}


real    MinimumRealMatrixValue(RealMatrix *M){
  real min;
  int p,n;

  n = M->ncols*M->nrows;
  min = M->val[0][0];
  for(p=1; p<n; p++)
    if(M->val[0][p] < min)
      min = M->val[0][p];

  return(min);
}


void    SetRealMatrix(RealMatrix *M, 
		      real value){
  int p,n;

  n = M->ncols*M->nrows;
  for(p=0; p<n; p++)
    M->val[0][p] = value;
}


void    ChangeValueRealMatrix(RealMatrix *M, 
			      real old_value,
			      real new_value){
  int p,n;

  n = M->ncols*M->nrows;
  for(p=0; p<n; p++)
    if(M->val[0][p] == old_value)
      M->val[0][p] = new_value;
}


Image  *ConvertRealMatrix2Image(RealMatrix *M){
  Image *img;
  real max,min;
  int p,n;

  n = M->ncols*M->nrows;
  img = CreateImage(M->ncols, M->nrows);
  max = MaximumRealMatrixValue(M);
  min = MinimumRealMatrixValue(M);
  for(p=0; p<n; p++)
    img->val[p] = ROUND(255.0*(M->val[0][p]-min)/(max-min));

  return img;
}


Image      *ThresholdRealMatrix(RealMatrix *M,
				real lower, 
				real higher){
  Image *bin=NULL;
  int p,n;

  bin = CreateImage(M->ncols,M->nrows);
  n = M->ncols*M->nrows;
  for (p=0; p < n; p++)
    if ((M->val[0][p] >= lower)&&(M->val[0][p] <= higher))
      bin->val[p]=1;
  return(bin);
}


RealMatrix* RotationMatrix3(int axis, // options: 0 (x) / 1 (y) / 2 (z)
			    double th)
{
  RealMatrix *m;
  m = CreateRealMatrix(4,4);
  if (axis==0) {
    m->val[0][0] = 1.0;    m->val[0][1] = 0.0;    m->val[0][2] = 0.0;    m->val[0][3] = 0.0;
    m->val[1][0] = 0.0;    m->val[1][1] = cos(th);    m->val[1][2] = -sin(th);    m->val[1][3] = 0.0;
    m->val[2][0] = 0.0;    m->val[2][1] = sin(th);    m->val[2][2] = cos(th);    m->val[2][3] = 0.0;
    m->val[3][0] = 0.0;    m->val[3][1] = 0.0;    m->val[3][2] = 0.0;    m->val[3][3] = 1.0;
  }
  if (axis==1) {
    m->val[0][0] = cos(th);    m->val[0][1] = 0.0;    m->val[0][2] = sin(th);    m->val[0][3] = 0.0;
    m->val[1][0] = 0.0;    m->val[1][1] = 1;    m->val[1][2] = 0.0;    m->val[1][3] = 0.0;
    m->val[2][0] = -sin(th);    m->val[2][1] = 0;    m->val[2][2] = cos(th);    m->val[2][3] = 0.0;
    m->val[3][0] = 0.0;    m->val[3][1] = 0.0;    m->val[3][2] = 0.0;    m->val[3][3] = 1.0;
    
  }
  if (axis==2) {
    m->val[0][0] = cos(th);    m->val[0][1] = -sin(th);    m->val[0][2] = 0.0;    m->val[0][3] = 0.0;
    m->val[1][0] = sin(th);    m->val[1][1] = cos(th);    m->val[1][2] = 0.0;    m->val[1][3] = 0.0;
    m->val[2][0] = 0.0;    m->val[2][1] = 0.0;    m->val[2][2] = 1.0;    m->val[2][3] = 0.0;
    m->val[3][0] = 0.0;    m->val[3][1] = 0.0;    m->val[3][2] = 0.0;    m->val[3][3] = 1.0;
    
  }
  return m;
}



RealMatrix* TranslationMatrix3(float dx, float dy, float dz)
{
  RealMatrix *m;
  m = CreateRealMatrix(4,4);
  m->val[0][0] = 1.0;    m->val[0][1] = 0.0;    m->val[0][2] = 0.0;    m->val[0][3] = dx;
  m->val[1][0] = 0.0;    m->val[1][1] = 1.0;    m->val[1][2] = 0.0;    m->val[1][3] = dy;
  m->val[2][0] = 0.0;    m->val[2][1] = 0.0;    m->val[2][2] = 1.0;    m->val[2][3] = dz;
  m->val[3][0] = 0.0;    m->val[3][1] = 0.0;    m->val[3][2] = 0.0;    m->val[3][3] = 1.0;
  return m;
}

RealMatrix* ScaleMatrix3(float Sx, float Sy, float Sz)
{
  RealMatrix *m;
  m = CreateRealMatrix(4,4);
  m->val[0][0] = Sx;    m->val[0][1] = 0.0;    m->val[0][2] = 0.0;    m->val[0][3] = 0;
  m->val[1][0] = 0.0;    m->val[1][1] = Sy;    m->val[1][2] = 0.0;    m->val[1][3] = 0;
  m->val[2][0] = 0.0;    m->val[2][1] = 0.0;    m->val[2][2] = Sz;    m->val[2][3] = 0;
  m->val[3][0] = 0.0;    m->val[3][1] = 0.0;    m->val[3][2] = 0.0;    m->val[3][3] = 1.0;
  return m;
}


RealMatrix* ShearMatrix3(float SHxy,float SHxz,float SHyx,float SHyz,float SHzx,float SHzy)
{
  RealMatrix *m;
  m = CreateRealMatrix(4,4);
  m->val[0][0] = 1.0;    m->val[0][1] = SHxy;    m->val[0][2] = SHxz;    m->val[0][3] = 0;
  m->val[1][0] = SHyx;    m->val[1][1] = 1.0;    m->val[1][2] = SHyz;    m->val[1][3] = 0;
  m->val[2][0] = SHzx;    m->val[2][1] = SHzy;    m->val[2][2] = 1.0;    m->val[2][3] = 0;
  m->val[3][0] = 0.0;    m->val[3][1] = 0.0;    m->val[3][2] = 0.0;    m->val[3][3] = 1.0;
  return m;
}




RealMatrix* TransformVoxel(RealMatrix *m, Voxel v)
{
  RealMatrix *vm,*res;
  vm = CreateRealMatrix(1,4);
  vm->val[0][0]=v.x;
  vm->val[1][0]=v.y;
  vm->val[2][0]=v.z;
  vm->val[3][0]=1.0;
  res=MultRealMatrix(m,vm);
  DestroyRealMatrix(&vm);
  return res;
}
