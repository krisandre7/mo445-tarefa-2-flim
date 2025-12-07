
#include "bia_matrix.h"

namespace bia{
  namespace Matrix{

    Matrix *Create(int ncols,int nrows){
      Matrix *mat=NULL;
      float *aux;
      int i;
      
      mat = (Matrix *) calloc(1,sizeof(Matrix));
      if(mat == NULL)
	bia::Error((char *)MSG1,(char *)"Matrix::Create");
      
      aux = (float *)calloc(nrows*ncols, sizeof(float));
      mat->val = (float **) calloc(nrows, sizeof(float *));
      if(mat->val == NULL || aux == NULL)
	bia::Error((char *)MSG1,(char *)"Matrix::Create");
      
      mat->val[0] = aux;
      for(i=1; i<nrows; i++) 
	mat->val[i] = mat->val[i-1] + ncols;
    
      mat->ncols = ncols;
      mat->nrows = nrows;
      return(mat);
    }


    void   Destroy(Matrix **mat){
      Matrix *aux;
      
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


    Matrix *Clone(Matrix *mat){
      Matrix *matc;

      matc = Create(mat->ncols, mat->nrows);
      memcpy(matc->val[0], mat->val[0],
	     mat->ncols*mat->nrows*sizeof(float));
      return(matc);
    }
    

    void     Copy(Matrix *dest, 
		  Matrix *src){
      if(dest->ncols!=src->ncols ||
	 dest->nrows!=src->nrows)
	bia::Error((char *)"Incompatible matrices",
		   (char *)"Matrix::Copy");
      
      memcpy(dest->val[0], src->val[0], 
	     src->ncols*src->nrows*sizeof(float));
    }
    

    void   Print(Matrix *M){
      int x, y;
  
      printf("\n");
      for(y=0; y<M->nrows; y++){
	for(x=0; x<M->ncols; x++)
	  printf("%3.5lf ", M->val[y][x]);
	printf("\n");
      }
      printf("\n");
    }


    void   PrintDimension(Matrix *M){
      printf("\n%d x %d\n", M->nrows, M->ncols);
    }
    

    Matrix *Invert(Matrix *A){
      Matrix *B=NULL;
      Matrix *I=NULL;
      int i,j,k;
      float m;
  
      if(A->ncols!=A->nrows)
	bia::Error((char *)"Matrix dimension error",
		   (char *)"Matrix::Invert");
  
      I = Create(A->ncols, A->nrows);
      B = Clone(A);
      
      for(i=0; i<A->nrows; i++)
	I->val[i][i] = 1.0;

      for(k=0; k<A->nrows; k++){
	m = B->val[k][k];
	//if(m < 0.0000000000001)
	// bia::Error((char *)"Singular matrix",
	//            (char *)"Matrix::Invert"); THIS IS WRONG
	
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
      Destroy(&B);
      return I;
    }


    float        GetTrace(Matrix *M){
      float sum;
      int i;
      
      if(M->ncols!=M->nrows)
	bia::Error((char *)"Matrix dimension error",
		   (char *)"Matrix::GetTrace");
      sum = 0.0;
      for(i=0; i<M->nrows; i++)
	sum += M->val[i][i];
      return sum;
    }
    
    
    Matrix *Mult(Matrix *A, 
		 Matrix *B){
      Matrix *M = NULL;
      int i,j,k;
      
      if(A->ncols!=B->nrows)
	bia::Error((char *)"Matrix dimension error",
		   (char *)"Matrix::Mult");
  
      M = Create(B->ncols, A->nrows);
      for(i=0; i<M->nrows; i++){
	for(j=0; j<M->ncols; j++){
	  M->val[i][j] = 0.0;
	  for (k=0; k<A->ncols; k++)
	    M->val[i][j] += A->val[i][k]*B->val[k][j];
	}
      }
      return(M);
    }
    
    
    Matrix *MultByScalar(Matrix *A, 
			 float k){
      Matrix *M = NULL;
      int i,j;
      
      M = Create(A->ncols, A->nrows);
      for(i=0; i<M->nrows; i++){
	for(j=0; j<M->ncols; j++){
	  M->val[i][j] = k*A->val[i][j];
	}
      }
      return(M);
    }


    Matrix *Sub(Matrix *A, 
		Matrix *B){
      Matrix *M = NULL;
      int i,j;
      
      if((A->ncols!=B->ncols)||(A->nrows!=B->nrows))
	bia::Error((char *)"Matrix dimension error",
		   (char *)"Matrix::Sub");
      M = Create(A->ncols, A->nrows);
      for(i=0; i<M->nrows; i++){
	for(j=0; j<M->ncols; j++){
	  M->val[i][j] = A->val[i][j] - B->val[i][j];
	}
      }
      return(M);
    }


    Matrix *Add(Matrix *A, 
		Matrix *B){
      Matrix *M = NULL;
      int i,j;
      
      if((A->ncols!=B->ncols)||(A->nrows!=B->nrows))
	bia::Error((char *)"Matrix dimension error",
		   (char *)"Matrix::Add");
      
      M = Create(A->ncols, A->nrows);
      for(i=0; i<M->nrows; i++){
	for(j=0; j<M->ncols; j++){
	  M->val[i][j] = A->val[i][j] + B->val[i][j];
	}
      }
      return(M);
    }


    Matrix *Transpose(Matrix *A){
      Matrix *M = NULL;
      int i,j;
      
      M = Create(A->nrows, A->ncols);
      for(i=0; i<M->nrows; i++){
	for(j=0; j<M->ncols; j++){
	  M->val[i][j] = A->val[j][i];
	}
      }
      return(M);
    }


    float ComputeDistanceL2(Matrix *Y, 
			    Matrix *X){
      Matrix *A,*B,*R;
      float d;
  
      A = Sub(X, Y);
      B = Transpose(A);
      R = Mult(A, B);
      d = GetTrace(R);
      d = sqrtf(d);
      Destroy(&A);
      Destroy(&B);
      Destroy(&R);
      return (d);
    }

    
    Matrix *Read(char *filename){
      Matrix *M;
      char msg[512];
      int  ncols,nrows,size,n,p;
      double *daux=NULL;
      FILE *fp;

      fp = fopen(filename,"rb");
      if(fp == NULL){
	sprintf(msg,"Cannot open %s",filename);
	bia::Error((char *)msg,(char *)"Matrix::Read");
      }
      fread(&ncols, sizeof(int), 1, fp);
      fread(&nrows, sizeof(int), 1, fp);
      fread(&size,  sizeof(int), 1, fp);

      M = Create(ncols, nrows);
      n = ncols*nrows;
      if(size==sizeof(float))
	fread(M->val[0], sizeof(float), n, fp);
      else if(size==sizeof(double)){
	daux = AllocDoubleArray(n);
	fread(daux, sizeof(double), n, fp);
	for(p=0; p<n; p++)
	  M->val[0][p] = (float)daux[p];
	free(daux);
      }
      else
	bia::Error((char *)"Bad or corrupted file",
		   (char *)"Matrix::Read");
      fclose(fp);
      return M;
    }


    void    Write(Matrix *M,
		  char *filename){
      char msg[512];
      int n,ncols,nrows,size;
      FILE *fp;
      
      fp = fopen(filename,"wb");
      if(fp == NULL){
	sprintf(msg,"Cannot open %s",filename);
	bia::Error((char *)msg,
		   (char *)"Matrix::Write");
      }
      size  = sizeof(float);
      ncols = M->ncols;
      nrows = M->nrows;
      n = ncols*nrows;
      fwrite(&ncols,    sizeof(int),  1, fp);
      fwrite(&nrows,    sizeof(int),  1, fp);
      fwrite(&size,     sizeof(int),  1, fp);
      fwrite(M->val[0], sizeof(float), n ,fp);
      fclose(fp);
    }


    bool   IsValidEntry(Matrix *M,
			int i, int j){
      if((j >= 0)&&(j < M->ncols)&&
	 (i >= 0)&&(i < M->nrows))
	return(true);
      else
	return(false);
    }


    float   GetMaximumValue(Matrix *M){
      float max;
      int p,n;
      
      n = M->ncols*M->nrows;
      max = M->val[0][0];
      for(p=1; p<n; p++)
	if(M->val[0][p] > max)
	  max = M->val[0][p];
      return(max);
    }
    
    
    float    GetMinimumValue(Matrix *M){
      float min;
      int p,n;
      
      n = M->ncols*M->nrows;
      min = M->val[0][0];
      for(p=1; p<n; p++)
	if(M->val[0][p] < min)
	  min = M->val[0][p];
      return(min);
    }


    void    Fill(Matrix *M, float value){
      int p,n;
      n = M->ncols*M->nrows;
      for(p=0; p<n; p++)
	M->val[0][p] = value;
    }
    

    void    ChangeValue(Matrix *M, 
			float old_value,
			float new_value){
      int p,n;
      
      n = M->ncols*M->nrows;
      for(p=0; p<n; p++)
	if(M->val[0][p] == old_value)
	  M->val[0][p] = new_value;
    }


    Image  *Convert2Image(Matrix *M){
      Image *img;
      float max,min;
      int p,n;
      
      n = M->ncols*M->nrows;
      img = CreateImage(M->ncols, M->nrows);
      max = GetMaximumValue(M);
      min = GetMinimumValue(M);
      for(p=0; p<n; p++)
	img->val[p] = ROUND(255.0*(M->val[0][p]-min)/(max-min));
      
      return img;
    }


    Image      *Threshold(Matrix *M,
			  float lower, 
			  float higher){
      Image *bin=NULL;
      int p,n;
      bin = CreateImage(M->ncols,M->nrows);
      n = M->ncols*M->nrows;
      for (p=0; p < n; p++)
	if ((M->val[0][p] >= lower)&&(M->val[0][p] <= higher))
	  bin->val[p]=1;
      return(bin);
    }


    // options: 0 (x) / 1 (y) / 2 (z)
    Matrix* RotationMatrix3(int axis, 
			    float th){
      Matrix *m;
      m = Create(4,4);
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



    Matrix* TranslationMatrix3(float dx, float dy, float dz){
      Matrix *m;
      m = Create(4,4);
      m->val[0][0] = 1.0;    m->val[0][1] = 0.0;    m->val[0][2] = 0.0;    m->val[0][3] = dx;
      m->val[1][0] = 0.0;    m->val[1][1] = 1.0;    m->val[1][2] = 0.0;    m->val[1][3] = dy;
      m->val[2][0] = 0.0;    m->val[2][1] = 0.0;    m->val[2][2] = 1.0;    m->val[2][3] = dz;
      m->val[3][0] = 0.0;    m->val[3][1] = 0.0;    m->val[3][2] = 0.0;    m->val[3][3] = 1.0;
      return m;
    }


    Matrix* TransformVoxel(Matrix *m, Voxel v){
      Matrix *vm,*res;
      vm = Create(1,4);
      vm->val[0][0]=v.c.x;
      vm->val[1][0]=v.c.y;
      vm->val[2][0]=v.c.z;
      vm->val[3][0]=1.0;
      res=Mult(m,vm);
      Destroy(&vm);
      return res;
    }


  } //end Matrix namespace
} //end bia namespace
