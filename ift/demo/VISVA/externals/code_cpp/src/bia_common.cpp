
#include "bia_common.h"

namespace bia{

  int *AllocIntArray(int n){
    int *v=NULL;

    v = (int *)_mm_malloc(sizeof(int)*n, sizeof(int)*4);
    if(v==NULL)
      Error((char *)MSG1,(char *)"bia::AllocIntArray");
    memset((void *)v, 0, sizeof(int)*n);
    return(v);
  }

  float *AllocFloatArray(int n){
    float *v=NULL;

    //if(posix_memalign((void **)&v, 16, sizeof(float)*n)!=0){
    //  printf("memalign error\n");
    //  exit(1);
    //}
    v = (float *)_mm_malloc(sizeof(float)*n, sizeof(float)*4);
    if(v==NULL) 
      Error((char *)MSG1,(char *)"bia::AllocFloatArray");
    memset((void *)v, 0, sizeof(float)*n);
    return(v);
  }


  uchar  *AllocUCharArray(int n){
    uchar *v=NULL;
    
    v = (uchar *)_mm_malloc(sizeof(uchar)*n, 16);
    if(v==NULL)
      Error((char *)MSG1,(char *)"bia::AllocUCharArray");
    memset((void *)v, 0, sizeof(uchar)*n);
    return(v);
  }

  ushort *AllocUShortArray(int n){
    ushort *v=NULL;
    
    v = (ushort *)_mm_malloc(sizeof(ushort)*n, 16);
    if(v==NULL)
      Error((char *)MSG1,(char *)"bia::AllocUShortArray");
    memset((void *)v, 0, sizeof(ushort)*n);
    return(v);
  }

  void    FreeIntArray(int **a){
    _mm_free(*a);
    *a = NULL;
  }

  void    FreeFloatArray(float **a){
    _mm_free(*a);
    *a = NULL;
  }

  void    FreeUCharArray(uchar **a){
    _mm_free(*a);
    *a = NULL;
  }

  void    FreeUShortArray(ushort **a){
    _mm_free(*a);
    *a = NULL;
  }

  void Error(char *msg,char *func){ 
    fprintf(stderr,"Error: %s in %s\n",msg,func);
    exit(-1);
  }
  
  void Warning(char *msg,char *func){ 
    fprintf(stdout,"Warning: %s in %s\n",msg,func);
  }


} //end bia namespace

