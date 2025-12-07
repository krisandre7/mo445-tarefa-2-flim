#include "floatadjacency3.h"

FloatAdjRel3 *CreateFloatAdjRel3(int n)
{
  FloatAdjRel3 *A=NULL;

  A = (FloatAdjRel3 *) calloc(1,sizeof(FloatAdjRel3));
  if (A != NULL){
    A->dx = AllocFloatArray(n);
    A->dy = AllocFloatArray(n);
    A->dz = AllocFloatArray(n);
    A->n  = n;
  } else {
    Error(MSG1,"CreateFloatAdjRel3");
  }

  return(A);
}

void DestroyFloatAdjRel3(FloatAdjRel3 **A)
{
  FloatAdjRel3 *aux;

  aux = *A;
  if (aux != NULL){
    if (aux->dx != NULL) free(aux->dx);
    if (aux->dy != NULL) free(aux->dy);
    if (aux->dz != NULL) free(aux->dz);
    free(aux);
    *A = NULL;
  }   
}


FloatAdjRel3 *CloneFloatAdjRel3(FloatAdjRel3 *A){
  FloatAdjRel3 *C;
  int i;

  C = CreateFloatAdjRel3(A->n);
  for(i=0; i < A->n; i++){
    C->dx[i] = A->dx[i];
    C->dy[i] = A->dy[i];
    C->dz[i] = A->dz[i];
  }

  return C;
}



FloatAdjRel3 *ChangeOrientationToLPSFloatAdjRel3(FloatAdjRel3 *A,
						 char *ori){
  FloatAdjRel3 *lps=NULL;
  int i;

  lps = CreateFloatAdjRel3(A->n);
  for(i=0; i < A->n; i++){
    if     (ori[0]=='L'){ lps->dx[i] =  A->dx[i]; }
    else if(ori[0]=='R'){ lps->dx[i] = -A->dx[i]; }
    else if(ori[0]=='P'){ lps->dy[i] =  A->dx[i]; }
    else if(ori[0]=='A'){ lps->dy[i] = -A->dx[i]; }
    else if(ori[0]=='S'){ lps->dz[i] =  A->dx[i]; }
    else if(ori[0]=='I'){ lps->dz[i] = -A->dx[i]; }
    else{ Error("Invalid orientation",
		"ChangeOrientationToLPSFloatAdjRel3"); }

    if     (ori[1]=='L'){ lps->dx[i] =  A->dy[i]; }
    else if(ori[1]=='R'){ lps->dx[i] = -A->dy[i]; }
    else if(ori[1]=='P'){ lps->dy[i] =  A->dy[i]; }
    else if(ori[1]=='A'){ lps->dy[i] = -A->dy[i]; }
    else if(ori[1]=='S'){ lps->dz[i] =  A->dy[i]; }
    else if(ori[1]=='I'){ lps->dz[i] = -A->dy[i]; }
    else{ Error("Invalid orientation",
		"ChangeOrientationToLPSFloatAdjRel3"); }

    if     (ori[2]=='L'){ lps->dx[i] =  A->dz[i]; }
    else if(ori[2]=='R'){ lps->dx[i] = -A->dz[i]; }
    else if(ori[2]=='P'){ lps->dy[i] =  A->dz[i]; }
    else if(ori[2]=='A'){ lps->dy[i] = -A->dz[i]; }
    else if(ori[2]=='S'){ lps->dz[i] =  A->dz[i]; }
    else if(ori[2]=='I'){ lps->dz[i] = -A->dz[i]; }
    else{ Error("Invalid orientation",
		"ChangeOrientationToLPSFloatAdjRel3"); }
  }

  return lps;
}

