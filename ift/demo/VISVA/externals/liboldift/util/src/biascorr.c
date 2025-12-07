#include "ift.h"


Scene *BiasCorrection(Scene *scn)
{
  Scene  *V=NULL,*R=NULL;
  GQueue *Q=NULL;
  int i,p,q,tmp,n,sz,Imax;
  Voxel u,v;
  AdjRel3 *A=Spheric(1.0);
  
  n     = scn->xsize*scn->ysize*scn->zsize;
  sz    = (scn->xsize * scn->ysize);
  V     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  R     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  Q     = CreateGQueue((Imax=MaximumValue3(scn))+1,n,V->data);
  SetRemovalPolicy(Q,MAXVALUE);

  for (p=0; p < n; p++) {
    V->data[p]=MAX(scn->data[p]-1,0);
    R->data[p]=p;
    InsertGQueue(&Q,p);
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    if (R->data[p]==p) {
      V->data[p]=scn->data[p];
    }
    u.z =  p/sz;
    u.x = (p%sz)%scn->xsize;
    u.y = (p%sz)/scn->xsize;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){	
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (V->data[q] < V->data[p]){
	  tmp = MIN(V->data[p],scn->data[q]);
	  if (tmp > V->data[q]){
	    UpdateGQueue(&Q,q,tmp);
	    R->data[q] = R->data[p];
	  }
	}
      }
    }
  }
 
  DestroyGQueue(&Q);

  for (p=0; p < n; p++) {
    V->data[p] = (int)(scn->data[p]*pow((float)Imax/scn->data[R->data[p]],0.25));
  }
  DestroyScene(&R);

  return(V);
}

int main(int argc, char **argv) {
  Scene *src, *dest;

  if (argc!=2) {
    fprintf(stderr,"biascorr <filename.scn> \n");
    return 1;
  }

  src = ReadScene(argv[1]);
  if (!src) {
    fprintf(stderr,"unable to read %s\n",argv[1]);
    return 2;
  }

  dest = BiasCorrection(src);
  WriteScene(dest,"scene-corr.scn");

  DestroyScene(&dest);
  DestroyScene(&src);
  return 0;
}
