#include "context.h"
#include "common.h"

Context *CreateContext(Scene *scn)
{
  return(NewContext(scn->xsize,scn->ysize,scn->zsize));
}

Context *NewContext(int xsize, int ysize, int zsize)
{
  Context *cxt=NULL;
  int i,diag;
  AdjRel *A = Circular(1.5);

  cxt = (Context *) calloc(1,sizeof(Context));
  if (cxt == NULL) 
    Error(MSG1,"CreateContext");
  cxt->xsize = xsize;
  cxt->ysize = ysize;
  cxt->zsize = zsize;

  diag = (int)sqrt((float)(cxt->xsize*cxt->xsize+cxt->ysize*cxt->ysize+cxt->zsize*cxt->zsize));
  cxt->width = cxt->height = cxt->depth = diag;

  /*  
  if((xsize <= ysize) && (xsize <= zsize))
    usize = ysize + zsize;
  else 
    if((ysize <= xsize) && (ysize <= zsize))
      usize = xsize + zsize;
    else
      usize = ysize + xsize;
  
  vsize = usize;
  */
  cxt->zbuff = CreateZBuffer(diag,diag);
  cxt->Su = cxt->Sv = cxt->ISu = cxt->ISv = NULL;
  cxt->oct = 1;
  cxt->fr = CreateFrame(cxt->xsize,cxt->ysize,cxt->zsize);
  SetClip(cxt,0,cxt->xsize-1,0,cxt->ysize-1,0,cxt->zsize-1);
  SetAngles(cxt,0.0,0.0);
  SetPhong(cxt,0,0.7,0.2,0.1,2);
  SetObjectColor(cxt,0,0.0,0.0,0.0);
  for (i=1; i < 256; i++) {
    SetPhong(cxt,(uchar)i,0.7,0.2,0.1,2);
    SetObjectColor(cxt,(uchar)i,1.0,1.0,1.0);
    cxt->obj[i].xoffset =0;
    cxt->obj[i].yoffset =0;
    cxt->obj[i].zoffset =0;
    cxt->obj[i].opac = 1.0;
  }
  cxt->footprint = GaussianKernel(A,1.0);
  DestroyAdjRel(&A);
  cxt->xaxis.x = 1;  cxt->yaxis.x = 0;  cxt->zaxis.x = 0;
  cxt->xaxis.y = 0;  cxt->yaxis.y = 1;  cxt->zaxis.y = 0;
  cxt->xaxis.z = 0;  cxt->yaxis.z = 0;  cxt->zaxis.z = 1;
  cxt->viewer.x = 0; cxt->view_up.x = 0;
  cxt->viewer.y = 0; cxt->view_up.y =-1;
  cxt->viewer.z =-1; cxt->view_up.z = 0;

  cxt->R[0][0] = 1.0;
  cxt->R[0][1] = 0.0;
  cxt->R[0][2] = 0.0;
  cxt->R[0][3] = 0.0;

  cxt->R[1][0] = 0.0;
  cxt->R[1][1] = 1.0;
  cxt->R[1][2] = 0.0;
  cxt->R[1][3] = 0.0;

  cxt->R[2][0] = 0.0;
  cxt->R[2][1] = 0.0;
  cxt->R[2][2] = 1.0;
  cxt->R[2][3] = 0.0;

  cxt->R[3][0] = 0.0;
  cxt->R[3][1] = 0.0;
  cxt->R[3][2] = 0.0;
  cxt->R[3][3] = 1.0;

  return(cxt);
}

void DestroyContext(Context **cxt)
{
  Context *aux;
  
  aux = *cxt;
  if (aux != NULL) {
    DestroyZBuffer(&aux->zbuff);
    if (aux->Su != NULL) free(aux->Su);
    if (aux->Sv != NULL) free(aux->Sv);
    if (aux->ISu != NULL) free(aux->ISu);
    if (aux->ISv != NULL) free(aux->ISv);
    if (aux->fopac != NULL) free(aux->fopac);
    if (aux->fr  != NULL) DestroyFrame(&aux->fr);
    DestroyKernel(&(aux->footprint));
    DestroyAdjPxl(&(aux->fprint));
    free(aux);
    *cxt = NULL;
  }
}


void SetAxisIndices(Context *cxt) 
{

  switch (cxt->oct) {
  case 1:
    cxt->xi = cxt->xmin; cxt->xf = cxt->xmax+1; cxt->dx = 1;
    cxt->yi = cxt->ymin; cxt->yf = cxt->ymax+1; cxt->dy = 1;
    cxt->zi = cxt->zmin; cxt->zf = cxt->zmax+1; cxt->dz = 1;
    break;
  case 2:
    cxt->xi = cxt->xmax; cxt->xf = cxt->xmin-1; cxt->dx = -1;
    cxt->yi = cxt->ymin; cxt->yf = cxt->ymax+1; cxt->dy = 1;
    cxt->zi = cxt->zmin; cxt->zf = cxt->zmax+1; cxt->dz = 1;
    break;
  case 3:
    cxt->xi = cxt->xmin; cxt->xf = cxt->xmax+1; cxt->dx = 1;
    cxt->yi = cxt->ymax; cxt->yf = cxt->ymin-1; cxt->dy = -1;
    cxt->zi = cxt->zmin; cxt->zf = cxt->zmax+1; cxt->dz = 1;
    break;
  case 4:
    cxt->xi = cxt->xmax; cxt->xf = cxt->xmin-1; cxt->dx = -1;
    cxt->yi = cxt->ymax; cxt->yf = cxt->ymin-1; cxt->dy = -1;
    cxt->zi = cxt->zmin; cxt->zf = cxt->zmax+1; cxt->dz = 1;
    break;
  case 5:
    cxt->xi = cxt->xmin; cxt->xf = cxt->xmax+1; cxt->dx = 1;
    cxt->yi = cxt->ymin; cxt->yf = cxt->ymax+1; cxt->dy = 1;
    cxt->zi = cxt->zmax; cxt->zf = cxt->zmin-1; cxt->dz = -1;
    break;
  case 6:
    cxt->xi = cxt->xmax; cxt->xf = cxt->xmin-1; cxt->dx = -1;
    cxt->yi = cxt->ymin; cxt->yf = cxt->ymax+1; cxt->dy = 1;
    cxt->zi = cxt->zmax; cxt->zf = cxt->zmin-1; cxt->dz = -1;
    break;
  case 7:
    cxt->xi = cxt->xmin; cxt->xf = cxt->xmax+1; cxt->dx = 1;
    cxt->yi = cxt->ymax; cxt->yf = cxt->ymin-1; cxt->dy = -1;
    cxt->zi = cxt->zmax; cxt->zf = cxt->zmin-1; cxt->dz = -1;
    break;
  case 8:
    cxt->xi = cxt->xmax; cxt->xf = cxt->xmin-1; cxt->dx = -1;
    cxt->yi = cxt->ymax; cxt->yf = cxt->ymin-1; cxt->dy = -1;
    cxt->zi = cxt->zmax; cxt->zf = cxt->zmin-1; cxt->dz = -1;
    break;
  }

}

void AddAngles (Context *cxt, float thx, float thy)
{
  float M[4][4],S[4][4],IS[4][4],P[4][4],IP[4][4];
  float Rx[4][4],Ry[4][4];
  int k;
  double trash;

  cxt->thx += thx;
  cxt->thy += thy;

  /* set the value of thx and thy between 0 and 360 in the context*/
  cxt->thx = 360.*modf(cxt->thx/360.,&trash);
  if (cxt->thx < 0.0)
    cxt->thx += 360.0;
  cxt->thy = 360.*modf(cxt->thy/360.,&trash);
  if (cxt->thy < 0.0)
    cxt->thy += 360.0;

  /* set the value of thx and thy between 0 and 360 for computation*/
  thx = 360.*modf(thx/360.,&trash);
  if (thx < 0.0)
    thx += 360.0;
  thy = 360.*modf(thy/360.,&trash);
  if (thy < 0.0)
    thy += 360.0;
  
  thx      = thx*PI/180.;
  thy      = thy*PI/180.;

  /* Compute Rotation Matrix : R = Tuv*M1*Txyz, where M1 is the
     rotation around the origin. */

  Rx[0][0] = 1.0;
  Rx[0][1] = 0.0;
  Rx[0][2] = 0.0;
  Rx[0][3] = 0.0;
  Rx[1][0] = 0.0;
  Rx[1][1] = cos(thx);
  Rx[1][2] = -sin(thx);
  Rx[1][3] = 0.0;
  Rx[2][0] = 0.0;
  Rx[2][1] = sin(thx);
  Rx[2][2] = cos(thx);
  Rx[2][3] = 0.0;
  Rx[3][0] = 0.0;
  Rx[3][1] = 0.0;
  Rx[3][2] = 0.0;
  Rx[3][3] = 1.0;

  Ry[0][0] = cos(thy);
  Ry[0][1] = 0.0;
  Ry[0][2] = sin(thy);
  Ry[0][3] = 0.0;
  Ry[1][0] = 0.0;
  Ry[1][1] = 1.0;
  Ry[1][2] = 0.0;
  Ry[1][3] = 0.0; 
  Ry[2][0] = -sin(thy);
  Ry[2][1] = 0.0;
  Ry[2][2] = cos(thy);
  Ry[2][3] = 0.0;
  Ry[3][0] = 0.0;
  Ry[3][1] = 0.0;
  Ry[3][2] = 0.0;
  Ry[3][3] = 1.0;

  MultMatrices(Rx,cxt->R,M);
  MultMatrices(Ry,M,cxt->R);
  TransMatrix(cxt->R,cxt->IR);

  cxt->viewer.x = -cxt->IR[0][2];
  cxt->viewer.y = -cxt->IR[1][2];
  cxt->viewer.z = -cxt->IR[2][2];

  /* Compute octant and axis indices */

  cxt->oct = Octant(cxt->viewer);
  SetAxisIndices(cxt);
       
 /* Compute Shear Factors Si, Sj, Ti, Tj and the look-up tables for
     shear and inverted shear transformations */

  cxt->PAxis = PAxis(cxt->viewer);
  switch (cxt->PAxis) {
  case 'x':
    cxt->Si =  -cxt->viewer.y / cxt->viewer.x;
    cxt->Sj =  -cxt->viewer.z / cxt->viewer.x;
    cxt->isize = cxt->ysize + abs((int)(cxt->Si * cxt->xsize));
    cxt->jsize = cxt->zsize + abs((int)(cxt->Sj * cxt->xsize));
    cxt->ksize = cxt->xsize;
    cxt->ki = cxt->xi;
    cxt->kf = cxt->xf;
    cxt->dk = cxt->dx;
    break;
  case 'y':
    cxt->Si = -cxt->viewer.z / cxt->viewer.y;
    cxt->Sj = -cxt->viewer.x / cxt->viewer.y;
    cxt->isize = cxt->zsize + abs((int)(cxt->Si * cxt->ysize));
    cxt->jsize = cxt->xsize + abs((int)(cxt->Sj * cxt->ysize));
    cxt->ksize = cxt->ysize;
    cxt->ki = cxt->yi;
    cxt->kf = cxt->yf;
    cxt->dk = cxt->dy;
    break;
  case 'z':
    cxt->Si =  -cxt->viewer.x / cxt->viewer.z;
    cxt->Sj =  -cxt->viewer.y / cxt->viewer.z;
    cxt->isize = cxt->xsize + abs((int)(cxt->Si * cxt->zsize));
    cxt->jsize = cxt->ysize + abs((int)(cxt->Sj * cxt->zsize));
    cxt->ksize = cxt->zsize;
    cxt->ki = cxt->zi;
    cxt->kf = cxt->zf;
    cxt->dk = cxt->dz;
  }

  if (cxt->Su != NULL) free(cxt->Su); 
  cxt->Su  = AllocFloatArray(cxt->ksize);
  if (cxt->Sv != NULL) free(cxt->Sv); 
  cxt->Sv  = AllocFloatArray(cxt->ksize);
  if (cxt->ISu != NULL) free(cxt->ISu); 
  cxt->ISu = AllocFloatArray(cxt->ksize);
  if (cxt->ISv != NULL) free(cxt->ISv); 
  cxt->ISv = AllocFloatArray(cxt->ksize);
  
  cxt->Ti = -cxt->Si*cxt->ksize/2.; 
  cxt->Tj = -cxt->Sj*cxt->ksize/2.;

  for (k = cxt->ki; k != cxt->kf; k += cxt->dk) {
    cxt->Su[k] = k*cxt->Si + cxt->Ti;
    cxt->Sv[k] = k*cxt->Sj + cxt->Tj;
    cxt->ISu[k] = -k*cxt->Si - cxt->Ti;
    cxt->ISv[k] = -k*cxt->Sj - cxt->Tj; 
  }

  /* Compute Shear and Inverted Shear Matrices */

  S[0][0] = 1.0; S[0][1] = 0.0; S[0][2] = cxt->Si;  S[0][3] = cxt->Ti;
  S[1][0] = 0.0; S[1][1] = 1.0; S[1][2] = cxt->Sj;  S[1][3] = cxt->Tj; 
  S[2][0] = 0.0; S[2][1] = 0.0; S[2][2] = 1.0; S[2][3] = 0.0;
  S[3][0] = 0.0; S[3][1] = 0.0; S[3][2] = 0.0; S[3][3] = 1.0;
  
  IS[0][0] = 1.0; IS[0][1] = 0.0; IS[0][2] = -cxt->Si; IS[0][3] = -cxt->Ti;
  IS[1][0] = 0.0; IS[1][1] = 1.0; IS[1][2] = -cxt->Sj; IS[1][3] = -cxt->Tj;
  IS[2][0] = 0.0; IS[2][1] = 0.0; IS[2][2] = 1.0; IS[2][3] = 0.0;
  IS[3][0] = 0.0; IS[3][1] = 0.0; IS[3][2] = 0.0; IS[3][3] = 1.0;

  /* Compute Permutation and Inverted Permutation Matrices. IP =
     Transposed P */
  
  switch (cxt->PAxis) {
  case 'x':
    P[0][0] = 0.0; P[0][1] = 1.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 0.0; P[1][2] = 1.0; P[1][3] = 0.0;
    P[2][0] = 1.0; P[2][1] = 0.0; P[2][2] = 0.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 0.0; IP[0][1] = 0.0; IP[0][2] = 1.0; IP[0][3] = 0.0;
    IP[1][0] = 1.0; IP[1][1] = 0.0; IP[1][2] = 0.0; IP[1][3] = 0.0;
    IP[2][0] = 0.0; IP[2][1] = 1.0; IP[2][2] = 0.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  case 'y':
    P[0][0] = 0.0; P[0][1] = 0.0; P[0][2] = 1.0; P[0][3] = 0.0;
    P[1][0] = 1.0; P[1][1] = 0.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 1.0; P[2][2] = 0.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 0.0; IP[0][1] = 1.0; IP[0][2] = 0.0; IP[0][3] = 0.0;
    IP[1][0] = 0.0; IP[1][1] = 0.0; IP[1][2] = 1.0; IP[1][3] = 0.0;
    IP[2][0] = 1.0; IP[2][1] = 0.0; IP[2][2] = 0.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  case 'z':
    P[0][0] = 1.0; P[0][1] = 0.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 1.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 0.0; P[2][2] = 1.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 1.0; IP[0][1] = 0.0; IP[0][2] = 0.0; IP[0][3] = 0.0;
    IP[1][0] = 0.0; IP[1][1] = 1.0; IP[1][2] = 0.0; IP[1][3] = 0.0;
    IP[2][0] = 0.0; IP[2][1] = 0.0; IP[2][2] = 1.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  }

 /* Compute Warp and Inverted Warp Matrices. R = W*S*P => W = R*IP*IS
    and IW = S*P*IR */
  
  MultMatrices(IP,IS,M);
  MultMatrices(cxt->R,M,cxt->W);
  MultMatrices(P,cxt->IR,M);
  MultMatrices(S,M,cxt->IW);
}


void SetAngles(Context *cxt, float thx, float thy)
{  
  float M[4][4],S[4][4],IS[4][4],P[4][4],IP[4][4];
  float Rx[4][4],Ry[4][4];
  int k;
  double trash;

  cxt->thx = 360.*modf(thx/360.,&trash);
  if (cxt->thx < 0.0)
    cxt->thx += 360.0;
  cxt->thy = 360.*modf(thy/360.,&trash);
  if (cxt->thy < 0.0)
    cxt->thy += 360.0;
  
  thx      = cxt->thx*PI/180.;
  thy      = cxt->thy*PI/180.;

  /* Compute Rotation Matrix : R = Tuv*M1*Txyz, where M1 is the
     rotation around the origin. */

  Rx[0][0] = 1.0;
  Rx[0][1] = 0.0;
  Rx[0][2] = 0.0;
  Rx[0][3] = 0.0;
  Rx[1][0] = 0.0;
  Rx[1][1] = cos(thx);
  Rx[1][2] = -sin(thx);
  Rx[1][3] = 0.0;
  Rx[2][0] = 0.0;
  Rx[2][1] = sin(thx);
  Rx[2][2] = cos(thx);
  Rx[2][3] = 0.0;
  Rx[3][0] = 0.0;
  Rx[3][1] = 0.0;
  Rx[3][2] = 0.0;
  Rx[3][3] = 1.0;

  Ry[0][0] = cos(thy);
  Ry[0][1] = 0.0;
  Ry[0][2] = sin(thy);
  Ry[0][3] = 0.0;
  Ry[1][0] = 0.0;
  Ry[1][1] = 1.0;
  Ry[1][2] = 0.0;
  Ry[1][3] = 0.0; 
  Ry[2][0] = -sin(thy);
  Ry[2][1] = 0.0;
  Ry[2][2] = cos(thy);
  Ry[2][3] = 0.0;
  Ry[3][0] = 0.0;
  Ry[3][1] = 0.0;
  Ry[3][2] = 0.0;
  Ry[3][3] = 1.0;

  MultMatrices(Ry,Rx,cxt->R);
  TransMatrix(cxt->R,cxt->IR);

  cxt->viewer.x = -cxt->IR[0][2];
  cxt->viewer.y = -cxt->IR[1][2];
  cxt->viewer.z = -cxt->IR[2][2];

  /* Compute octant and axis indices */

  cxt->oct = Octant(cxt->viewer);
  SetAxisIndices(cxt);
       
 /* Compute Shear Factors Si, Sj, Ti, Tj and the look-up tables for
     shear and inverted shear transformations */

  cxt->PAxis = PAxis(cxt->viewer);
  switch (cxt->PAxis) {
  case 'x':
    cxt->Si =  -cxt->viewer.y / cxt->viewer.x;
    cxt->Sj =  -cxt->viewer.z / cxt->viewer.x;
    cxt->isize = cxt->ysize + abs((int)(cxt->Si * cxt->xsize));
    cxt->jsize = cxt->zsize + abs((int)(cxt->Sj * cxt->xsize));
    cxt->ksize = cxt->xsize;
    cxt->ki = cxt->xi;
    cxt->kf = cxt->xf;
    cxt->dk = cxt->dx;
    break;
  case 'y':
    cxt->Si = -cxt->viewer.z / cxt->viewer.y;
    cxt->Sj = -cxt->viewer.x / cxt->viewer.y;
    cxt->isize = cxt->zsize + abs((int)(cxt->Si * cxt->ysize));
    cxt->jsize = cxt->xsize + abs((int)(cxt->Sj * cxt->ysize));
    cxt->ksize = cxt->ysize;
    cxt->ki = cxt->yi;
    cxt->kf = cxt->yf;
    cxt->dk = cxt->dy;
    break;
  case 'z':
    cxt->Si =  -cxt->viewer.x / cxt->viewer.z;
    cxt->Sj =  -cxt->viewer.y / cxt->viewer.z;
    cxt->isize = cxt->xsize + abs((int)(cxt->Si * cxt->zsize));
    cxt->jsize = cxt->ysize + abs((int)(cxt->Sj * cxt->zsize));
    cxt->ksize = cxt->zsize;
    cxt->ki = cxt->zi;
    cxt->kf = cxt->zf;
    cxt->dk = cxt->dz;
  }

  if (cxt->Su != NULL) free(cxt->Su); 
  cxt->Su  = AllocFloatArray(cxt->ksize);
  if (cxt->Sv != NULL) free(cxt->Sv); 
  cxt->Sv  = AllocFloatArray(cxt->ksize);
  if (cxt->ISu != NULL) free(cxt->ISu); 
  cxt->ISu = AllocFloatArray(cxt->ksize);
  if (cxt->ISv != NULL) free(cxt->ISv); 
  cxt->ISv = AllocFloatArray(cxt->ksize);
  
  cxt->Ti = -cxt->Si*cxt->ksize/2.; 
  cxt->Tj = -cxt->Sj*cxt->ksize/2.;

  for (k = cxt->ki; k != cxt->kf; k += cxt->dk) {
    cxt->Su[k] = k*cxt->Si + cxt->Ti;
    cxt->Sv[k] = k*cxt->Sj + cxt->Tj;
    cxt->ISu[k] = -k*cxt->Si - cxt->Ti;
    cxt->ISv[k] = -k*cxt->Sj - cxt->Tj; 
  }

  /* Compute Shear and Inverted Shear Matrices */

  S[0][0] = 1.0; S[0][1] = 0.0; S[0][2] = cxt->Si;  S[0][3] = cxt->Ti;
  S[1][0] = 0.0; S[1][1] = 1.0; S[1][2] = cxt->Sj;  S[1][3] = cxt->Tj; 
  S[2][0] = 0.0; S[2][1] = 0.0; S[2][2] = 1.0; S[2][3] = 0.0;
  S[3][0] = 0.0; S[3][1] = 0.0; S[3][2] = 0.0; S[3][3] = 1.0;
  
  IS[0][0] = 1.0; IS[0][1] = 0.0; IS[0][2] = -cxt->Si; IS[0][3] = -cxt->Ti;
  IS[1][0] = 0.0; IS[1][1] = 1.0; IS[1][2] = -cxt->Sj; IS[1][3] = -cxt->Tj;
  IS[2][0] = 0.0; IS[2][1] = 0.0; IS[2][2] = 1.0; IS[2][3] = 0.0;
  IS[3][0] = 0.0; IS[3][1] = 0.0; IS[3][2] = 0.0; IS[3][3] = 1.0;

  /* Compute Permutation and Inverted Permutation Matrices. IP =
     Transposed P */
  
  switch (cxt->PAxis) {
  case 'x':
    P[0][0] = 0.0; P[0][1] = 1.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 0.0; P[1][2] = 1.0; P[1][3] = 0.0;
    P[2][0] = 1.0; P[2][1] = 0.0; P[2][2] = 0.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 0.0; IP[0][1] = 0.0; IP[0][2] = 1.0; IP[0][3] = 0.0;
    IP[1][0] = 1.0; IP[1][1] = 0.0; IP[1][2] = 0.0; IP[1][3] = 0.0;
    IP[2][0] = 0.0; IP[2][1] = 1.0; IP[2][2] = 0.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  case 'y':
    P[0][0] = 0.0; P[0][1] = 0.0; P[0][2] = 1.0; P[0][3] = 0.0;
    P[1][0] = 1.0; P[1][1] = 0.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 1.0; P[2][2] = 0.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 0.0; IP[0][1] = 1.0; IP[0][2] = 0.0; IP[0][3] = 0.0;
    IP[1][0] = 0.0; IP[1][1] = 0.0; IP[1][2] = 1.0; IP[1][3] = 0.0;
    IP[2][0] = 1.0; IP[2][1] = 0.0; IP[2][2] = 0.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  case 'z':
    P[0][0] = 1.0; P[0][1] = 0.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 1.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 0.0; P[2][2] = 1.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 1.0; IP[0][1] = 0.0; IP[0][2] = 0.0; IP[0][3] = 0.0;
    IP[1][0] = 0.0; IP[1][1] = 1.0; IP[1][2] = 0.0; IP[1][3] = 0.0;
    IP[2][0] = 0.0; IP[2][1] = 0.0; IP[2][2] = 1.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  }

 /* Compute Warp and Inverted Warp Matrices. R = W*S*P => W = R*IP*IS
    and IW = S*P*IR */
  
  MultMatrices(IP,IS,M);
  MultMatrices(cxt->R,M,cxt->W);
  MultMatrices(P,cxt->IR,M);
  MultMatrices(S,M,cxt->IW);
}


void SetPhong(Context *cxt, uchar oindex, float diff, float spec, float amb, int ns)
{
  cxt->obj[oindex].diff = diff;
  cxt->obj[oindex].spec = spec;
  cxt->obj[oindex].amb  = amb;
  cxt->obj[oindex].ns   = ns;
}

void SetObjectColor(Context *cxt, uchar oindex, float red, \
		    float green, float blue)
{
  cxt->obj[oindex].Y   = 0.257*red+0.504*green+0.098*blue+16.0/255.;
  cxt->obj[oindex].Cb  = -0.148*red-0.291*green+0.439*blue+128.0/255.;
  cxt->obj[oindex].Cr  = 0.439*red-0.368*green-0.071*blue+128.0/255.;
}

void SetObjectOpacity(Context *cxt, uchar oindex, float opac)
{
  cxt->obj[oindex].opac   = opac;
}

void SetClip(Context *cxt, int xmin, int xmax, int ymin, int ymax, \
	     int zmin, int zmax)
{

  if (xmin < 0) 
    xmin = 0;
  else
    if (xmin >= cxt->xsize) 
      xmin = cxt->xsize-1;
  if (ymin < 0) 
    ymin = 0;
  else
    if (ymin >= cxt->ysize) 
      ymin = cxt->ysize-1;
  if (zmin < 0) 
    zmin = 0;
  else
    if (zmin >= cxt->zsize) 
      zmin = cxt->zsize-1;

  if (xmax < 0) 
    xmax = 0;
  else
    if (xmax >= cxt->xsize) 
      xmax = cxt->xsize-1;
  if (ymax < 0) 
    ymax = 0;
  else
    if (ymax >= cxt->ysize) 
      ymax = cxt->ysize-1;
  if (zmax < 0) 
    zmax = 0;
  else
    if (zmax >= cxt->zsize) 
      zmax = cxt->zsize-1;

  if (xmin > xmax) 
    xmax=xmin;
  if (ymin > ymax) 
    ymax=ymin;
  if (zmin > zmax) 
    zmax=zmin;

  cxt->xmin = xmin;
  cxt->xmax = xmax;
  cxt->ymin = ymin;
  cxt->ymax = ymax;
  cxt->zmin = zmin;
  cxt->zmax = zmax;

  SetAxisIndices(cxt);  
  SetFrameVert(cxt->fr,xmin,xmax,ymin,ymax,zmin,zmax);
}

ZBuffer *CreateZBuffer(int usize, int vsize)
{
  ZBuffer *zbuff=NULL;
  int i;

  zbuff = (ZBuffer *) calloc(1,sizeof(ZBuffer));
  if (zbuff == NULL)
    Error(MSG1,"CreateZBuffer");

  zbuff->dist   = AllocFloatArray(usize*vsize);
  zbuff->object = AllocUCharArray(usize*vsize);
  zbuff->voxel  = AllocIntArray(usize*vsize);
  zbuff->usize  = usize;
  zbuff->vsize  = vsize;
  zbuff->tbv    = AllocIntArray(vsize);
  zbuff->tbv[0] = 0;
  for (i=1; i < vsize; i++) 
    zbuff->tbv[i] = zbuff->tbv[i-1] + usize;
  
  InitZBuffer(zbuff);

  return(zbuff);
}

void DestroyZBuffer(ZBuffer **zbuff)
{
  ZBuffer *aux;

  aux = *zbuff;

  if (aux != NULL) {
    if (aux->dist   != NULL) free(aux->dist);
    if (aux->object != NULL) free(aux->object);
    if (aux->voxel  != NULL) free(aux->voxel);
    if (aux->tbv    != NULL) free(aux->tbv);
    free(aux);
    *zbuff = NULL;
  }
}

void  InitZBuffer(ZBuffer *zbuff)
{
  int i,u,v;
  
  for(v=0; v < zbuff->vsize; v++) 
    for(u=0; u < zbuff->usize; u++) {
      i = u + zbuff->tbv[v];
      zbuff->dist[i]   = (float)zbuff->usize;
      zbuff->object[i] = 0;
      zbuff->voxel[i]  = NIL;
    }
}

void  InitZBufferObject(ZBuffer *zbuff, uchar obj)
{
  int i,u,v;
  
  for(v=0; v < zbuff->vsize; v++) 
    for(u=0; u < zbuff->usize; u++) {
      i = u + zbuff->tbv[v];
      zbuff->object[i] = obj;
    }
}

void  InitZBufferVoxel(ZBuffer *zbuff)
{
  int i,u,v;
  
  for(v=0; v < zbuff->vsize; v++) 
    for(u=0; u < zbuff->usize; u++) {
      i = u + zbuff->tbv[v];
      zbuff->voxel[i] = NIL;
    }
}

FBuffer *CreateFBuffer(int usize, int vsize)
{
  FBuffer *fbuff=NULL;
  int i;

  fbuff = (FBuffer *) calloc(1,sizeof(FBuffer));
  if (fbuff == NULL)
    Error(MSG1,"CreateFBuffer");

  fbuff->val    = AllocFloatArray(usize*vsize);
  fbuff->usize  = usize;
  fbuff->vsize  = vsize;
  fbuff->tbv    = AllocIntArray(vsize);
  fbuff->tbv[0] = 0;
  for (i=1; i < vsize; i++) 
    fbuff->tbv[i] = fbuff->tbv[i-1] + usize;
  
  return(fbuff);
}

void DestroyFBuffer(FBuffer **buff)
{
  FBuffer *aux = *buff;

  if (aux != NULL) {
    if (aux->val    != NULL) free(aux->val);
    if (aux->tbv    != NULL) free(aux->tbv);
    free(aux);
    *buff = NULL;
  }
}

void  InitFBuffer(FBuffer *fbuff, float f)
{
  int i,u,v;
  
  for(v=0; v < fbuff->vsize; v++) 
    for(u=0; u < fbuff->usize; u++) {
      i = u + fbuff->tbv[v];
      fbuff->val[i]   = f;
    }
}

float GetTilt (Context *cxt) {
  return (cxt->thx);
}
float GetSpin (Context *cxt) {
  return (cxt->thy);
}

int GetZBufferVoxel (ZBuffer *zbuff, int i) {
  return (zbuff->voxel[i]);
}

float GetZBufferDist (ZBuffer *zbuff, int i) {
  return (zbuff->dist[i]);
}

uchar GetZBufferObject (ZBuffer *zbuff, int i) {
  return (zbuff->object[i]);
}

void  SetZBufferDist(ZBuffer *zbuff,FBuffer *dist)
{
  int i,u,v;
  
  for(v=0; v < zbuff->vsize; v++) 
    for(u=0; u < zbuff->usize; u++) {
      i = u + zbuff->tbv[v];
      zbuff->dist[i] = dist->val[i];
    }
}

void AddAnglesCut (Context *cxt, float thx, float thy) {

  float M[4][4],S[4][4],IS[4][4],P[4][4],IP[4][4];
  float Rx[4][4],Ry[4][4];
  int k;
  double trash;

  cxt->thx += thx;
  cxt->thy += thy;

  /* set the value of thx and thy between 0 and 360 in the context*/
  cxt->thx = 360.*modf(cxt->thx/360.,&trash);
  if (cxt->thx < 0.0)
    cxt->thx += 360.0;
  cxt->thy = 360.*modf(cxt->thy/360.,&trash);
  if (cxt->thy < 0.0)
    cxt->thy += 360.0;

  /* set the value of thx and thy between 0 and 360 for computation*/
  thx = 360.*modf(thx/360.,&trash);
  if (thx < 0.0)
    thx += 360.0;
  thy = 360.*modf(thy/360.,&trash);
  if (thy < 0.0)
    thy += 360.0;
  
  thx      = thx*PI/180.;
  thy      = thy*PI/180.;

  /* Compute Rotation Matrix : R = Tuv*M1*Txyz, where M1 is the
     rotation around the origin. */

  Rx[0][0] = 1.0;
  Rx[0][1] = 0.0;
  Rx[0][2] = 0.0;
  Rx[0][3] = 0.0;
  Rx[1][0] = 0.0;
  Rx[1][1] = cos(thx);
  Rx[1][2] = -sin(thx);
  Rx[1][3] = 0.0;
  Rx[2][0] = 0.0;
  Rx[2][1] = sin(thx);
  Rx[2][2] = cos(thx);
  Rx[2][3] = 0.0;
  Rx[3][0] = 0.0;
  Rx[3][1] = 0.0;
  Rx[3][2] = 0.0;
  Rx[3][3] = 1.0;

  Ry[0][0] = cos(thy);
  Ry[0][1] = 0.0;
  Ry[0][2] = sin(thy);
  Ry[0][3] = 0.0;
  Ry[1][0] = 0.0;
  Ry[1][1] = 1.0;
  Ry[1][2] = 0.0;
  Ry[1][3] = 0.0; 
  Ry[2][0] = -sin(thy);
  Ry[2][1] = 0.0;
  Ry[2][2] = cos(thy);
  Ry[2][3] = 0.0;
  Ry[3][0] = 0.0;
  Ry[3][1] = 0.0;
  Ry[3][2] = 0.0;
  Ry[3][3] = 1.0;

  MultMatrices(Rx,cxt->R,M);
  MultMatrices(Ry,M,cxt->R);
  TransMatrix(cxt->R,cxt->IR);

  cxt->viewer.x = -cxt->IR[0][2];
  cxt->viewer.y = -cxt->IR[1][2];
  cxt->viewer.z = -cxt->IR[2][2];

  /* Compute octant and axis indices */

  cxt->oct = Octant(cxt->viewer);
  SetAxisIndices(cxt);
       
 /* Compute Shear Factors Si, Sj, Ti, Tj and the look-up tables for
     shear and inverted shear transformations */

  cxt->PAxis = PAxis(cxt->viewer);
  switch (cxt->PAxis) {
  case 'x':
    cxt->Si =  -cxt->viewer.y / cxt->viewer.x;
    cxt->Sj =  -cxt->viewer.z / cxt->viewer.x;
    cxt->isize = cxt->ysize + abs((int)(cxt->Si * cxt->xsize));
    cxt->jsize = cxt->zsize + abs((int)(cxt->Sj * cxt->xsize));
    cxt->ksize = cxt->xsize;
    cxt->ki = cxt->xi;
    cxt->kf = cxt->xf;
    cxt->dk = cxt->dx;
    break;
  case 'y':
    cxt->Si = -cxt->viewer.z / cxt->viewer.y;
    cxt->Sj = -cxt->viewer.x / cxt->viewer.y;
    cxt->isize = cxt->zsize + abs((int)(cxt->Si * cxt->ysize));
    cxt->jsize = cxt->xsize + abs((int)(cxt->Sj * cxt->ysize));
    cxt->ksize = cxt->ysize;
    cxt->ki = cxt->yi;
    cxt->kf = cxt->yf;
    cxt->dk = cxt->dy;
    break;
  case 'z':
    cxt->Si =  -cxt->viewer.x / cxt->viewer.z;
    cxt->Sj =  -cxt->viewer.y / cxt->viewer.z;
    cxt->isize = cxt->xsize + abs((int)(cxt->Si * cxt->zsize));
    cxt->jsize = cxt->ysize + abs((int)(cxt->Sj * cxt->zsize));
    cxt->ksize = cxt->zsize;
    cxt->ki = cxt->zi;
    cxt->kf = cxt->zf;
    cxt->dk = cxt->dz;
  }

  if (cxt->Su != NULL) free(cxt->Su); 
  cxt->Su  = AllocFloatArray(cxt->ksize);
  if (cxt->Sv != NULL) free(cxt->Sv); 
  cxt->Sv  = AllocFloatArray(cxt->ksize);
  if (cxt->ISu != NULL) free(cxt->ISu); 
  cxt->ISu = AllocFloatArray(cxt->ksize);
  if (cxt->ISv != NULL) free(cxt->ISv); 
  cxt->ISv = AllocFloatArray(cxt->ksize);
  
  cxt->Ti = -cxt->Si*cxt->ksize/2.; 
  cxt->Tj = -cxt->Sj*cxt->ksize/2.;

  for (k = cxt->ki; k != cxt->kf; k += cxt->dk) {
    cxt->Su[k] = k*cxt->Si + cxt->Ti;
    cxt->Sv[k] = k*cxt->Sj + cxt->Tj;
    cxt->ISu[k] = -k*cxt->Si - cxt->Ti;
    cxt->ISv[k] = -k*cxt->Sj - cxt->Tj; 
  }

  /* Compute Shear and Inverted Shear Matrices */

  S[0][0] = 1.0; S[0][1] = 0.0; S[0][2] = cxt->Si;  S[0][3] = cxt->Ti;
  S[1][0] = 0.0; S[1][1] = 1.0; S[1][2] = cxt->Sj;  S[1][3] = cxt->Tj; 
  S[2][0] = 0.0; S[2][1] = 0.0; S[2][2] = 1.0; S[2][3] = 0.0;
  S[3][0] = 0.0; S[3][1] = 0.0; S[3][2] = 0.0; S[3][3] = 1.0;
  
  IS[0][0] = 1.0; IS[0][1] = 0.0; IS[0][2] = -cxt->Si; IS[0][3] = -cxt->Ti;
  IS[1][0] = 0.0; IS[1][1] = 1.0; IS[1][2] = -cxt->Sj; IS[1][3] = -cxt->Tj;
  IS[2][0] = 0.0; IS[2][1] = 0.0; IS[2][2] = 1.0; IS[2][3] = 0.0;
  IS[3][0] = 0.0; IS[3][1] = 0.0; IS[3][2] = 0.0; IS[3][3] = 1.0;

  /* Compute Permutation and Inverted Permutation Matrices. IP =
     Transposed P */
  
  switch (cxt->PAxis) {
  case 'x':
    P[0][0] = 0.0; P[0][1] = 1.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 0.0; P[1][2] = 1.0; P[1][3] = 0.0;
    P[2][0] = 1.0; P[2][1] = 0.0; P[2][2] = 0.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 0.0; IP[0][1] = 0.0; IP[0][2] = 1.0; IP[0][3] = 0.0;
    IP[1][0] = 1.0; IP[1][1] = 0.0; IP[1][2] = 0.0; IP[1][3] = 0.0;
    IP[2][0] = 0.0; IP[2][1] = 1.0; IP[2][2] = 0.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  case 'y':
    P[0][0] = 0.0; P[0][1] = 0.0; P[0][2] = 1.0; P[0][3] = 0.0;
    P[1][0] = 1.0; P[1][1] = 0.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 1.0; P[2][2] = 0.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 0.0; IP[0][1] = 1.0; IP[0][2] = 0.0; IP[0][3] = 0.0;
    IP[1][0] = 0.0; IP[1][1] = 0.0; IP[1][2] = 1.0; IP[1][3] = 0.0;
    IP[2][0] = 1.0; IP[2][1] = 0.0; IP[2][2] = 0.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  case 'z':
    P[0][0] = 1.0; P[0][1] = 0.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 1.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 0.0; P[2][2] = 1.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    IP[0][0] = 1.0; IP[0][1] = 0.0; IP[0][2] = 0.0; IP[0][3] = 0.0;
    IP[1][0] = 0.0; IP[1][1] = 1.0; IP[1][2] = 0.0; IP[1][3] = 0.0;
    IP[2][0] = 0.0; IP[2][1] = 0.0; IP[2][2] = 1.0; IP[2][3] = 0.0;
    IP[3][0] = 0.0; IP[3][1] = 0.0; IP[3][2] = 0.0; IP[3][3] = 1.0;
    break;
  }

 /* Compute Warp and Inverted Warp Matrices. R = W*S*P => W = R*IP*IS
    and IW = S*P*IR */
  
  MultMatrices(IP,IS,M);
  MultMatrices(cxt->R,M,cxt->W);
  MultMatrices(P,cxt->IR,M);
  MultMatrices(S,M,cxt->IW);
}


