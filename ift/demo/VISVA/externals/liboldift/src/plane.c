#include "plane.h"
#include "algebra.h"


Plane    *CreatePlane(Context *cxt)
{
  Plane *pl=NULL;

  pl = (Plane *) calloc(1,sizeof(Plane));
  if (pl == NULL) 
    Error(MSG1,"CreatePlane");
  pl->normal.x = 0.0; pl->normal.y = 0.0; pl->normal.z = 1.0;
  pl->po.x = 0.0;
  pl->po.y = 0.0;
  pl->po.z = 0.0;
  pl->dx=cxt->dx;
  pl->dy=cxt->dy;
  pl->dz=cxt->dz;

  pl->R[0][0] = 1.0;
  pl->R[0][1] = 0.0;
  pl->R[0][2] = 0.0;
  pl->R[0][3] = 0.0;

  pl->R[1][0] = 0.0;
  pl->R[1][1] = 1.0;
  pl->R[1][2] = 0.0;
  pl->R[1][3] = 0.0;

  pl->R[2][0] = 0.0;
  pl->R[2][1] = 0.0;
  pl->R[2][2] = 1.0;
  pl->R[2][3] = 0.0;

  pl->R[3][0] = 0.0;
  pl->R[3][1] = 0.0;
  pl->R[3][2] = 0.0;
  pl->R[3][3] = 1.0;

  pl->thx = 0.0;
  pl->thy = 0.0;

  return(pl);
}

Plane    *CreatePlaneaux()
{
  Plane *pl=NULL;

  pl = (Plane *) calloc(1,sizeof(Plane));
  if (pl == NULL) 
    Error(MSG1,"CreatePlane");
  pl->normal.x = 0.0; pl->normal.y = 0.0; pl->normal.z = 1.0;
  pl->po.x = 0.0;
  pl->po.y = 0.0;
  pl->po.z = 0.0;
  return(pl);
}

Plane *CopyPlane(Plane *pl){
  Plane *plaux = CreatePlaneaux();
  plaux->po.x=pl->po.x;
  plaux->po.y=pl->po.y;
  plaux->po.z=pl->po.z;
  plaux->normal.x=pl->normal.x;
  plaux->normal.y=pl->normal.y;
  plaux->normal.z=pl->normal.z;
  plaux->dx=pl->dx;
  plaux->dy=pl->dy;
  plaux->dz=pl->dz;
  return plaux;
}

void  DestroyPlane(Plane **pl)
{
  Plane *aux=*pl;

  if (aux != NULL) {
    free(aux);
    *pl = NULL;
  }
}

void RotatePlaneAbs (Plane *pl, float thx, float thy)
{
  float Rx[4][4],Ry[4][4];
  double trash;


  thx = 360.*modf(thx/360.,&trash);
  if (thx < 0.0)
    thx += 360.0;
  thy = 360.*modf(thy/360.,&trash);
  if (thy < 0.0)
    thy += 360.0;

  pl->thx = thx;
  pl->thy = thy;
  
  thx = thx*PI/180.;
  thy = thy*PI/180.;

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


  /* Ry*Rx */

  MultMatrices(Ry,Rx,pl->R);
  
  pl->normal.x = pl->R[0][2];
  pl->normal.y = pl->R[1][2];
  pl->normal.z = pl->R[2][2];
}

void RotatePlane (Plane *pl, float thx, float thy)
{
  float Rx[4][4],Ry[4][4],R[4][4];
  double trash;


  thx = 360.*modf(thx/360.,&trash);
  if (thx < 0.0)
    thx += 360.0;
  thy = 360.*modf(thy/360.,&trash);
  if (thy < 0.0)
    thy += 360.0;

  pl->thx = thx;
  pl->thy = thy;
  
  thx = thx*PI/180.;
  thy = thy*PI/180.;

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


  /* Ry*Rx */

  MultMatrices(Rx,pl->R,R);
  MultMatrices(Ry,R,pl->R);

  pl->normal.x = pl->R[0][2];
  pl->normal.y = pl->R[1][2];
  pl->normal.z = pl->R[2][2];
}

void LockPlane(Context *cxt, Plane *pl) {

  pl->R[0][0] = cxt->R[0][0];
  pl->R[0][1] = cxt->R[0][1];
  pl->R[0][2] = cxt->R[0][2];
  pl->R[0][3] = cxt->R[0][3];

  pl->R[1][0] = cxt->R[1][0];
  pl->R[1][1] = cxt->R[1][1];
  pl->R[1][2] = cxt->R[1][2];
  pl->R[1][3] = cxt->R[1][3];

  pl->R[2][0] = cxt->R[2][0];
  pl->R[2][1] = cxt->R[2][1];
  pl->R[2][2] = cxt->R[2][2];
  pl->R[2][3] = cxt->R[2][3];

  pl->R[3][0] = cxt->R[3][0];
  pl->R[3][1] = cxt->R[3][1];
  pl->R[3][2] = cxt->R[3][2];
  pl->R[3][3] = cxt->R[3][3];

}


void TranslatePlane(Plane *pl, float alpha)
{

  /* Use o translatePlaneAux para ficar melhor */
  
  if (pl->normal.x < 0)
    pl->po.x -= -(alpha*pl->normal.x)*pl->dx;
  else
    pl->po.x += (alpha*pl->normal.x)*pl->dx;
  
  if (pl->normal.y < 0)
  pl->po.y -= -(alpha*pl->normal.y)*pl->dy;
  else
    pl->po.y += (alpha*pl->normal.y)*pl->dy;
  

  if (pl->normal.z < 0)
    pl->po.z -= (alpha*pl->normal.z)*pl->dz;
  else
    pl->po.z += (alpha*pl->normal.z)*pl->dz;
}

void TranslatePlaneAbs(Plane *pl, float alpha)
{
  pl->po.x=0.0;
  pl->po.y=0.0;
  pl->po.z=0.0;
  TranslatePlane(pl,alpha);
}


void TranslatePlaneAux(Plane *pl, float alpha){
  pl->po.x=0.0;
  pl->po.y=0.0;
  pl->po.z=0.0;
  TranslatePlane(pl,alpha);
}

Plane *FacePlane(Vertex *vert, int n) 
{
  Plane *pl=NULL;
  Vector v1,v2,normal;
  float mag;
  int i;

  pl   = CreatePlaneaux();
  v2.x = vert[0].x - vert[1].x;
  v2.y = vert[0].y - vert[1].y;
  v2.z = vert[0].z - vert[1].z;
  v1.x = vert[2].x - vert[1].x;
  v1.y = vert[2].y - vert[1].y;
  v1.z = vert[2].z - vert[1].z;  
  normal = VectorProd(v1,v2);
  mag    = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
  pl->normal.x = normal.x/mag;
  pl->normal.y = normal.y/mag;
  pl->normal.z = normal.z/mag;
  for (i=0; i < n; i++) {
    pl->po.x += vert[i].x;
    pl->po.y += vert[i].y;
    pl->po.z += vert[i].z;
  }
  pl->po.x /= n;
  pl->po.y /= n;
  pl->po.z /= n;

  return(pl);
}

float GetPlaneX(Plane *pl){
  return pl->normal.x;
}


float GetPlaneY(Plane *pl){
  return pl->normal.y;
}

float GetPlaneTilt(Plane *pl){
  return pl->thx;
}


float GetPlaneSpin(Plane *pl){
  return pl->thy;
}


void SetPlaneZBuffer(ZBuffer *zbuff, Plane *pl) 
{

  int i;
  Pixel p,v,c;
  float z,tanx=0.0,tany=0.0;
  
  c.x = zbuff->usize/2;
  c.y = zbuff->vsize/2;

  if (pl->normal.x != 0.0) {
    tanx = pl->normal.z/pl->normal.x;
  } 
  if (pl->normal.y != 0.0) {
    tany = pl->normal.z/pl->normal.y;
  }   
    
  for (v.y = 0; v.y < zbuff->vsize; v.y ++)
    for (v.x = 0; v.x < zbuff->usize; v.x ++) {
      p.x  = v.x - c.x;
      p.y  = v.y - c.y;
      i = zbuff->tbv[v.y] + v.x;
	z = p.x * tanx + p.y * tany + pl->po.z;
	if (zbuff->dist[i] > c.x - z)
	  zbuff->dist[i] = c.x - z;   
	zbuff->voxel[i] = NIL;   
    }
}
  

void SetPlaneShearZBuffer(Context *cxt, ZBuffer *zbuff, Plane *pl) 
{

  int i;
  Pixel p,v,c;
  float z,Si=0.0,Sj=0.0,offset=0.0;
  float Ti=0.0,Tj=0.0;
  float S[4][4],IR[4][4],M[4][4],P[4][4],IW[4][4];

  c.x = zbuff->usize/2;
  c.y = zbuff->vsize/2;

 switch (cxt->PAxis) {
  case 'x':
    P[0][0] = 0.0; P[0][1] = 1.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 0.0; P[1][2] = 1.0; P[1][3] = 0.0;
    P[2][0] = 1.0; P[2][1] = 0.0; P[2][2] = 0.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    if (pl->normal.x != 0.0) {
      Si = -pl->normal.y/pl->normal.x;
    } 
    if (pl->normal.x != 0.0) {
      Sj = -pl->normal.z/pl->normal.x;
    }
    offset = pl->po.x;
    break;
  case 'y':
    P[0][0] = 0.0; P[0][1] = 0.0; P[0][2] = 1.0; P[0][3] = 0.0;
    P[1][0] = 1.0; P[1][1] = 0.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 1.0; P[2][2] = 0.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    if (pl->normal.y != 0.0) {
      Si = -pl->normal.x/pl->normal.y;
    } 
    if (pl->normal.y != 0.0) {
      Sj = -pl->normal.z/pl->normal.y;
    }   
    offset = pl->po.y;
    break;
  case 'z':
    P[0][0] = 1.0; P[0][1] = 0.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 1.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 0.0; P[2][2] = 1.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
    if (pl->normal.z != 0.0) {
      Si = -pl->normal.z/pl->normal.x;
    } 
    if (pl->normal.z != 0.0) {
      Sj = -pl->normal.z/pl->normal.y;
    }   
    offset = pl->po.z;
    break;
  }

  Ti = -Si*cxt->ksize/2.; 
  Tj = -Sj*cxt->ksize/2.;

  S[0][0] = 1.0; S[0][1] = 0.0; S[0][2] = Si;  S[0][3] = Ti;
  S[1][0] = 0.0; S[1][1] = 1.0; S[1][2] = Sj;  S[1][3] = Tj; 
  S[2][0] = 0.0; S[2][1] = 0.0; S[2][2] = 1.0; S[2][3] = 0.0;
  S[3][0] = 0.0; S[3][1] = 0.0; S[3][2] = 0.0; S[3][3] = 1.0;

  TransMatrix(pl->R,IR);
  MultMatrices(P,IR,M);
  MultMatrices(S,M,IW);
    
    for (v.y = 0; v.y < zbuff->vsize; v.y ++)
      for (v.x = 0; v.x < zbuff->usize; v.x ++) {
	p.x  = v.x - c.x;
	p.y  = v.y - c.y;
	i = zbuff->tbv[v.y] + v.x;
	z = p.x * IW[2][0] + p.y * IW[2][1] + offset * IW[2][2];
	if (zbuff->dist[i] > c.x - z)
	  zbuff->dist[i] = c.x - z;   
	zbuff->voxel[i] = NIL;   
      }
}
  
Plane *ClipPlane(Context *cxt,char axis) 
{
  Plane *pl=NULL;

  pl = CreatePlane(cxt);

  switch (axis) {
  case 'x':
    pl->normal.x = cxt->R[0][0];
    pl->normal.y = cxt->R[1][0];
    pl->normal.z = cxt->R[2][0];
    TranslatePlaneAbs(pl,(float)(cxt->xsize/2));
    break;
  case 'y':
    pl->normal.x = cxt->R[0][1];
    pl->normal.y = cxt->R[1][1];
    pl->normal.z = cxt->R[2][1];
    TranslatePlaneAbs(pl,(float)(cxt->ysize/2));
    break;
  case 'z':
    pl->normal.x = cxt->R[0][2];
    pl->normal.y = cxt->R[1][2];
    pl->normal.z = cxt->R[2][2];
    TranslatePlaneAbs(pl,(float)(cxt->zsize/2));
    break;
  default:
    break;
  }

  return(pl);
}
