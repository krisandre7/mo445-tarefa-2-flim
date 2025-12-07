#include "reslicing.h"
#include "image.h"
#include "geometry.h"


Scene *ResliceScene(Scene *scn, Context *cxt, Plane *pl, \
		    float alpha, int nslices)
{
  Scene *res;
  float u1,v1;
  int u,v,i,usize,vsize;
  Voxel w,w1;
  float Rx[4][4],Ry[4][4],R[4][4],d;

  d = sqrt(pl->normal.y*pl->normal.y + pl->normal.z*pl->normal.z);
  R[0][0] = Ry[0][0] = d;
  R[0][1] = Ry[0][1] = 0.0;
  R[0][2] = Ry[0][2] = pl->normal.x;
  R[0][3] = Ry[0][3] = 0.0;
  R[1][0] = Ry[1][0] = 0.0;
  R[1][1] = Ry[1][1] = 1.0;
  R[1][2] = Ry[1][2] = 0.0;
  R[1][3] = Ry[1][3] = 0.0;
  R[2][0] = Ry[2][0] = -pl->normal.x;
  R[2][1] = Ry[2][1] = 0.0;
  R[2][2] = Ry[2][2] = d;
  R[2][3] = Ry[2][3] = 0.0;
  R[3][0] = Ry[3][0] = 0.0;
  R[3][1] = Ry[3][1] = 0.0;
  R[3][2] = Ry[3][2] = 0.0;
  R[3][3] = Ry[3][3] = 1.0;

  if (d != 0.0){
    Rx[0][0] = 1.0;
    Rx[0][1] = 0.0;
    Rx[0][2] = 0.0;
    Rx[0][3] = 0.0;
    Rx[1][0] = 0.0;
    Rx[1][1] = pl->normal.z/d;
    Rx[1][2] = pl->normal.y/d;
    Rx[1][3] = 0.0;
    Rx[2][0] = 0.0;
    Rx[2][1] = -pl->normal.y/d;
    Rx[2][2] = pl->normal.z/d;
    Rx[2][3] = 0.0;
    Rx[3][0] = 0.0;
    Rx[3][1] = 0.0;
    Rx[3][2] = 0.0;
    Rx[3][3] = 1.0;    
    MultMatrices(Rx,Ry,R);
  } 

  usize = cxt->zbuff->usize;
  vsize = cxt->zbuff->vsize;
  res = CreateScene(usize,vsize,nslices);
 
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=alpha;
  
  for (i=0; i < nslices; i++) {

    
    for (v=0; v < vsize; v++) 
      for (u=0; u < usize; u++) {
	u1   = u - usize/2;
	v1   = v - vsize/2;
	w1.x  =(u1*R[0][0] + v1*R[0][1] + pl->po.x); 
	w1.y  =(u1*R[1][0] + v1*R[1][1] + pl->po.y);
	w.x = w1.x + scn->xsize/2;
	w.y = w1.y + scn->ysize/2;
	if ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&&
	    (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)){
	  w1.z = (int) (u1*R[2][0] + v1*R[2][1] + pl->po.z);
	  w.z  = w1.z + scn->zsize/2;
	  if ((w.z >= cxt->zmin)&&(w.z <= cxt->zmax)){
	    res->data[u + res->tby[v] + res->tbz[i]] = 
	      scn->data[w.x + scn->tby[w.y] + scn->tbz[w.z]];
	  }
	}
      }
    TranslatePlane(pl,alpha);
  }
  return(res);
}

Image *SliceSceneParallel(Scene *scn,Context *cxt,Plane *pl,Spline *spline){
  
  float l,delta,dy,alpha;
  int usize,vsize,u,v;
  Vector v1;
  Voxel w;
  Plane *plaux;
  Image *img;
  plaux = CopyPlane(pl);
  plaux->dx=pl->dx;
  plaux->dy=pl->dy;
  plaux->dz=pl->dz;
  
  Compute3DSpline(scn,cxt,plaux,spline);
  l = LengthSpline(spline,1.0);
  
  delta = spline->dx;
  if(spline->dy < delta) 
    delta = spline->dy;
  if(spline->dz < delta) 
    delta = spline->dz;


  usize=(int)l/delta;
  vsize=(int)usize;
  
  img = CreateImage(usize,vsize);
  dy=-((vsize-1)/2);
  TranslatePlane(plaux,dy);
  Compute3DSpline(scn,cxt,plaux,spline);
  
  for(v=0;v<vsize;v++){
    alpha=0.0;
    for(u=0;u<usize;u++){
      alpha=(float)u/(usize-1);
      CalculateVoxel(spline,alpha,&v1);
      w.x=v1.x;
      w.y=v1.y;
      w.z=v1.z;
      
      if ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&&
	  (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&&
	  (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)){
	img->val[u + v*usize] = 
	  scn->data[w.x + scn->tby[w.y] + scn->tbz[w.z]];
      }
      
    }
    TranslatePlane(plaux,1.0);
    Compute3DSpline(scn,cxt,plaux,spline);
  }
  return img;
};


Image *SliceSceneParallel2(Scene *scn,Context *cxt,Plane *pl,Spline *spline,int width){
  
  float l,delta,dy,alpha;
  int usize,vsize,u,v;
  Vector v1;
  Voxel w;
  Plane *plaux;
  Image *img;
  plaux = CopyPlane(pl);
  plaux->dx=pl->dx;
  plaux->dy=pl->dy;
  plaux->dz=pl->dz;
  
  Compute3DSpline(scn,cxt,plaux,spline);
  l = LengthSpline(spline,1.0);
  
  delta = spline->dx;
  if(spline->dy < delta) 
    delta = spline->dy;
  if(spline->dz < delta) 
    delta = spline->dz;


  usize=(int)l/delta;
  vsize=(int)2*width;
  
  img = CreateImage(width,vsize);
  dy=-((vsize-1)/2);
  TranslatePlane(plaux,dy);
  Compute3DSpline(scn,cxt,plaux,spline);
 
  for(v=0;v<vsize;v++){
    alpha=0.0;
    for(u=0;u<width;u++){
      if(u < usize) {
      alpha=(float)u/(usize-1);
      CalculateVoxel(spline,alpha,&v1);
      w.x=v1.x;
      w.y=v1.y;
      w.z=v1.z;
      
      if ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&&
	  (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&&
	  (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)){
	img->val[u + v*width] = 
	  scn->data[w.x + scn->tby[w.y] + scn->tbz[w.z]];
      }
      
      } else img->val[u + v*width] = 0;
    }

    TranslatePlane(plaux,1.0);
    Compute3DSpline(scn,cxt,plaux,spline);
  }
  return img;
};

Image *SliceScene(Scene *scn, Context *cxt, Plane *pl)
{
  Image *img;
  int u,v,u1,v1,usize,vsize,i,j;
  Voxel w,w1;
  float Rx[4][4],Ry[4][4],R[4][4],d;

  usize = cxt->zbuff->usize;
  vsize = cxt->zbuff->vsize;

  d = sqrt(pl->normal.y*pl->normal.y + pl->normal.z*pl->normal.z);
  R[0][0] = Ry[0][0] = d;
  R[0][1] = Ry[0][1] = 0.0;
  R[0][2] = Ry[0][2] = pl->normal.x;
  R[0][3] = Ry[0][3] = 0.0;
  R[1][0] = Ry[1][0] = 0.0;
  R[1][1] = Ry[1][1] = 1.0;
  R[1][2] = Ry[1][2] = 0.0;
  R[1][3] = Ry[1][3] = 0.0;
  R[2][0] = Ry[2][0] = -pl->normal.x;
  R[2][1] = Ry[2][1] = 0.0;
  R[2][2] = Ry[2][2] = d;
  R[2][3] = Ry[2][3] = 0.0;
  R[3][0] = Ry[3][0] = 0.0;
  R[3][1] = Ry[3][1] = 0.0;
  R[3][2] = Ry[3][2] = 0.0;
  R[3][3] = Ry[3][3] = 1.0;

  if (d != 0.0){
    Rx[0][0] = 1.0;
    Rx[0][1] = 0.0;
    Rx[0][2] = 0.0;
    Rx[0][3] = 0.0;
    Rx[1][0] = 0.0;
    Rx[1][1] = pl->normal.z/d;
    Rx[1][2] = pl->normal.y/d;
    Rx[1][3] = 0.0;
    Rx[2][0] = 0.0;
    Rx[2][1] = -pl->normal.y/d;
    Rx[2][2] = pl->normal.z/d;
    Rx[2][3] = 0.0;
    Rx[3][0] = 0.0;
    Rx[3][1] = 0.0;
    Rx[3][2] = 0.0;
    Rx[3][3] = 1.0;    
    MultMatrices(Rx,Ry,R);
  } 

  img = CreateImage(usize,vsize);
  
  for (v=0; v < vsize; v++) 
    for (u=0; u < usize; u++) {
      u1   = u - usize/2;
      v1   = v - vsize/2;
      w1.x  = u1*R[0][0] + v1*R[0][1] + pl->po.x; 
      w1.y  = u1*R[1][0] + v1*R[1][1] + pl->po.y;
      w.x = w1.x + scn->xsize/2;
      w.y = w1.y + scn->ysize/2;
      if ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&&
	  (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)){
	w1.z = (int) (u1*R[2][0] + v1*R[2][1] + pl->po.z);
	w.z  = w1.z + scn->zsize/2;
	if ((w.z >= cxt->zmin)&&(w.z <= cxt->zmax)){
	  i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	  j = u + img->tbrow[v];
	  img->val[j] = scn->data[i];
	  if (cxt->zbuff->voxel[j] == NIL) {
	    cxt->zbuff->object[j] = scn->data[i];
	    cxt->zbuff->voxel[j] = i;
	    cxt->zbuff->dist[j] = w1.z + usize;
	  }
	}
      }
    }
  return(img);
}

Scene *ResliceSceneNormal(Scene *scn, Context *cxt, Plane *pl,Spline *spline, \
float alpha, int nslices)
{
  Scene *res;
  float l,delta,u,i;
  int n,usize,vsize;
  Plane *plaux;
  Image *imgaux;

  l = LengthSpline(spline,1.0);
  delta = alpha/l;
  u=0.0;
  n = (l/alpha)+1;
  if(n > nslices) n = nslices;  
  
  usize = cxt->zbuff->usize;
  vsize = cxt->zbuff->vsize;
  
  res = CreateScene(usize,vsize,n);
  
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=alpha;
 
  for(i=1;i<=n;i++){
    plaux = NormalPlane(scn,spline,u);
    u+=delta;
    imgaux = SliceScene(scn,cxt,plaux);
    PutSlice(imgaux,res,i-1);
  }
  return res;
}

Scene *ResliceSceneParallel(Scene *scn,Context *cxt,Plane *pl,Spline *spline,float alpha,int nslices){
 
  Scene *res;
  float l1,l2,l,delta,u,i;
  int usize,vsize;
  Image *imgaux;
  Spline *base,*aux;
  
  
  u=-(float)((nslices-1)/2)*alpha;
  aux=TranslateSpline(spline,u);
  Compute3DSpline(scn,cxt,pl,aux);
  l1 = LengthSpline(aux,1.0);
  base=aux;
  
  u=(float)(nslices-1)*alpha;
  aux=TranslateSpline(aux,u);
  Compute3DSpline(scn,cxt,pl,aux);
  l2 = LengthSpline(aux,1.0);
  DestroySpline(&base);
  DestroySpline(&aux);

  l=l1;
  if(l2 > l) l = l2;


  delta = spline->dx;
  if(spline->dy < delta) 
    delta = spline->dy;
  if(spline->dz < delta) 
    delta = spline->dz;
  
  usize=(int)l/delta;
  vsize=(int)2*usize;
  res = CreateScene(usize,vsize,nslices);

  res->dx=delta;
  res->dy=delta;
  res->dz=alpha;

  u=-(float)((nslices-1)/2)*alpha;
  base=TranslateSpline(spline,u);
  imgaux = SliceSceneParallel2(scn,cxt,pl,base,usize);
  PutSlice(imgaux,res,0);
   
  aux=base;
  u=0.0;
  
  for(i=1;i<nslices;i++){
    u=alpha*i;
    aux=TranslateSpline(base,u);
    imgaux = SliceSceneParallel2(scn,cxt,pl,aux,usize);
    PutSlice(imgaux,res,i);
    DestroySpline(&aux);
  }
  return res;
}
