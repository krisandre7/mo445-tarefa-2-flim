#include "splatting.h"
#include "geometry.h"
#include "adjacency.h"
#include "comptime.h"
#include "shading.h"

Image *SplatScene(Scene *scn, Context *cxt)
{
  int u,v,i,j,k;
  
  Voxel w,w1;
  Image *img=NULL;
  ZBuffer *tmp;
  Image foo;
  AdjPxl *fprint;

  foo.ncols = cxt->zbuff->usize;
  fprint = AdjPixels(&foo,cxt->footprint->adj);

  tmp = CreateZBuffer(cxt->zbuff->usize,cxt->zbuff->vsize);

  for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
    for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) 
      for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx){
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	if (scn->data[i] != 0){
	  w1.x = w.x - cxt->xsize/2;
	  w1.y = w.y - cxt->ysize/2;
	  w1.z = w.z - cxt->zsize/2;
	  u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + cxt->R[0][2]*w1.z + \
		     cxt->R[0][3] + tmp->usize/2);
	  v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + cxt->R[1][2]*w1.z + \
		     cxt->R[1][3] + tmp->vsize/2);
	  k = u + tmp->tbv[v];
	  if (tmp->voxel[k] == NIL){
	    tmp->dist[k] = (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + cxt->R[2][2]*w1.z + cxt->R[2][3])+tmp->usize/2;	    
	    tmp->voxel[k] = i;
	    tmp->object[k] = (uchar)scn->data[i];
	  }
	}
      }


  InitZBuffer(cxt->zbuff);
  
  for (v=1; v < cxt->zbuff->vsize-1; v++) { /* for non-border pixels */
    for (u=1; u < cxt->zbuff->usize-1; u++) {
      i = u + tmp->tbv[v];
      if (tmp->voxel[i] != NIL) { /* a voxel was projected on this pixel */
	cxt->zbuff->dist[i]  = tmp->dist[i];
	cxt->zbuff->object[i] = tmp->object[i];
	cxt->zbuff->voxel[i] = tmp->voxel[i];
	for (j=0; j < fprint->n; j++){ /* for the 8 neighbour pixels */
	  k = i+fprint->dp[j];
	  if ((tmp->voxel[k] == NIL)&&(cxt->zbuff->voxel[k] == NIL)){/* no voxel was projected/spread on that neighbour pixel (hole) */
	    cxt->zbuff->dist[k]=tmp->dist[i];
	    cxt->zbuff->object[k]=tmp->object[i];
	    cxt->zbuff->voxel[k]=tmp->voxel[i];
	  }
	}
      }
    }
  }

  DestroyZBuffer(&tmp);
  DestroyAdjPxl(&fprint);
  img = PhongShading(cxt);
  return(img);
}

Image *SWSplatScene(Scene *scn, Context *cxt)
{
  int u,v,i,j,k,uw,vw;
  
  Voxel w,w1;
  Image *img=NULL;
  float Su,Sv;
  ZBuffer *tmp1,*tmp2;
  Image foo;
  AdjPxl *fprint;

  foo.ncols = cxt->zbuff->usize;
  fprint = AdjPixels(&foo,cxt->footprint->adj);

  Su = cxt->zbuff->usize/2; 
  Sv = cxt->zbuff->vsize/2; 
  InitZBuffer(cxt->zbuff);
  
  tmp1 = CreateZBuffer(cxt->zbuff->usize,cxt->zbuff->vsize);
  tmp2 = CreateZBuffer(cxt->zbuff->usize,cxt->zbuff->vsize);

  switch(cxt->PAxis) {
  case 'x':
    for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx)
      for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
	for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy){ 
	  i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	  if (scn->data[i] != 0){
	    w1.x = w.x;
	    w1.y = w.y - cxt->ysize/2;
	    w1.z = w.z - cxt->zsize/2;
	    u = (int) (w1.y + cxt->Su[w1.x] + Su); 
	    v = (int) (w1.z + cxt->Sv[w1.x] + Sv); 
	    k = u + cxt->zbuff->tbv[v];
	    if (tmp1->voxel[k]==NIL){
	      tmp1->dist[k] = (cxt->R[2][0]*(w1.x-cxt->xsize/2) + cxt->R[2][1]*w1.y + cxt->R[2][2]*w1.z + cxt->R[2][3])+cxt->zbuff->usize/2;
	      tmp1->voxel[k] = i;
	      tmp1->object[k] = (uchar)scn->data[i];
	    }
	  }
	}
    break;
  case 'y':
    for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) 
      for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx)
	for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz){ 
	  i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	  if (scn->data[i] != 0){
	    w1.x = w.x - cxt->xsize/2;
	    w1.y = w.y;
	    w1.z = w.z - cxt->zsize/2;
	    u = (int) (w1.z + cxt->Su[w1.y] + Su); 
	    v = (int) (w1.x + cxt->Sv[w1.y] + Sv); 
	    k = u + cxt->zbuff->tbv[v];
	    if (tmp1->voxel[k]==NIL){
	      tmp1->dist[k] = (cxt->R[2][0]*w1.x + cxt->R[2][1]*(w1.y-cxt->ysize/2) + cxt->R[2][2]*w1.z + cxt->R[2][3])+cxt->zbuff->usize/2;
	      tmp1->voxel[k] = i;
	      tmp1->object[k] = (uchar)scn->data[i];
	    }
	  }
	}
    break;
  case 'z':
    for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
      for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) 
	for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx){
	  i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	  if (scn->data[i] != 0){
	    w1.x = w.x - cxt->xsize/2;
	    w1.y = w.y - cxt->ysize/2;
	    w1.z = w.z;
	    u = (int) (w1.x + cxt->Su[w1.z] + Su); 
	    v = (int) (w1.y + cxt->Sv[w1.z] + Sv); 
	    k = u + cxt->zbuff->tbv[v];
	    if (tmp1->voxel[k]==NIL){
	      tmp1->dist[k] = (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + cxt->R[2][2]*(w1.z-cxt->zsize/2) + cxt->R[2][3])+cxt->zbuff->usize/2;
	      tmp1->voxel[k] = i;
	      tmp1->object[k] = (uchar) scn->data[i];
	    }
	  }
	}
    break;
  }

  /* Warping 2D */

  for (v=0; v<cxt->zbuff->vsize ; v++)
    for(u=0; u<cxt->zbuff->usize ; u++) {
      uw = ROUND(cxt->W[0][0] * (u-tmp1->usize/2) + cxt->W[0][1] * (v-tmp1->vsize/2)) + cxt->zbuff->usize/2 ;

      vw = ROUND(cxt->W[1][0] * (u-tmp1->usize/2) + cxt->W[1][1] * (v-tmp1->vsize/2)) + cxt->zbuff->vsize/2 ;
      
      if ((uw>=0)&&(uw<cxt->zbuff->usize)&&(vw>=0)&&(vw<cxt->zbuff->vsize)) {
	k = uw + cxt->zbuff->tbv[vw];
	if (tmp2->voxel[k]==NIL) {
	  i = u + cxt->zbuff->tbv[v];
	  tmp2->dist[k] = tmp1->dist[i];
	  tmp2->object[k] = tmp1->object[i];
	  tmp2->voxel[k] = tmp1->voxel[i];
	}
      }
    }

  InitZBuffer(cxt->zbuff);
  
  for (v=1; v < cxt->zbuff->vsize-1; v++) { // for non-border pixels
    for (u=1; u < cxt->zbuff->usize-1; u++) {
      i = u + tmp2->tbv[v];
      if (tmp2->voxel[i] != NIL) { // a voxel was projected on this pixel
	cxt->zbuff->dist[i]  = tmp2->dist[i];
	cxt->zbuff->object[i] = tmp2->object[i];
	cxt->zbuff->voxel[i] = tmp2->voxel[i];
	for (j=0; j < fprint->n; j++){ // for the 8 neighbour pixels
	  k = i+fprint->dp[j];
	  if ((tmp2->voxel[k] == NIL)&&(cxt->zbuff->voxel[k] == NIL)){// no voxel was projected/spread on that neighbour pixel (hole)
	    cxt->zbuff->dist[k]=tmp2->dist[i];
	    cxt->zbuff->object[k]=tmp2->object[i];
	    cxt->zbuff->voxel[k]=tmp2->voxel[i];
	  }
	}
      }
    }
  }
  
  DestroyZBuffer(&tmp1);
  DestroyZBuffer(&tmp2);
  DestroyAdjPxl(&fprint);  
  img = PhongShading(cxt);
  
  return(img);
}


Image *SplatBorder(Scene *scn, Context *cxt, Border *border)
{
  float dist;
  int u,v,i,j,k,aux,xysize;
  Voxel w,w1;
  Image *img=NULL;
  ZBuffer *tmp;
  Image foo;
  AdjPxl *fprint;

  foo.ncols = cxt->zbuff->usize;
  fprint = AdjPixels(&foo,cxt->footprint->adj);

  xysize = cxt->xsize * cxt->ysize;
  tmp = CreateZBuffer(cxt->zbuff->usize,cxt->zbuff->vsize);
  i = border->first;

  while(i != NIL) { // in the list of border voxel
    //    if (scn->data[i] != 0){// border never has a 0-label
    aux = i   % xysize;
    w.x = aux % cxt->xsize;
    w.y = aux / cxt->xsize;
    w.z = i   / xysize;
    w1.x = w.x - cxt->xsize/2;
    w1.y = w.y - cxt->ysize/2;
    w1.z = w.z - cxt->zsize/2;
    u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + cxt->R[0][2]*w1.z + \
      cxt->R[0][3] + tmp->usize/2);
    v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + cxt->R[1][2]*w1.z + \
      cxt->R[1][3] + tmp->vsize/2);
    k = u + tmp->tbv[v];
    dist = (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + cxt->R[2][2]*w1.z + cxt->R[2][3])+tmp->usize/2;
    if (dist < tmp->dist[k]){
      tmp->dist[k] = dist;
      tmp->voxel[k] = i;
      tmp->object[k] = (uchar)scn->data[i];
    }
    i = border->voxels[i].next; // to next border voxel
  }

  InitZBuffer(cxt->zbuff);
  
  for (v=1; v < cxt->zbuff->vsize-1; v++) { /* for non-border pixels */
    for (u=1; u < cxt->zbuff->usize-1; u++) {
      i = u + tmp->tbv[v];
      if (tmp->voxel[i] != NIL) { /* a voxel was projected on this pixel */
        cxt->zbuff->dist[i]  = tmp->dist[i];
        cxt->zbuff->object[i] = tmp->object[i];
        cxt->zbuff->voxel[i] = tmp->voxel[i];
        for (j=0; j < fprint->n; j++){ /* for the 8 neighbour pixels */
          k = i+fprint->dp[j];
          if ((tmp->voxel[k] == NIL)&&(cxt->zbuff->voxel[k] == NIL)){/* no voxel was projected/spread on that neighbour pixel (hole) */
            cxt->zbuff->dist[k]=tmp->dist[i];
            cxt->zbuff->object[k]=tmp->object[i];
            cxt->zbuff->voxel[k]=tmp->voxel[i];
          }
        }
      }
    }
  }

  DestroyZBuffer(&tmp);
  DestroyAdjPxl(&fprint);
  img = PhongShading(cxt);
  return(img);
}


