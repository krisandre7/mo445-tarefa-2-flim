#include "raycasting.h"
#include "geometry.h"
#include "shading.h"
#include "shell.h"

Image *SWRaycastScene(Scene *scn, Context *cxt)
{
  Image *img=NULL;
  int i,j,k,u,v;
  Point p,q;
  Voxel w;
  float u1,v1,Su,Sv;

  Su = fabs(cxt->ISu[cxt->ki]);
  Sv = fabs(cxt->ISv[cxt->ki]);

  switch(cxt->PAxis){
  case 'x':
    for (v=0; v < cxt->zbuff->vsize; v++)
      for (u=0; u < cxt->zbuff->usize; u++){
	j   = u+cxt->zbuff->tbv[v];
	cxt->zbuff->object[j] = 0;
	cxt->zbuff->voxel[j]  = NIL;
	cxt->zbuff->dist[j]   = cxt->zbuff->usize;
	u1 = u - cxt->zbuff->usize/2;
	v1 = v - cxt->zbuff->vsize/2;
	p.x = u1*cxt->IW[0][0] + v1*cxt->IW[0][1] + cxt->ysize/2;
	p.y = u1*cxt->IW[1][0] + v1*cxt->IW[1][1] + cxt->zsize/2;
	if (!(((p.x + Su) < cxt->ymin)||
	      ((p.x - Su) >= cxt->ymax)||
	      ((p.y + Sv) < cxt->zmin)||
	      ((p.y - Sv) >= cxt->zmax))){
	  for (k=cxt->ki; k != cxt->kf; k += cxt->dk) {
	    w.y = ROUND(p.x + cxt->ISu[k]);
	    w.z = ROUND(p.y + cxt->ISv[k]);
	    if ((w.y >= cxt->ymin)&&(w.y < cxt->ymax)&&
		(w.z >= cxt->zmin)&&(w.z < cxt->zmax)){
	      w.x = k;
	      i   = w.x + scn->tby[w.y] + scn->tbz[w.z];
	      if (scn->data[i] != 0) {
		q.x = w.x - cxt->xsize/2;
		q.y = w.y - cxt->ysize/2;
		q.z = w.z - cxt->zsize/2;
		cxt->zbuff->object[j] = (uchar)scn->data[i];
		cxt->zbuff->voxel[j]  = i;
		cxt->zbuff->dist[j]   = 
		  (cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + cxt->R[2][3]) + cxt->zbuff->usize/2; 
		break;
	      }
	    }
	  }
	}
      }
    break;
  case 'y':
    for (v=0; v < cxt->zbuff->vsize; v++)
      for (u=0; u < cxt->zbuff->usize; u++){
	j   = u+cxt->zbuff->tbv[v];
	cxt->zbuff->object[j] = 0;
	cxt->zbuff->voxel[j] = NIL;
	cxt->zbuff->dist[j]  = cxt->zbuff->usize;
	u1 = u - cxt->zbuff->usize/2;
	v1 = v - cxt->zbuff->vsize/2;
	p.x = u1*cxt->IW[0][0] + v1*cxt->IW[0][1] + cxt->zsize/2;
	p.y = u1*cxt->IW[1][0] + v1*cxt->IW[1][1] + cxt->xsize/2;
	if (!(((p.x + Su) < cxt->zmin)||
	      ((p.x - Su) >= cxt->zmax)||
	      ((p.y + Sv) < cxt->xmin)||
	      ((p.y - Sv) >= cxt->xmax))){
	  for (k=cxt->ki; k != cxt->kf; k += cxt->dk) {
	    w.z = ROUND(p.x + cxt->ISu[k]); 
	    w.x = ROUND(p.y + cxt->ISv[k]); 
	    if ((w.z >= cxt->zmin)&&(w.z < cxt->zmax)&&
		(w.x >= cxt->xmin)&&(w.x < cxt->xmax)){	  
	      w.y = k;
	      i   = w.x + scn->tby[w.y] + scn->tbz[w.z];
	      if (scn->data[i] != 0) {
		q.x = w.x - cxt->xsize/2;
		q.y = w.y - cxt->ysize/2;
		q.z = w.z - cxt->zsize/2;
		cxt->zbuff->object[j] = (uchar)scn->data[i];
		cxt->zbuff->voxel[j] = i;
		cxt->zbuff->dist[j]  = 
		  (cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + cxt->R[2][3]) + cxt->zbuff->usize/2; 
		break;
	      }
	    }
	  }
	}
      }
    break;
  case 'z':
    for (v=0; v < cxt->zbuff->vsize; v++)
      for (u=0; u < cxt->zbuff->usize; u++){
	j   = u+cxt->zbuff->tbv[v];
	cxt->zbuff->object[j] = 0;
	cxt->zbuff->voxel[j] = NIL;
	cxt->zbuff->dist[j]  = cxt->zbuff->usize;
	u1 = u - cxt->zbuff->usize/2;
	v1 = v - cxt->zbuff->vsize/2;
	p.x = u1*cxt->IW[0][0] + v1*cxt->IW[0][1] + cxt->xsize/2;
	p.y = u1*cxt->IW[1][0] + v1*cxt->IW[1][1] + cxt->ysize/2;
	
	if (!(((p.x + Su) < cxt->xmin)||
	      ((p.x - Su) >= cxt->xmax)||
	      ((p.y + Sv) < cxt->ymin)||
	      ((p.y - Sv) >= cxt->ymax))){
	  for (k=cxt->ki; k != cxt->kf; k += cxt->dk) {
	    w.x = ROUND(p.x + cxt->ISu[k]); 
	    w.y = ROUND(p.y + cxt->ISv[k]); 
	    if ((w.x >= cxt->xmin)&&(w.x < cxt->xmax)&&
		(w.y >= cxt->ymin)&&(w.y < cxt->ymax)){	  	  
	      w.z = k;
	      i   = w.x + scn->tby[w.y] + scn->tbz[w.z];
	      if (scn->data[i] != 0) {
		q.x = w.x - cxt->xsize/2;
		q.y = w.y - cxt->ysize/2;
		q.z = w.z - cxt->zsize/2;
		cxt->zbuff->object[j] = (uchar)scn->data[i];
		cxt->zbuff->voxel[j] = i;
		cxt->zbuff->dist[j]  = 
		  (cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + cxt->R[2][3]) + cxt->zbuff->usize/2;
		break;
	      }
	    }
   	  }
	}
      }
    break;
  }

  img = PhongShading(cxt);

  return(img);

}

Image *RaycastScene(Scene *scn, Context *cxt)
{
  Image *img=NULL;
  int i,j,u,v,u1,v1,k1;
  Point p,p0;
  Vector n;
  float dx=0.,dy=0.,dz=0.,advance_factor,aux;
  Voxel w,w1;

  k1  = -cxt->zbuff->usize/2;
  n.x = cxt->IR[0][2];
  n.y = cxt->IR[1][2];
  n.z = cxt->IR[2][2];
  switch (cxt->PAxis) {
  case 'x':
    if (n.x > 0) 
      dx = 1.;
    else
      dx = -1.;
    dy = dx*(n.y/n.x); /* correcao */
    dz = dx*(n.z/n.x);
    break;
  case 'y':
    if (n.y > 0) 
      dy = 1.;
    else
      dy = -1.;
    dx = dy*(n.x/n.y);
    dz = dy*(n.z/n.y);
    break;
  case 'z':
    if (n.z > 0) 
      dz = 1.;
    else
      dz = -1.;
    dx = dz*(n.x/n.z);
    dy = dz*(n.y/n.z);
    break;
  }
  
  for (v=0; v < cxt->zbuff->vsize; v++)
    for (u=0; u < cxt->zbuff->usize; u++){
      j   = u+cxt->zbuff->tbv[v];
      cxt->zbuff->object[j] = 0;
      cxt->zbuff->voxel[j]  = NIL;
      cxt->zbuff->dist[j]   = cxt->zbuff->usize;
      u1 = u - cxt->zbuff->usize/2;
      v1 = v - cxt->zbuff->vsize/2;     
      p.x = u1*cxt->IR[0][0] + v1*cxt->IR[0][1] + k1*cxt->IR[0][2]; 
      p.y = u1*cxt->IR[1][0] + v1*cxt->IR[1][1] + k1*cxt->IR[1][2];
      p.z = u1*cxt->IR[2][0] + v1*cxt->IR[2][1] + k1*cxt->IR[2][2];

      p0.x = p.x;
      p0.y = p.y;
      p0.z = p.z;
      /* p0 pertence ao view plane */

      /* procura-se p, o ponto de intersecçao com a cena */
      advance_factor = 0.;

      if ((n.x > Epsilon) || (n.x < -Epsilon)) {
	if (p0.x > cxt->xsize/2 - 1) {
	  p.x = cxt->xsize/2 - 1; 
	  advance_factor  = (p.x - p0.x)/n.x;
	  }
	else
	  if (p0.x < -cxt->xsize/2) {
	    p.x = -cxt->xsize/2;
	    advance_factor  = (p.x - p0.x)/n.x;
	  }
      } else {
	if ((p0.x > cxt->xsize/2-1)||(p0.x < -cxt->xsize/2))
	  goto caifora;
      } 

      if ((n.y > Epsilon) || (n.y < -Epsilon)) {
	if (p0.y > cxt->ysize/2 - 1) {
	  p.y = cxt->ysize/2 - 1;
	  aux = (p.y - p0.y)/n.y;
	  if (advance_factor < aux)
	    advance_factor = aux;
	}
	else
	  if (p0.y < -cxt->ysize/2) {
	    p.y = -cxt->ysize/2;
	    aux = (p.y - p0.y)/n.y;	    
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.y > cxt->ysize/2-1)||(p0.y < -cxt->ysize/2))
	  goto caifora;
      } 
      
      if ((n.z > Epsilon) || (n.z < -Epsilon)) {
      	if (p0.z > cxt->zsize/2 - 1) {
	  p.z = cxt->zsize/2 - 1; /* cuidado com dim. impares */
	  aux = (p.z - p0.z)/n.z;
	  if (advance_factor < aux )
	    advance_factor = aux;
	}
	else
	  if (p0.z < -cxt->zsize/2) {
	    p.z = -cxt->zsize/2; 
	    aux = (p.z - p0.z)/n.z;   
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.z > cxt->zsize/2-1)||(p0.z < -cxt->zsize/2))
	  goto caifora;
      } 

      p.x = p0.x + advance_factor * n.x;
      p.y = p0.y + advance_factor * n.y;
      p.z = p0.z + advance_factor * n.z;

      w1.x = ROUND(p.x); /* o (ROUND) seria melhor que o (int) */
      w1.y = ROUND(p.y);
      w1.z = ROUND(p.z);
      w.x  = w1.x + cxt->xsize/2;
      w.y  = w1.y + cxt->ysize/2;
      w.z  = w1.z + cxt->zsize/2;

      while ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&& 
	     (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&& 
	     (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)) {
		  
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	if (scn->data[i] != 0){
	  cxt->zbuff->dist[j] = sqrt((w1.x-p0.x)*(w1.x-p0.x) +
				     (w1.y-p0.y)*(w1.y-p0.y) +
				     (w1.z-p0.z)*(w1.z-p0.z));
	  cxt->zbuff->voxel[j] = i;
	  cxt->zbuff->object[j] = scn->data[i];
	  break;
	}
	else {
	  p.x += dx; 
	  p.y += dy;
	  p.z += dz;
	  w1.x = (int)p.x;
	  w1.y = (int)p.y;
	  w1.z = (int)p.z;
	  w.x  = w1.x + cxt->xsize/2;
	  w.y  = w1.y + cxt->ysize/2;
	  w.z  = w1.z + cxt->zsize/2;
	}
      } /* end while */
    caifora:
      ;
    } /* end for u */
  img = PhongShading(cxt);

  return(img);
}

Image *RaycastGrayScene(Scene *scn, Context *cxt, Scene *nscn)
{
  Image *img=NULL;
  int i,j,u,v,u1,v1,k1;
  Point p,p0;
  Vector n;
  float dx=0.,dy=0.,dz=0.,advance_factor,aux;
  Voxel w,w1;

  k1  = -cxt->zbuff->usize/2;
  n.x = cxt->IR[0][2];
  n.y = cxt->IR[1][2];
  n.z = cxt->IR[2][2];
  switch (cxt->PAxis) {
  case 'x':
    if (n.x > 0) 
      dx = 1.;
    else
      dx = -1.;
    dy = dx*(n.y/n.x); /* correcao */
    dz = dx*(n.z/n.x);
    break;
  case 'y':
    if (n.y > 0) 
      dy = 1.;
    else
      dy = -1.;
    dx = dy*(n.x/n.y);
    dz = dy*(n.z/n.y);
    break;
  case 'z':
    if (n.z > 0) 
      dz = 1.;
    else
      dz = -1.;
    dx = dz*(n.x/n.z);
    dy = dz*(n.y/n.z);
    break;
  }
  
  for (v=0; v < cxt->zbuff->vsize; v++)
    for (u=0; u < cxt->zbuff->usize; u++){
      j   = u+cxt->zbuff->tbv[v];
      cxt->zbuff->object[j] = 0;
      cxt->zbuff->voxel[j]  = NIL;
      cxt->zbuff->dist[j]   = cxt->zbuff->usize;
      u1 = u - cxt->zbuff->usize/2;
      v1 = v - cxt->zbuff->vsize/2;     
      p.x = u1*cxt->IR[0][0] + v1*cxt->IR[0][1] + k1*cxt->IR[0][2]; 
      p.y = u1*cxt->IR[1][0] + v1*cxt->IR[1][1] + k1*cxt->IR[1][2];
      p.z = u1*cxt->IR[2][0] + v1*cxt->IR[2][1] + k1*cxt->IR[2][2];

      p0.x = p.x;
      p0.y = p.y;
      p0.z = p.z;
      /* p0 pertence ao view plane */

      /* procura-se p, o ponto de intersecçao com a cena */
      advance_factor = 0.;

      if ((n.x > Epsilon) || (n.x < -Epsilon)) {
	if (p0.x > cxt->xsize/2 - 1) {
	  p.x = cxt->xsize/2 - 1; 
	  advance_factor  = (p.x - p0.x)/n.x;
	  }
	else
	  if (p0.x < -cxt->xsize/2) {
	    p.x = -cxt->xsize/2;
	    advance_factor  = (p.x - p0.x)/n.x;
	  }
      } else {
	if ((p0.x > cxt->xsize/2-1)||(p0.x < -cxt->xsize/2))
	  goto caifora;
      } 

      if ((n.y > Epsilon) || (n.y < -Epsilon)) {
	if (p0.y > cxt->ysize/2 - 1) {
	  p.y = cxt->ysize/2 - 1;
	  aux = (p.y - p0.y)/n.y;
	  if (advance_factor < aux)
	    advance_factor = aux;
	}
	else
	  if (p0.y < -cxt->ysize/2) {
	    p.y = -cxt->ysize/2;
	    aux = (p.y - p0.y)/n.y;	    
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.y > cxt->ysize/2-1)||(p0.y < -cxt->ysize/2))
	  goto caifora;
      } 
      
      if ((n.z > Epsilon) || (n.z < -Epsilon)) {
      	if (p0.z > cxt->zsize/2 - 1) {
	  p.z = cxt->zsize/2 - 1; /* cuidado com dim. impares */
	  aux = (p.z - p0.z)/n.z;
	  if (advance_factor < aux )
	    advance_factor = aux;
	}
	else
	  if (p0.z < -cxt->zsize/2) {
	    p.z = -cxt->zsize/2; 
	    aux = (p.z - p0.z)/n.z;   
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.z > cxt->zsize/2-1)||(p0.z < -cxt->zsize/2))
	  goto caifora;
      } 

      p.x = p0.x + advance_factor * n.x;
      p.y = p0.y + advance_factor * n.y;
      p.z = p0.z + advance_factor * n.z;

      w1.x = ROUND(p.x); /* o (ROUND) seria melhor que o (int) */
      w1.y = ROUND(p.y);
      w1.z = ROUND(p.z);
      w.x  = w1.x + cxt->xsize/2;
      w.y  = w1.y + cxt->ysize/2;
      w.z  = w1.z + cxt->zsize/2;

      while ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&& 
	     (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&& 
	     (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)) {
		  
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	if (scn->data[i] != 0){
	  cxt->zbuff->dist[j] = sqrt((w1.x-p0.x)*(w1.x-p0.x) +
				     (w1.y-p0.y)*(w1.y-p0.y) +
				     (w1.z-p0.z)*(w1.z-p0.z));
	  cxt->zbuff->voxel[j] = i;
	  cxt->zbuff->object[j] = scn->data[i];
	  break;
	}
	else {
	  p.x += dx; 
	  p.y += dy;
	  p.z += dz;
	  w1.x = (int)p.x;
	  w1.y = (int)p.y;
	  w1.z = (int)p.z;
	  w.x  = w1.x + cxt->xsize/2;
	  w.y  = w1.y + cxt->ysize/2;
	  w.z  = w1.z + cxt->zsize/2;
	}
      } /* end while */
    caifora:
      ;
    } /* end for u */
  img = PhongShadingScene(cxt,nscn);

  return(img);
}

Image *BRaycastScene(Scene *scn, Context *cxt)
{
  /* classic raycasting using 3D Bresenham´s algorithm */

  Image *img=NULL;
  int i,j,u,v,u1,v1,k1;
  Point p,p0,q0;
  Vector n;
  float advance_factor;
  Voxel w,w1;
  int lx,ly,lz,alx,aly,alz,x_inc,y_inc,z_inc,err_1,err_2,lx2,ly2,lz2;

  k1  = -cxt->zbuff->usize/2;
  n.x = cxt->IR[0][2];
  n.y = cxt->IR[1][2];
  n.z = cxt->IR[2][2];

  x_inc = (n.x < 0) ? -1 : 1; /* sign of integer Bresenham increment */
  y_inc = (n.y < 0) ? -1 : 1;
  z_inc = (n.z < 0) ? -1 : 1;

  lx = ROUND(n.x * 10 * cxt->zbuff->usize); /* ray-vector components*/
  ly = ROUND(n.y * 10 * cxt->zbuff->usize);
  lz = ROUND(n.z * 10 * cxt->zbuff->usize);

  alx = abs(lx);
  aly = abs(ly);
  alz = abs(lz);

  lx2 = (alx) << 1; /* multiplicate by 2 */
  ly2 = (aly) << 1;
  lz2 = (alz) << 1;


  switch (cxt->PAxis) {
  case 'x':

  for (v=0; v < cxt->zbuff->vsize; v++)
    for (u=0; u < cxt->zbuff->usize; u++){
      j   = u+cxt->zbuff->tbv[v];
      cxt->zbuff->object[j] = 0;
      cxt->zbuff->voxel[j]  = NIL;
      cxt->zbuff->dist[j]   = cxt->zbuff->usize;
      u1 = u - cxt->zbuff->usize/2;
      v1 = v - cxt->zbuff->vsize/2;     
      p.x = u1*cxt->IR[0][0] + v1*cxt->IR[0][1] + k1*cxt->IR[0][2]; 
      p.y = u1*cxt->IR[1][0] + v1*cxt->IR[1][1] + k1*cxt->IR[1][2];
      p.z = u1*cxt->IR[2][0] + v1*cxt->IR[2][1] + k1*cxt->IR[2][2];

      p0.x = p.x;      /* p0 pertence ao view plane, eh a origem do raio em coord. esp. objeto */
      p0.y = p.y;
      p0.z = p.z;

      q0.x = p0.x + cxt->xsize/2; /* eh a origem do raio em coord. esp. voxel */
      q0.y = p0.y + cxt->ysize/2;
      q0.z = p0.z + cxt->zsize/2;

      /* procura-se p, o ponto de intersecçao com a cena */
      advance_factor = 0.;

      if ((n.x > Epsilon) || (n.x < -Epsilon)) {
	if (p0.x > (cxt->xsize/2 - 1)) {
	  p.x = cxt->xsize/2 - 1; /* para funcionar com dimensoes impares teria que botar cxt->xsize/2 - 1 + cxt->xsize%2 */
	  advance_factor  = (p.x - p0.x)/n.x;
	  }
	else
	  if (p0.x < -cxt->xsize/2) {
	    p.x = -cxt->xsize/2;
	    advance_factor  = (p.x - p0.x)/n.x;
	  }
      }

      if ((n.y > Epsilon) || (n.y < -Epsilon)) {
	if (p0.y > (cxt->ysize/2 - 1)) {
	  p.y = cxt->ysize/2 - 1; /* cuidado com dim. impares */
	  if (advance_factor < (p.y - p0.y)/n.y)
         	   advance_factor = (p.y - p0.y)/n.y;
	}
	else
	  if (p0.y < -cxt->ysize/2) {
	    p.y = -cxt->ysize/2;
	    if (advance_factor < (p.y - p0.y)/n.y)
	      advance_factor = (p.y - p0.y)/n.y;
	  }
      }
      
      if ((n.z > Epsilon) || (n.z < -Epsilon)) {
	if (p0.z > (cxt->zsize/2 - 1)) {
	  p.z = cxt->zsize/2 - 1; /* cuidado com dim. impares */
	  if (advance_factor < (p.z - p0.z)/n.z)
	    advance_factor = (p.z - p0.z)/n.z;
	}
	else
	  if (p0.z < -cxt->zsize/2) {
	    p.z = -cxt->zsize/2;    
	    if (advance_factor < (p.z - p0.z)/n.z)
	      advance_factor = (p.z - p0.z)/n.z;
	  }
      }

      p.x = p0.x + advance_factor * n.x;
      p.y = p0.y + advance_factor * n.y;
      p.z = p0.z + advance_factor * n.z;

      w1.x = ROUND(p.x); 
      w1.y = ROUND(p.y);
      w1.z = ROUND(p.z);
      w.x  = w1.x + cxt->xsize/2;
      w.y  = w1.y + cxt->ysize/2;
      w.z  = w1.z + cxt->zsize/2;

      err_1 = ly2 - alx;
      err_2 = lz2 - alx;

      while ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&& 
	     (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&& 
	     (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)) {
		  
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	if (scn->data[i] != 0){
	  cxt->zbuff->dist[j] = sqrt((w.x-q0.x)*(w.x-q0.x) +
				     (w.y-q0.y)*(w.y-q0.y) +
				     (w.z-q0.z)*(w.z-q0.z));
	  /* distance distortion may occure because of integer Bresenham coordinates */
	  cxt->zbuff->voxel[j] = i;
	  cxt->zbuff->object[j] = scn->data[i];
	  break;
	}
	else {
	  if (err_1 > 0) {
	    w.y += y_inc;
	    err_1 -= lx2;
	  }
	  if (err_2 > 0) {
	    w.z += z_inc;
	    err_2 -= lx2;
	  }

	  err_1 += ly2;
	  err_2 += lz2;
	  w.x += x_inc;
	} /* end else */
      } /* end while */
    } /* end for u */

    break;


  case 'y':
    
  for (v=0; v < cxt->zbuff->vsize; v++)
    for (u=0; u < cxt->zbuff->usize; u++){
      j   = u+cxt->zbuff->tbv[v];
      cxt->zbuff->object[j] = 0;
      cxt->zbuff->voxel[j]  = NIL;
      cxt->zbuff->dist[j]   = cxt->zbuff->usize;
      u1 = u - cxt->zbuff->usize/2;
      v1 = v - cxt->zbuff->vsize/2;     
      p.x = u1*cxt->IR[0][0] + v1*cxt->IR[0][1] + k1*cxt->IR[0][2]; 
      p.y = u1*cxt->IR[1][0] + v1*cxt->IR[1][1] + k1*cxt->IR[1][2];
      p.z = u1*cxt->IR[2][0] + v1*cxt->IR[2][1] + k1*cxt->IR[2][2];

      p0.x = p.x;      /* p0 pertence ao view plane, eh a origem do raio em coord. esp. objeto */
      p0.y = p.y;
      p0.z = p.z;

      q0.x = p0.x + cxt->xsize/2; /* eh a origem do raio em coord. esp. voxel */
      q0.y = p0.y + cxt->ysize/2;
      q0.z = p0.z + cxt->zsize/2;

      /* procura-se p, o ponto de intersecçao com a cena */
      advance_factor = 0.;

      if ((n.x > Epsilon) || (n.x < -Epsilon)) {
	if (p0.x > (cxt->xsize/2 - 1)) {
	  p.x = cxt->xsize/2 - 1; /* para funcionar com dimensoes impares teria que botar cxt->xsize/2 - 1 + cxt->xsize%2 */
	  advance_factor  = (p.x - p0.x)/n.x;
	  }
	else
	  if (p0.x < -cxt->xsize/2) {
	    p.x = -cxt->xsize/2;
	    advance_factor  = (p.x - p0.x)/n.x;
	  }
      }

      if ((n.y > Epsilon) || (n.y < -Epsilon)) {
	if (p0.y > (cxt->ysize/2 - 1)) {
	  p.y = cxt->ysize/2 - 1; /* cuidado com dim. impares */
	  if (advance_factor < (p.y - p0.y)/n.y)
         	   advance_factor = (p.y - p0.y)/n.y;
	}
	else
	  if (p0.y < -cxt->ysize/2) {
	    p.y = -cxt->ysize/2;
	    if (advance_factor < (p.y - p0.y)/n.y)
	      advance_factor = (p.y - p0.y)/n.y;
	  }
      }
      
      if ((n.z > Epsilon) || (n.z < -Epsilon)) {
	if (p0.z > (cxt->zsize/2 - 1)) {
	  p.z = cxt->zsize/2 - 1; /* cuidado com dim. impares */
	  if (advance_factor < (p.z - p0.z)/n.z)
	    advance_factor = (p.z - p0.z)/n.z;
	}
	else
	  if (p0.z < -cxt->zsize/2) {
	    p.z = -cxt->zsize/2;    
	    if (advance_factor < (p.z - p0.z)/n.z)
	      advance_factor = (p.z - p0.z)/n.z;
	  }
      }

      p.x = p0.x + advance_factor * n.x;
      p.y = p0.y + advance_factor * n.y;
      p.z = p0.z + advance_factor * n.z;

      w1.x = ROUND(p.x); 
      w1.y = ROUND(p.y);
      w1.z = ROUND(p.z);
      w.x  = w1.x + cxt->xsize/2;
      w.y  = w1.y + cxt->ysize/2;
      w.z  = w1.z + cxt->zsize/2;
 
      err_1 = lx2 - aly;
      err_2 = lz2 - aly;

      while ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&& 
	     (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&& 
	     (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)) {
		  
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	if (scn->data[i] != 0){
	  cxt->zbuff->dist[j] = sqrt((w.x-q0.x)*(w.x-q0.x) +
				     (w.y-q0.y)*(w.y-q0.y) +
				     (w.z-q0.z)*(w.z-q0.z));
	  /* distance distortion may occure because of integer Bresenham coordinates */
	  cxt->zbuff->voxel[j] = i;
	  cxt->zbuff->object[j] = scn->data[i];
	  break;
	}
	else {
	  if (err_1 > 0) {
	    w.x += x_inc;
	    err_1 -= ly2;
	  }
	  if (err_2 > 0) {
	    w.z += z_inc;
	    err_2 -= ly2;
	  }
	  err_1 += lx2;
	  err_2 += lz2;
	  w.y += y_inc;
	} /* end else */
      } /* end while */
    } /* end for u */

    break;


  case 'z':
    
  for (v=0; v < cxt->zbuff->vsize; v++)
    for (u=0; u < cxt->zbuff->usize; u++){
      j   = u+cxt->zbuff->tbv[v];
      cxt->zbuff->object[j] = 0;
      cxt->zbuff->voxel[j]  = NIL;
      cxt->zbuff->dist[j]   = cxt->zbuff->usize;
      u1 = u - cxt->zbuff->usize/2;
      v1 = v - cxt->zbuff->vsize/2;     
      p.x = u1*cxt->IR[0][0] + v1*cxt->IR[0][1] + k1*cxt->IR[0][2]; 
      p.y = u1*cxt->IR[1][0] + v1*cxt->IR[1][1] + k1*cxt->IR[1][2];
      p.z = u1*cxt->IR[2][0] + v1*cxt->IR[2][1] + k1*cxt->IR[2][2];

      p0.x = p.x;      /* p0 pertence ao view plane, eh a origem do raio em coord. esp. objeto */
      p0.y = p.y;
      p0.z = p.z;

      q0.x = p0.x + cxt->xsize/2; /* eh a origem do raio em coord. esp. voxel */
      q0.y = p0.y + cxt->ysize/2;
      q0.z = p0.z + cxt->zsize/2;

      /* procura-se p, o ponto de intersecçao com a cena */
      advance_factor = 0.;

      if ((n.x > Epsilon) || (n.x < -Epsilon)) {
	if (p0.x > (cxt->xsize/2 - 1)) {
	  p.x = cxt->xsize/2 - 1; /* para funcionar com dimensoes impares teria que botar cxt->xsize/2 - 1 + cxt->xsize%2 */
	  advance_factor  = (p.x - p0.x)/n.x;
	  }
	else
	  if (p0.x < -cxt->xsize/2) {
	    p.x = -cxt->xsize/2;
	    advance_factor  = (p.x - p0.x)/n.x;
	  }
      }

      if ((n.y > Epsilon) || (n.y < -Epsilon)) {
	if (p0.y > (cxt->ysize/2 - 1)) {
	  p.y = cxt->ysize/2 - 1; /* cuidado com dim. impares */
	  if (advance_factor < (p.y - p0.y)/n.y)
         	   advance_factor = (p.y - p0.y)/n.y;
	}
	else
	  if (p0.y < -cxt->ysize/2) {
	    p.y = -cxt->ysize/2;
	    if (advance_factor < (p.y - p0.y)/n.y)
	      advance_factor = (p.y - p0.y)/n.y;
	  }
      }
      
      if ((n.z > Epsilon) || (n.z < -Epsilon)) {
	if (p0.z > (cxt->zsize/2 - 1)) {
	  p.z = cxt->zsize/2 - 1; /* cuidado com dim. impares */
	  if (advance_factor < (p.z - p0.z)/n.z)
	    advance_factor = (p.z - p0.z)/n.z;
	}
	else
	  if (p0.z < -cxt->zsize/2) {
	    p.z = -cxt->zsize/2;    
	    if (advance_factor < (p.z - p0.z)/n.z)
	      advance_factor = (p.z - p0.z)/n.z;
	  }
      }

      p.x = p0.x + advance_factor * n.x;
      p.y = p0.y + advance_factor * n.y;
      p.z = p0.z + advance_factor * n.z;

      w1.x = ROUND(p.x); 
      w1.y = ROUND(p.y);
      w1.z = ROUND(p.z);
      w.x  = w1.x + cxt->xsize/2;
      w.y  = w1.y + cxt->ysize/2;
      w.z  = w1.z + cxt->zsize/2;

      err_1 = ly2 - alz;
      err_2 = lx2 - alz;
      
      while ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&& 
	     (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&& 
	     (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)) {
	
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	if (scn->data[i] != 0){
	  cxt->zbuff->dist[j] = sqrt((w.x-q0.x)*(w.x-q0.x) +
				     (w.y-q0.y)*(w.y-q0.y) +
				     (w.z-q0.z)*(w.z-q0.z));
	  /* distance distortion may occure because of integer Bresenham coordinates */
	  cxt->zbuff->voxel[j] = i;
	  cxt->zbuff->object[j] = scn->data[i];
	  break;
	}
	else {
	  if (err_1 > 0) {
	    w.y += y_inc;
	    err_1 -= lz2;
	  }
	  if (err_2 > 0) {
	    w.x += x_inc;
	    err_2 -= lz2;
	  }
	  err_1 += ly2;
	  err_2 += lx2;
	  w.z += z_inc;	
	} /* end else */
      } /* end while */
    } /* end for u */

    break;
  } /* end switch */


  img = PhongShading(cxt);
  return(img);
}


Image *RaytracingScene(Scene *scn, Scene *alpha, Scene *normal, Vector *normaltable, Context *cxt)
{
  float amb,dist,idist,diff,spec,cos_a,cos_2a,pow,shading;
  float dx,dy,dz,advance_factor,aux;
  float nopac[256];
  int i,j,k,u,v,u1,v1,k1,ni;
  Point p,p0,bt,st;
  Vector n;
  Voxel w,w1;
  Image *img=NULL;
  FBuffer *cumopac = NULL;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;
  
  img= CreateImage(cxt->width, cxt->height);
  cumopac = CreateFBuffer(cxt->width, cxt->height);  
  InitFBuffer(cumopac,1.0);
  
  dx = dy = dz = 0.0;

  // scene translation
  st.x = cxt->xsize/2;
  st.y = cxt->ysize/2;
  st.z = cxt->zsize/2;

  // buffer translation
  bt.x = cxt->width/2;
  bt.y = cxt->height/2;
  bt.z = cxt->depth/2;

  k1  = -bt.x;
  n.x = cxt->IR[0][2];
  n.y = cxt->IR[1][2];
  n.z = cxt->IR[2][2];

  switch (cxt->PAxis) {
  case 'x':
    if (n.x > 0) 
      dx = 1.;
    else
      dx = -1.;
    dy = dx*(n.y/n.x); /* correcao */
    dz = dx*(n.z/n.x);
    break;
  case 'y':
    if (n.y > 0) 
      dy = 1.;
    else
      dy = -1.;
    dx = dy*(n.x/n.y);
    dz = dy*(n.z/n.y);
    break;
  case 'z':
    if (n.z > 0) 
      dz = 1.;
    else
      dz = -1.;
    dx = dz*(n.x/n.z);
    dy = dz*(n.y/n.z);
    break;
  }
  
  for (v=0; v <cxt->height; v++)
    for (u=0; u <cxt->width; u++){
      j  = u+img->tbrow[v];
      u1 = u - bt.x;
      v1 = v - bt.y;     
      p.x = u1*cxt->IR[0][0] + v1*cxt->IR[0][1] + k1*cxt->IR[0][2]; 
      p.y = u1*cxt->IR[1][0] + v1*cxt->IR[1][1] + k1*cxt->IR[1][2];
      p.z = u1*cxt->IR[2][0] + v1*cxt->IR[2][1] + k1*cxt->IR[2][2];

      p0.x = p.x;
      p0.y = p.y;
      p0.z = p.z;
      /* p0 pertence ao view plane */

      /* procura-se p, o ponto de intersecçao com a cena */
      advance_factor = 0.;

      if ((n.x > Epsilon) || (n.x < -Epsilon)) {
	if (p0.x > st.x - 1) {
	  p.x = st.x - 1; 
	  advance_factor  = (p.x - p0.x)/n.x;
	  }
	else
	  if (p0.x < -st.x) {
	    p.x = -st.x;
	    advance_factor  = (p.x - p0.x)/n.x;
	  }
      } else {
	if ((p0.x > st.x-1)||(p0.x < -st.x))
	  goto caifora;
      } 

      if ((n.y > Epsilon) || (n.y < -Epsilon)) {
	if (p0.y > st.y - 1) {
	  p.y = st.y - 1;
	  aux = (p.y - p0.y)/n.y;
	  if (advance_factor < aux)
	    advance_factor = aux;
	}
	else
	  if (p0.y < -st.y) {
	    p.y = -st.y;
	    aux = (p.y - p0.y)/n.y;	    
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.y > st.y-1)||(p0.y < -st.y))
	  goto caifora;
      } 
      
      if ((n.z > Epsilon) || (n.z < -Epsilon)) {
      	if (p0.z > st.z - 1) {
	  p.z = st.z - 1; /* cuidado com dim. impares */
	  aux = (p.z - p0.z)/n.z;
	  if (advance_factor < aux )
	    advance_factor = aux;
	}
	else
	  if (p0.z < -st.z) {
	    p.z = -st.z; 
	    aux = (p.z - p0.z)/n.z;   
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.z > st.z-1)||(p0.z < -st.z))
	  goto caifora;
      } 

      p.x = p0.x + advance_factor * n.x;
      p.y = p0.y + advance_factor * n.y;
      p.z = p0.z + advance_factor * n.z;

      w1.x = ROUND(p.x);
      w1.y = ROUND(p.y);
      w1.z = ROUND(p.z);
      w.x  = w1.x + st.x;
      w.y  = w1.y + st.y;
      w.z  = w1.z + st.z;

      while ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&& 
	     (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&& 
	     (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)) {
	
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];

	if (alpha->data[i] != 0 && scn->data[i] != 0){
	 
	  ni = normal->data[i];
	  	   
	  cos_a = fabs(cxt->viewer.x * normaltable[ni].x +\
		       cxt->viewer.y * normaltable[ni].y +\
		       cxt->viewer.z * normaltable[ni].z);

	  if (cos_a > 0.0) {
	  
	    dist = sqrt((w1.x-p0.x)*(w1.x-p0.x) +
			(w1.y-p0.y)*(w1.y-p0.y) +
			(w1.z-p0.z)*(w1.z-p0.z));
	    
	    idist = 1.0 - dist/(float)cxt->depth;
	    
	    cos_2a = 2*cos_a*cos_a - 1.0;
	    
	    if (cos_2a < 0.) {
	      pow = 0.;
	      spec = 0.;
	    } else {
	      pow = 1.;
	      for (k=0; k < cxt->obj[0].ns; k++) 
		pow = pow * cos_2a;
	      spec = cxt->obj[0].spec * pow;
	    }
	    
	    amb  = cxt->obj[0].amb;
	    diff = cxt->obj[0].diff * cos_a;
	    
	    
	    shading = (amb + idist * (diff + spec));
	    img->val[j] += (int)((float)(alpha->data[i]) * shading * cumopac->val[j]);
	    cumopac->val[j] *= 1.0 - nopac[alpha->data[i]];	  
	  }
	}
	p.x += dx; 
	p.y += dy;
	p.z += dz;
	w1.x = ROUND(p.x);
	w1.y = ROUND(p.y);
	w1.z = ROUND(p.z);
	w.x  = w1.x + st.x;
	w.y  = w1.y + st.y;
	w.z  = w1.z + st.z;
	  
      } /* end while */
    caifora:
      ;
    } /* end for u */
  DestroyFBuffer(&cumopac);
  return(img);

}

CImage *CRaytracingScene(Scene *scn, Scene *alpha, Scene *normal, Vector *normaltable, Context *cxt)
{
  float amb,dist,idist,diff,spec,cos_a,cos_2a,pow,shading;
  float dx,dy,dz,advance_factor,aux;
  float nopac[256];
  int i,j,k,u,v,u1,v1,k1,ni;
  Point p,p0,bt,st;
  Vector n;
  Voxel w,w1;
  CImage *cimg=NULL;
  FBuffer *cumopac = NULL;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;
  
  cimg= CreateCImage(cxt->width, cxt->height);
  cumopac = CreateFBuffer(cxt->width, cxt->height);  
  InitFBuffer(cumopac,1.0);
  
  dx = dy = dz = 0.0;

  // scene translation
  st.x = cxt->xsize/2;
  st.y = cxt->ysize/2;
  st.z = cxt->zsize/2;

  // buffer translation
  bt.x = cxt->width/2;
  bt.y = cxt->height/2;
  bt.z = cxt->depth/2;

  k1  = -bt.x;
  n.x = cxt->IR[0][2];
  n.y = cxt->IR[1][2];
  n.z = cxt->IR[2][2];

  switch (cxt->PAxis) {
  case 'x':
    if (n.x > 0) 
      dx = 1.;
    else
      dx = -1.;
    dy = dx*(n.y/n.x); /* correcao */
    dz = dx*(n.z/n.x);
    break;
  case 'y':
    if (n.y > 0) 
      dy = 1.;
    else
      dy = -1.;
    dx = dy*(n.x/n.y);
    dz = dy*(n.z/n.y);
    break;
  case 'z':
    if (n.z > 0) 
      dz = 1.;
    else
      dz = -1.;
    dx = dz*(n.x/n.z);
    dy = dz*(n.y/n.z);
    break;
  }
  
  for (v=0; v <cxt->height; v++)
    for (u=0; u <cxt->width; u++){
      j  = u+ cimg->C[0]->tbrow[v];
      u1 = u - bt.x;
      v1 = v - bt.y;     
      p.x = u1*cxt->IR[0][0] + v1*cxt->IR[0][1] + k1*cxt->IR[0][2]; 
      p.y = u1*cxt->IR[1][0] + v1*cxt->IR[1][1] + k1*cxt->IR[1][2];
      p.z = u1*cxt->IR[2][0] + v1*cxt->IR[2][1] + k1*cxt->IR[2][2];

      p0.x = p.x;
      p0.y = p.y;
      p0.z = p.z;
      /* p0 pertence ao view plane */

      /* procura-se p, o ponto de intersecçao com a cena */
      advance_factor = 0.;

      if ((n.x > Epsilon) || (n.x < -Epsilon)) {
	if (p0.x > st.x - 1) {
	  p.x = st.x - 1; 
	  advance_factor  = (p.x - p0.x)/n.x;
	  }
	else
	  if (p0.x < -st.x) {
	    p.x = -st.x;
	    advance_factor  = (p.x - p0.x)/n.x;
	  }
      } else {
	if ((p0.x > st.x-1)||(p0.x < -st.x))
	  goto caifora;
      } 

      if ((n.y > Epsilon) || (n.y < -Epsilon)) {
	if (p0.y > st.y - 1) {
	  p.y = st.y - 1;
	  aux = (p.y - p0.y)/n.y;
	  if (advance_factor < aux)
	    advance_factor = aux;
	}
	else
	  if (p0.y < -st.y) {
	    p.y = -st.y;
	    aux = (p.y - p0.y)/n.y;	    
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.y > st.y-1)||(p0.y < -st.y))
	  goto caifora;
      } 
      
      if ((n.z > Epsilon) || (n.z < -Epsilon)) {
      	if (p0.z > st.z - 1) {
	  p.z = st.z - 1; /* cuidado com dim. impares */
	  aux = (p.z - p0.z)/n.z;
	  if (advance_factor < aux )
	    advance_factor = aux;
	}
	else
	  if (p0.z < -st.z) {
	    p.z = -st.z; 
	    aux = (p.z - p0.z)/n.z;   
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.z > st.z-1)||(p0.z < -st.z))
	  goto caifora;
      } 

      p.x = p0.x + advance_factor * n.x;
      p.y = p0.y + advance_factor * n.y;
      p.z = p0.z + advance_factor * n.z;

      w1.x = ROUND(p.x);
      w1.y = ROUND(p.y);
      w1.z = ROUND(p.z);
      w.x  = w1.x + st.x;
      w.y  = w1.y + st.y;
      w.z  = w1.z + st.z;

      while ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&& 
	     (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&& 
	     (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)) {
	
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];

	if (alpha->data[i] != 0 && scn->data[i] != 0){
	 
	  ni = normal->data[i];
	  	   
	  cos_a = cxt->viewer.x * normaltable[ni].x +\
	    cxt->viewer.y * normaltable[ni].y +\
	    cxt->viewer.z * normaltable[ni].z;

	  if (cos_a > 0.0) {
	  
	    dist = sqrt((w1.x-p0.x)*(w1.x-p0.x) +
			(w1.y-p0.y)*(w1.y-p0.y) +
			(w1.z-p0.z)*(w1.z-p0.z));
	    
	    idist = 1.0 - dist/(float)cxt->depth;
	    
	    cos_2a = 2*cos_a*cos_a - 1.0;
	    
	    if (cos_2a < 0.) {
	      pow = 0.;
	      spec = 0.;
	    } else {
	      pow = 1.;
	      for (k=0; k < cxt->obj[0].ns; k++) 
		pow = pow * cos_2a;
	      spec = cxt->obj[0].spec * pow;
	    }
	    
	    amb  = cxt->obj[0].amb;
	    diff = cxt->obj[0].diff * cos_a;
	    
	    
	    shading = (amb + idist * (diff + spec));
	    AccVoxelColor(cimg,cxt,j,shading,(float)alpha->data[i]* cumopac->val[j],0);
	    cumopac->val[j] *= 1.0 - nopac[alpha->data[i]];	  
	  }
	}
	p.x += dx; 
	p.y += dy;
	p.z += dz;
	w1.x = ROUND(p.x);
	w1.y = ROUND(p.y);
	w1.z = ROUND(p.z);
	w.x  = w1.x + st.x;
	w.y  = w1.y + st.y;
	w.z  = w1.z + st.z;
	  
      } /* end while */
    caifora:
      ;
    } /* end for u */
  DestroyFBuffer(&cumopac);
  return(cimg);

}

Image *MIPScene(Scene *scn, Context *cxt)
{
  /*WITH BUG*/
  Image *img=NULL;
  int i,j,u,v,u1,v1,k1;
  Point p,p0;
  Vector n;
  float dx=0.,dy=0.,dz=0.,advance_factor,aux;
  Voxel w,w1;

  img = CreateImage(cxt->width,cxt->height);

  k1  = -cxt->width/2;
  n.x = cxt->IR[0][2];
  n.y = cxt->IR[1][2];
  n.z = cxt->IR[2][2];
  switch (cxt->PAxis) {
  case 'x':
    if (n.x > 0) 
      dx = 1.;
    else
      dx = -1.;
    dy = dx*(n.y/n.x); /* correcao */
    dz = dx*(n.z/n.x);
    break;
  case 'y':
    if (n.y > 0) 
      dy = 1.;
    else
      dy = -1.;
    dx = dy*(n.x/n.y);
    dz = dy*(n.z/n.y);
    break;
  case 'z':
    if (n.z > 0) 
      dz = 1.;
    else
      dz = -1.;
    dx = dz*(n.x/n.z);
    dy = dz*(n.y/n.z);
    break;
  }
  
  for (v=0; v < cxt->height; v++)
    for (u=0; u < cxt->width; u++){
      j = u+img->tbrow[v];
      u1 = u - cxt->width/2;
      v1 = v - cxt->height/2;     
      p.x = u1*cxt->IR[0][0] + v1*cxt->IR[0][1] + k1*cxt->IR[0][2]; 
      p.y = u1*cxt->IR[1][0] + v1*cxt->IR[1][1] + k1*cxt->IR[1][2];
      p.z = u1*cxt->IR[2][0] + v1*cxt->IR[2][1] + k1*cxt->IR[2][2];

      p0.x = p.x;
      p0.y = p.y;
      p0.z = p.z;
      /* p0 pertence ao view plane */

      /* procura-se p, o ponto de intersecçao com a cena */
      advance_factor = 0.;

      if ((n.x > Epsilon) || (n.x < -Epsilon)) {
	if (p0.x > cxt->xsize/2 - 1) {
	  p.x = cxt->xsize/2 - 1; 
	  advance_factor  = (p.x - p0.x)/n.x;
	  }
	else
	  if (p0.x < -cxt->xsize/2) {
	    p.x = -cxt->xsize/2;
	    advance_factor  = (p.x - p0.x)/n.x;
	  }
      } else {
	if ((p0.x > cxt->xsize/2-1)||(p0.x < -cxt->xsize/2))
	  goto caifora;
      } 

      if ((n.y > Epsilon) || (n.y < -Epsilon)) {
	if (p0.y > cxt->ysize/2 - 1) {
	  p.y = cxt->ysize/2 - 1;
	  aux = (p.y - p0.y)/n.y;
	  if (advance_factor < aux)
	    advance_factor = aux;
	}
	else
	  if (p0.y < -cxt->ysize/2) {
	    p.y = -cxt->ysize/2;
	    aux = (p.y - p0.y)/n.y;	    
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.y > cxt->ysize/2-1)||(p0.y < -cxt->ysize/2))
	  goto caifora;
      } 
      
      if ((n.z > Epsilon) || (n.z < -Epsilon)) {
      	if (p0.z > cxt->zsize/2 - 1) {
	  p.z = cxt->zsize/2 - 1; /* cuidado com dim. impares */
	  aux = (p.z - p0.z)/n.z;
	  if (advance_factor < aux )
	    advance_factor = aux;
	}
	else
	  if (p0.z < -cxt->zsize/2) {
	    p.z = -cxt->zsize/2; 
	    aux = (p.z - p0.z)/n.z;   
	    if (advance_factor < aux)
	      advance_factor = aux;
	  }
      } else {
	if ((p0.z > cxt->zsize/2-1)||(p0.z < -cxt->zsize/2))
	  goto caifora;
      } 

      p.x = p0.x + advance_factor * n.x;
      p.y = p0.y + advance_factor * n.y;
      p.z = p0.z + advance_factor * n.z;

      w1.x = ROUND(p.x); 
      w1.y = ROUND(p.y);
      w1.z = ROUND(p.z);
      w.x  = w1.x + cxt->xsize/2;
      w.y  = w1.y + cxt->ysize/2;
      w.z  = w1.z + cxt->zsize/2;

      while ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&& 
	     (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)&& 
	     (w.z >= cxt->zmin)&&(w.z <= cxt->zmax)) {
	
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	
	if (scn->data[i] != 0){
	  if (scn->data[i] > img->val[j]) 
	    img->val[j] = scn->data[i];	    
	}
	
	p.x += dx; 
	p.y += dy;
	p.z += dz;
	w1.x = ROUND(p.x);
	w1.y = ROUND(p.y);
	w1.z = ROUND(p.z);
	w.x  = w1.x + cxt->xsize/2;
	w.y  = w1.y + cxt->ysize/2;
	w.z  = w1.z + cxt->zsize/2;
	
      } /* end while */
    caifora:
      ;
    } /* end for u */


  return(img);
}
