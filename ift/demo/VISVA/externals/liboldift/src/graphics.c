#include "graphics.h"
#include "geometry.h"
#include "shading.h"
#include "reslicing.h"

void DrawPixel(Image *img, Pixel p, float d, ZBuffer *zbuff)
{  
  int i;

  if (ValidPixel(img,p.x,p.y)){
    i = p.x+img->tbrow[p.y];
    if (zbuff->dist[i] > d){
      img->val[i] = 255;
      zbuff->dist[i]=d;
      zbuff->voxel[i]=NIL;
    }
  }
}

void DrawLine(Image *img, Pixel p1, Pixel pn, float d1, float dn, \
	      ZBuffer *zbuff)
{
  int dx = abs(pn.x - p1.x), dy = abs (pn.y - p1.y);
  int pk = 2*dy-dx; /* decision variable for lines y=mx+b where -1 <= m <= 1 */
  int twody = 2*dy , twodydx = 2*(dy-dx);
  int qk = 2*dx-dy; /* decision variable for lines y=mx+b where m < -1
                      or m > 1 */
  int twodx = 2*dx ,twodxdy = 2*(dx-dy);
  float d,dd;
  int npts;
  Pixel p,pend;
  bool increase; /* true for quadrants 1 and 2 and false for quadrants
		    3 and 4 */
  
  if(pn.x==p1.x) 
    increase=true;
  else 
    if(pn.x>p1.x){
      if(pn.y>p1.y) /* quadrant 1 */ 
	increase=true;
      else /* quadrant 4 */
	increase=false;
    } else { /* pn.x<p1.x */
      if(pn.y>p1.y) /* quadrant 3 */
	increase=false;
      else /* quadrant 4 */
	increase=true;
    }
  
  /* Find start and end points */
  
 if(dx >= dy){ /* -1 <= m <= 1 */
   if(p1.x > pn.x) {
     p.x    = pn.x;
     p.y    = pn.y;
     d      = dn;
     npts   = p1.x - pn.x + 1;     
     dd = (d1-dn)/(npts-1);
     pend.x = p1.x;
   }
   else { /* p1.x <= pn.x */
     p.x = p1.x;
     p.y = p1.y;
     d   = d1;
     npts = pn.x - p1.x + 1;
     dd = (dn-d1)/(npts-1);
     pend.x = pn.x;
   }
   DrawPixel(img,p,d,zbuff);
   while(p.x < pend.x){
     p.x++;
     if(pk < 0){
       pk +=twody;
     } else{
       if(increase) 
	 p.y++;
       else 
	 p.y--;
       pk += twodydx;
     }
     d += dd; 
     DrawPixel(img,p,d,zbuff);
   }
 } else { /* dx < dy: m > 1 or m < -1 */
  
   if(p1.y > pn.y) {
     p.x = pn.x;
     p.y = pn.y;
     d      = dn;
     npts   = p1.y - pn.y + 1;      
     dd = (d1-dn)/(npts-1);
     pend.y = p1.y;
   }
   else { /* p1.y <= pn.y */
     p.x = p1.x;
     p.y = p1.y;
     d      = d1;
     npts   = pn.y - p1.y + 1;     
     dd = (dn-d1)/(npts-1);
     pend.y = pn.y;
   }	 
   DrawPixel(img,p,d,zbuff);
   while( p.y < pend.y){
     p.y++;
     if( qk < 0){
       qk +=twodx;
     } else{
      if(increase) 
	p.x++;
      else 
	p.x--;
      qk += twodxdy;
     }
     d += dd;
     DrawPixel(img,p,d,zbuff);
    }
 }

}


void IntensifyPixel (Image *img, int x, int y, double w, float d, ZBuffer *zbuff)
{  
  int i;
  double theta;

  if (ValidPixel(img,x,y)){
    i = x+img->tbrow[y];
    if (zbuff->dist[i] > d){
      theta = acos(d);
      img->val[i] = (int)(255.*(2.*theta+d*sin(theta)/3.));
      zbuff->dist[i]=d;
      zbuff->voxel[i]=NIL;
    }
  }
}


void DrawAALine(Image *img, Pixel p1, Pixel pn, float d1, float dn, \
	      ZBuffer *zbuff)
{
  int dx = abs(pn.x - p1.x), dy = abs (pn.y - p1.y);
  int pk = 2*dy-dx; /* decision variable for lines y=mx+b where -1 <= m <= 1 */
  int twody = 2*dy , twodydx = 2*(dy-dx);
  int qk = 2*dx-dy; /* decision variable for lines y=mx+b where m < -1
                      or m > 1 */
  int twodx = 2*dx ,twodxdy = 2*(dx-dy);
  int two_v_dx = 0, two_v_dy = 0;
  double inv_den = 1.0/(2.0*sqrt(dx*dx+dy*dy));
  double two_dx_inv_den = 2.0* dx * inv_den;
  double two_dy_inv_den = 2.0* dy * inv_den;
  float d,dd;
  int npts;
  double D=0.0;
  Pixel p,pend;
  bool increase; /* true for quadrants 1 and 2 and false for quadrants
		    3 and 4 */
  
  if(pn.x==p1.x) 
    increase=true;
  else 
    if(pn.x>p1.x){
      if(pn.y>p1.y) /* quadrant 1 */ 
	increase=true;
      else /* quadrant 4 */
	increase=false;
    } else { /* pn.x<p1.x */
      if(pn.y>p1.y) /* quadrant 3 */
	increase=false;
      else /* quadrant 4 */
	increase=true;
    }
  
  /* Find start and end points */
  
 if(dx >= dy){ /* -1 <= m <= 1 */
   if(p1.x > pn.x) {
     p.x    = pn.x;
     p.y    = pn.y;
     d      = dn;
     npts   = p1.x - pn.x + 1;     
     dd = (d1-dn)/(npts-1);
     pend.x = p1.x;
   }
   else { /* p1.x <= pn.x */
     p.x = p1.x;
     p.y = p1.y;
     d   = d1;
     npts = pn.x - p1.x + 1;
     dd = (dn-d1)/(npts-1);
     pend.x = pn.x;
   }
   IntensifyPixel(img,p.x,p.y,0.0,d,zbuff);
   IntensifyPixel(img,p.x,p.y+1,two_dx_inv_den,d,zbuff);
   IntensifyPixel(img,p.x,p.y-1,two_dx_inv_den,d,zbuff);
   while(p.x < pend.x){
     p.x++;
     if(pk < 0){
       two_v_dx = pk + dx;
       pk +=twody;
     } else{
       if(increase) 
	 p.y++;
       else 
	 p.y--;
       two_v_dx = pk - dx;
       pk += twodydx;
     }
     d += dd; 
     D = two_v_dx * inv_den;
     IntensifyPixel(img,p.x,p.y,D,d,zbuff);
     IntensifyPixel(img,p.x,p.y+1,two_dx_inv_den - D,d,zbuff);
     IntensifyPixel(img,p.x,p.y-1,two_dx_inv_den + D,d,zbuff);
   }
 } else { /* dx < dy: m > 1 or m < -1 */
  
   if(p1.y > pn.y) {
     p.x = pn.x;
     p.y = pn.y;
     d      = dn;
     npts   = p1.y - pn.y + 1;      
     dd = (d1-dn)/(npts-1);
     pend.y = p1.y;
   }
   else { /* p1.y <= pn.y */
     p.x = p1.x;
     p.y = p1.y;
     d      = d1;
     npts   = pn.y - p1.y + 1;     
     dd = (dn-d1)/(npts-1);
     pend.y = pn.y;
   }	 
   IntensifyPixel(img,p.x,p.y,0.0,d,zbuff);
   IntensifyPixel(img,p.x+1,p.y,two_dy_inv_den,d,zbuff);
   IntensifyPixel(img,p.x-1,p.y,two_dy_inv_den,d,zbuff);
   while( p.y < pend.y){
     p.y++;
     if( qk < 0){
       two_v_dy = qk + dy;
       qk +=twodx;
     } else{
      if(increase) 
	p.x++;
      else 
	p.x--;
      two_v_dy = qk - dy;
      qk += twodxdy;
     }
     d += dd;
     D = two_v_dy * inv_den;
     IntensifyPixel(img,p.x,p.y,D,d,zbuff);
     IntensifyPixel(img,p.x+1,p.y,two_dy_inv_den - D,d,zbuff);
     IntensifyPixel(img,p.x-1,p.y,two_dy_inv_den + D,d,zbuff);
   }
 }
}




void DrawFrame(Image *img, Context *cxt)
{
  Pixel p[8];
  float d[8],D;
  int  i;
  
  D = cxt->zbuff->usize;

  for(i=0;i<8;i++){
    p[i].x = (int)(cxt->R[0][0]*cxt->fr->vert[i].x + \
		   cxt->R[0][1]*cxt->fr->vert[i].y + \
		   cxt->R[0][2]*cxt->fr->vert[i].z + \
		   cxt->R[0][3] + D/2);
    p[i].y = (int)(cxt->R[1][0]*cxt->fr->vert[i].x + \
		   cxt->R[1][1]*cxt->fr->vert[i].y + \
		   cxt->R[1][2]*cxt->fr->vert[i].z + \
		   cxt->R[1][3] + D/2);
    d[i] = cxt->R[2][0]*cxt->fr->vert[i].x + \
           cxt->R[2][1]*cxt->fr->vert[i].y + \
           cxt->R[2][2]*cxt->fr->vert[i].z + \
           cxt->R[2][3] + D/2;
  }

  DrawLine(img,p[0],p[1],d[0]-5.0,d[1]-5.0,cxt->zbuff);
  DrawLine(img,p[0],p[2],d[0]-5.0,d[2]-5.0,cxt->zbuff);
  DrawLine(img,p[0],p[4],d[0]-5.0,d[4]-5.0,cxt->zbuff);
  DrawLine(img,p[1],p[3],d[1]-5.0,d[3]-5.0,cxt->zbuff);
  DrawLine(img,p[1],p[5],d[1]-5.0,d[5]-5.0,cxt->zbuff);
  DrawLine(img,p[2],p[3],d[2]-5.0,d[3]-5.0,cxt->zbuff);
  DrawLine(img,p[2],p[6],d[2]-5.0,d[6]-5.0,cxt->zbuff);
  DrawLine(img,p[3],p[7],d[3]-5.0,d[7]-5.0,cxt->zbuff);
  DrawLine(img,p[4],p[5],d[4]-5.0,d[5]-5.0,cxt->zbuff);
  DrawLine(img,p[4],p[6],d[4]-5.0,d[6]-5.0,cxt->zbuff);
  DrawLine(img,p[5],p[7],d[5]-5.0,d[7]-5.0,cxt->zbuff);
  DrawLine(img,p[6],p[7],d[6]-5.0,d[7]-5.0,cxt->zbuff);

}

void DrawAAFrame(Image *img, Context *cxt)
{
  Pixel p[8];
  float d[8],D;
  int  i;
  
  D = cxt->zbuff->usize;

  for(i=0;i<8;i++){
    p[i].x = (int)(cxt->R[0][0]*cxt->fr->vert[i].x + \
		   cxt->R[0][1]*cxt->fr->vert[i].y + \
		   cxt->R[0][2]*cxt->fr->vert[i].z + \
		   cxt->R[0][3] + D/2);
    p[i].y = (int)(cxt->R[1][0]*cxt->fr->vert[i].x + \
		   cxt->R[1][1]*cxt->fr->vert[i].y + \
		   cxt->R[1][2]*cxt->fr->vert[i].z + \
		   cxt->R[1][3] + D/2);
    d[i] = cxt->R[2][0]*cxt->fr->vert[i].x + \
           cxt->R[2][1]*cxt->fr->vert[i].y + \
           cxt->R[2][2]*cxt->fr->vert[i].z + \
           cxt->R[2][3] + D/2;
  }

  DrawAALine(img,p[0],p[1],d[0]-5.0,d[1]-5.0,cxt->zbuff);
  DrawAALine(img,p[0],p[2],d[0]-5.0,d[2]-5.0,cxt->zbuff);
  DrawAALine(img,p[0],p[4],d[0]-5.0,d[4]-5.0,cxt->zbuff);
  DrawAALine(img,p[1],p[3],d[1]-5.0,d[3]-5.0,cxt->zbuff);
  DrawAALine(img,p[1],p[5],d[1]-5.0,d[5]-5.0,cxt->zbuff);
  DrawAALine(img,p[2],p[3],d[2]-5.0,d[3]-5.0,cxt->zbuff);
  DrawAALine(img,p[2],p[6],d[2]-5.0,d[6]-5.0,cxt->zbuff);
  DrawAALine(img,p[3],p[7],d[3]-5.0,d[7]-5.0,cxt->zbuff);
  DrawAALine(img,p[4],p[5],d[4]-5.0,d[5]-5.0,cxt->zbuff);
  DrawAALine(img,p[4],p[6],d[4]-5.0,d[6]-5.0,cxt->zbuff);
  DrawAALine(img,p[5],p[7],d[5]-5.0,d[7]-5.0,cxt->zbuff);
  DrawAALine(img,p[6],p[7],d[6]-5.0,d[7]-5.0,cxt->zbuff);

}

void DrawCutFrame(Image *img, Context *cxt, Plane *pl)
{
  
  Pixel p1,p2;
  float D,d1,d2;
  int  i,j,k,l,m,npts=0;
  Vertex v0,v1,v[4],vert[4];
  Vector v0p0,v1p0;
  float ipnum,ipden,alpha;
  Plane *fpl;

  D = cxt->zbuff->usize;

  /* Compute intersections between each edge and the cut plane */

  for (i=0,j=0; i < 12; i++) {
    v0.x = cxt->fr->vert[cxt->fr->edge[i].vert[0]].x;
    v0.y = cxt->fr->vert[cxt->fr->edge[i].vert[0]].y;
    v0.z = cxt->fr->vert[cxt->fr->edge[i].vert[0]].z;
    v1.x = cxt->fr->vert[cxt->fr->edge[i].vert[1]].x;
    v1.y = cxt->fr->vert[cxt->fr->edge[i].vert[1]].y;
    v1.z = cxt->fr->vert[cxt->fr->edge[i].vert[1]].z;
    ipden = pl->normal.x*(v1.x-v0.x) + pl->normal.y*(v1.y-v0.y) + pl->normal.z*(v1.z-v0.z);
    if (ipden != 0.0){
      ipnum = -(pl->normal.x*(v0.x-pl->po.x) + pl->normal.y*(v0.y-pl->po.y) + \
	pl->normal.z*(v0.z-pl->po.z)); 
      alpha = ipnum/ipden;
      if ((alpha >= 0.0)&&(alpha <= 1.0)){
	v[j].x = v0.x + alpha*(v1.x-v0.x);
	v[j].y = v0.y + alpha*(v1.y-v0.y);
	v[j].z = v0.z + alpha*(v1.z-v0.z);
	j++;
      }
    }
  }
  npts = j;

  /* Draw intersection between the frame and the plane */

  for (i=0; i < npts; i++){
    v0.x = v[i].x; v0.y = v[i].y; v0.z = v[i].z;
    for (k=0,j=(i+1)%npts; k < npts-1; k++,j=(j+1)%npts){
      v1.x = v[j].x; v1.y = v[j].y; v1.z = v[j].z; 

      for (l=0; l < 6; l++) {
	for (m=0; m < 4; m++) {
	  vert[m].x = cxt->fr->vert[cxt->fr->face[l].vert[m]].x;
	  vert[m].y = cxt->fr->vert[cxt->fr->face[l].vert[m]].y;
	  vert[m].z = cxt->fr->vert[cxt->fr->face[l].vert[m]].z;
	}
	fpl = FacePlane(vert,4);
	if (!((fabs(pl->po.x-fpl->po.x)<1.0)&&
	      (fabs(pl->po.y-fpl->po.y)<1.0)&&
	      (fabs(pl->po.z-fpl->po.z)<1.0))){ /* fpl != pl */

	  v0p0.x = v0.x - fpl->po.x; 
	  v0p0.y = v0.y - fpl->po.y; 
	  v0p0.z = v0.z - fpl->po.z; 
	  v1p0.x = v1.x - fpl->po.x; 
	  v1p0.y = v1.y - fpl->po.y; 
	  v1p0.z = v1.z - fpl->po.z; 
	  if ((ScalarProd(v0p0,fpl->normal)==0.0)&&
	      (ScalarProd(v1p0,fpl->normal)==0.0)){
	    p1.x = (int)(cxt->R[0][0]*v0.x + \
			 cxt->R[0][1]*v0.y + \
			 cxt->R[0][2]*v0.z + \
			 cxt->R[0][3] + D/2);
	    p1.y = (int)(cxt->R[1][0]*v0.x + \
			 cxt->R[1][1]*v0.y + \
			 cxt->R[1][2]*v0.z + \
			 cxt->R[1][3] + D/2);
	    d1 = cxt->R[2][0]*v0.x + \
	      cxt->R[2][1]*v0.y + \
	      cxt->R[2][2]*v0.z + \
	      cxt->R[2][3] + D/2;
	    p2.x = (int)(cxt->R[0][0]*v1.x + \
			 cxt->R[0][1]*v1.y + \
			 cxt->R[0][2]*v1.z + \
			 cxt->R[0][3] + D/2);
	    p2.y = (int)(cxt->R[1][0]*v1.x + \
			 cxt->R[1][1]*v1.y + \
			 cxt->R[1][2]*v1.z + \
			 cxt->R[1][3] + D/2);
	    d2 = cxt->R[2][0]*v1.x + \
	      cxt->R[2][1]*v1.y + \
	      cxt->R[2][2]*v1.z + \
	      cxt->R[2][3] + D/2;
	    DrawLine(img,p1,p2,d1-5.0,d2-5.0,cxt->zbuff);
	  }
	}
	  DestroyPlane(&fpl);
      }
    }
  }

  /* Draw frame */

  DrawFrame(img,cxt);
}




void DrawPlane(Scene *scn, Context *cxt, Plane *pl, Image *img)
{
  float dist;
  int k,n,i,j,u,v,u1,v1,u2,v2,shift[3],usize,vsize;
  Voxel w,w1;
  float Rx[4][4],Ry[4][4],R[4][4],d;
  AdjPxl *fprint;
  ZBuffer *tmp;
  int Imax , s;

  Imax = scn->maxval;
  s = 0;
  
  while (Imax > 255) {
    s++; Imax>>=1;
  }
  fprint=AdjPixels(img,cxt->footprint->adj);
  d = sqrt(pl->normal.y*pl->normal.y + pl->normal.z*pl->normal.z);  
  usize = cxt->zbuff->usize;
  vsize = cxt->zbuff->vsize;
  
  tmp = CreateZBuffer(cxt->zbuff->usize,cxt->zbuff->vsize);

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
	  u2=(int)(w1.x*cxt->R[0][0] + w1.y*cxt->R[0][1] + \
		   w1.z*cxt->R[0][2] + tmp->usize/2);
	  v2=(int)(w1.x*cxt->R[1][0] + w1.y*cxt->R[1][1] + \
		   w1.z*cxt->R[1][2] + tmp->vsize/2);
	  dist=(w1.x*cxt->R[2][0] + w1.y*cxt->R[2][1] + w1.z*cxt->R[2][2]) + 
	    tmp->usize/2;
	  k = u2 + tmp->tbv[v2];
	  if (tmp->dist[k] >= dist) {
	    i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	    if ( s > 0 ) {
	      img->val[k]   = scn->data[i]>>s;
	    } else {
	      img->val[k]   = scn->data[i];
	    }	      
	    tmp->dist[k]  = dist;
	    tmp->voxel[k] = i;
	  }
	}
      }
    }
  
  shift[0] = 1; shift[1] = cxt->zbuff->usize; shift[2] = shift[1]+1;
  n = cxt->zbuff->usize*(cxt->zbuff->vsize-1);
  for (i=0; i < n-1; i++) 
    if (tmp->voxel[i] != NIL){
      if (cxt->zbuff->dist[i] > tmp->dist[i]){
	cxt->zbuff->dist[i]  = tmp->dist[i];
	cxt->zbuff->voxel[i] = tmp->voxel[i];
	for (j=0; j < fprint->n; j++){
	  k = i+fprint->dp[j];
	  if (tmp->voxel[k] == NIL){
	    if (cxt->zbuff->dist[k] > tmp->dist[i]){	    
	      cxt->zbuff->dist[k]=tmp->dist[i];
	      cxt->zbuff->voxel[k]=tmp->voxel[i];
	      img->val[k]=img->val[i];
	    }
	  }
	}
      }
    }

  DestroyZBuffer(&tmp);
}



Image *DrawScene(Scene *scn, Context *cxt)
{
  Image *img;
  float dist;
  int i,k,u,v;
  Voxel w,w1;
  int Imax, s;

  Imax = scn->maxval;
  s = 0;
  
  while (Imax > 255) {
    s++; Imax>>=1;
  }
  
  InitZBuffer(cxt->zbuff);
  img = CreateImage(cxt->zbuff->usize,cxt->zbuff->vsize);

  w.x = cxt->xi;
  for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
    for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) {
      i = w.x + scn->tby[w.y] + scn->tbz[w.z];
      w1.x = w.x - cxt->xsize/2;
      w1.y = w.y - cxt->ysize/2;
      w1.z = w.z - cxt->zsize/2;
      u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + \
		 cxt->R[0][2]*w1.z + cxt->R[0][3] + cxt->zbuff->usize/2);
      v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + \
		 cxt->R[1][2]*w1.z + cxt->R[1][3] + cxt->zbuff->vsize/2);
      dist =  (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + 
	       cxt->R[2][2]*w1.z + cxt->R[2][3]) + cxt->zbuff->usize/2;
      k = u + cxt->zbuff->tbv[v];
      if (cxt->zbuff->dist[k] > dist){
	cxt->zbuff->dist[k] = dist;
	cxt->zbuff->voxel[k] = i;
	if ( s > 0 ) 
	  img->val[k] = scn->data[i]>>s;
	else
	  img->val[k] = scn->data[i];
      }
    }
  w.y = cxt->yi;
  for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
    for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx) {
      i = w.x + scn->tby[w.y] + scn->tbz[w.z];
      w1.x = w.x - cxt->xsize/2;
      w1.y = w.y - cxt->ysize/2;
      w1.z = w.z - cxt->zsize/2;
      u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + \
		 cxt->R[0][2]*w1.z + cxt->R[0][3] + cxt->zbuff->usize/2);
      v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + \
		 cxt->R[1][2]*w1.z + cxt->R[1][3] + cxt->zbuff->vsize/2);
      dist =  (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + 
	       cxt->R[2][2]*w1.z + cxt->R[2][3]) + cxt->zbuff->usize/2;
      k = u + cxt->zbuff->tbv[v];
      if (cxt->zbuff->dist[k] > dist){
	cxt->zbuff->dist[k] = dist;
	cxt->zbuff->voxel[k] = i;
	if ( s > 0 ) 
	  img->val[k] = scn->data[i]>>s;
	else
	  img->val[k] = scn->data[i];
      }
    }
  w.z = cxt->zi;
  for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) 
    for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx) {
      i = w.x + scn->tby[w.y] + scn->tbz[w.z];
      w1.x = w.x - cxt->xsize/2;
      w1.y = w.y - cxt->ysize/2;
      w1.z = w.z - cxt->zsize/2;
      u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + \
		 cxt->R[0][2]*w1.z + cxt->R[0][3] + cxt->zbuff->usize/2);
      v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + \
		 cxt->R[1][2]*w1.z + cxt->R[1][3] + cxt->zbuff->vsize/2);
      dist =  (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + 
	       cxt->R[2][2]*w1.z + cxt->R[2][3]) + cxt->zbuff->usize/2;
      k = u + cxt->zbuff->tbv[v];
      if (cxt->zbuff->dist[k] > dist){
	cxt->zbuff->dist[k] = dist;
	cxt->zbuff->voxel[k] = i;
	if ( s > 0 )
	  img->val[k] = scn->data[i]>>s;
	else
	  img->val[k] = scn->data[i];
      }
    }
  
  DrawFrame(img,cxt);
  return(img);
}


Image *DrawCutScene(Scene *scn, Context *cxt, Plane *pl)
{
  Image *img;
  int u,v,i,k;
  Voxel w,w1;
  float iprod,dist;
  int Imax,s;
  
  
  
  Imax = scn->maxval;
  s       = 0;
  

  while(Imax > 255){
    s++; Imax>>=1;
  }
  
  
  InitZBuffer(cxt->zbuff);
  img = CreateImage(cxt->zbuff->usize,cxt->zbuff->vsize);
  DrawPlane(scn,cxt,pl,img); /* Tem que ser chamado antes do DrawCutFrame? */
  DrawCutFrame(img,cxt,pl);
  w.x = cxt->xi;
  for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
    for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) {
      i = w.x + scn->tby[w.y] + scn->tbz[w.z];
      w1.x = w.x - cxt->xsize/2;
      w1.y = w.y - cxt->ysize/2;
      w1.z = w.z - cxt->zsize/2;
      iprod = (w1.x-pl->po.x)*pl->normal.x + (w1.y-pl->po.y)*pl->normal.y + \
	(w1.z-pl->po.z)*pl->normal.z;
      if (iprod >= 0.0){
	u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + \
		   cxt->R[0][2]*w1.z + cxt->R[0][3] + cxt->zbuff->usize/2);
	v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + \
		   cxt->R[1][2]*w1.z + cxt->R[1][3] + cxt->zbuff->vsize/2);
	dist =  (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + 
		 cxt->R[2][2]*w1.z + cxt->R[2][3]) + cxt->zbuff->usize/2;
	k = u + cxt->zbuff->tbv[v];
	if (cxt->zbuff->dist[k] > dist){
	  cxt->zbuff->dist[k] = dist;
	  cxt->zbuff->voxel[k] = i;
	  if (s > 0) {
	    img->val[k] = scn->data[i]>>s;
	  } else {
	    img->val[k] = scn->data[i];
	  }
	  
	}
      }
    }
  w.y = cxt->yi;
  for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
    for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx) {
      i = w.x + scn->tby[w.y] + scn->tbz[w.z];
      w1.x = w.x - cxt->xsize/2;
      w1.y = w.y - cxt->ysize/2;
      w1.z = w.z - cxt->zsize/2;
      iprod = (w1.x-pl->po.x)*pl->normal.x + (w1.y-pl->po.y)*pl->normal.y + \
	(w1.z-pl->po.z)*pl->normal.z;
      if (iprod >= 0.0){
	u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + \
		   cxt->R[0][2]*w1.z + cxt->R[0][3] + cxt->zbuff->usize/2);
	v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + \
		   cxt->R[1][2]*w1.z + cxt->R[1][3] + cxt->zbuff->vsize/2);
	dist =  (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + 
		 cxt->R[2][2]*w1.z + cxt->R[2][3]) + cxt->zbuff->usize/2;
	k = u + cxt->zbuff->tbv[v];
	if (cxt->zbuff->dist[k] > dist){
	  cxt->zbuff->dist[k] = dist;
	  cxt->zbuff->voxel[k] = i;
	  if (s > 0) {
	    img->val[k] = scn->data[i]>>s;
	  } else {
	    img->val[k] = scn->data[i];
	  }
	}
      }
    }
  w.z = cxt->zi;
  for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) 
    for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx) {
      i = w.x + scn->tby[w.y] + scn->tbz[w.z];
      w1.x = w.x - cxt->xsize/2;
      w1.y = w.y - cxt->ysize/2;
      w1.z = w.z - cxt->zsize/2;
      iprod = (w1.x-pl->po.x)*pl->normal.x + (w1.y-pl->po.y)*pl->normal.y + \
	(w1.z-pl->po.z)*pl->normal.z;
      if (iprod >= 0.0){
	u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + \
		   cxt->R[0][2]*w1.z + cxt->R[0][3] + cxt->zbuff->usize/2);
	v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + \
		   cxt->R[1][2]*w1.z + cxt->R[1][3] + cxt->zbuff->vsize/2);
	dist =  (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + 
		 cxt->R[2][2]*w1.z + cxt->R[2][3]) + cxt->zbuff->usize/2;
	k = u + cxt->zbuff->tbv[v];
	if (cxt->zbuff->dist[k] > dist){
	  cxt->zbuff->dist[k] = dist;
	  cxt->zbuff->voxel[k] = i;
	  if (s > 0) {
	    img->val[k] = scn->data[i]>>s;
	  } else {
	    img->val[k] = scn->data[i];
	  }
	}
      }
    }
  
  return(img);
}

Image *DrawCutObjects(AnnScn *ascn, Context *cxt, Plane *pl)
{
  int u,v,i,j,k,n,shift[3];
  Voxel w,w1;
  Image *img=NULL;
  ZBuffer *tmp1;
  int *tmp2,*tmp3;
  float iprod;
  Scene *scn=ascn->scn;
  int Imax, s;

  
  Imax = scn->maxval;
  s    = 0;
  while ( Imax > 255 ) {
    s++; Imax >>= 1;
  }
  
  tmp1 = CreateZBuffer(cxt->zbuff->usize,cxt->zbuff->vsize);
  tmp2 = AllocIntArray(cxt->zbuff->usize*cxt->zbuff->vsize);
  tmp3 = AllocIntArray(cxt->zbuff->usize*cxt->zbuff->vsize);

  n = cxt->zbuff->usize*cxt->zbuff->vsize;
  for (i=0; i < n; i++) 
    tmp2[i]=tmp3[i]=NIL;

  for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
    for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) 
      for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx){
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	if (ascn->label->data[i] != 0){
	  w1.x = w.x - cxt->xsize/2;
	  w1.y = w.y - cxt->ysize/2;
	  w1.z = w.z - cxt->zsize/2;
	  iprod = (w1.x-pl->po.x)*pl->normal.x + (w1.y-pl->po.y)*pl->normal.y + \
	    (w1.z-pl->po.z)*pl->normal.z;
	  if (iprod >= 0.0){
	    u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + cxt->R[0][2]*w1.z + \
		       cxt->R[0][3] + tmp1->usize/2);
	    v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + cxt->R[1][2]*w1.z + \
		       cxt->R[1][3] + tmp1->vsize/2);
	    k = u + tmp1->tbv[v];
	    if (tmp1->voxel[k] == NIL){
	      tmp1->dist[k]   = (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + cxt->R[2][2]*w1.z + cxt->R[2][3])+tmp1->usize/2;
	      tmp1->voxel[k]  = i;
	      tmp1->object[k] = (uchar)ascn->label->data[i];
	      if (iprod == 0.0) {
		tmp2[k]  = i;

	      }
	    }
	  }
	}
      }

  /* Splatting 2x2 */

  InitZBuffer(cxt->zbuff);
  shift[0] = 1; shift[1] = cxt->zbuff->usize; shift[2] = shift[1]+1;
  n = cxt->zbuff->usize*(cxt->zbuff->vsize-1);
  for (i=0; i < n-1; i++) {
    if (tmp1->voxel[i] != NIL){
      cxt->zbuff->dist[i]   = tmp1->dist[i];
      cxt->zbuff->object[i] = tmp1->object[i];
      cxt->zbuff->voxel[i]  = tmp1->voxel[i];
      for (j=0; j < 3; j++){
	k = i+shift[j];
	if ((tmp1->voxel[k] == NIL)&&(cxt->zbuff->voxel[k]==NIL)){
	  cxt->zbuff->dist[k]   = tmp1->dist[i];
	  cxt->zbuff->object[k] = tmp1->object[i];
	  cxt->zbuff->voxel[k]  = tmp1->voxel[i];
	}
      }
    }
    if (tmp2[i] != NIL){
      tmp3[i]   = tmp2[i];
      for (j=0; j < 3; j++){
	k = i+shift[j];
	if (tmp2[k] == NIL){
	  tmp3[k]   = tmp2[i];
	}
      }
    }
  }
  img = DepthShading(cxt);
  n = cxt->zbuff->usize*cxt->zbuff->vsize;
  for (i=0; i < n; i++) 
    if (tmp3[i] != NIL){
      if ( s > 0 ) { 
	img->val[i] = scn->data[tmp3[i]] >> s;
      } else {
	img->val[i] = scn->data[tmp3[i]];
      }
    }
  
  DestroyZBuffer(&tmp1);
  free(tmp2);
  free(tmp3);
  
  DrawFrame(img,cxt);

  return(img);
}

CImage *DrawCutObjectsAux(Scene *scn, Scene *label, Context *cxt, Plane *pl)
{
  int u,v,i,j,k,n,shift[3];
  Voxel w,w1;
  Image *img=NULL;
  CImage *cimg=NULL;
  ZBuffer *tmp1;
  int *tmp2,*tmp3;
  float iprod;
  int Imax, s;

  
  Imax = scn->maxval;
  s    = 0;
  while ( Imax > 255 ) {
    s++; Imax >>= 1;
  }
  
  tmp1 = CreateZBuffer(cxt->zbuff->usize,cxt->zbuff->vsize);
  tmp2 = AllocIntArray(cxt->zbuff->usize*cxt->zbuff->vsize);
  tmp3 = AllocIntArray(cxt->zbuff->usize*cxt->zbuff->vsize);

  n = cxt->zbuff->usize*cxt->zbuff->vsize;
  for (i=0; i < n; i++) 
    tmp2[i]=tmp3[i]=NIL;

  for (w.z=cxt->zi; w.z != cxt->zf; w.z += cxt->dz) 
    for (w.y=cxt->yi; w.y != cxt->yf; w.y += cxt->dy) 
      for (w.x=cxt->xi; w.x != cxt->xf; w.x += cxt->dx){
	i = w.x + scn->tby[w.y] + scn->tbz[w.z];
	if (label->data[i] != 0){
	  w1.x = w.x - cxt->xsize/2;
	  w1.y = w.y - cxt->ysize/2;
	  w1.z = w.z - cxt->zsize/2;
	  iprod = (w1.x-pl->po.x)*pl->normal.x + (w1.y-pl->po.y)*pl->normal.y +
	    (w1.z-pl->po.z)*pl->normal.z;
	  if (iprod >= 0.0){
	    u = (int) (cxt->R[0][0]*w1.x + cxt->R[0][1]*w1.y + cxt->R[0][2]*w1.z + cxt->R[0][3] + tmp1->usize/2);
	    v = (int) (cxt->R[1][0]*w1.x + cxt->R[1][1]*w1.y + cxt->R[1][2]*w1.z + cxt->R[1][3] + tmp1->vsize/2);
	    k = u + tmp1->tbv[v];
	    if (tmp1->voxel[k] == NIL){
	      tmp1->dist[k]   = (cxt->R[2][0]*w1.x + cxt->R[2][1]*w1.y + cxt->R[2][2]*w1.z + cxt->R[2][3])+tmp1->usize/2;
	      tmp1->voxel[k]  = i;
	      tmp1->object[k] = (uchar)label->data[i];
	      if (iprod < 1.)
		tmp2[k]  = i;
	    }
	  }
	}
      }

  /* Splatting 2x2 */

  InitZBuffer(cxt->zbuff);
  shift[0] = 1; shift[1] = cxt->zbuff->usize; shift[2] = shift[1]+1;
  n = cxt->zbuff->usize*(cxt->zbuff->vsize-1);
  for (i=0; i < n-1; i++) {
    if (tmp1->voxel[i] != NIL){
      cxt->zbuff->dist[i]   = tmp1->dist[i];
      cxt->zbuff->object[i] = tmp1->object[i];
      cxt->zbuff->voxel[i]  = tmp1->voxel[i];
      for (j=0; j < 3; j++){
	k = i+shift[j];
	if ((tmp1->voxel[k] == NIL)&&(cxt->zbuff->voxel[k]==NIL)){
	  cxt->zbuff->dist[k]   = tmp1->dist[i];
	  cxt->zbuff->object[k] = tmp1->object[i];
	  cxt->zbuff->voxel[k]  = tmp1->voxel[i];
	}
      }
    }
    if (tmp2[i] != NIL){
      tmp3[i]   = tmp2[i];
      for (j=0; j < 3; j++){
	k = i+shift[j];
	if (tmp2[k] == NIL){
	  tmp3[k]   = tmp2[i];
	}
      }
    }
  }
  img = PhongShading(cxt);
  cimg = Colorize(cxt,img);
  n = cxt->zbuff->usize*cxt->zbuff->vsize;
  for (i=0; i < n; i++) 
    if (tmp3[i] != NIL){
      if ( s > 0 ) { 
	cimg->C[0]->val[i] = scn->data[tmp3[i]] >> s;
	cimg->C[1]->val[i] = scn->data[tmp3[i]] >> s;
	cimg->C[2]->val[i] = scn->data[tmp3[i]] >> s;
      } else {
	cimg->C[0]->val[i] = scn->data[tmp3[i]];
	cimg->C[1]->val[i] = scn->data[tmp3[i]];
	cimg->C[2]->val[i] = scn->data[tmp3[i]];
      }
    }
  
  DestroyZBuffer(&tmp1);
  DestroyImage(&img);
  free(tmp2);
  free(tmp3);  
  DrawCutFrame(cimg->C[0],cxt,pl);
  DrawCutFrame(cimg->C[1],cxt,pl);
  DrawCutFrame(cimg->C[2],cxt,pl);
  return(cimg);
}
 
bool FillLabel(Image *img, int p, int q, int label) 
{
  return ((img->val[q] != label) ? true: false);
}
 
bool FillAnyLabel(Image *img, int p, int q, int label) 
{
  return ((img->val[p] == img->val[q]) ? true: false);
}


void ift_FloodFill(Image *img, AdjRel *A, int pos, FloodFun fun, int label)
{
  Pixel u,v;
  int i, p, q, n = img->ncols * img->nrows;
  FIFOQ *Q = FIFOQNew(n);
  uchar *status = AllocUCharArray(n);
    
  FIFOQPush(Q, pos);
  status[pos] = 1;
  while( !FIFOQEmpty(Q) ){
    p = FIFOQPop(Q);
    status[p] = 2;
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for(i=1; i<A->n; i++) {      
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];	
      if( ValidPixel(img,v.x,v.y) ){
	q = img->tbrow[v.y] + v.x;
	if (status[q] == 0) {
	  if (fun(img,p,q,label)) {
	    status[q] = 1;
	    FIFOQPush(Q,q);
	  }
	}
      }
    }
  }
  for (p=0; p <n; p++)
    if (status[p] == 2) 
      img->val[p] = label;
  free(status);
  FIFOQDestroy(Q);
}
