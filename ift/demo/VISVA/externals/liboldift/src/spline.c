#include "spline.h"
#include "math.h"


void NormalizeVector(Vector *v){
  float size;
  
  size=(float)sqrt(v->x*v->x+v->y*v->y+v->z*v->z);
  v->x=(float)v->x/size;
  v->y=(float)v->y/size;
  v->z=(float)v->z/size;
  
}

Spline *CreateSpline(Scene *scn){
  Spline *spline = NULL;
  spline=(Spline*) calloc(1,sizeof(Spline));
  spline->numpts=0;
  spline->dx=scn->dx;
  spline->dy=scn->dy;
  spline->dz=scn->dz;
    
  return(spline);
};

void AddPoints(Spline *spline,int x,int y){
  Vector v1,v2;
  float cos,gama,d,a,b;

  
  if(spline->numpts-3 < 100){
    
    
    if(spline->numpts==0){ 
      spline->po[0].x=x;
      spline->po[0].y=y;
      spline->numpts=1;
    }
    else if(spline->numpts==1){
      spline->po[3].x=x;
      spline->po[3].y=y;
      v1.x=spline->po[3].x-spline->po[0].x;
      v1.y=spline->po[3].y-spline->po[0].y;
      v1.z=0.0;
      NormalizeVector(&v1);
      a=(spline->po[3].x-spline->po[0].x);
      b=(spline->po[3].y-spline->po[0].y);
      d=sqrt(a*a+b*b);
      
      spline->po[1].x=(int)(spline->po[0].x+(d/3)*v1.x);
      spline->po[1].y=(int)(spline->po[0].y+(d/3)*v1.y);
      spline->po[2].x=(int)(spline->po[0].x+(2*d/3)*v1.x);
      spline->po[2].y=(int)(spline->po[0].y+(2*d/3)*v1.y);
      spline->numpts=4;
    } else {
      spline->numpts+=3;
      spline->po[spline->numpts-1].x=x;
      spline->po[spline->numpts-1].y=y;
      v1.x=spline->po[spline->numpts-4].x-spline->po[spline->numpts-5].x;
      v1.y=spline->po[spline->numpts-4].y-spline->po[spline->numpts-5].y;
      v1.z=0.0;
      
      v2.x=spline->po[spline->numpts-1].x-spline->po[spline->numpts-4].x;
      v2.y=spline->po[spline->numpts-1].y-spline->po[spline->numpts-4].y;
      v2.z=0.0;
      
      NormalizeVector(&v1);
      NormalizeVector(&v2);
      cos=ScalarProd(v1,v2);
      a=(spline->po[spline->numpts-1].x-spline->po[spline->numpts-4].x);
      b=(spline->po[spline->numpts-1].y-spline->po[spline->numpts-4].y);
      d=sqrt(a*a+b*b);
      cos=sqrt(cos*cos)+0.25;
      gama = d/(4*cos);
      spline->po[spline->numpts-3].x=(int)spline->po[spline->numpts-4].x+gama*v1.x;
      spline->po[spline->numpts-3].y=(int)spline->po[spline->numpts-4].y+gama*v1.y;
      spline->po[spline->numpts-2].x=(int)spline->po[spline->numpts-3].x+d/2*v2.x;
      spline->po[spline->numpts-2].y=(int)spline->po[spline->numpts-3].y+d/2*v2.y;
    }
  }
  return;
  
};

void DestroySpline(Spline **spline){
  Spline *aux= *spline;
  if(aux != NULL){
    free(aux);
    *spline=NULL;
  }
}


float distance3D(Spline *spline,Vector v1,Vector v2) {
  
  float a,b,c;
  a=(float)(v1.x-v2.x)*spline->dx;
  b=(float)(v1.y-v2.y)*spline->dy;
  c=(float)(v1.z-v2.z)*spline->dz;
  a=a*a;
  b=b*b;
  c=c*c;

  return(sqrt(a+b+c));
}

void Convertcoords(Scene *scn, Context *cxt, Plane *pl,Pixel p,Voxel *v)

{
  int u1,v1,usize,vsize;
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
   
  //if (p.y < vsize) {
  //if (p.x < usize) {
  u1   = p.x - usize/2;
  v1   = p.y - vsize/2;
  w1.x  = u1*R[0][0] + v1*R[0][1] + pl->po.x; 
  w1.y  = u1*R[1][0] + v1*R[1][1] + pl->po.y;
  w.x = w1.x + scn->xsize/2;
  w.y = w1.y + scn->ysize/2;
  //if ((w.x >= cxt->xmin)&&(w.x <= cxt->xmax)&&
  //  (w.y >= cxt->ymin)&&(w.y <= cxt->ymax)){
  w1.z = (int) (u1*R[2][0] + v1*R[2][1] + pl->po.z);
  w.z  = w1.z + scn->zsize/2;
    
    //if ((w.z >= cxt->zmin)&&(w.z <= cxt->zmax)){
  v->x=w.x;
  v->y=w.y;
  v->z=w.z;
  return;
      //}
      //}
  //  }
  //}
  //v->x=-1;
  //v->y=-1;
  //v->z=-1;
  
}


void MovePoint(Spline *spline,int x,int y,int pointnumber) {
  if(pointnumber<=spline->numpts){
    spline->po[pointnumber].x=x;
    spline->po[pointnumber].y=y;
  }
  
}


int Section(Spline *spline,float alpha){
  float dist;
  int i;
  dist=alpha*spline->total;
  i=1;
  
  for(i=1;dist > spline->len[i];i++);
 
  return i;
}


void CalculateVoxel(Spline *spline,float alpha,Vector *v){
  
  float a,b,c,d;
  float u3,u2,u,dist;
  int i;
  
  dist=alpha*spline->total;
  i=Section(spline,alpha);
  u= (float)(dist - spline->len[i-1])/(spline->len[i]-spline->len[i-1]);
  i=3*i-2;
  u2=u*u;
  u3=u2*u;
 
  a=(float)(-u3+3*u2-3*u+1);
  b=(float)(3*u3-6*u2+3*u);
  c=(float)(-3*u3+3*u2);
  d=(float)(u3);

  v->x=a*spline->vo[i-1].x+b*spline->vo[i].x+c*spline->vo[i+1].x+d*spline->vo[i+2].x;
  v->y=a*spline->vo[i-1].y+b*spline->vo[i].y+c*spline->vo[i+1].y+d*spline->vo[i+2].y;
  v->z=a*spline->vo[i-1].z+b*spline->vo[i].z+c*spline->vo[i+1].z+d*spline->vo[i+2].z;
  
} 


void CalculateSectionVoxel(Spline *spline,int i,float alpha,Vector *v){
  
  float a,b,c,d;
  float u3,u2,u;
  
  u= (float)alpha;
  i=3*i-2;
  u2=u*u;
  u3=u2*u;
 
  a=(float)(-u3+3*u2-3*u+1);
  b=(float)(3*u3-6*u2+3*u);
  c=(float)(-3*u3+3*u2);
  d=(float)(u3);

  v->x=(float)a*spline->vo[i-1].x+b*spline->vo[i].x+c*spline->vo[i+1].x+d*spline->vo[i+2].x;
  v->y=(float)a*spline->vo[i-1].y+b*spline->vo[i].y+c*spline->vo[i+1].y+d*spline->vo[i+2].y;
  v->z=(float)a*spline->vo[i-1].z+b*spline->vo[i].z+c*spline->vo[i+1].z+d*spline->vo[i+2].z;
  
} 
/*
  void CalculateNormal(Spline *spline,float alpha,Vector *v){
  
  float u3,u2,u,a,b,c,d,size,dist;
  Vector aux;
  int i;
  
  dist=alpha*spline->total;
  i=Section(spline,alpha);
  u= (float)(dist - spline->len[i-1])/(spline->len[i]-spline->len[i-1]);
  i=3*i-2;
  u2=u*u;
  u3=u2*u;
  
  a=(float)(-3*u2+6*u-3);
  b=(float)(9*u2-12*u+3);
  c=(float)(-9*u2+6*u);
  d=(float)(3*u2);
 
  aux.x=(a*spline->vo[i-1].x+b*spline->vo[i].x+c*spline->vo[i+1].x+d*spline->vo[i+2].x);
  aux.y=(a*spline->vo[i-1].y+b*spline->vo[i].y+c*spline->vo[i+1].y+d*spline->vo[i+2].y);
  aux.z=(a*spline->vo[i-1].z+b*spline->vo[i].z+c*spline->vo[i+1].z+d*spline->vo[i+2].z);
 
  size=(float)sqrt(aux.x*aux.x+aux.y*aux.y+aux.z*aux.z);
  v->x=(float)aux.x/size;
  v->y=(float)aux.y/size;
  v->z=(float)aux.z/size;
    
  } 
*/

void CalculateNormal(Spline *spline,float alpha,Vector *v){
 
  float size;
  Vector aux,v1,v2;
  
  if (alpha == 1.00) alpha=0.999;
 
  CalculateVoxel(spline,alpha,&v1);
  CalculateVoxel(spline,alpha+0.001,&v2);
  aux.x=v2.x-v1.x;
  aux.y=v2.y-v1.y;
  aux.z=v2.z-v2.z;
 
  size=(float)sqrt(aux.x*aux.x+aux.y*aux.y+aux.z*aux.z);
  v->x=(float)aux.x/size;
  v->y=(float)aux.y/size;
  v->z=(float)aux.z/size;
    
}


void CalculateSectionNormal(Spline *spline,int i,float alpha,Vector *v){
 
  float size;
  Vector aux,v1,v2;
  
  CalculateSectionVoxel(spline,i,alpha,&v1);
  CalculateSectionVoxel(spline,i,alpha+0.001,&v2);
  aux.x=v2.x-v1.x;
  aux.y=v2.y-v1.y;
  aux.z=v2.z-v2.z;
 
  size=(float)sqrt(aux.x*aux.x+aux.y*aux.y+aux.z*aux.z);
  v->x=(float)aux.x/size;
  v->y=(float)aux.y/size;
  v->z=(float)aux.z/size;
    
}

float LengthSectionSpline(Spline *spline,int i,float alpha){
  float u,temp;
  int n;
  Vector v1,v2;
  
  temp=0.0;
  CalculateSectionVoxel(spline,i,0.0,&v1);
  
  for(n=1;n<=100*alpha;n++){
    u=(float)n/100;
    CalculateSectionVoxel(spline,i,u,&v2);
    temp+=distance3D(spline,v1,v2);
    v1.x=v2.x;
    v1.y=v2.y;
    v1.z=v2.z;
  }
  v1.x=spline->vo[3*(i-1)].x;
  v1.y=spline->vo[3*(i-1)].y;
  v1.z=spline->vo[3*(i-1)].z;
  v2.x=spline->vo[3*i].x;
  v2.y=spline->vo[3*i].y;
  v2.z=spline->vo[3*i].z;
  //temp=distance3D(spline,v1,v2);
  return temp;
}

void Compute3DSpline(Scene *scn,Context *cxt,Plane *pl,Spline *spline){
  int i,sections;
  float aux=0.0;
  
  spline->len[0]=0.0;
  
  for(i=0;i<=spline->numpts+1;i++){
    Convertcoords(scn,cxt,pl,spline->po[i],&spline->vo[i]);
  }

   
  if(spline->numpts < 4) 
    sections = 0;
  else if(spline->numpts == 4) 
    sections = 1;
  else sections = (spline->numpts - 4)/3+1;
  
    for(i=1;i<=sections;i++){ 
      aux+=LengthSectionSpline(spline,i,1.0);
      spline->len[i]=aux;
    }

  spline->total=aux;
}


Spline* TranslateSpline(Spline *spline,float alpha){
  int i,sections;
  float u;
  Vector v,n1;
   Spline *res;
   Scene *scn;
   n1.x=0.0;
   n1.y=0.0;
  scn = CreateScene(0,0,0);
  scn->dx=spline->dx;
  scn->dy=spline->dy;
  scn->dz=spline->dz;
  res=CreateSpline(scn);
  
  sections=((spline->numpts-4)/3)+1;
  
  for(i=0;i<sections;i++){
    u=(spline->len[i]/spline->total);
    CalculateNormal(spline,u,&n1);
  
    if(n1.x==0.0){
      if(n1.y>0.0) v.x=-1.0;
      else v.x=1.0;
      v.y=0.0;
  } else {
    
    if(n1.x>0.0) v.y=1.0;
    else v.y=-1.0;
    
    v.x=(float)-n1.y/n1.x*v.y;
  }
    v.z=0;
  
  NormalizeVector(&v);
  AddPoints(res,spline->po[3*i].x+(alpha*spline->dx*v.x),spline->po[3*i].y+(alpha*spline->dy*v.y));
  }
  
  CalculateNormal(spline,1.00,&n1);
  
    if(n1.x==0.0){
      if(n1.y>0.0) v.x=-1.0;
      else v.x=1.0;
      v.y=0.0;
  } else {
    
    if(n1.x>0.0) v.y=1.0;
    else v.y=-1.0;
    
    v.x=(float)-n1.y/n1.x*v.y;
  }
    v.z=0;
        
    NormalizeVector(&v);
    AddPoints(res,spline->po[3*i].x+alpha*spline->dx*v.x,spline->po[3*i].y+alpha*spline->dy*v.y);
    
    return res;
}

/*
Spline* TranslateSpline(Spline *spline,float alpha){
  int i;
  float u,deltax,deltay;
  Vector v,n1,sum;
  Spline *res;
  Scene *scn;
  n1.x=0.0;
  n1.y=0.0;
  scn = CreateScene(0,0,0);
  scn->dx=spline->dx;
  scn->dy=spline->dy;
  scn->dz=spline->dz;
  res=CreateSpline(scn);
  
  sum.x=0;
  sum.y=0;
  sum.z=0;
  
  for(i=0;i<100;i++){
    u=(float)(i/100);
    CalculateNormal(spline,u,&n1);
    
    if(n1.x==0.0){
      if(n1.y>0.0) v.x=-1.0;
      else v.x=1.0;
      v.y=0.0;
  } else {
    
    if(n1.x>0.0) v.y=1.0;
    else v.y=-1.0;
    
    v.x=(float)-n1.y/n1.x*v.y;
  }
    v.z=0;
  
  NormalizeVector(&v);
  sum.x+=v.x;
  sum.y+=v.y;
  }
  
  NormalizeVector(&sum);
  
  deltax=sum.x*alpha*spline->dx;
  deltay=sum.y*alpha*spline->dy;
  
  for(i=0;i<spline->numpts;i++) {
    res->po[i].x=spline->po[i].x+deltax;
    res->po[i].y=spline->po[i].y+deltay;
  }
  res->numpts=spline->numpts;
  return res;
}
*/

Plane *NormalPlane(Scene *scn,Spline *spline,float alpha){
  
  Vector n,v;
  Context *cxt;
  Plane *pl;
  //  float dist, u;
  //  int i;

  //  dist=alpha*spline->total;
  //  i=
  Section(spline,alpha);
  //  u= (float)(dist - spline->len[i-1])/(spline->len[i]-spline->len[i-1]);
  
  
  cxt = CreateContext(scn);  
  pl = CreatePlane(cxt);

  CalculateVoxel(spline,alpha,&v);
  CalculateNormal(spline,alpha,&n);
  
  
  pl->po.x=v.x-scn->xsize/2;
  pl->po.y=v.y-scn->ysize/2;
  pl->po.z=v.z-scn->zsize/2;
  
  pl->normal.x=n.x;
  pl->normal.y=n.y;
  pl->normal.z=n.z;
    
  return pl;
}

void DrawPixelAux(Scene *scn,Context *cxt,Plane *pl,Image *img,Pixel p){
  int k;
  k = p.x + img->tbrow[p.y];
  img->val[k]=255;
}

void DrawLineAux(Scene *scn,Context *cxt,Plane *pl,Image *img, Pixel p1, Pixel pn)
{
  int dx = abs(pn.x - p1.x), dy = abs (pn.y - p1.y);
  int pk = 2*dy-dx; /* decision variable for lines y=mx+b where -1 <= m <= 1 */
  int twody = 2*dy , twodydx = 2*(dy-dx);
  int qk = 2*dx-dy; /* decision variable for lines y=mx+b where m < -1
                      or m > 1 */
  int twodx = 2*dx ,twodxdy = 2*(dx-dy);
  //  int npts;
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
  
  /* Find start and end spline */
  
 if(dx >= dy){ /* -1 <= m <= 1 */
   if(p1.x > pn.x) {
     p.x    = pn.x;
     p.y    = pn.y;
     //     npts   = p1.x - pn.x + 1;     
     pend.x = p1.x;
   }
   else { /* p1.x <= pn.x */
     p.x = p1.x;
     p.y = p1.y;
     //     npts = pn.x - p1.x + 1;
     pend.x = pn.x;
   }
   DrawPixelAux(scn,cxt,pl,img,p);
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
     DrawPixelAux(scn,cxt,pl,img,p);
   }
 } else { /* dx < dy: m > 1 or m < -1 */
  
   if(p1.y > pn.y) {
     p.x = pn.x;
     p.y = pn.y;
     //     npts   = p1.y - pn.y + 1;      
     pend.y = p1.y;
   }
   else { /* p1.y <= pn.y */
     p.x = p1.x;
     p.y = p1.y;
     //     npts   = pn.y - p1.y + 1;     
     pend.y = pn.y;
   }	 
   DrawPixelAux(scn,cxt,pl,img,p);
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
     DrawPixelAux(scn,cxt,pl,img,p);
    }
 }

}

void CloseSpline(Spline *spline){
  
  if((spline->numpts > 1)&&(spline->numpts<18)){
  spline->po[0].x = spline->po[spline->numpts].x;
  spline->po[0].y = spline->po[spline->numpts].y;
  spline->numpts++;
  spline->po[spline->numpts].x=spline->po[1].x;
  spline->po[spline->numpts].y=spline->po[1].y;
  spline->po[spline->numpts+1].x=spline->po[2].x;
  spline->po[spline->numpts+1].y=spline->po[2].y;
  }
  
}

void CalculateSectionPixel(Spline *spline,int i,float alpha,Pixel *p){
  
  float a,b,c,d;
  float u3,u2,u; //dist;
  //int i;

  //dist=alpha*spline->total;
  //i=Section(spline,alpha);
  //u= (float)(dist - spline->len[i-1])/(spline->len[i]-spline->len[i-1]);
  i=3*i-2;
  u=alpha;
  u2=u*u;
  u3=u2*u;
  
  a=(float)(-u3+3*u2-3*u+1);
  b=(float)(3*u3-6*u2+3*u);
  c=(float)(-3*u3+3*u2);
  d=(float)(u3);

  p->x=(int)(a*spline->po[i-1].x+b*spline->po[i].x+c*spline->po[i+1].x+d*spline->po[i+2].x);
  p->y=(int)(a*spline->po[i-1].y+b*spline->po[i].y+c*spline->po[i+1].y+d*spline->po[i+2].y);
  
} 

void CalculatePixel(Spline *spline,float alpha,Pixel *p){
  
  float a,b,c,d;
  float u3,u2,u,dist;
  int i;

  dist=alpha*spline->total;
  i=Section(spline,alpha);
  u= (float)(dist - spline->len[i-1])/(spline->len[i]-spline->len[i-1]);
  i=3*i-2;
  u=alpha;
  u2=u*u;
  u3=u2*u;
  
  a=(float)(-u3+3*u2-3*u+1);
  b=(float)(3*u3-6*u2+3*u);
  c=(float)(-3*u3+3*u2);
  d=(float)(u3);

  p->x=(int)(a*spline->po[i-1].x+b*spline->po[i].x+c*spline->po[i+1].x+d*spline->po[i+2].x);
  p->y=(int)(a*spline->po[i-1].y+b*spline->po[i].y+c*spline->po[i+1].y+d*spline->po[i+2].y);
  
} 

void DrawSectionNormal(Scene *scn,Context *cxt,Plane *pl,Spline *spline,Image *img,int i,float alpha){
  Pixel p1,p2,p;
  Vector n1,v;
  
  CalculateSectionPixel(spline,i,alpha,&p);
  CalculateSectionNormal(spline,i,alpha,&n1);
  //n1.x=(int)(a*spline->po[i-1].x+b*spline->po[i].x+c*spline->po[i+1].x+d*spline->po[i+2].x);
  // n1.y=(int)(a*spline->po[i-1].y+b*spline->po[i].y+c*spline->po[i+1].y+d*spline->po[i+2].y);
  
  if(n1.x==0){
    v.x=1.0;
    v.y=0.0;
  } else {
    v.y=1.0;
    v.x=(float)-n1.y/n1.x;
  }
  v.z=0;
  
  NormalizeVector(&v);
  p1.x=p.x+60*v.x; 
  p1.y=p.y+60*v.y;
  p2.x=p.x-60*v.x; 
  p2.y=p.y-60*v.y;
  DrawLineAux(scn,cxt,pl,img,p1,p2);
}

void DrawNormal(Scene *scn,Context *cxt,Plane *pl,Spline *spline,Image *img,float alpha){
 
  float u,dist;
  int i;

  
  dist=alpha*spline->total;
  i=Section(spline,alpha);
  u= (float)(dist - spline->len[i-1])/(spline->len[i]-spline->len[i-1]);
   
  DrawSectionNormal(scn,cxt,pl,spline,img,i,u);
}

/*
void DrawNormal(Scene *scn,Context *cxt,Plane *pl,Spline *spline,Image *img,float alpha){
  Pixel p1,p2,p;
  Vector n1,v;
  float u,u2,a,b,c,d,dist;
  int i;

  
  dist=alpha*spline->total;
  i=Section(spline,alpha);
  u= (float)(dist - spline->len[i-1])/(spline->len[i]-spline->len[i-1]);
  i=3*i-2;
  u2=u*u;

  a=(float)(-3*u2+6*u-3);
  b=(float)(9*u2-12*u+3);
  c=(float)(-9*u2+6*u);
  d=(float)(3*u2);
  

  CalculatePixel(spline,i,alpha,&p);
  //CalculateNormal(spline,alpha,&n1);
  n1.x=(int)(a*spline->po[i-1].x+b*spline->po[i].x+c*spline->po[i+1].x+d*spline->po[i+2].x);
  n1.y=(int)(a*spline->po[i-1].y+b*spline->po[i].y+c*spline->po[i+1].y+d*spline->po[i+2].y);
  
  if(n1.x==0){
    v.x=1.0;
    v.y=0.0;
  } else {
    v.y=1.0;
    v.x=(float)-n1.y/n1.x;
  }
  v.z=0;
  
  NormalizeVector(&v);
  p1.x=p.x+60*v.x; 
  p1.y=p.y+60*v.y;
  p2.x=p.x-60*v.x; 
  p2.y=p.y-60*v.y;
  DrawLineAux(scn,cxt,pl,img,p1,p2);
}
*/
void DrawSpline(Scene *scn,Context *cxt,Plane *pl,Spline *spline,Image *img){
  int i,n,k,sections;
  float u;
  Pixel p;
 //Bezier Spline
 
  if(spline->numpts < 4) 
    sections = 0;
  else if(spline->numpts == 4) 
    sections = 1;
  else sections = (spline->numpts - 4)/3+1;
  
    for(i=1;i<=sections;i++){ 
      for(n=0;n<=1000;n++){
	u=(float)n/1000;
	CalculateSectionPixel(spline,i,u,&p);
	k =  p.x + img->tbrow[p.y];
	img->val[k]=255;
      }
      
    }
}

float LengthSpline(Spline *spline,float alpha){
  return (alpha*spline->total);
}

int getCoordx(Spline *spline,int pointnumber){
  if(pointnumber < spline->numpts) {
  return spline->po[pointnumber].x;
  } else return 0.0;
  
}


int getCoordy(Spline *spline,int pointnumber){
  if(pointnumber < spline->numpts) {
  return spline->po[pointnumber].y;
  } else return 0.0;
  
}

