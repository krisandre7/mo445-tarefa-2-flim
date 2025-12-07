#include "shading.h"
#include "queue.h"


Image *DepthShading(Context *cxt) 
{
  Image *img=NULL;
  int i,n;
  float dmin=(float)INT_MAX,dmax=(float)INT_MIN,coef;

  img = CreateImage(cxt->zbuff->usize,cxt->zbuff->vsize);
  n = img->ncols*img->nrows;
  for (i=0; i < n; i++){
    if (cxt->zbuff->voxel[i] != NIL)   {  /* not background */
      if (cxt->zbuff->dist[i] < dmin)
	dmin = cxt->zbuff->dist[i];
      if (cxt->zbuff->dist[i] > dmax)
	dmax = cxt->zbuff->dist[i];
    }
  }

  if (dmin != dmax) {
    coef = 255./(dmax-dmin);  
    for (i=0; i < n; i++){
      if ( cxt->zbuff->voxel[i] == NIL)  /* background */
      	img->val[i] = 0;
      else 
	img->val[i] = (int) (coef*(dmax-cxt->zbuff->dist[i]));
    }
  }
  
  return(img);
}

Image *PhongShading(Context *cxt) 
{
  Image *img=NULL;
  Vector normal,viewer;
  int i,k,n,value=0;
  float dmin=(float)INT_MAX,dmax=(float)INT_MIN,coef,amb;
  double cos2O,cosine,pow;
  uchar l;

  img = CreateImage(cxt->zbuff->usize,cxt->zbuff->vsize);
  n = img->ncols*img->nrows;


  // object space coordinates for viewer vector
  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
  
  for (i=0; i < n; i++){
    if (cxt->zbuff->voxel[i] != NIL)   {  /* not background */
      if (cxt->zbuff->dist[i] < dmin)
	dmin = cxt->zbuff->dist[i];
      if (cxt->zbuff->dist[i] > dmax)
	dmax = cxt->zbuff->dist[i];
    }
  }
  if ((dmax-dmin)>=Epsilon) { 
    coef = 255./(dmax-dmin);
    for (i=0; i < n; i++){
      if ( cxt->zbuff->voxel[i] == NIL) /* background */
      	img->val[i] = 0;
      else {
	ObjectNormal(i,cxt,&normal);	
	l = cxt->zbuff->object[i];      /* label */
	amb = cxt->obj[l].amb*255;
	cosine = ScalarProd(viewer,normal);/* viewer and normal
					      vectors are already
                                              normalized */
	if (cosine < -1.0) cosine = -1.0; /* necessary due to numeric errors in decimal digits */
	else if (cosine > 1.0) cosine = 1.0;
	
	/*
	  |angle|>90° impossible case because raycasting/splatting selected only visible voxels, i.e. |angle|< 90º*/
	if (cosine < -Epsilon)	
	  cosine = 0.;
	
	cos2O = 2*cosine*cosine-1;
	
	if (cos2O < 0.)	  /* |angle| >45° */
	  pow = 0.;
	else {
	  pow = 1.;
	  for (k=0; k < cxt->obj[l].ns; k++)
	    pow = pow*cos2O;
	}
	
	value = ROUND(amb + coef * (dmax-cxt->zbuff->dist[i]) * (cosine*cxt->obj[l].diff + pow*cxt->obj[l].spec));
	img->val[i] = ((value>255) ? 255 : (( value < 0 ) ? 0 : (unsigned char)value)); /* conversion with saturation */
      }
    }
  }
  else { // all visible parts of object are at the same distance (dmin == dmax)
    for (i=0; i < n; i++){
      if ( cxt->zbuff->voxel[i] == NIL) // background
      	img->val[i] = 0;
      else // object
	img->val[i] = 255;
    }
  }
  
  return(img);
}

Image *PhongShadingScene(Context *cxt,Scene *nscn) 
{
  Image *img=NULL;
  Vector normal,viewer;
  int i,k,n,value=0;
  float dmin=(float)INT_MAX,dmax=(float)INT_MIN,coef,amb;
  double cos2O,cosine,pow;
  uchar l;

  img = CreateImage(cxt->zbuff->usize,cxt->zbuff->vsize);
  n = img->ncols*img->nrows;


  // object space coordinates for viewer vector
  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
  
  for (i=0; i < n; i++){
    if (cxt->zbuff->voxel[i] != NIL)   {  /* not background */
      if (cxt->zbuff->dist[i] < dmin)
	dmin = cxt->zbuff->dist[i];
      if (cxt->zbuff->dist[i] > dmax)
	dmax = cxt->zbuff->dist[i];
    }
  }
  if ((dmax-dmin)>=Epsilon) { 
    coef = 255./(dmax-dmin);
    for (i=0; i < n; i++){
      if ( cxt->zbuff->voxel[i] == NIL) /* background */
      	img->val[i] = 0;
      else {
	SceneNormal(cxt->zbuff->voxel[i],nscn,&normal);	
	l = cxt->zbuff->object[i];      /* label */
	amb = cxt->obj[l].amb*255;
	cosine = ScalarProd(viewer,normal);/* viewer and normal
					      vectors are already
                                              normalized */
	if (cosine < -1.0) cosine = -1.0; /* necessary due to numeric errors in decimal digits */
	else if (cosine > 1.0) cosine = 1.0;
	
	/*
	  |angle|>90° impossible case because raycasting/splatting selected only visible voxels, i.e. |angle|< 90º*/
	if (cosine < -Epsilon)	
	  cosine = 0.;
	
	cos2O = 2*cosine*cosine-1;
	
	if (cos2O < 0.)	  /* |angle| >45° */
	  pow = 0.;
	else {
	  pow = 1.;
	  for (k=0; k < cxt->obj[l].ns; k++)
	    pow = pow*cos2O;
	}
	
	value = ROUND(amb + coef * (dmax-cxt->zbuff->dist[i]) * (cosine*cxt->obj[l].diff + pow*cxt->obj[l].spec));
	img->val[i] = ((value>255) ? 255 : (( value < 0 ) ? 0 : (unsigned char)value)); /* conversion with saturation */
      }
    }
  }
  else { // all visible parts of object are at the same distance (dmin == dmax)
    for (i=0; i < n; i++){
      if ( cxt->zbuff->voxel[i] == NIL) // background
      	img->val[i] = 0;
      else // object
	img->val[i] = 255;
    }
  }
  
  return(img);
}

void ObjectNormal(int Pind, Context *cxt, Vector *normal) {
  int u,v,xysize,r,Vind,N,M,i;
  int di[8];   // offsets for neighbours
  double modn;
  Vector p[3],w[3];
  
  normal->x = normal->y = normal->z = 0.0;
  xysize = cxt->xsize * cxt->ysize;
  u = Pind % (cxt->zbuff->usize);
  v = Pind / (cxt->zbuff->usize);
  
  if(u==0) { 
    if(v==0) {                      // top-left corner pixel
      N = 3;                        // number of neighbours
      di[0]=  1                    ; // E
      di[1]=  1 + cxt->zbuff->usize; // SE
      di[2]=      cxt->zbuff->usize; // S
    }
    else
      if (v==cxt->zbuff->vsize-1) { // bottom-left corner pixel
  	N = 3;
  	di[0]=    - cxt->zbuff->usize; // N
  	di[1]=  1 - cxt->zbuff->usize; // NE
  	di[2]=  1                    ; // E
      }
      else {                          // left edge pixel
  	N = 5;
  	di[0]=    - cxt->zbuff->usize; // N
  	di[1]=  1 - cxt->zbuff->usize; // NE
  	di[2]=  1                    ; // E
	di[3]=  1 + cxt->zbuff->usize; // SE
  	di[4]=      cxt->zbuff->usize; // S
      }
  }
  else
    if (u==cxt->zbuff->usize-1) {
      if(v==0) {                      // top-right corner pixel
  	N = 3;
  	di[0]=      cxt->zbuff->usize; // S
  	di[1]= -1 + cxt->zbuff->usize; // SW
  	di[2]= -1                    ; // W
      }
      else
  	if (v==cxt->zbuff->vsize-1) { // bottom-right corner pixel
  	  N = 3;
  	  di[0]= -1                    ; // W
  	  di[1]= -1 - cxt->zbuff->usize; // NW
  	  di[2]=    - cxt->zbuff->usize; // N
  	}
  	else {                        // right edge pixel
  	  N = 5;
  	  di[0]=      cxt->zbuff->usize; // S
  	  di[1]= -1 + cxt->zbuff->usize; // SW
  	  di[2]= -1                    ; // W
  	  di[3]= -1 - cxt->zbuff->usize; // NW
  	  di[4]=    - cxt->zbuff->usize; // N
  	}
    }
    else {
      if(v==0) {                      // top edge pixel
  	N = 5;
  	di[0]=  1                    ; // E
  	di[1]=  1 + cxt->zbuff->usize; // SE
  	di[2]=    + cxt->zbuff->usize; // S
  	di[3]= -1 + cxt->zbuff->usize; // SW
  	di[4]= -1                    ; // W
      }
      else
  	if (v==cxt->zbuff->vsize-1) { // bottom edge pixel
  	  N = 5;
  	  di[0]= -1                    ; // W
  	  di[1]= -1 - cxt->zbuff->usize; // NW
  	  di[2]=    - cxt->zbuff->usize; // N
  	  di[3]=  1 - cxt->zbuff->usize; // NE
  	  di[4]=  1                    ; // E
  	}
  	else {                        // central pixel
	  N = 8;
	  di[0]=    - cxt->zbuff->usize; // N
	  di[1]=  1 - cxt->zbuff->usize; // NE
	  di[2]=  1                    ; // E
	  di[3]=  1 + cxt->zbuff->usize; // SE
	  di[4]=      cxt->zbuff->usize; // S
	  di[5]= -1 + cxt->zbuff->usize; // SW
	  di[6]= -1                    ; // W
	  di[7]= -1 - cxt->zbuff->usize; // NW
	}
    }
  
  if(N == 8)
    M = N; // number of normals to be calculated 
  else
    M = N-1;
  
  // origin
  Vind = cxt->zbuff->voxel[Pind]; // voxel raster index
  p[0].z = (float) (Vind / xysize);         // voxel coordinates
  r      = Vind % xysize;
  p[0].y = (float) (r / cxt->xsize);
  p[0].x = (float) (r % cxt->xsize);
  
  // first neighbour
  Vind = cxt->zbuff->voxel[Pind+di[0]]; // voxel raster index
  if ( Vind == NIL) { /* background */
    // first vector
    w[1].x = 0.0;
    w[1].y = 0.0;
    w[1].z = 0.0;
  }
  else {
    p[1].z = (float) (Vind / xysize);              // voxel coordinates
    r = Vind % xysize;
    p[1].y = (float) (r / cxt->xsize);
    p[1].x = (float) (r % cxt->xsize);
    
    // first vector
    w[1].x = p[1].x - p[0].x;
    w[1].y = p[1].y - p[0].y;
    w[1].z = p[1].z - p[0].z;
  }
  
  for(i=1;i<=M;i++) {
    
    // second neighbour
    Vind = cxt->zbuff->voxel[Pind+di[i%N]];     // voxel raster index
    if ( Vind == NIL) { /* background */
      // second vector
      w[2].x = 0.0;
      w[2].y = 0.0;
      w[2].z = 0.0;
    }
    else {
      p[2].z = (float) (Vind / xysize);                   // voxel coordinates
      r = Vind % xysize;
      p[2].y = (float) (r / cxt->xsize);
      p[2].x = (float) (r % cxt->xsize);
      
      // second vector
      w[2].x = p[2].x - p[0].x;
      w[2].y = p[2].y - p[0].y; 
      w[2].z = p[2].z - p[0].z;
    }      
    
    // cross-product
    w[0] = VectorProd(w[2],w[1]);
    
    // normalization
    modn = sqrt((double) (w[0].x * w[0].x + w[0].y * w[0].y + w[0].z * w[0].z));
    if (modn > Epsilon) {
      w[0].x /= modn;
      w[0].y /= modn;
      w[0].z /= modn;
      
      normal->x += w[0].x;
      normal->y += w[0].y;
      normal->z += w[0].z;
    }
    
    //passar w[2] em w[1] para otimizar
    w[1].x = w[2].x;
    w[1].y = w[2].y;
    w[1].z = w[2].z;
  }
  
  // normalization
  modn = sqrt((double) (normal->x * normal->x + normal->y * normal->y + normal->z * normal->z));
  if (modn > Epsilon) {
    normal->x /= modn;
    normal->y /= modn;
    normal->z /= modn;
  }
  else {
    normal->x = 0.0;
    normal->y = 0.0;
    normal->z = 0.0;
  }
}

void SceneNormal(int Vind, Scene *scn, Vector *normal) {

  int u,v,w,xysize;
  int d[6]= {0,0,0,0,0,0};   // density for neighbours
  double modn;

  normal->x = normal->y = normal->z = 0.0;

  xysize = scn->xsize * scn->ysize;

  u = (Vind % xysize) % scn->xsize;
  v = (Vind % xysize) / scn->xsize;
  w = Vind / xysize;

  if (u > 0) {
    d[0] = scn->data[Vind - 1];
  }
  if (u < scn->xsize - 1) {
    d[1] = scn->data[Vind + 1];
  }

  if (v > 0) {
    d[2] = scn->data[Vind - scn->xsize];
  }
  if (v < scn->ysize - 1) {
    d[3] = scn->data[Vind + scn->xsize];
  }

  if (w > 0) {
    d[4] = scn->data[Vind - xysize];
  }
  if (w < scn->zsize - 1) {
    d[5] = scn->data[Vind + xysize];
  }

  normal->x = d[0] - d[1];
  normal->y = d[2] - d[3];
  normal->z = d[4] - d[5];

  // normalization
  modn = sqrt((double) (normal->x * normal->x + normal->y * normal->y + normal->z * normal->z));

  if (modn > Epsilon) {
    normal->x /= modn;
    normal->y /= modn;
    normal->z /= modn;
  }
  else {
    normal->x = 0.0;
    normal->y = 0.0;
    normal->z = 0.0;
  }
}


CImage *Colorize(Context *cxt, Image *img) 
{
  CImage *cimg=NULL;
  int i,j,n;
  float Y,Cb,Cr,R,G,B,max;

  cimg = CreateCImage(img->ncols,img->nrows);
  n = img->ncols*img->nrows;
  max = (float)MaximumValue(img);
  for(i=0; i < n; i++) 
    if (cxt->zbuff->voxel[i] != NIL) {
      j = cxt->zbuff->object[i];
      Y  = img->val[i] /max;// * cxt->obj[j].opac;
      Cb = cxt->obj[j].Cb;
      Cr = cxt->obj[j].Cr;

      R=296.82*(Y-0.062745098039)+
	406.98*(Cr-0.50196078431);
      
      G=296.82*(Y-0.062745098039)-
	207.315*(Cr-0.50196078431)-
	99.96*(Cb-0.50196078431);
      
      B=296.82*(Y-0.062745098039)+
	514.335*(Cb-0.50196078431);

/*       R=1.164*((float)Y-16.0)+ */
/* 	1.596*((float)Cr-128.0); */

/*       G=1.164*((float)Y-16.0)- */
/* 	0.813*((float)Cr-128.0)- */
/* 	0.392*((float)Cb-128.0); */
  
/*       B=1.164*((float)Y-16.0)+ */
/* 	2.017*((float)Cb-128.0); */

      if (R<0.0) R=0.0;
      if (G<0.0) G=0.0;
      if (B<0.0) B=0.0;
      if (R>255.0) R=255.0;
      if (G>255.0) G=255.0;
      if (B>255.0) B=255.0;
      cimg->C[0]->val[i]=(int)(R);
      cimg->C[1]->val[i]=(int)(G);
      cimg->C[2]->val[i]=(int)(B);
    } else { 
      if (img->val[i]==255){ /* Colorize Cube */
	cimg->C[0]->val[i]=255;
	cimg->C[1]->val[i]=255;
	cimg->C[2]->val[i]=255;
      }
    }
  return(cimg);
}

CImage *ColorizeSlice(Context *cxt, Image *img) 
{
  CImage *cimg=NULL;
  int i,j,n;
  float Y,Cb,Cr,R,G,B;
  cimg = CreateCImage(img->ncols,img->nrows);
  n = img->ncols*img->nrows;
  for(i=0; i < n; i++) {
    j = img->val[i];
    if (j > 0) {
      if (j > 255) j =0;
      Y  = cxt->obj[j].Y;
      Cb = cxt->obj[j].Cb;
      Cr = cxt->obj[j].Cr;

      R=296.82*(Y-0.062745098039)+
	406.98*(Cr-0.50196078431);
      
      G=296.82*(Y-0.062745098039)-
	207.315*(Cr-0.50196078431)-
	99.96*(Cb-0.50196078431);
      
      B=296.82*(Y-0.062745098039)+
	514.335*(Cb-0.50196078431);

      if (R<0.0) R=0.0;
      if (G<0.0) G=0.0;
      if (B<0.0) B=0.0;
      if (R>255.0) R=255.0;
      if (G>255.0) G=255.0;
      if (B>255.0) B=255.0;
      cimg->C[0]->val[i]=(int)(R);
      cimg->C[1]->val[i]=(int)(G);
      cimg->C[2]->val[i]=(int)(B);
    }
  }

  return(cimg);
}


void ColorizeCut(Context *cxt, CImage *cimg, Image *img) 
{
  int i,j,n;
  float Y,Cb,Cr,R,G,B;
  n = img->ncols*img->nrows;
  for(i=0; i < n; i++) {
    j = img->val[i];
    if (j > 0) {
      if (j > 255) j =0;
      Y  = cxt->obj[j].Y;
      Cb = cxt->obj[j].Cb;
      Cr = cxt->obj[j].Cr;

      R=296.82*(Y-0.062745098039)+
	406.98*(Cr-0.50196078431);
      
      G=296.82*(Y-0.062745098039)-
	207.315*(Cr-0.50196078431)-
	99.96*(Cb-0.50196078431);
      
      B=296.82*(Y-0.062745098039)+
	514.335*(Cb-0.50196078431);

      if (R<0.0) R=0.0;
      if (G<0.0) G=0.0;
      if (B<0.0) B=0.0;
      if (R>255.0) R=255.0;
      if (G>255.0) G=255.0;
      if (B>255.0) B=255.0;
      cimg->C[0]->val[i]=(int)(R);
      cimg->C[1]->val[i]=(int)(G);
      cimg->C[2]->val[i]=(int)(B);
    }
  }
}

CImage *OverlaySlice(Context *cxt, Image *obj, Image *img) 
{
  int n,i,j,k;
  float a;
  CImage *cimg;
  n = img->nrows*img->ncols;
  cimg = ColorizeSlice(cxt, obj);
  if (img != NULL) {
    for (i=0;i<n;i++) {
      k = obj->val[i];
      j = img->val[i] * 0.5; //(1.0-cxt->obj[k].opac);
      a = 0.5; //cxt->obj[k].opac;
      if (k != 0) {
	cimg->C[0]->val[i]=(int)(cimg->C[0]->val[i] * a + j);
	cimg->C[1]->val[i]=(int)(cimg->C[1]->val[i] * a + j);
	cimg->C[2]->val[i]=(int)(cimg->C[2]->val[i] * a + j);
      } else {
	cimg->C[0]->val[i] = img->val[i];
	cimg->C[1]->val[i] = img->val[i];
	cimg->C[2]->val[i] = img->val[i];
      }
    }
  }
  return(cimg);
}

void AccVoxelColor (CImage *cimg, Context *cxt, int i, float Y, float opac,uchar obj) {

  float Cb,Cr,R,G,B;
  
  Cb = cxt->obj[obj].Cb;
  Cr = cxt->obj[obj].Cr;
  
  R=296.82*(Y-0.062745098039)+
    406.98*(Cr-0.50196078431);
  
  G=296.82*(Y-0.062745098039)-
    207.315*(Cr-0.50196078431)-
    99.96*(Cb-0.50196078431);
  
  B=296.82*(Y-0.062745098039)+
    514.335*(Cb-0.50196078431);
   
  if (R<0.0) R=0.0;
  if (G<0.0) G=0.0;
  if (B<0.0) B=0.0;
  if (R>255.0) R=255.0;
  if (G>255.0) G=255.0;
  if (B>255.0) B=255.0;
  
  cimg->C[0]->val[i] += (uchar)(R*opac);
  cimg->C[1]->val[i] += (uchar)(G*opac);
  cimg->C[2]->val[i] += (uchar)(B*opac);
}

CImage *GetXCSlice(Scene *scn, int x)
{
  CImage *cimg=NULL;
  int i,j,n;
  Voxel v;

  v.x = x;
  cimg = CreateCImage(scn->ysize,scn->zsize);
  n    = (cimg->C[0]->ncols*cimg->C[0]->nrows)-1;            
  for (v.z=0;v.z<scn->zsize;v.z++)
    for (v.y=0;v.y<scn->ysize;v.y++) {
      i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
      if (scn->data[i] != 0) {
	j = cimg->C[0]->tbrow[v.z] + v.y;
	cimg->C[0]->val[n-j] = scn->data[i];
	cimg->C[1]->val[n-j] = scn->data[i];
	cimg->C[2]->val[n-j] = scn->data[i];
      }
    }
  return(cimg);
}

CImage *GetYCSlice(Scene *scn, int y)
{
  CImage *cimg=NULL;
  int i,j,n;
  Voxel v;

  v.y = y;
  cimg = CreateCImage(scn->xsize,scn->zsize);
  n    = (cimg->C[0]->ncols*cimg->C[0]->nrows)-1;
  for (v.z=0;v.z<scn->zsize;v.z++)
    for (v.x=0;v.x<scn->xsize;v.x++) {
      i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
      if (scn->data[i] != 0) {
	j = cimg->C[0]->tbrow[v.z] + v.x;
	cimg->C[0]->val[n-j] = scn->data[i];
	cimg->C[1]->val[n-j] = scn->data[i];
	cimg->C[2]->val[n-j] = scn->data[i];
      }
    }
  return(cimg);
}

CImage *GetZCSlice(Scene *scn, int z)
{
  CImage *cimg=NULL;
  int n; /* , Imax, s, n1; */
  int *data;
  
  
  cimg  = CreateCImage(scn->xsize,scn->ysize);
  n    = cimg->C[0]->ncols*cimg->C[0]->nrows;            
 
  data = scn->data + z*n;
  n    = n*sizeof(n);                 
  memcpy(cimg->C[0]->val, data, n);          
  memcpy(cimg->C[1]->val, data, n);          
  memcpy(cimg->C[2]->val, data, n);          
 

  return(cimg);
}

Image *FillSlice(Image *img) {

  Image *fimg=NULL,*fcost=NULL,*marker=NULL,*cost=NULL;
  AdjRel *A = NULL;
  int sz,imax,p,q,n,x,y,i,j,tmp;
  Queue *Q=NULL;
  AdjPxl *N=NULL;
  A = Circular(1.0);

  imax   = MaximumValue(img);
  marker   = CreateImage(img->ncols,img->nrows);
  SetImage(marker,imax+1);
  for (y=0; y < marker->nrows; y++) {
    i = marker->tbrow[y]; j = marker->ncols-1+marker->tbrow[y];
    marker->val[i] = img->val[i];
    marker->val[j] = img->val[j];
  }
  for (x=0; x < marker->ncols; x++) {
    i = x+marker->tbrow[0]; j = x+marker->tbrow[marker->nrows-1]; 
    marker->val[i] = img->val[i];
    marker->val[j] = img->val[j];
  }
  sz  = FrameSize(A);
  fimg = AddFrame(img,sz,INT_MIN);
  fcost = AddFrame(marker,sz,INT_MIN);
  N  = AdjPixels(fcost,A);
  n = fcost->ncols*fcost->nrows;
  Q = CreateQueue(imax+2,n);
  DestroyImage(&marker);
  DestroyAdjRel(&A);
  
  for(p = 0; p < n; p++) 
    if (fcost->val[p] != INT_MIN) {
      InsertQueue(Q,fcost->val[p],p);
    }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q]){
	if (fimg->val[p] && fimg->val[q]) {
	  tmp = fimg->val[q];
	} else {
	  tmp = MAX(fcost->val[p],fimg->val[q]);
	}
	if (tmp < fcost->val[q]){
	  UpdateQueue(Q,q,fcost->val[q],tmp);
	  fcost->val[q] = tmp;
	}	  
      }
    }
  }

  for(p = 0; p < n; p++) 
    if (fcost->val[p] > 255) {
      fcost->val[p] = 0;
    }
  
  cost = RemFrame(fcost,sz);

  DestroyQueue(&Q);
  DestroyImage(&fimg);
  DestroyImage(&fcost);
  DestroyAdjPxl(&N);
 
  return(cost);
}

Image *FillCutPlane(ZBuffer *zbuff, Image *img) {

  Image *fimg=NULL,*fcost=NULL,*marker=NULL,*cost=NULL;
  AdjRel *A = NULL;
  int sz,imax,p,q,n,x,y,i,j,tmp;
  Queue *Q=NULL;
  AdjPxl *N=NULL;
  A = Circular(1.0);
  
  imax   = MaximumValue(img);
  marker   = CreateImage(img->ncols,img->nrows);
  SetImage(marker,imax+1);
  for (y=0; y < marker->nrows; y++) {
    i = marker->tbrow[y]; j = marker->ncols-1+marker->tbrow[y];
    marker->val[i] = img->val[i];
    marker->val[j] = img->val[j];
  }
  for (x=0; x < marker->ncols; x++) {
    i = x+marker->tbrow[0]; j = x+marker->tbrow[marker->nrows-1]; 
    marker->val[i] = img->val[i];
    marker->val[j] = img->val[j];
  }
  sz  = FrameSize(A);
  fimg = AddFrame(img,sz,INT_MIN);
  fcost = AddFrame(marker,sz,INT_MIN);
  N  = AdjPixels(fcost,A);
  n = fcost->ncols*fcost->nrows;
  Q = CreateQueue(imax+2,n);
  DestroyImage(&marker);
  DestroyAdjRel(&A);
  
  for(p = 0; p < n; p++)
    if (fcost->val[p] != INT_MIN){
      InsertQueue(Q,fcost->val[p],p);
    }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q]){
	if (fimg->val[p] && fimg->val[q]) {
	  tmp = fimg->val[q];
	} else {
	  tmp = MAX(fcost->val[p],fimg->val[q]);
	}
	if (tmp < fcost->val[q]){
	  UpdateQueue(Q,q,fcost->val[q],tmp);
	  fcost->val[q] = tmp;
	}	  
      }
    }
  }
  
  cost = RemFrame(fcost,sz);

  DestroyQueue(&Q);
  DestroyImage(&fimg);
  DestroyImage(&fcost);
  DestroyAdjPxl(&N);
 
  return(cost);

}
void FillPlane(CImage *cimg, Context *cxt) {

  int n,i;
  float coef;
  n = cimg->C[0]->ncols*cimg->C[0]->nrows;
  coef = 255./(float)cxt->depth;
  for (i=0;i<n;i++) {
    cimg->C[2]->val[i] += (int) (coef*cxt->zbuff->dist[i]);
  }

}
