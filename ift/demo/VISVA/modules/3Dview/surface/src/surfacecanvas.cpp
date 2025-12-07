
#include "surfacecanvas.h"

namespace Surface{

  SurfaceCanvas :: SurfaceCanvas(wxWindow *parent) 
    : RenderCanvas(parent){
    light.x =  0;
    light.y =  0;
    light.z = -1;
    ka = 0.20;
    kd = 0.60;
    ks = 0.40;
    sre = 10;
    zgamma = 0.25;
    zstretch = 1.25;
    vxbuf = NULL;
    defaulthandler = new SurfaceHandler(this);
    SetInteractionHandler(defaulthandler);
  }


  void SurfaceCanvas :: SetInteractionHandler(InteractionHandler *handler){
    if(handler==NULL)
      this->handler = defaulthandler;
    else
      this->handler = handler;
  }

  int SurfaceCanvas :: GetSurfaceVoxel(int p){
    return this->vxbuf[p];
  }


  CImage *SurfaceCanvas :: Render2CImage(bool dataChanged, int skip){
    //int braincolor = 0xffeecc;
    Scene *label;
    static int *bmap=NULL;
    static int maplen=0;
    AdjRel3 *A;
    Voxel u,v;
    int i,p,q,n;
    CImage *cimg;

    if(!APP->Data.loaded) return NULL;

    if(skip<=0) skip = 1;

    n = APP->Data.w*APP->Data.h*APP->Data.nframes;
    label = APP->Data.label;

    if( dataChanged || bmap==NULL ){
      A = Spheric(1.0);
      if(bmap!=NULL) free(bmap);
      bmap = (int *)malloc(sizeof(int)*n);
      maplen = 0;

      for(p=0; p<n; p++){
	if(APP->GetLabelColour(label->data[p])==NIL)
	  continue;
	u.x = VoxelX(label, p);
	u.y = VoxelY(label, p);
	u.z = VoxelZ(label, p);

	for(i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if(ValidVoxel(label,v.x,v.y,v.z)){
	    q = VoxelAddress( label, v.x, v.y, v.z );
	    if(APP->GetLabelColour(label->data[q])==NIL){
	      bmap[maplen] = p;
	      maplen++;
	      break;
	    }
	  }
	  else{
	    bmap[maplen] = p;
	    maplen++;
	    break;
	  }
	}
      }
      DestroyAdjRel3(&A);
      bmap = (int *) realloc(bmap, sizeof(int) * maplen);
    }

    cimg = render_map(bmap, maplen, APP->Data.orig,
		      dataChanged, skip);
    return cimg;
  }


  void SurfaceCanvas :: drawRender(bool dataChanged, 
				   int skip){
    CImage *cimg_l,*cimg_r,*cimg;
    Matrix *brot;
    float delta = 3.7*(PI/180.0); //Toe-in angle.
    SurfaceView *view;
    bool is3D = false;
    
    if(!APP->Data.loaded) return;
    if(skip<=0) skip = 1;

    view = (SurfaceView *)this->GetParent();
    is3D = view->but3D->GetValue();

    if(is3D){
      brot = CopyMatrix(rot);
      RotateY(delta/2.0);
      cimg_l = Render2CImage(dataChanged, skip);
      //WriteCImage(cimg_l, "left.ppm");

      RotateY(-delta);
      cimg_r = Render2CImage(dataChanged, skip);
      //WriteCImage(cimg_r, "right.ppm");

      DestroyMatrix(&rot);
      rot = brot;
      this->rotChanged = true;
      cimg = Render2CImage(dataChanged, skip);
      //WriteCImage(cimg, "normal.ppm");

      DestroyCImage(&cimg);
      cimg = Anaglyph3D(cimg_l, cimg_r);
      DestroyCImage(&cimg_l);
      DestroyCImage(&cimg_r);
    }
    else{
      cimg = Render2CImage(dataChanged, skip);
    }

    DrawCImage(cimg);
    Refresh();
    
    DestroyCImage(&cimg);
  }


  CImage *SurfaceCanvas :: render_map(int *bmap, int maplen,
				      Scene *vol, 
				      bool dataChanged,
				      int skip){
    int w,h,w2,h2,W2,H2,D2,WH,W,H,D;
    int i,j,k,m,n,p,q;
    Scene *label;
    static float *xb=NULL,*yb=NULL,*normals=NULL,*zbuf=NULL;
    static int prevskip = NIL;
    static int prevCut = NIL;
    float fa,fb;
    Point A,B;
    Vector V1,V2;
    float diag,dz,kz,z,Y;
    int ca;
    int *pm;
    int splatsz;
    int px,py,pz;
    int vx,vy,vz;
    CImage *dest;
    float R[4][4];
    int maxval,c;
    Voxel Cut;
    SurfaceView *view;
    bool drawPlanes;

    if(skip<=0) skip = 1;

    label = APP->Data.label;

    view = (SurfaceView *)this->GetParent();
    drawPlanes = view->butPlanes->GetValue();

    Cut = APP->Get2DViewSlice();

    R[0][0] = rot->val[0]; R[0][1] = rot->val[1]; R[0][2] = rot->val[2]; R[0][3] = 0.0;
    R[1][0] = rot->val[3]; R[1][1] = rot->val[4]; R[1][2] = rot->val[5]; R[1][3] = 0.0;
    R[2][0] = rot->val[6]; R[2][1] = rot->val[7]; R[2][2] = rot->val[8]; R[2][3] = 0.0;
    R[3][0] = 0.0;         R[3][1] = 0.0;         R[3][2] = 0.0;         R[3][3] = 1.0;

    W = vol->xsize;
    H = vol->ysize;
    D = vol->zsize;

    maxval = MaximumValue3(vol);

    w = MAX(MAX(W,H),D);
    h = w;
    w2 = w/2;
    h2 = h/2;
    dest = CreateCImage(w,h);

    W2 = W / 2;
    H2 = H / 2;
    D2 = D / 2;
    WH = W*H;

    VectorNormalize(&light);

    if(this->rotChanged || dataChanged || 
       vxbuf==NULL || prevskip!=skip ||
       prevCut!=VoxelAddress(vol,Cut.x,Cut.y,Cut.z)){

      if(vxbuf!=NULL)   free(vxbuf);
      if(zbuf!=NULL)    free(zbuf);
      if(xb!=NULL)      free(xb);
      if(yb!=NULL)      free(yb);
      if(normals!=NULL) free(normals);

      /* init z-buffer and float (x,y) buffers (for normals) */
      vxbuf = (int *) malloc( w * h * sizeof(int) );
      zbuf = (float *) malloc( w * h * sizeof(float) );
      xb = (float *) malloc( w * h * sizeof(float) );
      yb = (float *) malloc( w * h * sizeof(float) );
      normals = (float *) malloc( w * h * sizeof(float) );

      for(i=0;i<w*h;i++){
	zbuf[i] = FLT_MAX;
	xb[i] = 0.0;
	yb[i] = 0.0;
	normals[i] = 0.0;
	vxbuf[i] = NIL;
      }

      /* make projection */
      pm = bmap;
      for(i=0;i<maplen;i+=skip,pm+=skip) {
      
	n = *pm;
      
	pz = n / WH;
	py = (n % WH) / W;
	px = (n % WH) % W;

	A.x = px - W2;
	A.y = py - H2;
	A.z = pz - D2;

	V1.x = A.x;
	V1.y = A.y;
	V1.z = A.z;
	V2 = VectorRotate(V1, R);
	B.x = V2.x;
	B.y = V2.y;
	B.z = V2.z;

	px = w2 + (int)B.x;
	py = h2 + (int)B.y;

	if (px < 0 || py < 0 || px >= w || py >= h) continue;
      
	p = px + py*w;
      
	if (B.z < zbuf[p]) {
	  zbuf[p] = B.z;
	  xb[p] = B.x;
	  yb[p] = B.y;
	  vxbuf[p] = n;
	}
      }

      /* forward-only 2x2 splatting */

      splatsz = 2;
      if (skip > 2) splatsz = 8;

      for(j=h-splatsz;j>=0;j--)
	for(i=w-splatsz;i>=0;i--) {
	  p = i + j*w;
	  if (zbuf[p] != FLT_MAX) {
	    fa = zbuf[p];
	    for(k=0;k<splatsz;k++)
	      for(m=0;m<splatsz;m++) {
		fb = zbuf[q = p + k + m*w];
		if (fa < fb) {
		  zbuf[q] = fa;
		  xb[q] = xb[p] + k;
		  yb[q] = yb[p] + m;
		  vxbuf[q] = vxbuf[p];
		}
	      }
	  }
	}

      /* compute normals */
    
      for(j=2;j<h-2;j++)
	for(i=2;i<w-2;i++)
	  if (zbuf[i+j*w] != FLT_MAX)
	    render_calc_normal(i,j,w,xb,yb,zbuf,normals);
    
    }

    /* render the monster */
    A.x = W;
    A.y = H;
    A.z = D;
    diag = VectorMagnitude((Vector *)&A) / 2.0;

    for(j=0;j<h;j++){
      for(i=0;i<w;i++) {
	p = i+j*w;

	if (zbuf[p] != FLT_MAX) {
	  z = zbuf[p];
	  dz = (diag - z);
	  kz = (dz*dz) / (4.0*diag*diag);

	  kz = zgamma + zstretch * kz;

	  fa = normals[p];
	  fb = phong_specular(fa, sre);
	
	  c = APP->GetLabelColour(label->data[vxbuf[p]]);
	
	  ca=RGB2YCbCr(c);
	  Y = (float) t0(ca);
	  Y /= 255.0;
	
	  Y = ka + kz * ( kd * Y * fa + ks * Y * fb);

	  Y *= 255.0;
	  if (Y > 255.0) Y=255.0;
	  if (Y < 0.0) Y=0.0;
	  ca = triplet((int) Y, t1(ca), t2(ca));
	  ca = YCbCr2RGB(ca);

	  dest->C[0]->val[p] = t0(ca);
	  dest->C[1]->val[p] = t1(ca);
	  dest->C[2]->val[p] = t2(ca);
	}
      }
    }

    /********* Draw Planes ********/
    if(drawPlanes){
      /* Plane XY - make projection */
      vz = Cut.z;
      for(vx = 0; vx < vol->xsize; vx++) {
	for(vy = 0; vy < vol->ysize; vy++){
	  n = VoxelAddress(vol,vx,vy,vz);
	  V1.x = vx - W2;
	  V1.y = vy - H2;
	  V1.z = vz - D2;
	  V2 = VectorRotate(V1, R);
	  B.x = V2.x;
	  B.y = V2.y;
	  B.z = V2.z;
	  
	  px = w2 + (int)B.x;
	  py = h2 + (int)B.y;
	  
	  if (px < 0 || py < 0 || px >= w || py >= h) continue;
	  p = px + py*w;
	  
	  if (B.z < zbuf[p]) {
	    zbuf[p] = B.z;
	    xb[p] = B.x;
	    yb[p] = B.y;
	    vxbuf[p] = n;
	  }
	}
      }

      /* Plane XZ - make projection */
      vy = Cut.y;
      for(vx = 0; vx < vol->xsize; vx++) {
	for(vz = 0; vz < vol->zsize; vz++){
	  n = VoxelAddress(vol,vx,vy,vz);
	  V1.x = vx - W2;
	  V1.y = vy - H2;
	  V1.z = vz - D2;
	  V2 = VectorRotate(V1, R);
	  B.x = V2.x;
	  B.y = V2.y;
	  B.z = V2.z;
	  
	  px = w2 + (int)B.x;
	  py = h2 + (int)B.y;
	
	  if (px < 0 || py < 0 || px >= w || py >= h) continue;
	  p = px + py*w;
	
	  if (B.z < zbuf[p]) {
	    zbuf[p] = B.z;
	    xb[p] = B.x;
	    yb[p] = B.y;
	    vxbuf[p] = n;
	  }
	}
      }
      
      /* Plane YZ - make projection */
      vx = Cut.x;
      for(vy = 0; vy < vol->ysize; vy++) {
	for(vz = 0; vz < vol->zsize; vz++){
	  n = VoxelAddress(vol,vx,vy,vz);
	  V1.x = vx - W2;
	  V1.y = vy - H2;
	  V1.z = vz - D2;
	  V2 = VectorRotate(V1, R);
	  B.x = V2.x;
	  B.y = V2.y;
	  B.z = V2.z;
	
	  px = w2 + (int)B.x;
	  py = h2 + (int)B.y;
	  
	  if (px < 0 || py < 0 || px >= w || py >= h) continue;
	  p = px + py*w;
	  
	  if (B.z < zbuf[p]) {
	    zbuf[p] = B.z;
	    xb[p] = B.x;
	    yb[p] = B.y;
	    vxbuf[p] = n;
	  }
	}
      }

    
      /* forward-only 2x2 splatting */
      if(this->rotChanged || dataChanged || prevskip!=skip ||
	 prevCut!=VoxelAddress(vol,Cut.x,Cut.y,Cut.z)){
	
	splatsz = 2;
	for(j=h-splatsz;j>=0;j--) {
	  for(i=w-splatsz;i>=0;i--) {
	    p = i + j*w;
	    if (zbuf[p] != FLT_MAX) {
	      c = APP->GetLabelColour(label->data[vxbuf[p]]);
	      if(c!=NIL) continue;
	      fa = zbuf[p];
	      for(k=0;k<splatsz;k++)
		for(m=0;m<splatsz;m++) {
		  fb = zbuf[q = p + k + m*w];
		  if (fa < fb-2) {
		    zbuf[q] = fa;
		    xb[q] = xb[p] + k;
		    yb[q] = yb[p] + m;
		    vxbuf[q] = vxbuf[p];
		  }
		}
	    }
	  }
	}
      }
      
      for(j=0;j<h;j++){
	for(i=0;i<w;i++) {
	  p = i+j*w;
	  
	  if (zbuf[p] != FLT_MAX) {
	    c = APP->GetLabelColour(label->data[vxbuf[p]]);
	    if(c == NIL){
	      ca = xgray(IntegerNormalize(vol->data[vxbuf[p]],0,maxval,0,255));
	      dest->C[0]->val[p] = t0(ca);
	      dest->C[1]->val[p] = t1(ca);
	      dest->C[2]->val[p] = t2(ca);
	    }
	  }
	}
      }

    }/*draw Planes.*/

    this->rotChanged = false;
    prevskip = skip;
    prevCut = VoxelAddress(vol, Cut.x, Cut.y, Cut.z);

    return dest;
  }


  void SurfaceCanvas :: render_calc_normal(int x,int y,int rw, float *xb,float *yb,float *zb,float *out){
    int nx[8] = {  0,  1,  1,  1,  0, -1, -1, -1 }; /* adj. viz 8 */
    int ny[8] = { -1, -1,  0,  1,  1,  1,  0, -1 };

    int qx[8] = {  1,  2,  2,  1, -1, -2, -2, -1 };
    int qy[8] = { -2, -1,  1,  2,  2,  1, -1, -2 };

    unsigned int i;
    int k, nv, a;
    float est;
  
    Vector p[3], w[3], normal = {0.0, 0.0, 0.0};

    nv = 0;
    k  = 1;
    for(i=0;i<8;i++) {

      a = x + y*rw;
      if (zb[a] == FLT_MAX) continue;
      p[0].x = xb[a];   p[0].y = yb[a];   p[0].z = zb[a];

      a = (x+nx[(i+k)%8]) + (y+ny[(i+k)%8])*rw;
      if (zb[a] == FLT_MAX) continue;
      p[1].x = xb[a];   p[1].y = yb[a];   p[1].z = zb[a];

      a = (x+nx[i]) + (y+ny[i])*rw;
      if (zb[a] == FLT_MAX) continue;
      p[2].x = xb[a];   p[2].y = yb[a];   p[2].z = zb[a];

      w[0].x = p[1].x - p[0].x;
      w[0].y = p[1].y - p[0].y;
      w[0].z = p[1].z - p[0].z;
    
      w[1].x = p[2].x - p[0].x;
      w[1].y = p[2].y - p[0].y;
      w[1].z = p[2].z - p[0].z;
    
      w[2] = VectorProd(w[0], w[1]);
      VectorNormalize(&w[2]);
    
      ++nv;
      normal.x += w[2].x;
      normal.y += w[2].y;
      normal.z += w[2].z;
    }

    for(i=0;i<8;i++) {

      a = x + y*rw;
      if (zb[a] == FLT_MAX) continue;
      p[0].x = xb[a];   p[0].y = yb[a];   p[0].z = zb[a];

      a = (x+qx[(i+k)%8]) + (y+qy[(i+k)%8])*rw;
      if (zb[a] == FLT_MAX) continue;
      p[1].x = xb[a];   p[1].y = yb[a];   p[1].z = zb[a];

      a = (x+qx[i]) + (y+qy[i])*rw;
      if (zb[a] == FLT_MAX) continue;
      p[2].x = xb[a];   p[2].y = yb[a];   p[2].z = zb[a];
    
      w[0].x = p[1].x - p[0].x;
      w[0].y = p[1].y - p[0].y;
      w[0].z = p[1].z - p[0].z;
    
      w[1].x = p[2].x - p[0].x;
      w[1].y = p[2].y - p[0].y;
      w[1].z = p[2].z - p[0].z;
    
      w[2] = VectorProd(w[0], w[1]);
      VectorNormalize(&w[2]);
    
      ++nv;
      normal.x += w[2].x;
      normal.y += w[2].y;
      normal.z += w[2].z;
    }

    if (!nv) {
      est = 1.0;
    } else {
      VectorNormalize(&normal);
      est = ScalarProd(light, normal);
      if (est < 0.0) est=0.0;
    }

    a = x + rw*y;
    out[a] = est;
  }


  float SurfaceCanvas :: phong_specular(float angcos, int n) {
    float a,r;

    a=acos(angcos);
    if (a > M_PI / 4.0)
      return 0.0;

    a=cos(2.0 * a);
    r=a;
    while(n!=1) { r*=a; --n; }
    return r;
  }



  CImage *SurfaceCanvas :: Anaglyph3D(CImage *cimg_l, CImage *cimg_r){
    CImage *cimg;
    int j,i,p;
    if(cimg_l->C[0]->ncols != cimg_r->C[0]->ncols ||
       cimg_l->C[0]->nrows != cimg_r->C[0]->nrows){
      Error((char *)"Incompatible images",(char *)"Anaglyph3D");
    }
    cimg = CreateCImage(cimg_l->C[0]->ncols, cimg_l->C[0]->nrows);
    for(i=0; i<cimg_l->C[0]->nrows; i++){
      for(j=0; j<cimg_l->C[0]->ncols; j++){
	p = j+i*cimg_l->C[0]->ncols;
	(cimg->C[0])->val[p] = (cimg_l->C[0])->val[p];
	(cimg->C[1])->val[p] = (cimg_r->C[1])->val[p];
	(cimg->C[2])->val[p] = (cimg_r->C[2])->val[p];
      }
    }
    return cimg;
  }


  int SurfaceCanvas :: xgray(int value) {
    float v,width,level,l1,l2;
    int B,C;

    APP->GetBriContr(&B, &C);
    v = (float) value;  
    level = (1.0 - (float)B/100.0) * 255.0; 
    width = (1.0 - (float)C/100.0) * 255.0;
    l1 = level - width/2.0;
    l2 = level + width/2.0;
    if(l1<0)   l1 = 0.0;
    if(l2>255) l2 = 255.0;

    if(value < l1) v = 0.0;
    else if(value >= l2) v = 255.0;
    else{
      v = (value - l1)/(l2-l1);
      v *= 255.0;
    }

    return(gray((int)v));
  }


} //end Surface namespace


