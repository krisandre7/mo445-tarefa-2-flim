
#include "rayleapingcanvas.h"

namespace Rayleaping{

  RayleapingCanvas :: RayleapingCanvas(wxWindow *parent) 
    : RenderCanvas(parent){
    Vpj.x = 1.0; Vpj.y = 0.0; Vpj.z = 0.0;
    Vpi.x = 0.0; Vpi.y = 1.0; Vpi.z = 0.0;
    light.x =  0;
    light.y =  0;
    light.z = -1;
    ka = 0.20;
    kd = 0.60;
    ks = 0.40;
    sre = 10;
    zgamma = 0.25;
    zstretch = 1.25;
    zbuf = NULL;
    vxbuf = NULL;
    normals = NULL;
    defaulthandler = new RayleapingHandler(this);
    SetInteractionHandler(defaulthandler);
  }


  void RayleapingCanvas :: SetInteractionHandler(InteractionHandler *handler){
    if(handler==NULL)
      this->handler = defaulthandler;
    else
      this->handler = handler;
  }

  int RayleapingCanvas :: GetSurfaceVoxel(int p){
    return this->vxbuf[p];
  }


  CImage *RayleapingCanvas :: Render2CImage(int skip){
    //int braincolor = 0xffeecc;
    Scene *label;
    Voxel v;
    float diag;
    float Dco; //distancia do observador
    float Dpo; //distancia focal
    float dx,dy,di,dj,num,den;
    float d,dmin,dmax;
    float T[3];
    int p,maxsize,i,j,k,kmin,kmax;
    CImage *cimg;
    Vector Voc,Voa,X;
    //Vector Vpj,Vpi;
    Face faces[6];
    Point P,O,C,A;
    Point Ray_enter,Ray_exit,Ray_point,Ray_back;
    float R[4][4];
    //int color[6];
    int voxel_color;
    float kz,dz,fa,fb,Y;
    int c,ca,step,r;
    int step_size = 2;
    int step_back_size = 2;

    if(!APP->Data.loaded) return NULL;

    if(skip<=0) skip = 1;

    //color[0] = 0xff0000; color[1] = 0x00ff00; color[2] = 0x0000ff;
    //color[3] = 0xffff00; color[4] = 0x00ffff; color[5] = 0xffffff;

    // Plane xz, y=0:
    faces[0].V.x = 0.0; faces[0].V.y =  0.0; faces[0].V.z = 0.0;
    faces[0].N.x = 0.0; faces[0].N.y = -1.0; faces[0].N.z = 0.0;
    faces[0].fixedindex = 1;
    faces[0].fixedval = 0;
    //Plane xz, y=max:
    faces[1].V.x = 0.0; faces[1].V.y = APP->Data.h-1; faces[1].V.z = 0.0;
    faces[1].N.x = 0.0; faces[1].N.y = 1.0; faces[1].N.z = 0.0;
    faces[1].fixedindex = 1;
    faces[1].fixedval = APP->Data.h-1;
    //Plane yz, x=0:
    faces[2].V.x =  0.0; faces[2].V.y = 0.0; faces[2].V.z = 0.0;
    faces[2].N.x = -1.0; faces[2].N.y = 0.0; faces[2].N.z = 0.0;
    faces[2].fixedindex = 0;
    faces[2].fixedval = 0;
    //Plane yz, x=max:
    faces[3].V.x = APP->Data.w-1; faces[3].V.y = 0.0; faces[3].V.z = 0.0;
    faces[3].N.x = 1.0; faces[3].N.y = 0.0; faces[3].N.z = 0.0;
    faces[3].fixedindex = 0;
    faces[3].fixedval = APP->Data.w-1;
    //Plane xy, z=0:
    faces[4].V.x = 0.0; faces[4].V.y = 0.0; faces[4].V.z =  0.0;
    faces[4].N.x = 0.0; faces[4].N.y = 0.0; faces[4].N.z = -1.0;
    faces[4].fixedindex = 2;
    faces[4].fixedval = 0;
    //Plane xy, z=max:
    faces[5].V.x = 0.0; faces[5].V.y = 0.0; faces[5].V.z = APP->Data.nframes-1;
    faces[5].N.x = 0.0; faces[5].N.y = 0.0; faces[5].N.z = 1.0;
    faces[5].fixedindex = 2;
    faces[5].fixedval = APP->Data.nframes-1;

    R[0][0] = rot->val[0]; R[0][1] = rot->val[1]; R[0][2] = rot->val[2]; R[0][3] = 0.0;
    R[1][0] = rot->val[3]; R[1][1] = rot->val[4]; R[1][2] = rot->val[5]; R[1][3] = 0.0;
    R[2][0] = rot->val[6]; R[2][1] = rot->val[7]; R[2][2] = rot->val[8]; R[2][3] = 0.0;
    R[3][0] = 0.0;         R[3][1] = 0.0;         R[3][2] = 0.0;         R[3][3] = 1.0;

    maxsize = MAX(MAX(APP->Data.w, APP->Data.h), APP->Data.nframes);
    if(maxsize%2 == 0) maxsize++;
    diag = sqrtf((float)APP->Data.w*APP->Data.w + 
		 (float)APP->Data.h*APP->Data.h + 
		 (float)APP->Data.nframes*APP->Data.nframes);
    Dco = (diag/2.0) * 1.5; //3.0;
    Dpo = Dco - (diag/2.0) * (-0.5); //1.0;
    dx = dy = 1.0;

    Voc.x = 0.0; Voc.y = 0.0; Voc.z = 1.0;
    Vpj.x = 1.0; Vpj.y = 0.0; Vpj.z = 0.0;
    Vpi.x = 0.0; Vpi.y = 1.0; Vpi.z = 0.0;

    cimg = CreateCImage(maxsize, maxsize);

    if(vxbuf == NULL){
      vxbuf = (int *) malloc( cimg->C[0]->nrows * cimg->C[0]->ncols * sizeof(int) );
    }
    if(normals == NULL){
      normals = (float *) malloc( cimg->C[0]->nrows * cimg->C[0]->ncols * sizeof(float) );
    }
    if(zbuf == NULL){
      zbuf = (float *) malloc( cimg->C[0]->nrows * cimg->C[0]->ncols * sizeof(float) );
    }

    label = APP->Data.label;

    //if( dataChanged ){}
    //if(APP->GetLabelColour(label->data[p])==NIL)
    //---------------------------
    Voc = VectorRotate(Voc, R);
    Vpj = VectorRotate(Vpj, R);
    Vpi = VectorRotate(Vpi, R);
    C.x = APP->Data.w/2.0;
    C.y = APP->Data.h/2.0;
    C.z = APP->Data.nframes/2.0;
    O.x = C.x + Dco*(-Voc.x);
    O.y = C.y + Dco*(-Voc.y);
    O.z = C.z + Dco*(-Voc.z);
    P.x = O.x + Voc.x*Dpo;
    P.y = O.y + Voc.y*Dpo;
    P.z = O.z + Voc.z*Dpo;

    light.x = -Voc.x;
    light.y = -Voc.y;
    light.z = -Voc.z;
    VectorNormalize(&light);

    for(i = 0; i < cimg->C[0]->nrows; i++){
      for(j = 0; j < cimg->C[0]->ncols; j++){
	p = j + i*cimg->C[0]->ncols;
	zbuf[p] = FLT_MAX;
	vxbuf[p] = NIL;
	normals[p] = 0.0;

	di = (i - cimg->C[0]->nrows/2)*dy;
	dj = (j - cimg->C[0]->ncols/2)*dx;
	A.x = P.x + dj*Vpj.x + di*Vpi.x;
	A.y = P.y + dj*Vpj.y + di*Vpi.y;
	A.z = P.z + dj*Vpj.z + di*Vpi.z;
	Voa.x = A.x - O.x;
	Voa.y = A.y - O.y;
	Voa.z = A.z - O.z;
	VectorNormalize(&Voa);

	dmin = FLT_MAX;
	dmax = -FLT_MAX;
	kmin = kmax = -1;
	for(k = 0; k < 6; k++){
	  X.x = faces[k].V.x - O.x;
	  X.y = faces[k].V.y - O.y;
	  X.z = faces[k].V.z - O.z;

	  num = ScalarProd(X, faces[k].N);
	  den = ScalarProd(Voa, faces[k].N);

	  if(den >  0.000001 ||
	     den < -0.000001 ){
	    d = num/den;
	    T[0] = d*Voa.x + O.x;
	    T[1] = d*Voa.y + O.y;
	    T[2] = d*Voa.z + O.z;
	    T[faces[k].fixedindex] = faces[k].fixedval;
	    v.x = ROUND(T[0]);
	    v.y = ROUND(T[1]);
	    v.z = ROUND(T[2]);
	    if( ValidVoxel(label, v.x, v.y, v.z) ){
	      if(d < dmin){ 
		Ray_enter.x = T[0];
		Ray_enter.y = T[1];
		Ray_enter.z = T[2];
		zbuf[p] = d;
		dmin = d;
		kmin = k;
	      }
	      if(d > dmax){ 
		Ray_exit.x = T[0];
		Ray_exit.y = T[1];
		Ray_exit.z = T[2];
		dmax = d; 
		kmax = k; 
	      }
	    }
	  }
	}
	if(kmin != -1 && kmax != -1 && kmin != kmax){
	  Ray_point.x = Ray_enter.x + Voa.x * step_size;
	  Ray_point.y = Ray_enter.y + Voa.y * step_size;
	  Ray_point.z = Ray_enter.z + Voa.z * step_size;
	  v.x = ROUND(Ray_point.x);
	  v.y = ROUND(Ray_point.y);
	  v.z = ROUND(Ray_point.z);

	  while( ValidVoxel(label, v.x, v.y, v.z) ){
	    r = VoxelAddress(label, v.x, v.y, v.z);

	    if(APP->GetLabelColour(label->data[r])!=NIL){
	      break;
	    }

	    Ray_point.x = Ray_point.x + Voa.x * step_size;
	    Ray_point.y = Ray_point.y + Voa.y * step_size;
	    Ray_point.z = Ray_point.z + Voa.z * step_size;
	    v.x = ROUND(Ray_point.x);
	    v.y = ROUND(Ray_point.y);
	    v.z = ROUND(Ray_point.z);
	  }

	  if(!ValidVoxel(label, v.x, v.y, v.z)) continue;

	  Ray_back = Ray_point;
	  for(step = 1; step <= step_back_size; step++){
	    Ray_point.x = Ray_point.x - Voa.x;
	    Ray_point.y = Ray_point.y - Voa.y;
	    Ray_point.z = Ray_point.z - Voa.z;
	    v.x = ROUND(Ray_point.x);
	    v.y = ROUND(Ray_point.y);
	    v.z = ROUND(Ray_point.z);
	    if(ValidVoxel(label, v.x, v.y, v.z)){
	      r = VoxelAddress(label, v.x, v.y, v.z);
	      if(APP->GetLabelColour(label->data[r])!=NIL){
		Ray_back = Ray_point;
	      }
	      else break;
	    }
	    else break;
	  }

	  if(step <= step_back_size){ //draw_pixel
	    /*
	      v.x = ROUND(Ray_enter.x);
	      v.y = ROUND(Ray_enter.y);
	      v.z = ROUND(Ray_enter.z);
	      vxbuf[p] = VoxelAddress(label, v.x, v.y, v.z);
	      cimg->C[0]->val[p] = t0(color[kmin]);
	      cimg->C[1]->val[p] = t1(color[kmin]);
	      cimg->C[2]->val[p] = t2(color[kmin]);
	    */
	    v.x = ROUND(Ray_back.x);
	    v.y = ROUND(Ray_back.y);
	    v.z = ROUND(Ray_back.z);
	    if(ValidVoxel(label, v.x, v.y, v.z)){
	      r = VoxelAddress(label, v.x, v.y, v.z);
	      vxbuf[p] = r;
	      voxel_color = APP->GetLabelColour(label->data[r]);
	      cimg->C[0]->val[p] = t0(voxel_color);
	      cimg->C[1]->val[p] = t1(voxel_color);
	      cimg->C[2]->val[p] = t2(voxel_color);
	    }
	  }

	}
      }
    }
 
    // render the monster 

    for(i = 0; i < cimg->C[0]->nrows; i++){
      for(j = 0; j < cimg->C[0]->ncols; j++){
	p = j + i*cimg->C[0]->ncols;

	if(vxbuf[p]!=NIL){
	  render_calc_normal(j, i, cimg->C[0]->ncols, vxbuf, normals);

	  //dz = (diag/2.0 - zbuf[p]);
	  //kz = (dz*dz) / (diag*diag);

	  dz = 1.0 - (zbuf[p] - (Dco-diag/2.0))/diag;
	  kz = (dz*dz);

	  kz = zgamma + zstretch * kz;
	  
	  fa = normals[p];
	  fb = phong_specular(fa, sre);

	  //c = APP->GetLabelColour(label->data[vxbuf[p]]);
	  c = triplet(cimg->C[0]->val[p],
		      cimg->C[1]->val[p],
		      cimg->C[2]->val[p]);

	  ca = RGB2YCbCr(c);
	  Y = (float) t0(ca);
	  Y /= 255.0;
	
	  Y = ka + kz * ( kd * Y * fa + ks * Y * fb);

	  Y *= 255.0;
	  if (Y > 255.0) Y=255.0;
	  if (Y < 0.0) Y=0.0;
	  ca = triplet((int) Y, t1(ca), t2(ca));
	  ca = YCbCr2RGB(ca);

	  cimg->C[0]->val[p] = t0(ca);
	  cimg->C[1]->val[p] = t1(ca);
	  cimg->C[2]->val[p] = t2(ca);
	}
      }
    }
    this->rotChanged = false;
    return cimg;
  }


  void RayleapingCanvas :: drawRender(int skip){
    CImage *cimg_l,*cimg_r,*cimg;
    Matrix *brot;
    float delta = 3.7*(PI/180.0); //Toe-in angle.
    RayleapingView *view;
    bool is3D = false;

    if(!APP->Data.loaded) return;
    if(skip<=0) skip = 1;

    view = (RayleapingView *)this->GetParent();
    is3D = view->but3D->GetValue();

    if(is3D){
      brot = CopyMatrix(rot);
      Rotate( delta/2.0, Vpi);
      cimg_l = Render2CImage(skip);
      //WriteCImage(cimg_l, "left.ppm");

      Rotate(-delta, Vpi);
      cimg_r = Render2CImage(skip);
      //WriteCImage(cimg_r, "right.ppm");

      DestroyMatrix(&rot);
      rot = brot;
      this->rotChanged = true;
      cimg = Render2CImage(skip);
      //WriteCImage(cimg, "normal.ppm");

      DestroyCImage(&cimg);
      cimg = Anaglyph3D(cimg_l, cimg_r);
      DestroyCImage(&cimg_l);
      DestroyCImage(&cimg_r);
    }
    else{
      cimg = Render2CImage(skip);
    }

    DrawCImage(cimg);
    Refresh();

    DestroyCImage(&cimg);
  }

  /*
  void RayleapingCanvas :: render_map(int *bmap, int maplen,
				      Scene *vol, 
				      bool dataChanged,
				      int skip){
    int w,h,w2,h2,W2,H2,D2,WH,W,H,D;
    int i,j,k,m,n,p,q;
    Scene *label;
    static float *xb=NULL,*yb=NULL,*normals=NULL,*zbuf=NULL;
    static int prevskip = NIL;
    float fa,fb;
    Point A,B;
    Vector V1,V2;
    float diag,dz,kz,z,Y;
    int ca;
    int *pm;
    int splatsz;
    int px,py,pz;
    CImage *dest;
    float R[4][4];
    int maxval,c;

    if(skip<=0) skip = 1;

    label = APP->Data.label;

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
       vxbuf==NULL || prevskip!=skip){

      if(vxbuf!=NULL)   free(vxbuf);
      if(zbuf!=NULL)    free(zbuf);
      if(xb!=NULL)      free(xb);
      if(yb!=NULL)      free(yb);
      if(normals!=NULL) free(normals);

      // init z-buffer and float (x,y) buffers (for normals)
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

      // make projection
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

      // forward-only 2x2 splatting

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

      // compute normals
    
      for(j=2;j<h-2;j++)
	for(i=2;i<w-2;i++)
	  if (zbuf[i+j*w] != FLT_MAX)
	    render_calc_normal(i,j,w,xb,yb,zbuf,normals);
    
    }

    // render the monster 
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

    DrawCImage(dest);
    Refresh();
      
    DestroyCImage(&dest);
    this->rotChanged = false;
    prevskip = skip;
  }
  */

  void RayleapingCanvas :: render_calc_normal(int x,int y, int rw, 
					      int *vxbuf, float *out){
    int nx[8] = {  0,  1,  1,  1,  0, -1, -1, -1 }; // adj. viz 8 
    int ny[8] = { -1, -1,  0,  1,  1,  1,  0, -1 };

    int qx[8] = {  1,  2,  2,  1, -1, -2, -2, -1 };
    int qy[8] = { -2, -1,  1,  2,  2,  1, -1, -2 };

    Scene *label = APP->Data.label;
    unsigned int i;
    int k, nv, a;
    float est;
  
    Vector p[3], w[3], normal = {0.0, 0.0, 0.0};

    nv = 0;
    k  = 1;
    for(i=0;i<8;i++) {

      a = x + y*rw;
      if (vxbuf[a] == NIL) continue;
      p[0].x = VoxelX(label, vxbuf[a]);  
      p[0].y = VoxelY(label, vxbuf[a]);
      p[0].z = VoxelZ(label, vxbuf[a]);

      a = (x+nx[(i+k)%8]) + (y+ny[(i+k)%8])*rw;
      if (vxbuf[a] == NIL) continue;
      p[1].x = VoxelX(label, vxbuf[a]);
      p[1].y = VoxelY(label, vxbuf[a]);
      p[1].z = VoxelZ(label, vxbuf[a]);

      a = (x+nx[i]) + (y+ny[i])*rw;
      if (vxbuf[a] == NIL) continue;
      p[2].x = VoxelX(label, vxbuf[a]);
      p[2].y = VoxelY(label, vxbuf[a]);
      p[2].z = VoxelZ(label, vxbuf[a]);

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
      if (vxbuf[a] == NIL) continue;
      p[0].x = VoxelX(label, vxbuf[a]);
      p[0].y = VoxelY(label, vxbuf[a]);
      p[0].z = VoxelZ(label, vxbuf[a]);

      a = (x+qx[(i+k)%8]) + (y+qy[(i+k)%8])*rw;
      if (vxbuf[a] == NIL) continue;
      p[1].x = VoxelX(label, vxbuf[a]);
      p[1].y = VoxelY(label, vxbuf[a]);
      p[1].z = VoxelZ(label, vxbuf[a]);

      a = (x+qx[i]) + (y+qy[i])*rw;
      if (vxbuf[a] == NIL) continue;
      p[2].x = VoxelX(label, vxbuf[a]);
      p[2].y = VoxelY(label, vxbuf[a]);
      p[2].z = VoxelZ(label, vxbuf[a]);
    
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


  float RayleapingCanvas :: phong_specular(float angcos, int n) {
    float a,r;

    a=acos(angcos);
    if (a > M_PI / 4.0)
      return 0.0;

    a=cos(2.0 * a);
    r=a;
    while(n!=1) { r*=a; --n; }
    return r;
  }


  CImage *RayleapingCanvas :: Anaglyph3D(CImage *cimg_l, CImage *cimg_r){
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


} //end Rayleaping namespace


