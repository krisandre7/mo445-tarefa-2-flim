
#include "gradient.h"
#include "bz2_lib.h"


ScnGradient *CreateScnGradient(int xsize,int ysize,int zsize){
  ScnGradient *grad=NULL;

  grad = (ScnGradient *) calloc(1,sizeof(ScnGradient));
  if(grad == NULL)
    Error(MSG1,"CreateScnGradient");

  grad->Gx = CreateScene(xsize,ysize,zsize);
  grad->Gy = CreateScene(xsize,ysize,zsize);
  grad->Gz = CreateScene(xsize,ysize,zsize);
  grad->mag = NULL;
  return grad;
}


void       DestroyScnGradient(ScnGradient **grad){
  ScnGradient *aux;

  aux = *grad;
  if(aux != NULL){
    if(aux->Gx != NULL) DestroyScene(&aux->Gx);
    if(aux->Gy != NULL) DestroyScene(&aux->Gy);
    if(aux->Gz != NULL) DestroyScene(&aux->Gz);
    if(aux->mag != NULL) DestroyScene(&aux->mag);
    free(aux);    
    *grad = NULL;
  }
}


ScnGradient *RemScnGradientFrame(ScnGradient *fgrad, int sz){
  ScnGradient *grad=NULL;

  grad = (ScnGradient *) calloc(1,sizeof(ScnGradient));
  if(grad == NULL)
    Error(MSG1,"RemScnGradientFrame");

  grad->Gx = RemFrame3(fgrad->Gx, sz);
  grad->Gy = RemFrame3(fgrad->Gy, sz);
  grad->Gz = RemFrame3(fgrad->Gz, sz);
  if(fgrad->mag!=NULL){
    grad->mag = RemFrame3(fgrad->mag, sz);
    MaximumValue3(grad->mag);
  }
  else
    grad->mag = NULL;

  return grad;
}


ScnGradient *LinearInterpScnGradientCentr(ScnGradient *grad,
					  float dx,float dy,float dz){
  ScnGradient *interp=NULL;

  interp = (ScnGradient *) calloc(1,sizeof(ScnGradient));
  if(interp == NULL)
    Error(MSG1,"LinearInterpScnGradientCentr");

  interp->Gx = FastLinearInterpCentr3(grad->Gx, dx,dy,dz);
  interp->Gy = FastLinearInterpCentr3(grad->Gy, dx,dy,dz);
  interp->Gz = FastLinearInterpCentr3(grad->Gz, dx,dy,dz);
  interp->mag = NULL;

  return interp;
}


ScnGradient *ChangeOrientationToLPS_ScnGradient(ScnGradient *grad,
						char *ori){
  ScnGradient *lps=NULL;
  Scene *gx,*gy,*gz;

  lps = (ScnGradient *) calloc(1,sizeof(ScnGradient));
  if(lps == NULL)
    Error(MSG1,"ChangeOrientationToLPS_ScnGradient");

  gx = ChangeOrientationToLPS(grad->Gx, ori);
  gy = ChangeOrientationToLPS(grad->Gy, ori);
  gz = ChangeOrientationToLPS(grad->Gz, ori);
  lps->Gx  = NULL;
  lps->Gy  = NULL;
  lps->Gz  = NULL;
  lps->mag = NULL;

  if     (ori[0]=='L'){ lps->Gx = gx; }
  else if(ori[0]=='R'){ lps->Gx = gx; Negate3inplace(gx); }
  else if(ori[0]=='P'){ lps->Gy = gx; }
  else if(ori[0]=='A'){ lps->Gy = gx; Negate3inplace(gx); }
  else if(ori[0]=='S'){ lps->Gz = gx; }
  else if(ori[0]=='I'){ lps->Gz = gx; Negate3inplace(gx); }
  else{ Error("Invalid orientation",
	      "ChangeOrientationToLPS_ScnGradient"); }

  if     (ori[1]=='L'){ lps->Gx = gy; }
  else if(ori[1]=='R'){ lps->Gx = gy; Negate3inplace(gy); }
  else if(ori[1]=='P'){ lps->Gy = gy; }
  else if(ori[1]=='A'){ lps->Gy = gy; Negate3inplace(gy); }
  else if(ori[1]=='S'){ lps->Gz = gy; }
  else if(ori[1]=='I'){ lps->Gz = gy; Negate3inplace(gy); }
  else{ Error("Invalid orientation",
	      "ChangeOrientationToLPS_ScnGradient"); }

  if     (ori[2]=='L'){ lps->Gx = gz; }
  else if(ori[2]=='R'){ lps->Gx = gz; Negate3inplace(gz); }
  else if(ori[2]=='P'){ lps->Gy = gz; }
  else if(ori[2]=='A'){ lps->Gy = gz; Negate3inplace(gz); }
  else if(ori[2]=='S'){ lps->Gz = gz; }
  else if(ori[2]=='I'){ lps->Gz = gz; Negate3inplace(gz); }
  else{ Error("Invalid orientation",
	      "ChangeOrientationToLPS_ScnGradient"); }

  if(lps->Gx==NULL || lps->Gy==NULL || lps->Gz==NULL)
    Error("Invalid orientation",
	  "ChangeOrientationToLPS_ScnGradient");
  
  return lps;
}


ScnGradient *ReadScnGradient(char *filename){
  ScnGradient *grad=NULL;
  int xsize,ysize,zsize,n;
  float dx,dy,dz;
  char msg[512];
  FILE *fp;

  fp = fopen(filename,"rb");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"ReadScnGradient");
  }

  fread(&xsize, sizeof(int), 1, fp);
  fread(&ysize, sizeof(int), 1, fp);
  fread(&zsize, sizeof(int), 1, fp);
  grad = CreateScnGradient(xsize,ysize,zsize);
  fread(&dx, sizeof(float), 1, fp);
  fread(&dy, sizeof(float), 1, fp);
  fread(&dz, sizeof(float), 1, fp);
  SetVoxelSize(grad->Gx, dx, dy, dz);
  SetVoxelSize(grad->Gy, dx, dy, dz);
  SetVoxelSize(grad->Gz, dx, dy, dz);
  n = xsize*ysize*zsize;
  fread((grad->Gx)->data, sizeof(int), n, fp);
  fread((grad->Gy)->data, sizeof(int), n, fp);
  fread((grad->Gz)->data, sizeof(int), n, fp);
  fclose(fp);

  return grad;
}


void         WriteScnGradient(ScnGradient *grad, char *filename){
  int xsize,ysize,zsize,n;
  float dx,dy,dz;
  char msg[512];
  FILE *fp;

  fp = fopen(filename,"wb");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"WriteScnGradient");
  }

  xsize = (grad->Gx)->xsize;
  ysize = (grad->Gx)->ysize;
  zsize = (grad->Gx)->zsize;
  fwrite(&xsize, sizeof(int),  1, fp);
  fwrite(&ysize, sizeof(int),  1, fp);
  fwrite(&zsize, sizeof(int),  1, fp);
  dx = (grad->Gx)->dx;
  dy = (grad->Gx)->dy;
  dz = (grad->Gx)->dz;
  fwrite(&dx, sizeof(float),  1, fp);
  fwrite(&dy, sizeof(float),  1, fp);
  fwrite(&dz, sizeof(float),  1, fp);
  n = xsize*ysize*zsize;
  fwrite((grad->Gx)->data, sizeof(int), n, fp);
  fwrite((grad->Gy)->data, sizeof(int), n, fp);
  fwrite((grad->Gz)->data, sizeof(int), n, fp);
  fclose(fp);
}


ScnGradient *ReadCompressedScnGradient(char *filename){
  FILE   *fp;
  ift_BZFILE *b;
  int     bzerror;
  ScnGradient *grad=NULL;
  char msg[512];
  int   data_i[3];
  float data_f[3];
  int   nBuf;
  int xsize,ysize,zsize,n;
  float dx,dy,dz;

  fp = fopen(filename,"rb");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"ReadCompressedScnGradient");
    return NULL;
  }

  b = ift_BZ2_bzReadOpen ( &bzerror, fp, 0, 0, NULL, 0 );
  if( bzerror != ift_BZ_OK ){
    ift_BZ2_bzReadClose ( &bzerror, b );
    fclose(fp);
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"ReadCompressedScnGradient");
    return NULL;
  }

  nBuf = ift_BZ2_bzRead ( &bzerror, b, data_i, sizeof(int)*3 );
  if ( ( ( bzerror == ift_BZ_OK ) || 
	 ( bzerror == ift_BZ_STREAM_END ) ) && (nBuf > 0) ) {
    xsize = data_i[0];
    ysize = data_i[1];
    zsize = data_i[2];
    grad = CreateScnGradient(xsize,ysize,zsize);
  }
  else{
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"ReadCompressedScnGradient");
    return NULL;
  }

  nBuf = ift_BZ2_bzRead ( &bzerror, b, data_f, sizeof(float)*3 );
  if ( ( ( bzerror == ift_BZ_OK ) || 
	 ( bzerror == ift_BZ_STREAM_END ) ) && (nBuf > 0) ) {
    dx = data_f[0];
    dy = data_f[1];
    dz = data_f[2];
    SetVoxelSize(grad->Gx, dx, dy, dz);
    SetVoxelSize(grad->Gy, dx, dy, dz);
    SetVoxelSize(grad->Gz, dx, dy, dz);
  }
  else{
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"ReadCompressedScnGradient");
    return grad;
  }

  n = xsize*ysize*zsize;
  nBuf = ift_BZ2_bzRead ( &bzerror, b, (grad->Gx)->data, sizeof(int)*n );
  if( !(( ( bzerror == ift_BZ_OK ) || 
	  ( bzerror == ift_BZ_STREAM_END ) ) && (nBuf > 0)) ) {
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"ReadCompressedScnGradient");
    return grad;
  }
  nBuf = ift_BZ2_bzRead ( &bzerror, b, (grad->Gy)->data, sizeof(int)*n );
  if( !(( ( bzerror == ift_BZ_OK ) || 
	  ( bzerror == ift_BZ_STREAM_END ) ) && (nBuf > 0)) ) {
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"ReadCompressedScnGradient");
    return grad;
  }
  nBuf = ift_BZ2_bzRead ( &bzerror, b, (grad->Gz)->data, sizeof(int)*n );
  if( !(( ( bzerror == ift_BZ_OK ) || 
	  ( bzerror == ift_BZ_STREAM_END ) ) && (nBuf > 0)) ) {
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"ReadCompressedScnGradient");
    return grad;
  }

  ift_BZ2_bzReadClose( &bzerror, b );
  fclose(fp);

  return grad;
}


void         WriteCompressedScnGradient(ScnGradient *grad, char *filename){
  FILE   *fp;
  ift_BZFILE *b;
  int    bzerror;
  char msg[512];
  int xsize,ysize,zsize,n;
  float dx,dy,dz;
  int   data_i[3];
  float data_f[3];

  fp = fopen( filename, "wb" );
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"WriteCompressedScnGradient");
    return;
  }

  b = ift_BZ2_bzWriteOpen( &bzerror, fp, 1, 0, 30 ); //9, 0, 30 );
  if(bzerror != ift_BZ_OK){
    ift_BZ2_bzWriteClose ( &bzerror, b, 0, 0, 0 );
    fclose(fp);
    sprintf(msg,"Cannot open %s",filename);
    Warning(msg,"WriteCompressedScnGradient");
    return;
  }

  if(bzerror == ift_BZ_OK){
    xsize = (grad->Gx)->xsize;
    ysize = (grad->Gx)->ysize;
    zsize = (grad->Gx)->zsize;
    data_i[0] = xsize;
    data_i[1] = ysize;
    data_i[2] = zsize;
    ift_BZ2_bzWrite( &bzerror, b, data_i, sizeof(int)*3 );
    dx = (grad->Gx)->dx;
    dy = (grad->Gx)->dy;
    dz = (grad->Gx)->dz;
    data_f[0] = dx;
    data_f[1] = dy;
    data_f[2] = dz;
    ift_BZ2_bzWrite( &bzerror, b, data_f, sizeof(float)*3 );
    n = xsize*ysize*zsize;
    ift_BZ2_bzWrite( &bzerror, b, (grad->Gx)->data, sizeof(int)*n );
    ift_BZ2_bzWrite( &bzerror, b, (grad->Gy)->data, sizeof(int)*n );
    ift_BZ2_bzWrite( &bzerror, b, (grad->Gz)->data, sizeof(int)*n );
  }

  ift_BZ2_bzWriteClose( &bzerror, b, 0, 0, 0 );
  fclose(fp);

  if(bzerror == ift_BZ_IO_ERROR){
    sprintf(msg,"Error on writing to %s",filename);
    Error(msg,"WriteCompressedScnGradient");
  }
}


//------------------------------------------


Scene *LaplacianFilter3D(Scene *orig){
  Scene *lap;
  int dx,dy,dz,di,m,i,N;
  int acc;
  static int n0[9]={ -1,  0,  1, -1,  0,  1, -1,  0,  1};
  static int n1[9]={ -1, -1, -1,  0,  0,  0,  1,  1,  1};
  static int wt[9]={ -1, -1, -1, -1, -1, -1, -1, -1, -1};
  static int wc[9]={ -1, -1, -1, -1, 26, -1, -1, -1, -1};

  lap = CreateScene(orig->xsize,orig->ysize,orig->zsize);
  SetVoxelSize(lap, orig->dx, orig->dy, orig->dz);
  N = orig->xsize*orig->ysize*orig->zsize;

  dx = 1;
  dy = orig->xsize;
  dz = orig->xsize * orig->ysize;
  di = dx+dy+dz;

  for(i=di; i<N-di; i++){
    // xy plane
    acc = 0;
    for(m=0;m<9;m++){
      acc += wt[m]*orig->data[ i+dz+dx*n0[m]+dy*n1[m] ];
      acc += wc[m]*orig->data[ i+ 0+dx*n0[m]+dy*n1[m] ];
      acc += wt[m]*orig->data[ i-dz+dx*n0[m]+dy*n1[m] ];
    }
    
    lap->data[i] = acc;
  }

  return lap;
}



Scene *SobelFilter3D(Scene *scn) {
  Scene *grad;
  int   dx,dy,dz,di,m,i,N;
  int   acc[3];
  float facc[3];
  static int n0[9]={-1, 0, 1,-1, 0, 1,-1, 0, 1};
  static int n1[9]={-1,-1,-1, 0, 0, 0, 1, 1, 1};
  static int we[9]={ 1, 2, 1, 2, 4, 2, 1, 2, 1};

  grad = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  SetVoxelSize(grad, scn->dx, scn->dy, scn->dz);
  N = scn->xsize*scn->ysize*scn->zsize;

  dx = 1;
  dy = scn->xsize;
  dz = scn->xsize * scn->ysize;
  di = dx+dy+dz;

  for(i=di; i<N-di; i++){
    // xy plane
    acc[0] = 0;
    for(m=0; m<9; m++)
      acc[0] += we[m] * (scn->data[ i+dz+dx*n0[m]+dy*n1[m] ] -
			 scn->data[ i-dz+dx*n0[m]+dy*n1[m] ]);
    // yz plane
    acc[1] = 0;
    for(m=0; m<9; m++)
      acc[1] += we[m] * (scn->data[ i+dx+dz*n0[m]+dy*n1[m] ] -
			 scn->data[ i-dx+dz*n0[m]+dy*n1[m] ]);
    // xz plane
    acc[2] = 0;
    for(m=0; m<9; m++)
      acc[2] += we[m] * (scn->data[ i+dy+dx*n0[m]+dz*n1[m] ] -
			 scn->data[ i-dy+dx*n0[m]+dz*n1[m] ]);
    acc[0] >>= 4;
    acc[1] >>= 4;
    acc[2] >>= 4;
    facc[0] = (float)acc[0];
    facc[1] = (float)acc[1];
    facc[2] = (float)acc[2];
    grad->data[i] = ROUND(sqrtf((facc[0]*facc[0] + 
				 facc[1]*facc[1] +
				 facc[2]*facc[2])/3.0));
  }
  return grad;
}


Scene *SphericalGradient(Scene *scn, float r){
  AdjRel3 *A = SceneSphericalAdjRel3(scn, r);
  AdjVxl  *N;
  Scene *grad=NULL;
  int p,q,i,vp,vq;
  float *mg=NULL;
  float gx,gy,gz,d;
  int fx,fy,fz,dp;

  N  = AdjVoxels(scn, A);
  mg = AdjRel3SceneDistance(scn, A);
  FrameSizes3(A, &fx, &fy, &fz);
  dp  = fx*1;
  dp += fy*scn->xsize;
  dp += fz*scn->xsize*scn->ysize;

  grad = CreateScene(scn->xsize,
		     scn->ysize,
		     scn->zsize);
  SetVoxelSize(grad, scn->dx, scn->dy, scn->dz);

  for(p=dp; p<scn->n-dp; p++){
    vp = scn->data[p];
    gx = gy = gz = 0.0;
    for(i=1; i<N->n; i++){
      q = p + N->dp[i];
      vq = scn->data[q];
      d = (float)(vq-vp);
	
      gx  += (d*A->dx[i])/(mg[i]);
      gy  += (d*A->dy[i])/(mg[i]);
      gz  += (d*A->dz[i])/(mg[i]);
    }
    gx = ROUND(10.0*gx*scn->dx/fx);
    gy = ROUND(10.0*gy*scn->dy/fy);
    gz = ROUND(10.0*gz*scn->dz/fz);
    grad->data[p] = ROUND(sqrtf(gx*gx + gy*gy + gz*gz));
  }
  ClearSceneAdjFrame(grad, A);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);
  free(mg);

  return grad;
}


//------------------------------------------

ScnGradient *SphericalScnGradient(Scene *scn, float r){
  AdjRel3 *A = SceneSphericalAdjRel3(scn, r);
  AdjVxl  *N;
  ScnGradient *grad=NULL;
  int p,q,i,vp,vq;
  float *mg=NULL;
  float gx,gy,gz,d;
  int fx,fy,fz,dp;

  N  = AdjVoxels(scn, A);
  mg = AdjRel3SceneDistance(scn, A);
  FrameSizes3(A, &fx, &fy, &fz);
  dp  = fx*1;
  dp += fy*scn->xsize;
  dp += fz*scn->xsize*scn->ysize;

  grad = CreateScnGradient(scn->xsize,
			   scn->ysize,
			   scn->zsize);
  SetVoxelSize(grad->Gx, scn->dx, scn->dy, scn->dz);
  SetVoxelSize(grad->Gy, scn->dx, scn->dy, scn->dz);
  SetVoxelSize(grad->Gz, scn->dx, scn->dy, scn->dz);

  for(p=dp; p<scn->n-dp; p++){
    vp = scn->data[p];
    gx = gy = gz = 0.0;
    for(i=1; i<N->n; i++){
      q = p + N->dp[i];
      vq = scn->data[q];
      d = (float)(vq-vp);
	
      gx  += (d*A->dx[i])/(mg[i]);
      gy  += (d*A->dy[i])/(mg[i]);
      gz  += (d*A->dz[i])/(mg[i]);
    }
    (grad->Gx)->data[p] = ROUND(10.0*gx*scn->dx/fx);
    (grad->Gy)->data[p] = ROUND(10.0*gy*scn->dy/fy);
    (grad->Gz)->data[p] = ROUND(10.0*gz*scn->dz/fz);
  }
  ClearSceneAdjFrame(grad->Gx, A);
  ClearSceneAdjFrame(grad->Gy, A);
  ClearSceneAdjFrame(grad->Gz, A);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);
  free(mg);

  return grad;
}



void ComputeScnGradientMagnitude(ScnGradient *grad){
  Scene *mag=NULL;
  float gx,gy,gz,d;
  int p,m,Gmax=0;

  if(grad->mag!=NULL) return;

  mag = CreateScene((grad->Gx)->xsize,
		    (grad->Gx)->ysize,
		    (grad->Gx)->zsize);
  SetVoxelSize(mag, grad->Gx->dx, grad->Gx->dy, grad->Gx->dz);
  for(p=0; p<mag->n; p++){
    gx = (float)(grad->Gx->data[p]);
    gy = (float)(grad->Gy->data[p]);
    gz = (float)(grad->Gz->data[p]);
    d = sqrtf(gx*gx + gy*gy + gz*gz);
    m = ROUND(d);
    if(m>Gmax) Gmax = m;
    mag->data[p] = m;
  }
  mag->maxval = Gmax;
  grad->mag = mag;
}


int     ScnGradientMaximumMag(ScnGradient *grad){
  int Imax;
  if(grad->mag==NULL)
    ComputeScnGradientMagnitude(grad);
  Imax = MaximumValue3(grad->mag);
  return Imax;
}


void    ScnGradientNormalize(ScnGradient *grad,
			     int omin,int omax,
			     int nmin,int nmax){
  float gx,gy,gz,m,nm;
  int p;

  ComputeScnGradientMagnitude(grad);
  for(p=0; p<(grad->Gx)->n; p++){
    m  = (grad->mag)->data[p];
    nm = (float)IntegerNormalize((grad->mag)->data[p],
				 omin, omax, 
				 nmin, nmax);
    gx = grad->Gx->data[p];
    gy = grad->Gy->data[p];
    gz = grad->Gz->data[p];
    if((grad->mag)->data[p]>0){
      (grad->Gx)->data[p] = ROUND(gx*(nm/m));
      (grad->Gy)->data[p] = ROUND(gy*(nm/m));
      (grad->Gz)->data[p] = ROUND(gz*(nm/m));
      (grad->mag)->data[p] = ROUND(nm);
    }
    else{
      (grad->Gx)->data[p] = 0;
      (grad->Gy)->data[p] = 0;
      (grad->Gz)->data[p] = 0;
      (grad->mag)->data[p] = 0;
    }
  }
  (grad->mag)->maxval = IntegerNormalize((grad->mag)->maxval,
					 omin, omax, 
					 nmin, nmax);
}


void    PowerEnhancementScene(Scene *scn){
  int Imax,p;
  float v;

  Imax = MaximumValue3(scn);
  if(Imax==0) return;
  for(p=0; p<scn->n; p++){
    v = (float)scn->data[p];
    scn->data[p] = ROUND((v*v)/((float)Imax));
  }
}


void    PowerEnhancementScnGradient(ScnGradient *grad){
  Scene *pmag;
  float gx,gy,gz,m,pm;
  int p;
  ComputeScnGradientMagnitude(grad);
  pmag = CopyScene(grad->mag);
  PowerEnhancementScene(pmag);
  for(p=0; p<pmag->n; p++){
    gx = (grad->Gx)->data[p];
    gy = (grad->Gy)->data[p];
    gz = (grad->Gz)->data[p];
    m  = (grad->mag)->data[p];
    pm = pmag->data[p];
    if((grad->mag)->data[p]>0){
      grad->Gx->data[p] = ROUND((gx/m)*pm);
      grad->Gy->data[p] = ROUND((gy/m)*pm);
      grad->Gz->data[p] = ROUND((gz/m)*pm);
      grad->mag->data[p] = ROUND(pm);
    }
    else{
      grad->Gx->data[p] = 0;
      grad->Gy->data[p] = 0;
      grad->Gz->data[p] = 0;
      grad->mag->data[p] = 0;
    }
  }
  DestroyScene(&pmag);
}


//-------------------------------------------


ScnGradient *MRI_SphericalScnGradient(Scene *scn, float r, MRI_Info info){
  AdjRel3 *A = SceneSphericalAdjRel3(scn, r);
  AdjVxl  *N;
  ScnGradient *grad=NULL;
  int p,q,i,vp,vq;
  float *mg=NULL;
  float gx,gy,gz,d,w;
  int fx,fy,fz,dp;
  float *weight1,*weight2;
  int Tgm_dark = ((info.Tcsf+info.Ecsf)+info.Tgm)/2;

  N  = AdjVoxels(scn, A);
  mg = AdjRel3SceneDistance(scn, A);
  FrameSizes3(A, &fx, &fy, &fz);
  dp  = fx*1;
  dp += fy*scn->xsize;
  dp += fz*scn->xsize*scn->ysize;

  grad = CreateScnGradient(scn->xsize,
			   scn->ysize,
			   scn->zsize);
  SetVoxelSize(grad->Gx, scn->dx, scn->dy, scn->dz);
  SetVoxelSize(grad->Gy, scn->dx, scn->dy, scn->dz);
  SetVoxelSize(grad->Gz, scn->dx, scn->dy, scn->dz);

  //---- Likelihood for CSF and GM -----
  weight1 = SceneIntensity2Gaussian(scn, info.Tcsf,
				    (float)info.Ecsf,
				    (Tgm_dark-info.Tcsf));
  weight2 = SceneIntensity2Gaussian(scn, Tgm_dark,
				    (Tgm_dark-info.Tcsf),
				    (info.Tgm-Tgm_dark));
  //--------------------------------

  for(p=dp; p<scn->n-dp; p++){
    vp = scn->data[p];
    gx = gy = gz = 0.0;
    for(i=1; i<N->n; i++){
      q = p + N->dp[i];
      vq = scn->data[q];
      d = (float)(vq-vp);

      //---- Enhance CSF/GM transition ----
      if(vp>vq) w = (weight1[vq]+weight2[vp])/2.0;
      else      w = (weight1[vp]+weight2[vq])/2.0;
      d *= w;
      //-----------------------------------

      gx  += (d*A->dx[i])/(mg[i]);
      gy  += (d*A->dy[i])/(mg[i]);
      gz  += (d*A->dz[i])/(mg[i]);
    }
    (grad->Gx)->data[p] = ROUND(10.0*gx*scn->dx/fx);
    (grad->Gy)->data[p] = ROUND(10.0*gy*scn->dy/fy);
    (grad->Gz)->data[p] = ROUND(10.0*gz*scn->dz/fz);
  }
  ClearSceneAdjFrame(grad->Gx, A);
  ClearSceneAdjFrame(grad->Gy, A);
  ClearSceneAdjFrame(grad->Gz, A);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);
  free(weight1);
  free(weight2);
  free(mg);

  return grad;
}



Scene  *MRI_SphericalAccAbsDiff3(Scene *scn, float r, MRI_Info info){
  AdjRel3 *A = SceneSphericalAdjRel3(scn, r);
  AdjVxl  *N;
  Scene *diff=NULL;
  int p,q,i,vp,vq;
  float *mg=NULL;
  float w,sum;
  int fx,fy,fz,dp,d,dmax = (info.Twm-info.Tcsf);
  float *weight1,*weight2;
  int Tgm_dark = ((info.Tcsf+info.Ecsf)+info.Tgm)/2;

  N  = AdjVoxels(scn, A);
  mg = AdjRel3Distance(A);
  FrameSizes3(A, &fx, &fy, &fz);
  dp  = fx*1;
  dp += fy*scn->xsize;
  dp += fz*scn->xsize*scn->ysize;

  diff = CreateScene(scn->xsize,
		     scn->ysize,
		     scn->zsize);
  SetVoxelSize(diff, scn->dx, scn->dy, scn->dz);
  
  //---- Likelihood for CSF and GM -----
  weight1 = SceneIntensity2Gaussian(scn, info.Tcsf,
				    (float)info.Ecsf,
				    (Tgm_dark-info.Tcsf));
  weight2 = SceneIntensity2Gaussian(scn, Tgm_dark,
				    (Tgm_dark-info.Tcsf),
				    (info.Tgm-Tgm_dark));
  //--------------------------------

  for(p=dp; p<scn->n-dp; p++){
    vp = scn->data[p];
    sum = 0.0;
    for(i=1; i<N->n; i++){
      q = p + N->dp[i];
      vq = scn->data[q];

      d = vq-vp;

      //---- Enhance CSF/GM transition ----
      if(d<0){ w = (weight1[vq]+weight2[vp]); d=-d; }  // /2.0;
      else   { w = (weight1[vp]+weight2[vq]);       }  // *2.0;
      //-----------------------------------

      //--- limits the maximum difference ---
      if(d>dmax) d = MAX(0, 2*dmax-d);
      //-------------------------------------

      sum += (w*((float)d)/mg[i]);
    }
    diff->data[p] = ROUND(100.0*(sum/2.0));
  }
  ClearSceneAdjFrame(diff, A);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);
  free(weight1);
  free(weight2);
  free(mg);

  return diff;
}


//-------------------------------------------
//-------------------------------------------
//-------------------------------------------

int vclip(int v,int min,int max) {
  if (v<min) return min;
  if (v>max) return max;
  return v;
}


float MyDistance(float *f1, float *f2, int n){
  int i;
  float dist;
  
  dist = 0;
  for (i=0; i < n; i++) 
    dist += (f2[i]-f1[i]);///2.0;

  //  dist /= n;
  return(dist);//exp(-(dist-0.5)*(dist-0.5)/2.0));
}


Scene *TextGradient3(Scene *scn){
  float   dist,gx,gy,gz;  
  int     Imax,i,p,q,n=scn->xsize*scn->ysize*scn->zsize;
  Voxel   u,v;
  AdjRel3 *A=Spheric(1.0),*A6=Spheric(1.0);
  float   *mg=AllocFloatArray(A6->n);
  Scene   *grad=CreateScene(scn->xsize,scn->ysize,scn->zsize);

  typedef struct _features {
    float *f;
  } Features;

  Features *feat=(Features *)calloc(n,sizeof(Features));
  for (p=0; p < n; p++)
    feat[p].f = AllocFloatArray(A->n);

  Imax = MaximumValue3(scn);

  for (u.z=0; u.z < scn->zsize; u.z++) 
    for (u.y=0; u.y < scn->ysize; u.y++) 
      for (u.x=0; u.x < scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	for (i=0; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    feat[p].f[i]=(float)scn->data[q];///(float)Imax;
	  }
	}
      }

  for (i=0; i < A6->n; i++) 
    mg[i]=sqrt(A6->dx[i]*A6->dx[i]+A6->dy[i]*A6->dy[i]+A6->dz[i]*A6->dz[i]);

  for (u.z=0; u.z < scn->zsize; u.z++) 
    for (u.y=0; u.y < scn->ysize; u.y++) 
      for (u.x=0; u.x < scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	gx = gy = gz = 0.0;
	for (i=1; i < A6->n; i++) {
	  v.x = u.x + A6->dx[i];
	  v.y = u.y + A6->dy[i];
	  v.z = u.z + A6->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    dist = MyDistance(feat[p].f,feat[q].f,A->n);
	    gx  += dist*A6->dx[i]/mg[i]; 
	    gy  += dist*A6->dy[i]/mg[i]; 
	    gz  += dist*A6->dz[i]/mg[i]; 
	  }
	} 
	grad->data[p]=(int)sqrt(gx*gx + gy*gy + gz*gz);//(100000.0*sqrt(gx*gx + gy*gy + gz*gz));
      }


  for (p=0; p < n; p++)
    free(feat[p].f);
  free(feat);

  free(mg);
  DestroyAdjRel3(&A);
  DestroyAdjRel3(&A6);
  return(grad);
}


Scene * VarianceGrad3(Scene *input, float radius, int thres) {
  Scene *mean, *var, *cnt;
  AdjRel3 *A;
  int i,j,k,l,px,py,pz,W,H,D,WH,N,p,tmp;
  float norm;
  register int a1,a2;

  W = input->xsize;
  H = input->ysize;
  D = input->zsize;
  WH = W*H;
  N = W*H*D;

  mean = CreateScene(W,H,D);
  cnt  = CreateScene(W,H,D);
  var  = CreateScene(W,H,D);

  for(i=0;i<N;i++) {
    mean->data[i] = 0;
    var->data[i] = 0;
    cnt->data[i] = 0;
  }

  A = Spheric(radius);
  
  /* find mean */
  for(k=0;k<D;k++) {
    printf("grad mean %d of %d    \r",k+1,D);
    fflush(stdout);
    for(j=0;j<H;j++)
      for(i=0;i<W;i++) {
	a1 = a2 = 0;
	for(l=0;l<A->n;l++) {
	  px = i + A->dx[l]; if (px < 0 || px >=W) continue;
	  py = j + A->dy[l]; if (py < 0 || py >=H) continue;
	  pz = k + A->dz[l]; if (pz < 0 || pz >=D) continue;
	  a2++;
	  a1+=input->data[px+py*W+pz*WH];
	}
	p = i + j*W + k*WH;
	cnt->data[p] = a2;
	mean->data[p] = a1/a2;
      }
  }

  /* find stddev */
  for(k=0;k<D;k++) {
    printf("grad sdev %d of %d    \r",k+1,D);
    fflush(stdout);
    for(j=0;j<H;j++)
      for(i=0;i<W;i++) {
	a1 = 0;
	p = i+j*W+k*WH;
	a2 = mean->data[p];
	for(l=0;l<A->n;l++) {
	  px = i + A->dx[l]; if (px < 0 || px >=W) continue;
	  py = j + A->dy[l]; if (py < 0 || py >=H) continue;
	  pz = k + A->dz[l]; if (pz < 0 || pz >=D) continue;
	  tmp = a2 - input->data[px+py*W+pz*WH];
	  a1 += tmp*tmp;
	}
	var->data[p] = (int) sqrt(a1 / cnt->data[p]);
      }
  }


  printf("done                    \r");

  /* apply threshold */
  for(i=0;i<N;i++)
    if (input->data[i] < thres)
      var->data[i] = 0;

  /* normalize to 0-32767 */
  tmp = 0;
  for(i=0;i<N;i++)
    if (var->data[i] > tmp)
      tmp = var->data[i];

  norm = ((float)tmp) / 32767.0;

  for(i=0;i<N;i++)
    var->data[i] = (int) (((float)(var->data[i])) / norm);

  DestroyScene(&mean);
  DestroyScene(&cnt);
  DestroyAdjRel3(&A);
  return var;
}



Scene *BrainGrad3(Scene *scn){
  Scene *enha,*grad;
  int a,b,c;
  
  b     = ComputeOtsu3(scn);
  MeansAboveBelowT(scn,b,&c,&a);
  enha  = ApplySShape(scn,a,b,c);
  grad  = TextGradient3(enha);
  MaximumValue3(grad);
  DestroyScene(&enha);
  
  return grad;
}



Scene *FeatTextGrad3(Scene *scn, FeatMap *fmap){
  real    dist,gx,gy,gz;  
  int     i,p,q;
  Voxel   u,v;
  AdjRel3 *A6=Spheric(1.0);
  float   *mg=AllocFloatArray(A6->n);
  Scene   *grad=CreateScene(scn->xsize,scn->ysize,scn->zsize);

  for(i=0; i<A6->n; i++) 
    mg[i]=sqrt(A6->dx[i]*A6->dx[i] + A6->dy[i]*A6->dy[i] + A6->dz[i]*A6->dz[i]);

  for(u.z=0; u.z<scn->zsize; u.z++) 
    for(u.y=0; u.y<scn->ysize; u.y++) 
      for(u.x=0; u.x<scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	gx = gy = gz = 0.0;
	for(i=1; i<A6->n; i++) {
	  v.x = u.x + A6->dx[i];
	  v.y = u.y + A6->dy[i];
	  v.z = u.z + A6->dz[i];
	  if(ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    dist = DistanceSub(fmap->data[p],fmap->data[q],fmap->nfeat);
	    gx  += dist*A6->dx[i]/mg[i]; 
	    gy  += dist*A6->dy[i]/mg[i]; 
	    gz  += dist*A6->dz[i]/mg[i]; 
	  }
	}
	grad->data[p]=(int)(sqrt(gx*gx + gy*gy + gz*gz));
      }

  MaximumValue3(grad);
  DestroyAdjRel3(&A6);
  free(mg);

  return(grad);
}




Scene *FeatGrad3(Scene *scn, FeatMap *fmap){
  real    dist,maxdist;
  int     i,p,q;
  Voxel   u,v;
  AdjRel3 *A6=Spheric(1.0);
  Scene   *grad=CreateScene(scn->xsize,scn->ysize,scn->zsize);

  for(u.z=0; u.z<scn->zsize; u.z++) 
    for(u.y=0; u.y<scn->ysize; u.y++) 
      for(u.x=0; u.x<scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	maxdist = 0.0;
	for(i=1; i<A6->n; i++) {
	  v.x = u.x + A6->dx[i];
	  v.y = u.y + A6->dy[i];
	  v.z = u.z + A6->dz[i];
	  if(ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    dist = DistanceL2(fmap->data[p],fmap->data[q],fmap->nfeat);
	    if(dist > maxdist)
	      maxdist = dist;
	  }
	}
	grad->data[p]=(int)(maxdist);
      }

  MaximumValue3(grad);
  DestroyAdjRel3(&A6);
  
  return(grad);
}



