
#include "oldift.h"


Scene* msp_BinarizeByThreshold(Scene *scn, float threshold)
{						
  Scene *bin;
  int i;
  int n = scn->xsize * scn->ysize * scn->zsize; 
  bin = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (i=0;i<n;i++)
    if (scn->data[i]>threshold) bin->data[i]=1;
  bin->dx=scn->dx;
  bin->dy=scn->dy;
  bin->dz=scn->dz;
  return bin;
}


void msp_MakeMasksetArray(Scene *scn, Voxel **maskset, int *size)
{
  int p,count=0,n,i;
  Voxel v;
  n = scn->xsize * scn->ysize * scn->zsize; 
  for (i=0;i<n;i++)
    if (scn->data[i]>0) count++;
  *size = count;
  *maskset = (Voxel *) malloc(count*sizeof(Voxel));
  i=0;
  for (v.z=0; v.z < scn->zsize; v.z++)
    for (v.y=0; v.y < scn->ysize; v.y++)
      for (v.x=0; v.x < scn->xsize; v.x++) {
	p = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (scn->data[p]>0) {
	  (*maskset)[i] = v;
	  i++;
	}  
      }
}


double msp_GetZFromPlaneProjection(float x, float y, Point *p1, Point *p2, Point *p3)
{
        double A,B,C,D;
        double x1=p1->x, x2=p2->x, x3=p3->x;
        double y1=p1->y, y2=p2->y, y3=p3->y;
        double z1=p1->z, z2=p2->z, z3=p3->z;
        double res;
        A = y1*(z2 - z3) + y2*(z3 - z1) + y3*(z1 - z2);
        B = z1*(x2 - x3) + z2*(x3 - x1) + z3*(x1 - x2);
        C = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2);
        D = -(x1*(y2*z3 - y3*z2) + x2*(y3*z1 - y1*z3) + x3*(y1*z2 - y2*z1));
        res = - (D + B*y + A*x) / C;
        return res;
}

double msp_GetZFromPlaneProjection2(float x, float y, Plane *p)
{
        double A,B,C,D;
        double res;
        A = p->normal.x;
        B = p->normal.y;
        C = p->normal.z;
	D = -(A*p->po.x + B*p->po.y + C*p->po.z);
        res = - (D + B*y + A*x) / C;
        return res;
}







int msp_ThresholdToGetNMaxVoxels(Scene *scn, int n)
{
  int last=n,pos;
  Curve *hist = Histogram3(scn);
  pos= hist->n-1;
  while (last>0) {
    if (hist->Y[pos]<0) printf("error");
    if (hist->Y[pos]==0) pos--;
    else {
      hist->Y[pos]--;
      last--;
    }
    if (pos<0) return 0; // just in case
  }
  DestroyCurve(&hist);
  return pos;
}







// Returns a score of symmetry (higher score=higher symmetry)
// "mask" is a binary image of the relevant voxels (for binary images, scn and mask are usually the same)
// "maskset" is the same information but in the format of an array of voxels
float msp_SymmetryScoreBin(Scene *scn, Voxel maskset[], int maskset_size, Plane *p)
{
  int i,q;
  float dist,side;
  float A,B,C,D;
  Vector normal;
  Point symmetric;
  int symmetric_val;
  float score=0;  // Correlation -> score = V1 / sqrt(V2/V3); 
  int M=0,N=maskset_size;
  if (N==0) { 
    printf("Warning: empty binary image.\n"); 
    return 0;
  }
  // Set normal vector and plane equation
  normal = p->normal;
  VectorNormalize(&normal);
  A = normal.x;
  B = normal.y;
  C = normal.z;
  D = -(normal.x*p->po.x + normal.y*p->po.y + normal.z*p->po.z);
  for (i=0;i<maskset_size;i++) {    
       // Which side the point is?
       side = A*maskset[i].x + B*maskset[i].y + C*maskset[i].z + D; 
       // point-plane distance
       dist = fabs(side);
       // Find the symmetric point
       if (side>0) { // same side of the normal
	 symmetric.x = maskset[i].x - (2*dist)*normal.x;
	 symmetric.y = maskset[i].y - (2*dist)*normal.y;
	 symmetric.z = maskset[i].z - (2*dist)*normal.z;
       }
       else { // opposite side of the normal (or in the plane)
	 symmetric.x = maskset[i].x + (2*dist)*normal.x;
	 symmetric.y = maskset[i].y + (2*dist)*normal.y;
	 symmetric.z = maskset[i].z + (2*dist)*normal.z;
       }
       // If validvoxel
       if (dist>=1 && symmetric.x<=scn->xsize-1 
	   && symmetric.y<=scn->ysize-1 && symmetric.z<=scn->zsize-1 
	   && symmetric.x>=(float)0 && symmetric.y>=(float)0 && symmetric.z>=(float)0) {
	 //symmetric_val=GetVoxelValue_trilinear(scn,symmetric.x,symmetric.y,symmetric.z);
	 symmetric_val=GetVoxelValue_nn(scn,symmetric.x,symmetric.y,symmetric.z);
	 q = scn->tbz[maskset[i].z] + scn->tby[maskset[i].y] + maskset[i].x;
	 if (symmetric_val!=0 && scn->data[q]!=0) M++;  // if counterpart voxels are 1 then M++
       }
  }
  score = (float)M/(float)N;
  return score;
}






void msp_GetPlaneFrom3Points(Point *p1, Point *p2, Point *p3, Plane *p)
{
  
  float A,B,C,D;
  float x1=p1->x, x2=p2->x, x3=p3->x;
  float y1=p1->y, y2=p2->y, y3=p3->y;
  float z1=p1->z, z2=p2->z, z3=p3->z;
  A = y1*(z2 - z3) + y2*(z3 - z1) + y3*(z1 - z2);
  B = z1*(x2 - x3) + z2*(x3 - x1) + z3*(x1 - x2);
  C = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2);
  D = -(x1*(y2*z3 - y3*z2) + x2*(y3*z1 - y1*z3) + x3*(y1*z2 - y2*z1));
  p->normal.x=A;
  p->normal.y=B;
  p->normal.z=C;
  p->po.x=0;
  p->po.y=0;
  p->po.z=-D/C; // find z in (0,0,z)
}





Scene* msp_DrawMyPlaneAnyOrientation(Scene *scn, Plane *p)
{
  Scene *new = CopyScene(scn);
  int maxval = MaximumValue3(scn);
  int z,q;
  Voxel v;
  float A,B,C,D;

  A = p->normal.x;
  B = p->normal.y;
  C = p->normal.z;
  D = -(p->normal.x*p->po.x + p->normal.y*p->po.y + p->normal.z*p->po.z);
  for (v.y=0; v.y < scn->ysize; v.y++)
    for (v.x=0; v.x < scn->xsize; v.x++) {
      z = (int)(((-D-B*v.y-A*v.x)/C)+0.5);
      if (ValidVoxel(scn,v.x,v.y,z)) {
	q = v.x + scn->tby[v.y] + scn->tbz[z];
	new->data[q]=maxval;
      }
    }
  return new;
}


  

void msp_ZeroBorders(Scene *scn, int bordersize)
{
  Voxel u;
  int p;
  if (bordersize<1) return;
  for (u.z=0; u.z < scn->zsize; u.z++) 
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++) {
	if (u.x<bordersize || u.y<bordersize || u.z<bordersize ||
	    u.x>=scn->xsize-bordersize || u.y>=scn->ysize-bordersize || u.z>=scn->zsize-bordersize) {
	  p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	  scn->data[p]=0;
	}
      }
}



void msp_ZeroMask(Scene *scn,Scene *mask)
{
  int i,n;
  n = scn->xsize * scn->ysize * scn->zsize; 
  for (i=0;i<n;i++)
    if (mask->data[i]==0) scn->data[i]=0;
}


int CountNonZeroVoxels(Scene *scn)
{
  int i,n,count=0;
  n = scn->xsize * scn->ysize * scn->zsize; 
  for (i=0;i<n;i++)
    if (scn->data[i]>0) count++;
  return count;
}



/*
Scene* msp_CTSobelFilter3(Scene *scn)
{
  Scene *res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  Image *tmp1,*tmp2;
  int i;

  for (i=0;i<scn->ysize;i++) {
    tmp1 = GetYSlice(scn,i);
    tmp2 = SobelFilter(tmp1);
    PutYSlice(tmp2,res,i);
    DestroyImage(&tmp1);
    DestroyImage(&tmp2);
  }
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=scn->dz;
  return res;

}
*/




Scene* msp_SubsampleByN(Scene *scn, int n) // eg. n = 2 is half-subsample
{
  Scene *new;
  Voxel v,u;
  int p,acc;

  if (n==1) {
    new = CopyScene(scn);
    return new;
  }
  new = CreateScene(scn->xsize/n,scn->ysize/n,scn->zsize/n);
  new->dx=scn->dx*n;
  new->dy=scn->dy*n;
  new->dz=scn->dz*n;
  for (v.z=0; v.z < new->zsize; v.z++)
    for (v.y=0; v.y < new->ysize; v.y++)
      for (v.x=0; v.x < new->xsize; v.x++) {
        p = new->tbz[v.z] + new->tby[v.y] + v.x;
	acc=0;
	for (u.z=0; u.z < n; u.z++)
	  for (u.y=0; u.y < n; u.y++)
	    for (u.x=0; u.x < n; u.x++) {
	      acc += scn->data[scn->tbz[v.z*n+u.z] + scn->tby[v.y*n+u.y] + v.x*n+u.x];
	    }
	new->data[p] = acc/(n*n*n);
      }
  return new;
}


Scene* msp_SubsampleByN_bin(Scene *scn, int n) // eg. n = 2 is half-subsample
{
  Scene *new;
  Voxel v,u;
  int p,acc;

  if (n==1) {
    new = CopyScene(scn);
    return new;
  }
  new = CreateScene(scn->xsize/n,scn->ysize/n,scn->zsize/n);
  new->dx=scn->dx*n;
  new->dy=scn->dy*n;
  new->dz=scn->dz*n;
  for (v.z=0; v.z < new->zsize; v.z++)
    for (v.y=0; v.y < new->ysize; v.y++)
      for (v.x=0; v.x < new->xsize; v.x++) {
        p = new->tbz[v.z] + new->tby[v.y] + v.x;
	acc=0;
	for (u.z=0; u.z < n; u.z++)
	  for (u.y=0; u.y < n; u.y++)
	    for (u.x=0; u.x < n; u.x++) {
	      if (scn->data[scn->tbz[v.z*n+u.z] + scn->tby[v.y*n+u.y] + v.x*n+u.x]!=0)  acc++;
	    }
	if ((float)acc/(float)(n*n*n)>=0.5)
	  new->data[p] = 1;
	else
	  new->data[p] = 0;
      }
  return new;
}




float msp_TextGrad_Distance(float *f1, float *f2, int n){
  int i;
  float dist;
  
  dist = 0;
  for (i=0; i < n; i++) 
    dist += (f2[i]-f1[i]);///2.0;

  //  dist /= n;
  return(dist);//exp(-(dist-0.5)*(dist-0.5)/2.0));
}


Scene* msp_TextGradient3(Scene *scn){
  float   dist,gx,gy,gz;  
  int     i,p,q,n=scn->xsize*scn->ysize*scn->zsize;
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
	    dist = msp_TextGrad_Distance(feat[p].f,feat[q].f,A->n);
	    gx  += dist*A6->dx[i]/mg[i]; 
	    gy  += dist*A6->dy[i]/mg[i]; 
	    gz  += dist*A6->dz[i]/mg[i]; 
	  }
	} 
	grad->data[p]=(int)sqrt(gx*gx + gy*gy + gz*gz);
      }


  for (p=0; p < n; p++)
    free(feat[p].f);
  free(feat);

  free(mg);
  DestroyAdjRel3(&A);
  DestroyAdjRel3(&A6);
  return(grad);
}




Scene*  msp_Rotate3(Scene *scn, 
		double thx, double thy, double thz, // angles to rotate
		  int cx, int cy, int cz, // center of the rotation
                double transx, double transy, double transz) // translation after rotation
{
  int p;
  Voxel u;
  Scene *res;
  RealMatrix *trans1,*rot1,*rot2,*rot3,*trans2,*aux1,*aux2,*inv,*vox;
  trans1 = TranslationMatrix3(-cx,-cy,-cz);
  rot1 = RotationMatrix3(0,thx);
  rot2 = RotationMatrix3(1,thy);
  rot3 = RotationMatrix3(2,thz);
  trans2 = TranslationMatrix3(cx+transx,cy+transy,cz+transz);
  // Compose transform
  aux1 = MultRealMatrix(trans2,rot3);
  aux2 = MultRealMatrix(aux1,rot2);
  DestroyRealMatrix(&aux1);
  aux1 = MultRealMatrix(aux2,rot1);
  DestroyRealMatrix(&aux2);
  aux2 = MultRealMatrix(aux1,trans1);
  inv = InvertRealMatrix(aux2);
  DestroyRealMatrix(&trans1);
  DestroyRealMatrix(&rot1);
  DestroyRealMatrix(&rot2);
  DestroyRealMatrix(&rot3);
  DestroyRealMatrix(&trans2);
  DestroyRealMatrix(&aux2);
  DestroyRealMatrix(&aux1);
  // Applying transform for all voxels
  res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (u.z=0; u.z < res->zsize; u.z++) 
    for (u.y=0; u.y < res->ysize; u.y++)
      for (u.x=0; u.x < res->xsize; u.x++){
	p = u.x + res->tby[u.y] + res->tbz[u.z];
	vox = TransformVoxel(inv,u); // acelerar isso!!!!
	if ((vox->val[0][0]<=res->xsize-1)&&(vox->val[0][1]<=res->ysize-1)
	    &&(vox->val[0][2]<=res->zsize-1 && vox->val[0][0]>0 && vox->val[0][1]>0 && vox->val[0][2]>0))
	  {
	    res->data[p]=GetVoxelValue_trilinear(scn,vox->val[0][0],vox->val[0][1],vox->val[0][2]);
	  }
	else
	  res->data[p]=0;
	DestroyRealMatrix(&vox);
      }
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=scn->dz;
  return res;
}

Scene*  msp_RotateToMSP(Scene *scn, Plane *p)
{
    Point center;
    Scene *res;
    float thx,thy,transz;
    float A,B,C,D;
    A = p->normal.x;
    B = p->normal.y;
    C = p->normal.z;
    D = -(p->normal.x*p->po.x + p->normal.y*p->po.y + p->normal.z*p->po.z);
    center.x = scn->xsize/2.0;
    center.y = scn->ysize/2.0;
    center.z = (-D-B*center.y-A*center.x)/C;
    transz = ((int)scn->zsize/2) - center.z;  // put the MSP on the zsize/2 slice
    if (C==0) {printf("error: thx=0\n"); exit(1);} 
       else thx = atan(B/C); 
    if (C==0) {printf("error: thx=0\n"); exit(1);}
    else { 
      thy = - atan(A/(B*sin(thx)+C*cos(thx))); 
    }
    res = msp_Rotate3(scn,thx,thy,0, center.x,center.y,center.z, 0,0,transz);
    return res;
}



Scene* msp_RotateAxis(Scene *scn, int plane)  // plane= 1(X) 2(Y) 3(Z) -1(-X) -2(-Y) -3(-Z)
{
  Scene *res=NULL;
  int i,j,k,p,q;

  if (plane==1) {
      res = CreateScene(scn->xsize,scn->zsize,scn->ysize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++) {
              p = res->tbz[j] + res->tby[res->ysize-k-1] + i; // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==2) {
      res = CreateScene(scn->zsize,scn->ysize,scn->xsize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++) {
              p = res->tbz[res->zsize-i-1] + res->tby[j] + k;  // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==3)          {
      res = CreateScene(scn->ysize,scn->xsize,scn->zsize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++)     {
              p = res->tbz[k] + res->tby[i] + (res->xsize-j-1);  // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==-1) {
      res = CreateScene(scn->xsize,scn->zsize,scn->ysize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++) {
              p = res->tbz[res->zsize-j-1] + res->tby[k] + i; // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==-2) {
      res = CreateScene(scn->zsize,scn->ysize,scn->xsize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++) {
              p = res->tbz[i] + res->tby[j] + (res->xsize-k-1);  // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==-3)          {
      res = CreateScene(scn->ysize,scn->xsize,scn->zsize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++)     {
              p = res->tbz[k] + res->tby[res->ysize-i-1] + j;  // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=scn->dz;
  return res;
}


Plane* First_Step_Sagittal(Scene *in, Scene *mask, int factor1, float nbest1, int step1, float *max_score)
{
  // Parameters for the method
  int verbose=1;  // 1-verbose mode
  int savesteps=0; // 1-Store intermediate images
  int c;
  Scene *scn;
  Scene *scn1,*grad1,*bin1,*mask1;
  int num1;
  Voxel *maskset1;
  int maskset1_size;
  float aux,max=0;
  Plane *max_plane=NULL;
  float z;
  int count=0;
  Context *ctx;
  Plane *plane;
  Point p1,p2,p3;

  scn=in;
  // resizing
  if (verbose) { printf("   Preprocessing: [resizing] "); fflush(stdout); }
  scn1 = msp_SubsampleByN(scn,factor1);
  if (savesteps) WriteScene(scn1,"debug-stage1-A.resized.scn");
  // computing edges
  if (verbose) { printf("[egdes] "); fflush(stdout); }
  // grad1 = SobelFilter3(scn1);
  grad1 = msp_TextGradient3(scn1);
  msp_ZeroBorders(grad1,1); // zero voxels on the border (width 3)
  num1 = nbest1 * SceneLen(grad1); // number of voxels to use in the segmentation (nbest=0.05 means 5%)
  if (mask!=NULL) {
    mask1 = msp_SubsampleByN_bin(mask,factor1);
    if (savesteps) WriteScene(mask1,"debug-stage1-Aa.resized-mask.scn");
    msp_ZeroMask(grad1,mask1);
    c = CountNonZeroVoxels(mask1);    
    if (num1>(c/3)) num1 = c/3;
    DestroyScene(&mask1);
  }
  if (savesteps) WriteScene(grad1,"debug-stage1-B.edges.scn");
  // Binarize the gradient
  if (verbose) { printf("[thresholding] \n");  fflush(stdout); }
  //if (mask==NULL)
  bin1 = msp_BinarizeByThreshold(grad1,msp_ThresholdToGetNMaxVoxels(grad1,num1));
  DestroyScene(&grad1);
  if (savesteps) WriteScene(bin1,"debug-stage1-C.bin_edges.scn");
  msp_MakeMasksetArray(bin1,&maskset1,&maskset1_size);
  // Searching for the MSP... 
  if (verbose) { printf("   Searching... ");  fflush(stdout); }
  ctx = CreateContext(bin1);
  plane = CreatePlane(ctx);
  DestroyContext(&ctx);
  p1.x=0;          p1.y=0;
  p2.x=bin1->xsize; p2.y=0;
  p3.x=0;          p3.y=bin1->ysize;
  max=0;
  max_plane=NULL;

  if (verbose) { printf("(step_size=%d)\n",step1);  fflush(stdout); }
  for (p1.z=0; p1.z<bin1->zsize; p1.z+=step1)
    for (p2.z=0; p2.z<bin1->zsize; p2.z+=step1)
      for (p3.z=0; p3.z<bin1->zsize; p3.z+=step1) {
	// consider only planes close to the centroid
	z = msp_GetZFromPlaneProjection(bin1->xsize/2,bin1->ysize/2,&p1,&p2,&p3);
	z = fabs(z-bin1->zsize/2); // distance to the centroid
	if (z>(bin1->zsize/6))
	  continue;
	// process this plane
	msp_GetPlaneFrom3Points(&p1,&p2,&p3,plane);
	aux=msp_SymmetryScoreBin(bin1, maskset1,maskset1_size,plane);
	if (aux>max) { 
	  max=aux; 
	  if (max_plane!=NULL) DestroyPlane(&max_plane);
	  max_plane=CopyPlane(plane);
	}
	count++;
  }
  VectorNormalize(&max_plane->normal);
  if (verbose) printf("   %d iteractions - MaxScore=%.3f \n   Plane: po=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n",count,max,max_plane->po.x,max_plane->po.y,max_plane->po.z,max_plane->normal.x,max_plane->normal.y,max_plane->normal.z);

  if (max_score!=NULL) *max_score = max;

  // Save the 1st stage result
  if (savesteps) {
    Scene *drawscn1;
    drawscn1 = msp_DrawMyPlaneAnyOrientation(scn1,max_plane);
    WriteScene(drawscn1,"debug-stage1-D.result.scn");
    DestroyScene(&drawscn1);
  }
  DestroyScene(&bin1);
  DestroyScene(&scn1);
  free(maskset1);
  return max_plane;
}




// Parameters:
// in: Input scene
// mask_in: consider just the voxels within this mask (0=bg / 1=fg)
// input_ori: 1=sagittal / 2=axial / 3=coronal / 0=unknown
// quality: 1=best accuracy/slowest / 2=moderate / 3=medium accuracy/fastest
// detected_plane: when input_ori=0, it returns the detected orientation, 1=X 2=Y 3=Z
Plane* FindSymmetryPlane(Scene *in, Scene *mask_in, int input_ori, int quality, int *detected_plane)  
{
  // Parameters for the method
  int verbose=1;  // 1-verbose mode
  int savesteps=0; // 1-Store intermediate images
  int factor1 = 4;  // => 1/4 downsampling for stage 1
  int factor2 = 2;  // => 1/2 downsampling for stage 2
  int factor3 = 1;  // => 1/1 downsampling for stage 3
  double nbest1 = 0.08; // 8% best for stage 1
  double nbest2 = 0.05; // 5% best for stage 2
  double nbest3 = 0.05; // 5% best for stage 3
  int step1=2;
  int step2=1;
  int range2=step1*(factor1/factor2);
  int step3=1;
  int range3=step2*(factor2/factor3);
  int det_plane=0;

  Plane *max_plane=NULL;
  float max_score_sag;
  
  Scene *scn=in;
  Scene *mask=mask_in;

  //second stage
  Point p1_max, p2_max, p3_max;
  int count=0;
  Point p1,p2,p3;
  float aux,max=0;
  Scene *scn2,*grad2,*bin2,*mask2;
  int num2;
  Voxel *maskset2;
  int maskset2_size;
  int max1,max2,max3;
  Context *ctx;
  Plane *plane;

  //third
    Scene *scn3,*grad3,*bin3,*mask3;
    int num3;
    Voxel *maskset3;
    int maskset3_size;


  if (in->xsize>300 || in->ysize>300 || in->zsize>300) { // for too large images (e.g. CT)
    factor1=8;
    factor2=4;
    factor3=2;
  }
  
  if (quality<1 || quality>3) quality=1; //default=1

  if (input_ori<0 || input_ori>3) input_ori=0; //default=0
  
  if (input_ori==1) det_plane=3; // Z
  if (input_ori==2) det_plane=1; // X
  if (input_ori==3) det_plane=1; // X

  // Reslicing the scene to sagittal orientation
  if (input_ori==2) { // axial
	Scene *tmp;
    if (verbose) { printf("Reslicing to sagittal... ");  fflush(stdout); }
    tmp = msp_RotateAxis(in,1);
    scn  = msp_RotateAxis(tmp,2);
    DestroyScene(&tmp);
    if (mask_in!=NULL) { Scene *tmp2=msp_RotateAxis(mask_in,1); mask  = msp_RotateAxis(tmp2,2); DestroyScene(&tmp2);  }
    if (verbose) { printf("done.\n");  fflush(stdout); }
    if (savesteps) WriteScene(scn,"debug-stage0-reeslicing.scn");
  }
  if (input_ori==3) { // coronal
    Scene *tmp;
    if (verbose) { printf("Reslicing to sagittal... ");  fflush(stdout); }
    tmp = msp_RotateAxis(in,2);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (mask_in!=NULL) { Scene *tmp2=msp_RotateAxis(mask_in,2); mask = CopyScene(tmp2); DestroyScene(&tmp2);  }
    if (verbose) { printf("done.\n");  fflush(stdout); }
    if (savesteps) WriteScene(scn,"debug-stage0-reeslicing.scn");
  }
  if (mask_in!=NULL && savesteps) WriteScene(mask,"debug-stage0-reeslicing-mask.scn");


  // -----------------------------------------------------------
  // FIRST STAGE 
  // -----------------------------------------------------------

  if (verbose) { printf("FIRST STAGE:\n"); fflush(stdout); }

  max_plane=NULL;
  max_plane = First_Step_Sagittal(scn,mask,factor1,nbest1,step1,&max_score_sag);
  
  // TEST THE OTHER TWO ORIENTATIONS WHEN INPUT_ORI is UNKNOWN(0)
  if (input_ori==0) {
    Scene *scn_X,*scn_Y;
    Scene *mask_X=NULL,*mask_Y=NULL;
    float max_score_X,max_score_Y;    
    Plane *max_plane_X=NULL,*max_plane_Y=NULL;
    Scene *tmp;
    printf("  Detecting other orientations... \n");  fflush(stdout); 
    // ---------------------------------------------------------
    if (verbose) { printf("  Trying planes X=n: [reslicing] \n");  fflush(stdout); }
    tmp = msp_RotateAxis(in,2);    
    scn_X = CopyScene(tmp);    
    DestroyScene(&tmp);    
    if (mask!=NULL) {Scene *tmp2 = msp_RotateAxis(mask,2); mask_X = CopyScene(tmp2); DestroyScene(&tmp2);}
    max_plane_X = First_Step_Sagittal(scn_X, mask_X,factor1,nbest1,step1,&max_score_X);
    // ---------------------------------------------------------
    if (verbose) { printf("  Trying planes Y=n: [reslicing] \n");  fflush(stdout); }
    tmp = msp_RotateAxis(in,1); 
    scn_Y = CopyScene(tmp); 
    DestroyScene(&tmp);
    if (mask!=NULL) {Scene *tmp2 = msp_RotateAxis(mask,1); mask_Y = CopyScene(tmp2); DestroyScene(&tmp2);}
    max_plane_Y = First_Step_Sagittal(scn_Y,mask_Y,factor1,nbest1,step1,&max_score_Y);
    // ---------------------------------------------------------

    if (max_score_X>max_score_sag && max_score_X>max_score_Y) {
      if (verbose) { printf("  MSP is closer to plane X=Xsize/2 (Axial or Coronal orientations)\n");  fflush(stdout); }
      det_plane = 1;
      scn = scn_X;
      mask=mask_X;
      DestroyPlane(&max_plane);
      max_plane = CopyPlane(max_plane_X);
      DestroyScene(&scn_Y);
      if (mask_Y!=NULL) DestroyScene(&mask_Y);
    }
    else 
    if (max_score_Y>max_score_sag && max_score_Y>max_score_X) {
      if (verbose) { printf("MSP is closer to plane Y=Ysize/2 \n");  fflush(stdout); }
      det_plane = 2;
      scn = scn_Y;
      mask=mask_Y;
      DestroyPlane(&max_plane);
      max_plane = CopyPlane(max_plane_Y);
      DestroyScene(&scn_X);
      if (mask_X!=NULL) DestroyScene(&mask_X);
    }      
    else
      if (verbose) { printf("  MSP is closer to plane Z=Zsize/2 (Sagittal orientation)\n");  fflush(stdout); }

    if (detected_plane!=NULL) *detected_plane=det_plane;
  }
    

  // -----------------------------------------------------------
  // SECOND STAGE 
  // -----------------------------------------------------------
  ctx = CreateContext(scn);
  plane = CreatePlane(ctx);
  DestroyContext(&ctx);
  // Adjust previous result to the new scale
  max_plane->po.x=max_plane->po.x*(factor1/factor2)+(0.5*((factor1/factor2)-1));
  max_plane->po.y=max_plane->po.y*(factor1/factor2)+(0.5*((factor1/factor2)-1));
  max_plane->po.z=max_plane->po.z*(factor1/factor2)+(0.5*((factor1/factor2)-1));
  if (quality<=2) {
    if (verbose) { printf("SECOND STAGE:\n"); fflush(stdout); }
    p1_max.x=0; p1_max.y=0; 
    p2_max.x=scn->xsize/factor2; p2_max.y=0;
    p3_max.x=0; p3_max.y=scn->ysize/factor2;
    p1_max.z = msp_GetZFromPlaneProjection2(0,0,max_plane);
    p2_max.z = msp_GetZFromPlaneProjection2(scn->xsize/factor2,0,max_plane);
    p3_max.z = msp_GetZFromPlaneProjection2(0,scn->ysize/factor2,max_plane);
    // resizing
    if (verbose) { printf("   Preprocessing: [resizing] "); fflush(stdout); }
    scn2 = msp_SubsampleByN(scn,factor2);
    if (savesteps) WriteScene(scn2,"debug-stage2-A.resized.scn");
    // computing edges
    if (verbose) { printf("[egdes] "); fflush(stdout); }
    // grad2 = SobelFilter3(scn2);
    grad2 = msp_TextGradient3(scn2);
    msp_ZeroBorders(grad2,2); // zero voxels on the border (width 3)
    num2 = nbest2 * SceneLen(grad2); // number of voxels to use in the segmentation (nbest=0.05 means 5%)
    if (mask!=NULL) {
	  int c;
      mask2 = msp_SubsampleByN_bin(mask,factor2);
      if (savesteps) WriteScene(mask2,"debug-stage2-Aa.resized-mask.scn");
      msp_ZeroMask(grad2,mask2);
      c = CountNonZeroVoxels(mask2);    
      if (num2>c/3) num2 = c/3;
      DestroyScene(&mask2);
    }
    if (savesteps) WriteScene(grad2,"debug-stage2-B.edges.scn");
    // Binarize the gradient
    if (verbose) { printf("[thresholding] \n");  fflush(stdout); }
    bin2 = msp_BinarizeByThreshold(grad2,msp_ThresholdToGetNMaxVoxels(grad2,num2));
    DestroyScene(&grad2);
    if (savesteps) WriteScene(bin2,"debug-stage2-C.bin_edges.scn");
    msp_MakeMasksetArray(bin2,&maskset2,&maskset2_size);
    // Searching for the MSP... 
    if (verbose) { printf("   Searching... ");  fflush(stdout); }
    count=0; max=0;
    if (verbose) { printf("(step_size=%d range=%d)\n",step2,range2);  fflush(stdout); }
    p1.x=0;          p1.y=0;
    p2.x=bin2->xsize; p2.y=0;
    p3.x=0;          p3.y=bin2->ysize;
    max1=p1_max.z;  max2=p2_max.z;  max3=p3_max.z;
    for (p1.z=max1-range2; p1.z<=max1+range2; p1.z+=step2) 
      for (p2.z=max2-range2; p2.z<=max2+range2; p2.z+=step2)
	for (p3.z=max3-range2; p3.z<=max3+range2; p3.z+=step2) {
	  msp_GetPlaneFrom3Points(&p1,&p2,&p3,plane);
	  aux=msp_SymmetryScoreBin(bin2,maskset2,maskset2_size,plane);
	  count++;
	  if (aux>max) {
	    max=aux;
	    if (max_plane!=NULL) DestroyPlane(&max_plane);
	    max_plane=CopyPlane(plane);
	    p1_max=p1; p2_max=p2; p3_max=p3;
	  }
	}
    VectorNormalize(&max_plane->normal);
    if (verbose) printf("   %d iteractions - MaxScore=%.3f \n   Plane: po=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n",count,max,max_plane->po.x,max_plane->po.y,max_plane->po.z,max_plane->normal.x,max_plane->normal.y,max_plane->normal.z);
    // Save the 2st stage result
    if (savesteps) {
      Scene *drawscn2;
      drawscn2 = msp_DrawMyPlaneAnyOrientation(scn2,max_plane);
      WriteScene(drawscn2,"debug-stage2-D.result.scn");
      DestroyScene(&drawscn2);
    }
    if (mask!=NULL) DestroyScene(&mask2);
    DestroyScene(&scn2);
    DestroyScene(&bin2);
    free(maskset2);
  }




  // -----------------------------------------------------------
  // THIRD STAGE 
  // -----------------------------------------------------------
  // Adjust previous result to the new scale
  max_plane->po.x=max_plane->po.x*(factor2/factor3)+(0.5*((factor2/factor3)-1));
  max_plane->po.y=max_plane->po.y*(factor2/factor3)+(0.5*((factor2/factor3)-1));
  max_plane->po.z=max_plane->po.z*(factor2/factor3)+(0.5*((factor2/factor3)-1));

  if (quality<=1) {
    if (verbose) { printf("THIRD STAGE:\n"); fflush(stdout); }
    p1_max.x=0; p1_max.y=0; 
    p2_max.x=scn->xsize/factor3; p2_max.y=0;
    p3_max.x=0; p3_max.y=scn->ysize/factor3;
    p1_max.z = msp_GetZFromPlaneProjection2(0,0,max_plane);
    p2_max.z = msp_GetZFromPlaneProjection2(scn->xsize/factor3,0,max_plane);
    p3_max.z = msp_GetZFromPlaneProjection2(0,scn->ysize/factor3,max_plane);
    if (savesteps) {
      Scene *adrawscn3;
      adrawscn3 = msp_DrawMyPlaneAnyOrientation(scn,max_plane);
      WriteScene(adrawscn3,"tmp.scn");
      DestroyScene(&adrawscn3);
    }   
    // resizing
    if (verbose) { printf("   Preprocessing: [resizing] "); fflush(stdout); }
    scn3 = msp_SubsampleByN(scn,factor3);
    if (savesteps) WriteScene(scn3,"debug-stage3-A.resized.scn");
    // computing edges
    if (verbose) { printf("[egdes] "); fflush(stdout); }
    // grad3 = SobelFilter3(scn3);
    grad3 = msp_TextGradient3(scn3);
    msp_ZeroBorders(grad3,4); // zero voxels on the border (width 3)
    num3 = nbest3 * SceneLen(grad3); // number of voxels to use in the segmentation (nbest=0.05 means 5%)
    if (mask!=NULL) {
      int c;
      mask3 = msp_SubsampleByN_bin(mask,factor3);
      if (savesteps) WriteScene(mask3,"debug-stage3-Aa.resized-mask.scn");
      msp_ZeroMask(grad3,mask3);
      c = CountNonZeroVoxels(mask3);    
      if (num3>c/3) num3 = c/3;
      DestroyScene(&mask3);
    }
    if (savesteps) WriteScene(grad3,"debug-stage3-B.edges.scn");
    // Binarize the gradient
    if (verbose) { printf("[thresholding] \n");  fflush(stdout); }
    bin3 = msp_BinarizeByThreshold(grad3,msp_ThresholdToGetNMaxVoxels(grad3,num3));
    DestroyScene(&grad3);
    if (savesteps) WriteScene(bin3,"debug-stage3-C.bin_edges.scn");
    msp_MakeMasksetArray(bin3,&maskset3,&maskset3_size);
    // Searching for the MSP... 
    if (verbose) { printf("   Searching... ");  fflush(stdout); }
    count=0; max=0;
    if (verbose) { printf("(step_size=%d range=%d)\n",step3,range3);  fflush(stdout); }
    p1.x=0;          p1.y=0;
    p2.x=bin3->xsize; p2.y=0;
    p3.x=0;          p3.y=bin3->ysize;
    max1=p1_max.z;  max2=p2_max.z;  max3=p3_max.z;
    for (p1.z=max1-range3; p1.z<=max1+range3; p1.z+=step3) 
      for (p2.z=max2-range3; p2.z<=max2+range3; p2.z+=step3)
	for (p3.z=max3-range3; p3.z<=max3+range3; p3.z+=step3) {
	  //printf("%.3f ",p3.z);
	  msp_GetPlaneFrom3Points(&p1,&p2,&p3,plane);
	  aux=msp_SymmetryScoreBin(bin3,maskset3,maskset3_size,plane);
	  count++;
	  if (aux>max) {
	    max=aux;
	    if (max_plane!=NULL) DestroyPlane(&max_plane);
	    max_plane=CopyPlane(plane);
	    p1_max=p1; p2_max=p2; p3_max=p3;
	  }
	}
    // in case of fator3 is not 1, do this to correct:
    max_plane->po.x=max_plane->po.x*(factor3)+(0.5*((factor3)-1));
    max_plane->po.y=max_plane->po.y*(factor3)+(0.5*((factor3)-1));
    max_plane->po.z=max_plane->po.z*(factor3)+(0.5*((factor3)-1));
    VectorNormalize(&max_plane->normal);
    if (verbose) printf("   %d iteractions - MaxScore=%.3f \n   Plane: po=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n",count,max,max_plane->po.x,max_plane->po.y,max_plane->po.z,max_plane->normal.x,max_plane->normal.y,max_plane->normal.z);
    // Save the 3rd stage result
    if (savesteps) {
      Scene *drawscn3;
      drawscn3 = msp_DrawMyPlaneAnyOrientation(scn3,max_plane);
      WriteScene(drawscn3,"debug-stage3-D.result.scn");
      DestroyScene(&drawscn3);
    }
    if (mask!=NULL) DestroyScene(&mask3);
    DestroyScene(&scn3);
    DestroyScene(&bin3);    
    free(maskset3);
  }

  if (scn!=in) DestroyScene(&scn);
 
  return max_plane; 
}



// Parameters:
// scn: Input scene
// mask: consider just the voxels within this mask (0=bg / 1=fg) or NULL
// input_ori: 1=sagittal / 2=axial / 3=coronal / 0=unknown
// quality: 1=best accuracy/slowest / 2=moderate / 3=medium accuracy/fastest
Scene* MSP_Align(Scene *in, Scene *mask, int input_ori, int quality)
{
  int verbose=1;
  
  Plane *msp;
  Scene *out;
  int detected_plane;
  Scene *scn;
  scn = in;

  if (input_ori<1 || input_ori>3) input_ori=0; //default=0
  //if (output_ori<1 || output_ori>3) output_ori=1; //default=1  

  // Compute the MSP
  msp=FindSymmetryPlane(in,mask,input_ori,quality,&detected_plane);

  // Reslicing the scene to sagittal orientation
  if (input_ori==2) { // axial
    Scene *tmp;
    if (verbose) { printf("Reslicing to sagittal... ");  fflush(stdout); }
    tmp = msp_RotateAxis(scn,1);
    scn  = msp_RotateAxis(tmp,2);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==3) { // coronal
    Scene *tmp;
    if (verbose) { printf("Reslicing to sagittal... ");  fflush(stdout); }
    tmp = msp_RotateAxis(scn,2);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }

  if (input_ori==0 && detected_plane==1) {  // axial or coronal
    Scene *tmp;
    if (verbose) { printf("Reslicing to sagittal...");  fflush(stdout); }
    tmp = msp_RotateAxis(scn,2);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==0 && detected_plane==2) { // laid-down coronal
    Scene *tmp;
    if (verbose) { printf("Reslicing to sagittal...");  fflush(stdout); }
    tmp = msp_RotateAxis(scn,1);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  

  // Rotate the volume according to MSP found
  if (verbose) { printf("Rotating... ");  fflush(stdout); }
  out = msp_RotateToMSP(scn,msp);
  if (verbose) { printf("done.\n");  fflush(stdout); }

  // Reslicing the output
  if (input_ori==2) { // axial
    Scene *tmp;
    if (verbose) { printf("Reslicing back to axial... ");  fflush(stdout); }
    tmp = msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out  = msp_RotateAxis(tmp,-1);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==3) { // coronal
    Scene *tmp;
    if (verbose) { printf("Reslicing back to coronal... ");  fflush(stdout); }
    tmp = msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }

  if (input_ori==0 && detected_plane==1) {  // axial or coronal
    Scene *tmp;
    if (verbose) { printf("Reslicing back to X=Xsize/2...");  fflush(stdout); }
    tmp = msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==0 && detected_plane==2) { // laid-down coronal
    Scene *tmp;
    if (verbose) { printf("Reslicing back to Y=Ysize/2...");  fflush(stdout); }
    tmp = msp_RotateAxis(out,-1);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  
  return out;
}






/*
//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------



// Parameters:
// in: Input scene
// mask_in: consider just the voxels within this mask (0=bg / 1=fg)
// input_ori: 1=sagittal / 2=axial / 3=coronal / 0=unknown
// quality: 1=best accuracy/slowest / 2=moderate / 3=medium accuracy/fastest
// detected_plane: when input_ori=0, it returns the detected orientation, 1=X 2=Y 3=Z
Plane* MSGA_FindSymmetryPlane(Scene *in, Scene *mask_in, int input_ori, int quality, int *detected_plane)  
{
  // Parameters for the method
  int verbose=1;  // 1-verbose mode
  int savesteps=1; // 1-Store intermediate images
  int factor1 = 4;  // => 1/4 downsampling for stage 1
  int factor2 = 2;  // => 1/2 downsampling for stage 2
  int factor3 = 1;  // => 1/1 downsampling for stage 3
  double nbest1 = 0.08; // 8% best for stage 1
  double nbest2 = 0.05; // 5% best for stage 2
  double nbest3 = 0.05; // 5% best for stage 3
  int step1=2;
  int step2=1;
  int range2=step1*(factor1/factor2);
  int step3=1;
  int range3=step2*(factor2/factor3);

  int det_plane=0;

  if (in->xsize>300 || in->ysize>300 || in->zsize>300) { // for too large images (e.g. CT)
    factor1=8;
    factor2=4;
    factor3=2;
  }
  
  if (quality<1 || quality>3) quality=1; //default=1

  if (input_ori<0 || input_ori>3) input_ori=0; //default=0
  
  if (input_ori==1) det_plane=3; // Z
  if (input_ori==2) det_plane=1; // X
  if (input_ori==3) det_plane=1; // X

  Scene *scn=in;
  Scene *mask=mask_in;

  // Reslicing the scene to sagittal orientation
  if (input_ori==2) { // axial
    if (verbose) { printf("Reslicing to sagittal... ");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(in,1);
    scn  = msp_RotateAxis(tmp,2);
    DestroyScene(&tmp);
    if (mask_in!=NULL) { Scene *tmp2=msp_RotateAxis(mask_in,1); mask  = msp_RotateAxis(tmp2,2); DestroyScene(&tmp2);  }
    if (verbose) { printf("done.\n");  fflush(stdout); }
    if (savesteps) WriteScene(scn,"debug-stage0-reeslicing.scn");
  }
  if (input_ori==3) { // coronal
    if (verbose) { printf("Reslicing to sagittal... ");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(in,2);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (mask_in!=NULL) { Scene *tmp2=msp_RotateAxis(mask_in,2); mask = CopyScene(tmp2); DestroyScene(&tmp2);  }
    if (verbose) { printf("done.\n");  fflush(stdout); }
    if (savesteps) WriteScene(scn,"debug-stage0-reeslicing.scn");
  }
  if (mask_in!=NULL && savesteps) WriteScene(mask,"debug-stage0-reeslicing-mask.scn");

  return NULL;
}



void MSGA_Iteration(Scene *bin, Voxel maskset[], int maskset_size, Plane *init_plane, float init_score, Plane *final_plane, float *final_score)
{
  // MSGA PARAMETERS
  int parRX[]={0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 75.0, 89.0, -0.5, -1.0, -3.0, -6.0, -12.0, -24.0, -48.0, -75.0, -89.0}
  int parRY[]={0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 75.0, 89.0, -0.5, -1.0, -3.0, -6.0, -12.0, -24.0, -48.0, -75.0, -89.0}
  int parRZ[]={0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 75.0, 89.0, -0.5, -1.0, -3.0, -6.0, -12.0, -24.0, -48.0, -75.0, -89.0}
  int parTX[]={0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 75.0, 89.0, -0.5, -1.0, -3.0, -6.0, -12.0, -24.0, -48.0, -75.0, -89.0}
  int parTY[]={0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 75.0, 89.0, -0.5, -1.0, -3.0, -6.0, -12.0, -24.0, -48.0, -75.0, -89.0}
  int parTZ[]={0.5, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 75.0, 89.0, -0.5, -1.0, -3.0, -6.0, -12.0, -24.0, -48.0, -75.0, -89.0}
  int* par_array[]={*parRX, *parRY, *parRZ, *parTX, *parTY, *parTZ}
  int par_size[]={18, 18, 18, 18, 18, 18}

  final_score=init_score;
  *final_plane=*init_plane;

  int i,k;
  Plane p;
  float score, bestscore;
  float bestparm[6]; 
  Plane bestplane; 
  for (k=0;k<6;k++) { 
    bestscore=0;
    bestparm[k]=0; 
    for (i=0; i<par_size[k]; i++) {
      p = *init_plane;
      // p=transf(p)
      // score = calc_score(p)
      if (score>bestscore) { bestscore = score;  bestparm[k] = par_array[k][i]; } // best for this scale
      if (score>*final_score) { *final_score=score;  *final_plane=p; } // best overall
    }
  }

  p = *initPlane;
  //transf(p)
  //score = 
  if (score>*final_score) { *final_score=score;  *final_plane=p; } // best overall
  
      
}
*/  


