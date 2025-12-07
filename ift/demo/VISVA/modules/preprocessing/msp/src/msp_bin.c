
#include "oldift.h"

/*
#include <sys/time.h>
double GetTimestamp()
{
  struct timeval t;
  double res;
  gettimeofday(&t,NULL);
  res = t.tv_sec + 0.000001*t.tv_usec;
  return res;
};
*/


int verbose = 0;
int savesteps = 0;
int draw=0;
int orientation=0; // sagittal=1, axial=2, coronal=2, unknown=0 (default)
float vs=1.0;   
int interp=0; //false  
int txt=0; 
int ct=0; //false

double time_preproc,time_search;



/* Scene* BinarizeByThreshold(Scene *scn, float threshold) */
/* {						 */
/*   Scene *bin; */
/*   int i; */
/*   int n = scn->xsize * scn->ysize * scn->zsize;  */
/*   bin = CreateScene(scn->xsize,scn->ysize,scn->zsize); */
/*   for (i=0;i<n;i++) */
/*     if (scn->data[i]>threshold) bin->data[i]=1; */
/*   return bin; */
/* } */


/* void MakeMasksetArray(Scene *scn, Voxel **maskset, int *size) */
/* { */
/*   int p,count=0,n,i; */
/*   Voxel v; */
/*   n = scn->xsize * scn->ysize * scn->zsize;  */
/*   for (i=0;i<n;i++) */
/*     if (scn->data[i]>0) count++; */
/*   *size = count; */
/*   *maskset = (Voxel *) malloc(count*sizeof(Voxel)); */
/*   i=0; */
/*   for (v.z=0; v.z < scn->zsize; v.z++) */
/*     for (v.y=0; v.y < scn->ysize; v.y++) */
/*       for (v.x=0; v.x < scn->xsize; v.x++) { */
/* 	p = v.x + scn->tby[v.y] + scn->tbz[v.z]; */
/* 	if (scn->data[p]>0) { */
/* 	  (*maskset)[i] = v; */
/* 	  i++; */
/* 	}   */
/*       } */
/* } */


/* Scene* MakeMask(Scene *scn) */
/* { */
/*   int n,i; */
/*   Scene *mask = CreateScene(scn->xsize,scn->ysize,scn->zsize); */
/*   n = scn->xsize * scn->ysize * scn->zsize;  */
/*   for (i=0;i<n;i++) */
/*     if (scn->data[i]>0) mask->data[i]=1; */
/*   return mask; */
/* } */





/* double GetZFromPlaneProjection(float x, float y, Point *p1, Point *p2, Point *p3) */
/* { */
/*         double A,B,C,D; */
/*         double x1=p1->x, x2=p2->x, x3=p3->x; */
/*         double y1=p1->y, y2=p2->y, y3=p3->y; */
/*         double z1=p1->z, z2=p2->z, z3=p3->z; */
/*         double res; */
/*         A = y1*(z2 - z3) + y2*(z3 - z1) + y3*(z1 - z2); */
/*         B = z1*(x2 - x3) + z2*(x3 - x1) + z3*(x1 - x2); */
/*         C = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2); */
/*         D = -(x1*(y2*z3 - y3*z2) + x2*(y3*z1 - y1*z3) + x3*(y1*z2 - y2*z1)); */
/*         res = - (D + B*y + A*x) / C; */
/*         return res; */
/* } */

/* double GetZFromPlaneProjection2(float x, float y, Plane *p) */
/* { */
/*         double A,B,C,D; */
/*         double res; */
/*         A = p->normal.x; */
/*         B = p->normal.y; */
/*         C = p->normal.z; */
/* 	D = -(A*p->po.x + B*p->po.y + C*p->po.z); */
/*         res = - (D + B*y + A*x) / C; */
/*         return res; */
/* } */







/* int ThresholdToGetNMaxVoxels(Scene *scn, int n) */
/* { */
/*   int last=n,pos; */
/*   Curve *hist = Histogram3(scn); */
/*   pos= hist->n-1; */
/*   while (last>0) { */
/*     if (hist->Y[pos]<0) printf("error"); */
/*     if (hist->Y[pos]==0) pos--; */
/*     else { */
/*       hist->Y[pos]--; */
/*       last--; */
/*     } */
/*     if (pos<0) return 0; // just in case */
/*   } */
/*   DestroyCurve(&hist); */
/*   return pos; */
/* } */



/* // Returns a score of symmetry (higher score=higher symmetry) */
/* // "mask" is a binary image of the relevant voxels (for binary images, scn and mask are usually the same) */
/* // "maskset" is the same information but in the format of an array of voxels */
/* /\* float SymmetryScore(Scene *scn, Scene *mask, Voxel maskset[], int maskset_size, Plane *p) *\/ */
/* /\* { *\/ */
/* /\*   int i,q; *\/ */
/* /\*   float dist,side; *\/ */
/* /\*   float A,B,C,D; *\/ */
/* /\*   Vector normal; *\/ */
/* /\*   Point symmetric; *\/ */
/* /\*   int symmetric_val; *\/ */
/* /\*   float score=0;  // Correlation -> score = V1 / sqrt(V2/V3);  *\/ */
/* /\*   float V1=0,V2=0,V3=0; *\/ */
/* /\*   // Set normal vector and plane equation *\/ */
/* /\*   normal = p->normal; *\/ */
/* /\*   VectorNormalize(&normal); *\/ */
/* /\*   A = normal.x; *\/ */
/* /\*   B = normal.y; *\/ */
/* /\*   C = normal.z; *\/ */
/* /\*   D = -(normal.x*p->po.x + normal.y*p->po.y + normal.z*p->po.z); *\/ */
/* /\*   for (i=0;i<maskset_size;i++) {     *\/ */
/* /\*        // Which side the point is? *\/ */
/* /\*        side = A*maskset[i].x + B*maskset[i].y + C*maskset[i].z + D;  *\/ */
/* /\*        // point-plane distance *\/ */
/* /\*        dist = fabs(side); *\/ */
/* /\*        // Find the symmetric point *\/ */
/* /\*        if (side>0) { // same side of the normal *\/ */
/* /\* 	 symmetric.x = maskset[i].x - (2*dist)*normal.x; *\/ */
/* /\* 	 symmetric.y = maskset[i].y - (2*dist)*normal.y; *\/ */
/* /\* 	 symmetric.z = maskset[i].z - (2*dist)*normal.z; *\/ */
/* /\*        } *\/ */
/* /\*        else { // opposite side of the normal (or in the plane) *\/ */
/* /\* 	 symmetric.x = maskset[i].x + (2*dist)*normal.x; *\/ */
/* /\* 	 symmetric.y = maskset[i].y + (2*dist)*normal.y; *\/ */
/* /\* 	 symmetric.z = maskset[i].z + (2*dist)*normal.z; *\/ */
/* /\*        } *\/ */
/* /\*        // If validvoxel *\/ */
/* /\*        if (dist>=1 && symmetric.x<=scn->xsize-1  *\/ */
/* /\* 	   && symmetric.y<=scn->ysize-1 && symmetric.z<=scn->zsize-1  *\/ */
/* /\* 	   && symmetric.x>=0 && symmetric.y>=0 && symmetric.z>=0) { *\/ */
/* /\* 	 //symmetric_val=GetVoxelValue_trilinear(scn,symmetric.x,symmetric.y,symmetric.z); *\/ */
/* /\* 	 symmetric_val=GetVoxelValue_nn(scn,symmetric.x,symmetric.y,symmetric.z); *\/ */
/* /\* 	 q = scn->tbz[maskset[i].z] + scn->tby[maskset[i].y] + maskset[i].x; *\/ */
/* /\* 	 V1 += symmetric_val * scn->data[q]; *\/ */
/* /\* 	 V2 += symmetric_val * symmetric_val; *\/ */
/* /\* 	 V3 += scn->data[q] * scn->data[q]; *\/ */
/* /\*        } *\/ */
/* /\*   } *\/ */
/* /\*   if (V2==0 || V3==0) *\/ */
/* /\*     score =0; *\/ */
/* /\*   else *\/ */
/* /\*     score = V1 / sqrt(V2*V3); *\/ */
/* /\*   return score; *\/ */
/* /\* } *\/ */




/* // Returns a score of symmetry (higher score=higher symmetry) */
/* // "mask" is a binary image of the relevant voxels (for binary images, scn and mask are usually the same) */
/* // "maskset" is the same information but in the format of an array of voxels */
/* float SymmetryScoreBin(Scene *scn, Scene *mask, Voxel maskset[], int maskset_size, Plane *p) */
/* { */
/*   int i,q; */
/*   float dist,side; */
/*   float A,B,C,D; */
/*   Vector normal; */
/*   Point symmetric; */
/*   int symmetric_val; */
/*   float score=0;  // Correlation -> score = V1 / sqrt(V2/V3);  */
/*   int M=0,N=maskset_size; */
/*   if (N==0) {  */
/*     printf("Error: empty binary image.\n"); exit(1); */
/*   } */
/*   // Set normal vector and plane equation */
/*   normal = p->normal; */
/*   VectorNormalize(&normal); */
/*   A = normal.x; */
/*   B = normal.y; */
/*   C = normal.z; */
/*   D = -(normal.x*p->po.x + normal.y*p->po.y + normal.z*p->po.z); */
/*   for (i=0;i<maskset_size;i++) {     */
/*        // Which side the point is? */
/*        side = A*maskset[i].x + B*maskset[i].y + C*maskset[i].z + D;  */
/*        // point-plane distance */
/*        dist = fabs(side); */
/*        // Find the symmetric point */
/*        if (side>0) { // same side of the normal */
/* 	 symmetric.x = maskset[i].x - (2*dist)*normal.x; */
/* 	 symmetric.y = maskset[i].y - (2*dist)*normal.y; */
/* 	 symmetric.z = maskset[i].z - (2*dist)*normal.z; */
/*        } */
/*        else { // opposite side of the normal (or in the plane) */
/* 	 symmetric.x = maskset[i].x + (2*dist)*normal.x; */
/* 	 symmetric.y = maskset[i].y + (2*dist)*normal.y; */
/* 	 symmetric.z = maskset[i].z + (2*dist)*normal.z; */
/*        } */
/*        // If validvoxel */
/*        if (dist>=1 && symmetric.x<=scn->xsize-1  */
/* 	   && symmetric.y<=scn->ysize-1 && symmetric.z<=scn->zsize-1  */
/* 	   && symmetric.x>=0 && symmetric.y>=0 && symmetric.z>=0) { */
/* 	 //symmetric_val=GetVoxelValue_trilinear(scn,symmetric.x,symmetric.y,symmetric.z); */
/* 	 symmetric_val=GetVoxelValue_nn(scn,symmetric.x,symmetric.y,symmetric.z); */
/* 	 q = scn->tbz[maskset[i].z] + scn->tby[maskset[i].y] + maskset[i].x; */
/* 	 if (symmetric_val!=0 && scn->data[q]!=0) M++; */
/*        } */
/*   } */
/*   score = (float)M/(float)N; */
/*   return score; */
/* } */






/* void GetPlaneFrom3Points(Point *p1, Point *p2, Point *p3, Plane *p) */
/* { */
  
/*   float A,B,C,D; */
/*   float x1=p1->x, x2=p2->x, x3=p3->x; */
/*   float y1=p1->y, y2=p2->y, y3=p3->y; */
/*   float z1=p1->z, z2=p2->z, z3=p3->z; */
/*   A = y1*(z2 - z3) + y2*(z3 - z1) + y3*(z1 - z2); */
/*   B = z1*(x2 - x3) + z2*(x3 - x1) + z3*(x1 - x2); */
/*   C = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2); */
/*   D = -(x1*(y2*z3 - y3*z2) + x2*(y3*z1 - y1*z3) + x3*(y1*z2 - y2*z1)); */
/*   p->normal.x=A; */
/*   p->normal.y=B; */
/*   p->normal.z=C; */
/*   p->po.x=0; */
/*   p->po.y=0; */
/*   p->po.z=-D/C; // find z in (0,0,z) */
/* } */



/* Scene* DrawMyPlaneAnyOrientation(Scene *scn, Plane *p) */
/* { */
/*   Scene *new = CopyScene(scn); */
/*   int maxval = MaximumValue3(scn); */
/*   int z,q; */
/*   Voxel v; */
/*   float A,B,C,D; */

/*   A = p->normal.x; */
/*   B = p->normal.y; */
/*   C = p->normal.z; */
/*   D = -(p->normal.x*p->po.x + p->normal.y*p->po.y + p->normal.z*p->po.z); */
/*   for (v.y=0; v.y < scn->ysize; v.y++) */
/*     for (v.x=0; v.x < scn->xsize; v.x++) { */
/*       z = (int)(((-D-B*v.y-A*v.x)/C)+0.5); */
/*       if (ValidVoxel(scn,v.x,v.y,z)) { */
/* 	q = v.x + scn->tby[v.y] + scn->tbz[z]; */
/* 	new->data[q]=maxval; */
/*       } */
/*     } */
/*   return new; */
/* } */


/* void DrawMSP(Scene *scn)  */
/* { */
/*   Voxel v; */
/*   int p; */
/*   int maxval = MaximumValue3(scn); */
/*   int z=scn->zsize/2; */
/*   for (v.y=0; v.y < scn->ysize; v.y++) */
/*     for (v.x=0; v.x < scn->xsize; v.x++) { */
/*       p = v.x + scn->tby[v.y] + scn->tbz[z]; */
/*       scn->data[p]=maxval; */
/*     } */
/* } */


  

/* void ZeroBorders(Scene *scn, int bordersize) */
/* { */
/*   Voxel u; */
/*   int p; */
/*   if (bordersize<1) return; */
/*   for (u.z=0; u.z < scn->zsize; u.z++)  */
/*     for (u.y=0; u.y < scn->ysize; u.y++) */
/*       for (u.x=0; u.x < scn->xsize; u.x++) { */
/* 	if (u.x<bordersize || u.y<bordersize || u.z<bordersize || */
/* 	    u.x>=scn->xsize-bordersize || u.y>=scn->ysize-bordersize || u.z>=scn->zsize-bordersize) { */
/* 	  p = u.x + scn->tby[u.y] + scn->tbz[u.z]; */
/* 	  scn->data[p]=0; */
/* 	} */
/*       } */
/* } */


/* void PrintSum(Scene *scn) */
/* { */
/*   Voxel u; */
/*   int i, p, sum,max=0,max_i=0; */
/*   for (i=0;i<scn->ysize;i++) { */
/*     sum=0; */
/*     for (u.z=0; u.z < scn->zsize; u.z++)  */
/*       for (u.x=0; u.x < scn->xsize; u.x++) { */
/* 	p = u.x + scn->tby[i] + scn->tbz[u.z]; */
/* 	sum += scn->data[p]; */
/*       } */
/*     printf("slice%03d = %d\n",i,sum); */
/*     if (sum>max) { max=sum; max_i=i; } */
/*   } */
/*   printf("Max=%d (slice %d)\n",max,max_i); */
/* } */


/* Scene* CTSobelFilter3(Scene *scn) */
/* { */
/*   Scene *res = CreateScene(scn->xsize,scn->ysize,scn->zsize); */
/*   Image *tmp1,*tmp2; */
/*   int i; */

/*   for (i=0;i<scn->ysize;i++) { */
/*     tmp1 = GetYSlice(scn,i); */
/*     tmp2 = SobelFilter(tmp1); */
/*     PutYSlice(tmp2,res,i); */
/*     DestroyImage(&tmp1); */
/*     DestroyImage(&tmp2); */
/*   } */
/*   return res; */

/* } */





/* Scene* SubsampleByN(Scene *scn, int n) // eg. n = 2 is half-subsample */
/* { */
/*   Scene *new; */
/*   Voxel v,u; */
/*   int p,acc; */

/*   if (n==1) { */
/*     new = CopyScene(scn); */
/*     return new; */
/*   } */
/*   new = CreateScene(scn->xsize/n,scn->ysize/n,scn->zsize/n); */
/*   new->dx=scn->dx*n; */
/*   new->dy=scn->dy*n; */
/*   new->dz=scn->dz*n; */
/*   for (v.z=0; v.z < new->zsize; v.z++) */
/*     for (v.y=0; v.y < new->ysize; v.y++) */
/*       for (v.x=0; v.x < new->xsize; v.x++) { */
/*         p = new->tbz[v.z] + new->tby[v.y] + v.x; */
/* 	acc=0; */
/* 	for (u.z=0; u.z < n; u.z++) */
/* 	  for (u.y=0; u.y < n; u.y++) */
/* 	    for (u.x=0; u.x < n; u.x++) { */
/* 	      acc += scn->data[scn->tbz[v.z*n+u.z] + scn->tby[v.y*n+u.y] + v.x*n+u.x]; */
/* 	    } */
/* 	new->data[p] = acc/(n*n*n); */
/*       } */
/*   return new; */
/* } */






/* float Distance(float *f1, float *f2, int n){ */
/*   int i; */
/*   float dist; */
  
/*   dist = 0; */
/*   for (i=0; i < n; i++)  */
/*     dist += (f2[i]-f1[i]);///2.0; */

/*   //  dist /= n; */
/*   return(dist);//exp(-(dist-0.5)*(dist-0.5)/2.0)); */
/* } */


/* Scene *TextGradient3(Scene *scn){ */
/*   float   dist,gx,gy,gz;   */
/*   int     Imax,i,p,q,n=scn->xsize*scn->ysize*scn->zsize; */
/*   Voxel   u,v; */
/*   AdjRel3 *A=Spheric(1.0),*A6=Spheric(1.0); */
/*   float   *mg=AllocFloatArray(A6->n); */
/*   Scene   *grad=CreateScene(scn->xsize,scn->ysize,scn->zsize); */

/*   typedef struct _features { */
/*     float *f; */
/*   } Features; */

/*   Features *feat=(Features *)calloc(n,sizeof(Features)); */
/*   for (p=0; p < n; p++) */
/*     feat[p].f = AllocFloatArray(A->n); */

/*   Imax = MaximumValue3(scn); */

/*   for (u.z=0; u.z < scn->zsize; u.z++)  */
/*     for (u.y=0; u.y < scn->ysize; u.y++)  */
/*       for (u.x=0; u.x < scn->xsize; u.x++) { */
/* 	p = u.x + scn->tby[u.y] + scn->tbz[u.z]; */
/* 	for (i=0; i < A->n; i++) { */
/* 	  v.x = u.x + A->dx[i]; */
/* 	  v.y = u.y + A->dy[i]; */
/* 	  v.z = u.z + A->dz[i]; */
/* 	  if (ValidVoxel(scn,v.x,v.y,v.z)){ */
/* 	    q = v.x + scn->tby[v.y] + scn->tbz[v.z]; */
/* 	    feat[p].f[i]=(float)scn->data[q];///(float)Imax; */
/* 	  } */
/* 	} */
/*       } */

/*   for (i=0; i < A6->n; i++)  */
/*     mg[i]=sqrt(A6->dx[i]*A6->dx[i]+A6->dy[i]*A6->dy[i]+A6->dz[i]*A6->dz[i]); */

/*   for (u.z=0; u.z < scn->zsize; u.z++)  */
/*     for (u.y=0; u.y < scn->ysize; u.y++)  */
/*       for (u.x=0; u.x < scn->xsize; u.x++) { */
/* 	p = u.x + scn->tby[u.y] + scn->tbz[u.z]; */
/* 	gx = gy = gz = 0.0; */
/* 	for (i=1; i < A6->n; i++) { */
/* 	  v.x = u.x + A6->dx[i]; */
/* 	  v.y = u.y + A6->dy[i]; */
/* 	  v.z = u.z + A6->dz[i]; */
/* 	  if (ValidVoxel(scn,v.x,v.y,v.z)){ */
/* 	    q = v.x + scn->tby[v.y] + scn->tbz[v.z]; */
/* 	    dist = Distance(feat[p].f,feat[q].f,A->n); */
/* 	    gx  += dist*A6->dx[i]/mg[i];  */
/* 	    gy  += dist*A6->dy[i]/mg[i];  */
/* 	    gz  += dist*A6->dz[i]/mg[i];  */
/* 	  } */
/* 	}  */
/* 	grad->data[p]=(int)sqrt(gx*gx + gy*gy + gz*gz); */
/*       } */


/*   for (p=0; p < n; p++) */
/*     free(feat[p].f); */
/*   free(feat); */

/*   free(mg); */
/*   DestroyAdjRel3(&A); */
/*   DestroyAdjRel3(&A6); */
/*   return(grad); */
/* } */





/* Plane* FindSymmetry3(Scene *scn) */
/* { */
/*   double time1=0,time2=0,time3=0; */
/*   int factor1 = 4;  // => 1/4 downsampling for stage 1 */
/*   int factor2 = 2;  // => 1/2 downsampling for stage 2 */
/*   int factor3 = 1;  // => 1/1 downsampling for stage 3 */
  
/*   time1=GetTimestamp(); */

/*   // ----------------------------------------------------------- */
/*   // FIRST STAGE  */
/*   // ----------------------------------------------------------- */
/*   if (verbose) { printf("FIRST STAGE:\n"); fflush(stdout); } */
/*   // resizing */
/*   Scene *scn1,*grad1,*bin1; */
/*   if (verbose) { printf("   Preprocessing: [resizing] "); fflush(stdout); } */
/*   scn1 = SubsampleByN(scn,factor1); */
/*   if (savesteps) WriteScene(scn1,"debug-stage1-A.resized.scn"); */
/*   // computing edges */
/*   if (verbose) { printf("[egdes] "); fflush(stdout); } */
/*   if (ct) grad1 = CTSobelFilter3(scn1); */
/*   //else grad1 = SobelFilter3(scn1); */
/*   else grad1 = TextGradient3(scn1); */
/*   ZeroBorders(grad1,1); // zero voxels on the border (width 3) */
/*   if (savesteps) WriteScene(grad1,"debug-stage1-B.edges.scn"); */
/*   // Binarize the gradient */
/*   if (verbose) { printf("[thresholding] \n");  fflush(stdout); } */
/*   int num1 = 0.08 * SceneLen(grad1); // number of voxels to use in the segmentation (0.05=5%) */
/*   bin1 = BinarizeByThreshold(grad1,ThresholdToGetNMaxVoxels(grad1,num1)); */
/*   if (savesteps) WriteScene(bin1,"debug-stage1-C.bin_edges.scn"); */
/*   Voxel *maskset1; */
/*   int maskset1_size; */
/*   MakeMasksetArray(bin1,&maskset1,&maskset1_size); */
/*   // Searching for the MSP...  */
/*   if (verbose) { printf("   Searching... ");  fflush(stdout); } */
/*   Context *ctx = CreateContext(bin1); */
/*   Plane *plane = CreatePlane(ctx); */
/*   DestroyContext(&ctx); */
/*   Point p1,p2,p3; */
/*   p1.x=0;          p1.y=0; */
/*   p2.x=bin1->xsize; p2.y=0; */
/*   p3.x=0;          p3.y=bin1->ysize; */
/*   float aux,max=0; */
/*   Plane *max_plane=NULL; */
/*   Point p1_max, p2_max, p3_max; */
/*   p1_max.x=p2_max.x=p3_max.x=0;  */
/*   p1_max.y=p2_max.y=p3_max.y=0;  */
/*   p1_max.z=p2_max.z=p3_max.z=0;  */
/*   int step1=2; */
/*   float z; */
/*   int count=0; */
/*   if (verbose) { printf("(step_size=%d)\n",step1);  fflush(stdout); } */
/*   for (p1.z=0; p1.z<bin1->zsize; p1.z+=step1) */
/*     for (p2.z=0; p2.z<bin1->zsize; p2.z+=step1) */
/*       for (p3.z=0; p3.z<bin1->zsize; p3.z+=step1) { */
/* 	// consider only planes close to the centroid */
/* 	z = GetZFromPlaneProjection(bin1->xsize/2,bin1->ysize/2,&p1,&p2,&p3); */
/* 	z = fabs(z-bin1->zsize/2); // distance to the centroid */
/* 	if (z>(bin1->zsize/6)) */
/* 	  continue; */
/* 	// process this plane */
/* 	GetPlaneFrom3Points(&p1,&p2,&p3,plane); */
/* 	aux=SymmetryScoreBin(bin1,bin1,maskset1,maskset1_size,plane); */
/* 	if (aux>max) {  */
/* 	  max=aux;  */
/* 	  if (max_plane!=NULL) DestroyPlane(&max_plane); */
/* 	  max_plane=CopyPlane(plane); */
/* 	  p1_max=p1; p2_max=p2; p3_max=p3; */
/* 	} */
/* 	count++; */
/*       } */
/*   VectorNormalize(&max_plane->normal); */
/*   if (verbose) printf("   %d iteractions - MaxScore=%.3f \n   Plane: po=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n",count,max,max_plane->po.x,max_plane->po.y,max_plane->po.z,max_plane->normal.x,max_plane->normal.y,max_plane->normal.z); */
/*   // Save the 1st stage result */
/*   if (savesteps) { */
/*     Scene *drawscn1; */
/*     drawscn1 = DrawMyPlaneAnyOrientation(scn1,max_plane); */
/*     WriteScene(drawscn1,"debug-stage1-D.result.scn"); */
/*     DestroyScene(&drawscn1); */
/*   } */
/*   DestroyScene(&scn1); */
/*   DestroyScene(&bin1); */
/*   DestroyScene(&grad1); */
  

/*   // ----------------------------------------------------------- */
/*   // SECOND STAGE  */
/*   // ----------------------------------------------------------- */
/*   if (verbose) { printf("SECOND STAGE:\n"); fflush(stdout); } */
/*   // Adjust previous result to the new scale */
/*   max_plane->po.x=max_plane->po.x*(factor1/factor2)+(0.5*((factor1/factor2)-1)); */
/*   max_plane->po.y=max_plane->po.y*(factor1/factor2)+(0.5*((factor1/factor2)-1)); */
/*   max_plane->po.z=max_plane->po.z*(factor1/factor2)+(0.5*((factor1/factor2)-1)); */
/*   p1_max.x=0; p1_max.y=0;  */
/*   p2_max.x=scn->xsize/factor2; p2_max.y=0; */
/*   p3_max.x=0; p3_max.y=scn->ysize/factor2; */
/*   p1_max.z = GetZFromPlaneProjection2(0,0,max_plane); */
/*   p2_max.z = GetZFromPlaneProjection2(scn->xsize/factor2,0,max_plane); */
/*   p3_max.z = GetZFromPlaneProjection2(0,scn->ysize/factor2,max_plane); */
/*   // resizing */
/*   Scene *scn2,*grad2,*bin2; */
/*   if (verbose) { printf("   Preprocessing: [resizing] "); fflush(stdout); } */
/*   scn2 = SubsampleByN(scn,factor2); */
/*   if (savesteps) WriteScene(scn2,"debug-stage2-A.resized.scn"); */
/*   // computing edges */
/*   if (verbose) { printf("[egdes] "); fflush(stdout); } */
/*   if (ct) grad2 = CTSobelFilter3(scn2); */
/*   //else grad2 = SobelFilter3(scn2); */
/*   else grad2 = TextGradient3(scn2); */
/*   ZeroBorders(grad2,2); // zero voxels on the border (width 3) */
/*   if (savesteps) WriteScene(grad2,"debug-stage2-B.edges.scn"); */
/*   // Binarize the gradient */
/*   if (verbose) { printf("[thresholding] \n");  fflush(stdout); } */
/*   int num2 = 0.05 * SceneLen(grad2); // number of voxels to use in the segmentation (0.05=5%) */
/*   bin2 = BinarizeByThreshold(grad2,ThresholdToGetNMaxVoxels(grad2,num2)); */
/*   if (savesteps) WriteScene(bin2,"debug-stage2-C.bin_edges.scn"); */
/*   Voxel *maskset2; */
/*   int maskset2_size; */
/*   MakeMasksetArray(bin2,&maskset2,&maskset2_size); */
/*   // Searching for the MSP...  */
/*   if (verbose) { printf("   Searching... ");  fflush(stdout); } */
/*   count=0; max=0; */
/*   int step2=1; */
/*   int range2=step1*(factor1/factor2); */
/*   if (verbose) { printf("(step_size=%d range=%d)\n",step2,range2);  fflush(stdout); } */
/*   p1.x=0;          p1.y=0; */
/*   p2.x=bin2->xsize; p2.y=0; */
/*   p3.x=0;          p3.y=bin2->ysize; */
/*   int max1,max2,max3; */
/*   max1=p1_max.z;  max2=p2_max.z;  max3=p3_max.z; */
/*   for (p1.z=max1-range2; p1.z<=max1+range2; p1.z+=step2)  */
/*     for (p2.z=max2-range2; p2.z<=max2+range2; p2.z+=step2) */
/*       for (p3.z=max3-range2; p3.z<=max3+range2; p3.z+=step2) { */
/* 	GetPlaneFrom3Points(&p1,&p2,&p3,plane); */
/* 	aux=SymmetryScoreBin(bin2,bin2,maskset2,maskset2_size,plane); */
/* 	count++; */
/* 	if (aux>max) { */
/* 	  max=aux; */
/* 	  if (max_plane!=NULL) DestroyPlane(&max_plane); */
/* 	  max_plane=CopyPlane(plane); */
/* 	  p1_max=p1; p2_max=p2; p3_max=p3; */
/* 	} */
/*       } */
/*   VectorNormalize(&max_plane->normal); */
/*   if (verbose) printf("   %d iteractions - MaxScore=%.3f \n   Plane: po=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n",count,max,max_plane->po.x,max_plane->po.y,max_plane->po.z,max_plane->normal.x,max_plane->normal.y,max_plane->normal.z); */
/*   // Save the 2st stage result */
/*   if (savesteps) { */
/*     Scene *drawscn2; */
/*     drawscn2 = DrawMyPlaneAnyOrientation(scn2,max_plane); */
/*     WriteScene(drawscn2,"debug-stage2-D.result.scn"); */
/*     DestroyScene(&drawscn2); */
/*   } */
/*   DestroyScene(&scn2); */
/*   DestroyScene(&bin2); */
/*   DestroyScene(&grad2); */





/*   // ----------------------------------------------------------- */
/*   // THIRD STAGE  */
/*   // ----------------------------------------------------------- */
/*   if (verbose) { printf("THIRD STAGE:\n"); fflush(stdout); } */
/*   // Adjust previous result to the new scale */
/*   max_plane->po.x=max_plane->po.x*(factor2/factor3)+(0.5*((factor2/factor3)-1)); */
/*   max_plane->po.y=max_plane->po.y*(factor2/factor3)+(0.5*((factor2/factor3)-1)); */
/*   max_plane->po.z=max_plane->po.z*(factor2/factor3)+(0.5*((factor2/factor3)-1)); */
/*   p1_max.x=0; p1_max.y=0;  */
/*   p2_max.x=scn->xsize/factor3; p2_max.y=0; */
/*   p3_max.x=0; p3_max.y=scn->ysize/factor3; */
/*   p1_max.z = GetZFromPlaneProjection2(0,0,max_plane); */
/*   p2_max.z = GetZFromPlaneProjection2(scn->xsize/factor3,0,max_plane); */
/*   p3_max.z = GetZFromPlaneProjection2(0,scn->ysize/factor3,max_plane); */
/*   if (savesteps) { */
/*     Scene *adrawscn3; */
/*     adrawscn3 = DrawMyPlaneAnyOrientation(scn,max_plane); */
/*     WriteScene(adrawscn3,"tmp.scn"); */
/*     DestroyScene(&adrawscn3); */
/*   } */
/*   // resizing */
/*   Scene *scn3,*grad3,*bin3; */
/*   if (verbose) { printf("   Preprocessing: [resizing] "); fflush(stdout); } */
/*   scn3 = SubsampleByN(scn,factor3); */
/*   if (savesteps) WriteScene(scn3,"debug-stage3-A.resized.scn"); */
/*   // computing edges */
/*   if (verbose) { printf("[egdes] "); fflush(stdout); } */
/*   if (ct) grad3 = CTSobelFilter3(scn3); */
/*   //else grad3 = SobelFilter3(scn3); */
/*   else grad3 = TextGradient3(scn3); */
/*   ZeroBorders(grad3,4); // zero voxels on the border (width 3) */
/*   if (savesteps) WriteScene(grad3,"debug-stage3-B.edges.scn"); */
/*   // Binarize the gradient */
/*   if (verbose) { printf("[thresholding] \n");  fflush(stdout); } */
/*   int num3 = 0.05 * SceneLen(grad3); // number of voxels to use in the segmentation (0.05=5%) */
/*   bin3 = BinarizeByThreshold(grad3,ThresholdToGetNMaxVoxels(grad3,num3)); */
/*   if (savesteps) WriteScene(bin3,"debug-stage3-C.bin_edges.scn"); */
/*   Voxel *maskset3; */
/*   int maskset3_size; */
/*   MakeMasksetArray(bin3,&maskset3,&maskset3_size); */
/*   // Searching for the MSP...  */
/*   if (verbose) { printf("   Searching... ");  fflush(stdout); } */
/*   count=0; max=0; */
/*   int step3=1; */
/*   int range3=step2*(factor2/factor3); */
/*   if (verbose) { printf("(step_size=%d range=%d)\n",step3,range3);  fflush(stdout); } */
/*   p1.x=0;          p1.y=0; */
/*   p2.x=bin3->xsize; p2.y=0; */
/*   p3.x=0;          p3.y=bin3->ysize; */
/*   max1=p1_max.z;  max2=p2_max.z;  max3=p3_max.z; */
/*   for (p1.z=max1-range3; p1.z<=max1+range3; p1.z+=step3)  */
/*     for (p2.z=max2-range3; p2.z<=max2+range3; p2.z+=step3) */
/*       for (p3.z=max3-range3; p3.z<=max3+range3; p3.z+=step3) { */
/* 	//printf("%.3f ",p3.z); */
/* 	GetPlaneFrom3Points(&p1,&p2,&p3,plane); */
/* 	aux=SymmetryScoreBin(bin3,bin3,maskset3,maskset3_size,plane); */
/* 	count++; */
/* 	if (aux>max) { */
/* 	  max=aux; */
/* 	  if (max_plane!=NULL) DestroyPlane(&max_plane); */
/* 	  max_plane=CopyPlane(plane); */
/* 	  p1_max=p1; p2_max=p2; p3_max=p3; */
/* 	} */
/*       } */
/*   VectorNormalize(&max_plane->normal); */
/*   if (verbose) printf("   %d iteractions - MaxScore=%.3f \n   Plane: po=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n",count,max,max_plane->po.x,max_plane->po.y,max_plane->po.z,max_plane->normal.x,max_plane->normal.y,max_plane->normal.z); */


/*   // Save the 3rd stage result */
/*   if (savesteps) { */
/*     Scene *drawscn3; */
/*     drawscn3 = DrawMyPlaneAnyOrientation(scn3,max_plane); */
/*     WriteScene(drawscn3,"debug-stage3-D.result.scn"); */
/*     DestroyScene(&drawscn3); */
/*   } */

/*   DestroyScene(&scn3); */
/*   DestroyScene(&bin3); */
/*   DestroyScene(&grad3); */

/*   time3=GetTimestamp(); */
/*   time_preproc=time2-time1; // is 0 now */
/*   time_search=time3; //-time2; */


/*   return max_plane;  */
/* } */



/* Scene*  MyRotate3(Scene *scn,  */
/* 		double thx, double thy, double thz, // angles to rotate */
/* 		  int cx, int cy, int cz, // center of the rotation */
/*                 double transx, double transy, double transz) // translation after rotation */
/* { */
/*   int p; */
/*   Voxel u; */
/*   Scene *res; */
/*   RealMatrix *trans1,*rot1,*rot2,*rot3,*trans2,*aux1,*aux2,*inv,*vox; */
/*   trans1 = TranslationMatrix3(-cx,-cy,-cz); */
/*   rot1 = RotationMatrix3(0,thx); */
/*   rot2 = RotationMatrix3(1,thy); */
/*   rot3 = RotationMatrix3(2,thz); */
/*   trans2 = TranslationMatrix3(cx+transx,cy+transy,cz+transz); */
/*   // Compose transform */
/*   aux1 = MultRealMatrix(trans2,rot3); */
/*   aux2 = MultRealMatrix(aux1,rot2); */
/*   DestroyRealMatrix(&aux1); */
/*   aux1 = MultRealMatrix(aux2,rot1); */
/*   DestroyRealMatrix(&aux2); */
/*   aux2 = MultRealMatrix(aux1,trans1); */
/*   inv = InvertRealMatrix(aux2); */
/*   DestroyRealMatrix(&trans1); */
/*   DestroyRealMatrix(&rot1); */
/*   DestroyRealMatrix(&rot2); */
/*   DestroyRealMatrix(&rot3); */
/*   DestroyRealMatrix(&trans2); */
/*   DestroyRealMatrix(&aux2); */
/*   DestroyRealMatrix(&aux1); */
/*   // Applying transform for all voxels */
/*   res = CreateScene(scn->xsize,scn->ysize,scn->zsize); */
/*   for (u.z=0; u.z < res->zsize; u.z++)  */
/*     for (u.y=0; u.y < res->ysize; u.y++) */
/*       for (u.x=0; u.x < res->xsize; u.x++){ */
/* 	p = u.x + res->tby[u.y] + res->tbz[u.z]; */
/* 	vox = TransformVoxel(inv,u); // acelerar isso!!!! */
/* 	if ((vox->val[0][0]<=res->xsize-1)&&(vox->val[0][1]<=res->ysize-1) */
/* 	    &&(vox->val[0][2]<=res->zsize-1 && vox->val[0][0]>0 && vox->val[0][1]>0 && vox->val[0][2]>0)) */
/* 	  { */
/* 	    res->data[p]=GetVoxelValue_trilinear(scn,vox->val[0][0],vox->val[0][1],vox->val[0][2]); */
/* 	  } */
/* 	else */
/* 	  res->data[p]=0; */
/* 	DestroyRealMatrix(&vox); */
/*       } */
/*   return res; */
/* } */



/* Scene*  MyRotateToMSP(Scene *scn, Plane *p) */
/* { */
/*     Point center; */
/*     Scene *res; */
/*     float thx,thy,transz; */
/*     float A,B,C,D; */
/*     if (verbose) { printf("Rotating Scene... ");  fflush(stdout); } */
/*     A = p->normal.x; */
/*     B = p->normal.y; */
/*     C = p->normal.z; */
/*     D = -(p->normal.x*p->po.x + p->normal.y*p->po.y + p->normal.z*p->po.z); */
/*     center.x = scn->xsize/2.0; */
/*     center.y = scn->ysize/2.0; */
/*     center.z = (-D-B*center.y-A*center.x)/C; */
/*     transz = ((int)scn->zsize/2) - center.z; */
/*     if (C==0) {printf("error: thx=0\n"); exit(1);}  */
/*        else thx = atan(B/C);  */
/*     if (C==0) {printf("error: thx=0\n"); exit(1);} */
/*     else {  */
/*       thy = - atan(A/(B*sin(thx)+C*cos(thx)));  */
/*     } */
/*     if (verbose) printf("th_x=%.3frad -> th_y=%.3frad -> trans_z=%.3f\n",thx,thy,transz); */
/*     res = MyRotate3(scn,thx,thy,0, center.x,center.y,center.z, 0,0,transz); */
/*     return res; */
/* } */



/* Scene* RotateAxis(Scene *scn, int plane)  // plane= 1(X) 2(Y) 3(Z) -1(-X) -2(-Y) -3(-Z) */
/* { */
/*   Scene *res=NULL; */
/*   int i,j,k,p,q; */

/*   if (plane==1) { */
/*       res = CreateScene(scn->xsize,scn->zsize,scn->ysize); */
/*       for (i=0;i<scn->xsize;i++) */
/*         for (j=0;j<scn->ysize;j++) */
/*           for (k=0;k<scn->zsize;k++) { */
/*               p = res->tbz[j] + res->tby[res->ysize-k-1] + i; // res voxel */
/*               q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel */
/*               res->data[p]=scn->data[q]; */
/*             } */
/*     } */
/*   if (plane==2) { */
/*       res = CreateScene(scn->zsize,scn->ysize,scn->xsize); */
/*       for (i=0;i<scn->xsize;i++) */
/*         for (j=0;j<scn->ysize;j++) */
/*           for (k=0;k<scn->zsize;k++) { */
/*               p = res->tbz[res->zsize-i-1] + res->tby[j] + k;  // res voxel */
/*               q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel */
/*               res->data[p]=scn->data[q]; */
/*             } */
/*     } */
/*   if (plane==3)          { */
/*       res = CreateScene(scn->ysize,scn->xsize,scn->zsize); */
/*       for (i=0;i<scn->xsize;i++) */
/*         for (j=0;j<scn->ysize;j++) */
/*           for (k=0;k<scn->zsize;k++)     { */
/*               p = res->tbz[k] + res->tby[i] + (res->xsize-j-1);  // res voxel */
/*               q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel */
/*               res->data[p]=scn->data[q]; */
/*             } */
/*     } */
/*   if (plane==-1) { */
/*       res = CreateScene(scn->xsize,scn->zsize,scn->ysize); */
/*       for (i=0;i<scn->xsize;i++) */
/*         for (j=0;j<scn->ysize;j++) */
/*           for (k=0;k<scn->zsize;k++) { */
/*               p = res->tbz[res->zsize-j-1] + res->tby[k] + i; // res voxel */
/*               q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel */
/*               res->data[p]=scn->data[q]; */
/*             } */
/*     } */
/*   if (plane==-2) { */
/*       res = CreateScene(scn->zsize,scn->ysize,scn->xsize); */
/*       for (i=0;i<scn->xsize;i++) */
/*         for (j=0;j<scn->ysize;j++) */
/*           for (k=0;k<scn->zsize;k++) { */
/*               p = res->tbz[i] + res->tby[j] + (res->xsize-k-1);  // res voxel */
/*               q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel */
/*               res->data[p]=scn->data[q]; */
/*             } */
/*     } */
/*   if (plane==-3)          { */
/*       res = CreateScene(scn->ysize,scn->xsize,scn->zsize); */
/*       for (i=0;i<scn->xsize;i++) */
/*         for (j=0;j<scn->ysize;j++) */
/*           for (k=0;k<scn->zsize;k++)     { */
/*               p = res->tbz[k] + res->tby[res->ysize-i-1] + j;  // res voxel */
/*               q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel */
/*               res->data[p]=scn->data[q]; */
/*             } */
/*     } */
/*   return res; */
/* } */










Scene*  msp_Rotate3(Scene *scn, 
		double thx, double thy, double thz, // angles to rotate
		  int cx, int cy, int cz, // center of the rotation
		    double transx, double transy, double transz); // translation after rotation


void DrawMSP(Scene *scn);

Scene*  msp_RotateToMSP(Scene *scn, Plane *p);




Scene* msp_RotateAxis(Scene *scn, int plane);  // plane= 1(X) 2(Y) 3(Z) -1(-X) -2(-Y) -3(-Z)

Scene*  msp_Rotate3(Scene *scn, 
		double thx, double thy, double thz, // angles to rotate
		  int cx, int cy, int cz, // center of the rotation
		    double transx, double transy, double transz); // translation after rotation


void DrawMSP(Scene *scn)
{
  Voxel v;
  int p;
  int maxval = MaximumValue3(scn);
  int z=scn->zsize/2;
  for (v.y=0; v.y < scn->ysize; v.y++)
    for (v.x=0; v.x < scn->xsize; v.x++) {
      p = v.x + scn->tby[v.y] + scn->tbz[z];
      scn->data[p]=maxval;
    }
}

void SaveTxtFile(Plane *p, char * filename)
{
  FILE *fp;
  fp = fopen(filename,"wt");
  fprintf(fp,"%.10f %.10f %.10f\n",p->po.x,p->po.y,p->po.z);
  fprintf(fp,"%.10f %.10f %.10f\n",p->normal.x,p->normal.y,p->normal.z);
  fprintf(fp,"# OBS: First line is the point P, coordinates x, y and z.\n");
  fprintf(fp,"#      Second line is the normal vector V, coordinates x, y and z.\n");
  fclose(fp);
}







int main(int argc, char **argv)
{
  Scene *in,*out;
  Plane *msp;
  int i;
  int quality=1;

  if (argc>8 || argc<3) {
    printf("\n");
    printf("Usage: %s [options] input-scn output-scn\n\n",argv[0]);
    printf("Optional parameters: \n");
    printf(" -sag / -sagittal : input is in sagittal orientation (default=unknown)\n");
    printf(" -cor / -coronal  : input is in coronal orientation\n");
    printf(" -axi / -axial    : input is in axial orientation\n");
    printf(" -vs value        : interpolate image to given isotropic voxel size (mm).\n");
    printf(" -quality <n>     : Quality = 1(best/slower), 2 (medium), 3 (worse/fast).\n");
    printf("                      (default: do not interpolate)\n");
    printf(" -draw            : draw plane on output (default: no)\n");
    //printf(" -txt             : store the MSP in a txt file (default: no)\n\n");
    printf(" -v               : verbose operation (debug mode)\n");
    //printf(" -vv              : verbose + save intermediate step images (debug mode)\n");
    //printf(" -ct              : use this for CT images\n");
    exit(1);
  }

  // Parse command line parameters
  for(i=1;i<argc;i++) {
    if (!strcmp(argv[i],"-v")) { verbose=1; continue; }
    if (!strcmp(argv[i],"-vv")) { verbose=1; savesteps=1; continue; }
    //if (!strcmp(argv[i],"-ct")) { ct=1; continue; }
    if (!strcmp(argv[i],"-draw")) { draw=1; continue; }
    if (!strcmp(argv[i],"-txt")) { txt=1; continue; }
    if (i<argc-1 && !strcmp(argv[i],"-vs")) { vs=atof(argv[++i]); interp=1; continue; }
    if (i<argc-1 && !strcmp(argv[i],"-quality")) { quality=atoi(argv[++i]); continue; }
    if (!strcmp(argv[i],"-sag") || !strcmp(argv[i],"-sagittal")) {
      orientation = 1;
      continue;
    }
    if (!strcmp(argv[i],"-axi") || !strcmp(argv[i],"-axial")) {
      orientation = 2;
      continue;
    }
    if (!strcmp(argv[i],"-cor") || !strcmp(argv[i],"-coronal")) {
      orientation = 3;
      continue;
    }
  }

  int input_ori=orientation;
    
  // Reading input scene
  in = ReadScene(argv[argc-2]);

  // Interpolate to cubic voxels (option -vs)
  if (interp==1) { 
    if (verbose) { printf("Interpolating to %fmm... \n",vs);  fflush(stdout); }
    Scene *tmp;
    tmp = LinearInterp(in,vs,vs,vs);
    DestroyScene(&in);
    in = CopyScene(tmp);
    DestroyScene(&tmp);
    if (savesteps) WriteScene(in,"debug-step0a-interpolated.scn");
  }


  int detected_plane;

  // Compute the MSP
  msp=FindSymmetryPlane(in,NULL,input_ori,quality,&detected_plane);

  Scene *scn;
  scn = in;
  // Reslicing the scene to sagittal orientation
  if (input_ori==2) { // axial
    if (verbose) { printf("Reslicing to back to axial... ");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(scn,1);
    scn  = msp_RotateAxis(tmp,2);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==3) { // coronal
    if (verbose) { printf("Reslicing back to coronal... ");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(scn,2);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }

  if (input_ori==0 && detected_plane==1) {  // axial or coronal
    if (verbose) { printf("Reslicing back to axial/coronal...");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(scn,2);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==0 && detected_plane==2) { // laid-down coronal
    if (verbose) { printf("Reslicing back to rotated-coronal...");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(scn,1);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  

  // Rotate the volume according to MSP found
  if (verbose) { printf("Rotating... ");  fflush(stdout); }
  out = msp_RotateToMSP(scn,msp);
  if (verbose) { printf("done.\n");  fflush(stdout); }

  // Draw the msp (option -draw)
  if (draw) DrawMSP(out);

  // Reslicing the output
  if (input_ori==2) { // axial
    if (verbose) { printf("Reslicing back to axial... ");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out  = msp_RotateAxis(tmp,-1);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==3) { // coronal
    if (verbose) { printf("Reslicing back to coronal... ");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }

  if (input_ori==0 && detected_plane==1) {  // axial or coronal
    if (verbose) { printf("Reslicing back to axial/coronal...");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==0 && detected_plane==2) { // laid-down coronal
    if (verbose) { printf("Reslicing back to rotated-coronal...");  fflush(stdout); }
    Scene *tmp;
    tmp = msp_RotateAxis(out,-1);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  



  // Writing final scene
  WriteScene(out,argv[argc-1]);
  printf("Written %s \n",argv[argc-1]);

  // Store the plane in txt file
  if (txt) {
    char basename[200],filename[200];
    strcpy(basename,argv[argc-1]);
    basename[strlen(basename)-4]='\0';
    sprintf(filename,"%s.txt",basename);
    SaveTxtFile(msp,filename);
    printf("Written %s\n",filename);
  }

  if (txt) {
    char basename[200],filename[200];
    strcpy(basename,argv[argc-1]);
    basename[strlen(basename)-4]='\0';
    sprintf(filename,"%s_time.txt",basename);
    FILE *fp;
    fp = fopen(filename,"wt");
    fprintf(fp,"%f\n",time_preproc+time_search);
    fprintf(fp,"%f\n",time_preproc);
    fprintf(fp,"%f\n",time_search);
    fprintf(fp,"Obs: 1st number is total time (use just this)\n");
    fprintf(fp,"Obs: 2nd number is pre-processing time (not working)\n");
    fprintf(fp,"Obs: 3rd number is search time (not working)\n");
    fclose(fp);
  }
    
  return 0;
}



