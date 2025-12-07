#include <ift.h>

Scene *FeatureGradient3(Scene *scn, int nfeats, int maxval)
{
  float   dist,gx,gy,gz;
  int     i,j,p,q,s,n=scn->xsize*scn->ysize*scn->zsize,Imax;
  Voxel   u,v;
  AdjRel3 *A8=Spheric(1.5),*A=NULL;
  float  *mg=AllocFloatArray(A8->n);
  Scene  *grad=CreateScene(scn->xsize, scn->ysize, scn->zsize);
  float  **feat=(float **)calloc(nfeats,sizeof(float *));
  Scene *scn1,*scn2;

  /* Compute multiscale features with nfeats scales */

  for (i=0; i < nfeats; i++) 
    feat[i]=AllocFloatArray(n);

  Imax  = MaximumValue3(scn);
  scn1  = CopyScene(scn);

  for (s=1; s <= nfeats; s=s+1) {
    A  = Spheric(s);
    scn2 = AsfOCRec3(scn1,A);
    for (i=0; i < n; i++) {
      feat[s-1][i] = (float)scn2->data[i]/(float)Imax;
    }
    DestroyScene(&scn2);
    DestroyAdjRel3(&A);
  }
  DestroyScene(&scn1);

  /* Compute Gradient */

  for (i=0; i < A8->n; i++)
    mg[i]=sqrt(A8->dx[i]*A8->dx[i]+A8->dy[i]*A8->dy[i]+A8->dz[i]*A8->dz[i]);  // sqrt??? ou raiz cubica???

  for (u.z=0; u.z < scn->zsize; u.z++)
	  for (u.y=0; u.y < scn->ysize; u.y++)
  	  for (u.x=0; u.x < scn->xsize; u.x++) {
  	    p = u.x + scn->tby[u.y] + scn->tbz[u.z];
  	    gx = gy = gz = 0.0;
  	    for (i=1; i < A8->n; i++) {
					v.z = u.z + A8->dz[i];
					v.x = u.x + A8->dx[i];
					v.y = u.y + A8->dy[i];
					if (ValidVoxel(scn,v.x,v.y,v.z)){
	  				q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	  				dist = 0;
	  				for (j=0; j < nfeats; j++) 
	    				dist += feat[j][q]-feat[j][p];
            	gz  += dist*A8->dz[i]/mg[i];
            	gx  += dist*A8->dx[i]/mg[i];
            	gy  += dist*A8->dy[i]/mg[i];
          }
      }
      //      gx = gx/nfeats; gy = gy/nfeats;
      grad->data[p]=(int)(maxval*sqrt(gx*gx + gy*gy + gz*gz));
    }

  free(mg);
  DestroyAdjRel3(&A8);
  for (i=0; i < nfeats; i++)
    free(feat[i]);
  free(feat);
  return(grad);
}

float Distance(float *f1, float *f2, int n)
{
  int i;
  float dist;

  dist = 0;
  for (i=0; i < n; i++)
    dist += (f2[i]-f1[i]);//(f2[i]-f1[i])/2.0;
  //dist /= n;

  return(dist);//exp(-(dist-0.5)*(dist-0.5)/2.0));

}

Scene *TextGradient(Scene *scn)
{
  float   dist,gx,gy,gz;
  int     i,p,q,n=scn->xsize*scn->ysize*scn->zsize;//,Imax=MaximumValue3(scn);
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
            dist = Distance(feat[p].f,feat[q].f,A->n);
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

int main(int argc, char *argv[]) 
{
  timer    *t1=NULL,*t2=NULL;
  Scene    *scn=NULL,*grad=NULL,*handicap=NULL;
  Scene    *label=NULL;
//  CImage   *cimg1=NULL,*cimg2=NULL;
  char      ext[10],*pos;
  AdjRel3   *A=NULL;

  /*--------------------------------------------------------*/
#ifndef _WIN32
  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/

  if (argc!=4)
    Error("Usage must be: watergray <image> <di> <factor>","main");

  pos = strrchr(argv[1],'.') + 1;
  sscanf(pos,"%s",ext);

  t1 = Tic();

  if (strcmp(ext,"scn")==0){
    scn   = ReadScene(argv[1]);
    grad  = TextGradient(scn);
  }else{
    printf("Invalid image format: %s\n",ext);
    exit(-1);
  }
  
  A = Spheric(atof(argv[2])); 
  WriteScene(grad,"watergray-a.scn");
  //handicap = Add3(grad,(int)(atof(argv[3])*MaximumValue3(grad)));
  handicap = CTVolumeClose3(grad,(int)atof(argv[3]));
  WriteScene(handicap,"watergray-b.scn");
  label = WaterGray3(grad,handicap,A);
  
  DestroyAdjRel3(&A);
  // cimg2 = DrawLabeledRegions(img,label);    //ver como criar scena colorida
					      
  // WriteCImage(cimg2,"result.ppm");    
  Scene *scn2=DrawBorder3(scn,label,MaximumValue3(scn)+1);
  WriteScene(scn2,"result.scn");
  DestroyScene(&grad);  
  DestroyScene(&handicap);  

  DestroyScene(&scn);  
  DestroyScene(&scn2);  
  DestroyScene(&label);
 // DestroyCImage(&cimg2);  
  t2 = Toc();    
  
  fprintf(stdout,"watershed in %f ms\n",CTime(t1,t2));

  /* ---------------------------------------------------------- */
#ifndef _WIN32
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif

  return(0);
}




