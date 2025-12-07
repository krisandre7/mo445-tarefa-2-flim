#include "ift.h"

iftImage *DynamicOptPathCostToMarker(iftMImage *feat, iftImage *marker)
{
  iftImage   *pathval  = NULL, *root = NULL;
  iftMImage  *treeinfo = NULL;
  int diag = iftRound(sqrt(feat->xsize*feat->xsize +
			   feat->ysize*feat->ysize + feat->zsize*feat->zsize)); 
  int        *nnodes   = NULL;
  int Imax             = iftRound(sqrtf(feat->m)*iftMMaximumValue(feat,0)+diag);
  iftAdjRel *A         = NULL;

  if (iftIs3DImage(marker))
    A = iftSpheric(1.0);
  else
    A = iftCircular(1.0);
    
  for (int b=1; b < feat->m; b++)
    Imax = iftMax(Imax, iftMMaximumValue(feat,b) + diag);
  
  iftGQueue  *Q        = NULL;
  int         i, p, q, r, tmp;
  iftVoxel    u, v, w;

  /* Initialization */
  
  pathval    = iftCreateImage(feat->xsize, feat->ysize, feat->zsize);
  root       = iftCreateImage(feat->xsize, feat->ysize, feat->zsize);
  treeinfo   = iftCreateMImage(feat->xsize, feat->ysize, feat->zsize, feat->m);
  nnodes     = iftAllocIntArray(feat->n);
  Q          = iftCreateGQueue(Imax+1, feat->n, pathval->val);

  /* Initialize costs */
    
  for (p = 0; p < feat->n; p++)
  {
    pathval->val[p] = IFT_INFINITY_INT;
    if (marker->val[p] != 0){
      root->val[p]    = p;
      pathval->val[p] = 0;  
      iftInsertGQueue(&Q, p);
    }
  }

  /* Image Foresting Transform */

  while (!iftEmptyGQueue(Q))
  {
    p = iftRemoveGQueue(Q);
    r = root->val[p];
    w = iftMGetVoxelCoord(feat,r);
    
    for (int b=0; b < treeinfo->m; b++)
      treeinfo->val[r][b] += feat->val[p][b];
    nnodes[r]  += 1;
    u           = iftMGetVoxelCoord(feat, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(feat, v))
      {
        q = iftMGetVoxelIndex(feat, v);

	if (Q->L.elem[q].color != IFT_BLACK) {

	  float dist = 0.0;
	  for (int b=0; b < treeinfo->m; b++)
	    dist += powf((feat->val[q][b]-treeinfo->val[r][b]/nnodes[r]),2.0);

	  
	  int sdist = iftRound(0.2*sqrt((v.x-w.x)*(v.x-w.x) + (v.y-w.y)*(v.y-w.y)));
	  
          tmp = pathval->val[p] + iftRound(0.8*sqrt(dist)) + sdist ;

          if (tmp < pathval->val[q])  {
	    if (Q->L.elem[q].color == IFT_GRAY)
	      iftRemoveGQueueElem(Q,q);
            root->val[q]     = root->val[p];
            pathval->val[q]  = tmp;
            iftInsertGQueue(&Q, q);
          }
        }
      }
    }
  }

  iftDestroyGQueue(&Q);
  iftDestroyImage(&root);
  iftFree(nnodes);
  iftDestroyMImage(&treeinfo);
  iftDestroyAdjRel(&A);
  
  return (pathval);
}

int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;

  
  /* Example: iftSegmIris <image.*> <iris_border.*> 10 4000 */ 
  
  if (argc!=5)
    iftError("Usage: iftSegmIris <image.*> <iris_borders.*> <pupil_thres> <iris_cost_thres>","main");
  
  iftImage *img = iftReadImageByExt(argv[1]);    
  iftAdjRel *C  = iftCircular(5.0);
  iftImage *iris = iftMedianFilter(img, C);
  iftDestroyImage(&img);
  iftDestroyAdjRel(&C);
  
  t1     = iftTic();

  /* Segment the pupil */
  
  iftImage *bin    = iftThreshold(iris,0,atoi(argv[3]),255);
  iftImage *close  = iftCloseBin(bin,15.0);
  iftImage *pupil  = iftOpenBin(close,30.0);
  iftDestroyImage(&close);
  iftDestroyImage(&bin);
  
  /* Select a marker around the pupil */
  
  iftSet   *S      = NULL;
  iftImage *marker = iftDilateBin(pupil,&S,15.0);
  iftDestroySet(&S);
  bin              = iftSub(marker,pupil);
  iftDestroyImage(&marker);
  marker           = iftErodeBin(bin,&S,7.0);
  iftDestroySet(&S);
  iftDestroyImage(&bin);
  
  for (int p = 0; p < pupil->n; p++)
    if (marker->val[p] == 0)
      marker->val[p] = pupil->val[p];
  
  /* Segment the iris by a dynamic optimum-path cost to marker */
  
  iftMImage *miris = iftImageToMImage(iris,GRAY_CSPACE);
  iftAdjRel *A     = iftCircular(3.0);
  iftMImage *feat  = iftExtendMImageByAdjacency(miris,A);
  iftDestroyMImage(&miris);
  iftDestroyAdjRel(&A);
  iftImage *cost   = DynamicOptPathCostToMarker(feat,marker);
  bin              = iftThreshold(cost,0,atoi(argv[4]),255);

  for (int p=0; p < bin->n; p++){
    if (pupil->val[p] != 0)
      bin->val[p] = 0;
  }

  iftAdjRel *B     = iftCircular(1.0);
  iftColor RGB, YCbCr;
  RGB.val[0] = 255;
  RGB.val[0] = 0;
  RGB.val[0] = 0;
  YCbCr = iftRGBtoYCbCr(RGB,255);
  iftDrawBorders(iris, bin, B, YCbCr, B);
 
  iftDestroyMImage(&feat);
  iftDestroyImage(&marker);
  iftDestroyAdjRel(&B);
    
  t2     = iftToc();
  fprintf(stdout,"Iris segmented in %f ms\n",iftCompTime(t1,t2));

  iftWriteImageByExt(iris,argv[2]);
  iftDestroyImage(&iris);
  iftDestroyImage(&pupil);
  iftDestroyImage(&cost);
  iftDestroyImage(&bin);
  
  /* ---------------------------------------------------------- */


  return(0);
}

