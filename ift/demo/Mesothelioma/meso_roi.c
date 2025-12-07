#include "ift.h"

iftImage *iftMesoRegionOfInterest(iftImage *img, iftImage *mask, float max_dist) {
  iftAdjRel *A         = iftSpheric(1.75);
  iftImage  *dist      = iftCreateImageFromImage(img);
  iftImage  *root      = iftCreateImageFromImage(img);
  iftImage  *roi       = iftCreateImageFromImage(img);
  iftGQueue *Q         = iftCreateGQueue(IFT_QSIZE, img->n, dist->val);
  
  max_dist             = max_dist*max_dist;
  
  for (int p = 0; p < img->n; p++) {
    dist->val[p] = IFT_INFINITY_INT;
    
    if (mask->val[p] != 0) {
      dist->val[p] = 0;
      root->val[p] = p;
      iftInsertGQueue(&Q, p);
    }
  }
  
  while(!iftEmptyGQueue(Q)) {
    int p = iftRemoveGQueue(Q);

    if (dist->val[p] < max_dist) {
      
      iftVoxel u = iftGetVoxelCoord(img, p);
      iftVoxel r = iftGetVoxelCoord(img, root->val[p]);
    
      for (int i = 1; i < A->n; i++) {
	iftVoxel v = iftGetAdjacentVoxel(A, u, i);
	
	if (iftValidVoxel(img, v)) {
	  int q = iftGetVoxelIndex(img, v);
	  
	  if (dist->val[q] > dist->val[p]) {
	    
	    float tmp = (v.x-r.x)*(v.x-r.x) + (v.y-r.y)*(v.y-r.y) + (v.z-r.z)*(v.z-r.z);
	    
	    if (tmp < dist->val[q]) {
	      if (dist->val[q] != IFT_INFINITY_INT)
		iftRemoveGQueueElem(Q, q);
	      dist->val[q]      = tmp;
	      root->val[q]      = root->val[p];
	      //if (img->val[q] > 1000) 
	      roi->val[q]     = 1;	      
	      iftInsertGQueue(&Q, q);
	    }
	  }
	}
      }
    } 
  }

  for (int p=0; p < roi->n; p++){
    if (mask->val[p]!=0)
      roi->val[p] = 0;
  }
  
  iftDestroyGQueue(&Q);

  iftDestroyImage(&root);
  iftDestroyImage(&dist);
  iftDestroyAdjRel(&A);

  return(roi);
}


int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc!=5)
      iftError("Usage: extract_mesothelioma <...>\n"
	       "[1] reference image .nii.gz \n"
	       "[2] fluid mask .nii.gz \n"
	       "[3] Dilation radius \n"
	       "[4] output roi mask .nii.gz\n",
	       "main");

    tstart = iftTic();

    iftImage  *img      = iftReadImageByExt(argv[1]); 
    iftImage  *mask     = iftReadImageByExt(argv[2]);
    float      max_dist = atof(argv[3]);
    iftImage  *meso     = iftMesoRegionOfInterest(img, mask, max_dist);
    iftWriteImageByExt(meso,argv[4]);
    iftDestroyImage(&img);
    iftDestroyImage(&mask);
    iftDestroyImage(&meso);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return (0);
}

