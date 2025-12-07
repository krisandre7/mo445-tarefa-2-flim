#include "ift.h"

#define CMAX     4095

typedef struct iftDynTrees_ {
  iftMImage *img;          /* original image in some color space */
  iftAdjRel *A;            /* adjacency relation */
  iftImage  *label;        /* label map */
  iftImage  *root;         /* root map */
  iftMImage *cumfeat;      /* cumulative feature vector map */
  int       *treesize;     /* tree size map */
  iftImage  *cost;         /* path cost map */
  iftImage  *pred;         /* predecessor map */
  iftGQueue *Q;            /* priority queue with integer costs */
  float      maxfeatdist;  /* maximum feature distance */
} iftDynTrees;

iftDynTrees *iftCreateDynTrees(iftMImage *img, iftAdjRel *A);
void         iftDestroyDynTrees(iftDynTrees **dt);
void         iftExecDiffDynTrees(iftDynTrees *dt, iftLabeledSet **S, iftSet **M);
iftSet      *iftDiffDynTreeRemoval(iftDynTrees *dt, iftSet **M);
void         iftDiffDynSubtreeUpdate(iftDynTrees *dt, int q); 
  
iftDynTrees *iftCreateDynTrees(iftMImage *img, iftAdjRel *A)
{
  iftDynTrees *dt = (iftDynTrees *)calloc(1,sizeof(iftDynTrees));

  dt->img      = img; dt->A = A;
  dt->label    = iftCreateImage(img->xsize,img->ysize,img->zsize); 
  dt->root     = iftCreateImage(img->xsize,img->ysize,img->zsize); 
  dt->cost     = iftCreateImage(img->xsize,img->ysize,img->zsize); 
  dt->pred     = iftCreateImage(img->xsize,img->ysize,img->zsize); 
  dt->cumfeat  = iftCreateMImage(img->xsize,img->ysize,img->zsize,img->m);
  dt->treesize = iftAllocIntArray(img->n);
  dt->Q        = iftCreateGQueue(CMAX+1, img->n, dt->cost->val);

  /* compute the maximum feature distance */

  dt->maxfeatdist = 0.0;
  for (int p=0; p < img->n; p++){
    iftVoxel u = iftMGetVoxelCoord(img,p);
    for (int i=1; i < A->n; i++) {
      iftVoxel v = iftGetAdjacentVoxel(A,u,i);
      if (iftMValidVoxel(img,v)){
	int q = iftMGetVoxelIndex(img,v);
	float dist = 0.0;
	for (int b=0; b < img->m; b++){
	  dist += (img->val[q][b]-img->val[p][b])*
	    (img->val[q][b]-img->val[p][b]);
	}
	if (dist > dt->maxfeatdist)
	  dt->maxfeatdist = dist;
      }
    }
  }

  /* initialize maps */

  for (int p=0; p < img->n; p++){
    dt->cost->val[p] = IFT_INFINITY_INT;
    dt->pred->val[p] = IFT_NIL;
  }

  return(dt);
}

void iftDestroyDynTrees(iftDynTrees **dt)
{
  iftDynTrees *aux = *dt;

  if (aux != NULL) {
    iftDestroyGQueue(&aux->Q);
    iftDestroyImage(&aux->cost);
    iftDestroyImage(&aux->pred);
    iftDestroyImage(&aux->label);
    iftDestroyImage(&aux->root);
    iftDestroyMImage(&aux->cumfeat);
    iftFree(aux->treesize);
    iftFree(aux);
    *dt = NULL;
  }
}

void iftExecDiffDynTrees(iftDynTrees *dt, iftLabeledSet **S, iftSet **M)
{
  iftSet *F = iftDiffDynTreeRemoval(dt, M); /* remove trees and return their
				       frontier set, which is empty
				       when the marking set M is
				       empty */

  iftMImage *img      = dt->img;
  iftAdjRel *A        = dt->A; 
  iftImage  *cost     = dt->cost;
  iftImage  *label    = dt->label;
  iftImage  *pred     = dt->pred;
  iftImage  *root     = dt->root;
  iftGQueue *Q        = dt->Q;
  iftMImage *cumfeat  = dt->cumfeat;
  int       *treesize = dt->treesize;
  
  /* Remove seeds from F, if it is the case since seeds have higher
     priority */

  /* if (F != NULL) { /\* non-empty set *\/ */
  /*   iftLabeledSet *seed = *S; */
  /*   while (seed != NULL){ */
  /*     int p = seed->elem; */
  /*     iftRemoveSetElem(&F,p); */
  /*     seed  = seed->next; */
  /*   } */
  /* } */

  /* Reinitialize maps for seed voxels and insert seeds and frontier
     in the queue */

  while (*S != NULL) {
    int lambda;
    int p         = iftRemoveLabeledSet(S,&lambda);
    cost->val[p]  = 0;
    label->val[p] = lambda;
    pred->val[p]  = IFT_NIL;
    root->val[p]  = p;
    for (int b=0; b < img->m; b++)
      cumfeat->val[p][b] = 0;
    treesize[p] = 0;
    iftUnionSetElem(&F,p);
  }
  
  while (F != NULL){
    int p = iftRemoveSet(&F);
    iftInsertGQueue(&Q,p);
  }

  /* Execute the Image Foresting Transform of DDT */

  while (!iftEmptyGQueue(Q)){
    int p      = iftRemoveGQueue(Q);
    iftVoxel u = iftMGetVoxelCoord(img,p);
    
    /* set / update dynamic tree of p */

    int r = root->val[p];
    /* if (pred->val[p] == IFT_NIL) { /\* p is a root voxel *\/ */
    /*   for (int b=0; b < img->m; b++) */
    /* 	cumfeat->val[r][b] = img->val[r][b]; */
    /*   treesize[r] = 1; */
    /* } else { */
    for (int b=0; b < img->m; b++)
      cumfeat->val[r][b] += img->val[r][b];
    treesize[r] += 1;
    /* } */
    
    /* visit the adjacent voxels for possible path extension */
    
    for (int i=1; i < A->n; i++) {
      iftVoxel v = iftGetAdjacentVoxel(A,u,i);

      if (iftMValidVoxel(img,v)){
	int q   = iftMGetVoxelIndex(img,v);
	if (Q->L.elem[q].color != IFT_BLACK){
	  float arcw = 0;
	  for (int b=0; b < img->m; b++){
	    arcw += (img->val[q][b] - cumfeat->val[r][b]/treesize[r])*
	      (img->val[q][b] - cumfeat->val[r][b]/treesize[r]);
	  }
	  arcw = CMAX * (arcw / dt->maxfeatdist); 
	  int tmp = iftMax(cost->val[p], iftMin(arcw,CMAX));
	  if (tmp < cost->val[q])  {
	    if (Q->L.elem[q].color == IFT_GRAY)
	      iftRemoveGQueueElem(Q, q);
	    cost->val[q]  = tmp; 
	    label->val[q] = label->val[p];
	    root->val[q]  = root->val[p];
	    pred->val[q]  = p;
	    iftInsertGQueue(&Q,q);
	  } else {
	    if ( (label->val[q] != label->val[p]) &&
		 (pred->val[q] == p) ) {
	      iftDiffDynSubtreeUpdate(dt,q);
	    }
	  }
	}
      }
    }
  }

  iftResetGQueue(Q);
}

iftSet *iftDiffDynTreeRemoval(iftDynTrees *dt, iftSet **M)
{
  iftSet *F = NULL, *R = NULL, *T = NULL, *Q = NULL;
  iftAdjRel *B = NULL;

  if (iftIs3DMImage(dt->img))
    B = iftSpheric(1.74);
  else
    B = iftCircular(1.42);
  
  /* Find the roots of the trees that contain elements in the marking
     set. If the root has not been inserted yet in a root set R, do it
     for its entire marker and reset their cost and predecessor
     information. Set R must contain the roots of all trees marked for
     removal. Set T starts being equal to R and next it stores
     elements from the trees rooted in R. */
  
  while(*M != NULL) {
    int p = iftRemoveSet(M);
    int r = dt->root->val[p];
    /* insert in R and T all roots in the marker of r for tree
       removal */ 
    iftInsertSet(&Q,r);
    while (Q != NULL) {
      int r = iftRemoveSet(&Q);
      if (dt->cost->val[r] != IFT_INFINITY_INT){ /* r is not in R and T yet */
	dt->cost->val[r]=IFT_INFINITY_INT; 
	iftInsertSet(&R,r); iftInsertSet(&T,r);
      }
      iftVoxel u = iftMGetVoxelCoord(dt->img,r);
      for (int i=1; i < B->n; i++) {
      	iftVoxel v = iftGetAdjacentVoxel(B,u,i);
      	if (iftMValidVoxel(dt->img,v)){
      	  int s = iftMGetVoxelIndex(dt->img, v);
	  /* s belongs to the same marker of r and it has not been
	     inserted in R and T yet. */
      	  if ((dt->root->val[s]==s)&&   
	      (dt->cost->val[s] != IFT_INFINITY_INT))
      	    iftInsertSet(&Q,s);
      	}
      }
    }
  }
  
  /* Visit all nodes in each tree with root in R, while removing the
     tree. It also identifies the frontier voxels of trees that have
     not been marked for removal. */

  while (T != NULL) {
    int      p = iftRemoveSet(&T);
    iftVoxel u = iftMGetVoxelCoord(dt->img,p);
    for (int i=1; i < dt->A->n; i++) {
      iftVoxel v = iftGetAdjacentVoxel(dt->A,u,i);
      if (iftMValidVoxel(dt->img,v)){
	int q = iftMGetVoxelIndex(dt->img, v);
	/* If q belongs to a tree for removal, then reset its cost and
	   predecessor information, inserting it in T to pursue tree
	   removal. */
	if (dt->pred->val[q]==p){ 
	  dt->cost->val[q]=IFT_INFINITY_INT;
	  dt->pred->val[q]=IFT_NIL;
	  iftInsertSet(&T,q);
	} else {
	  /* If q belongs to a tree that was not marked for removal,
	     then q is a frontier voxel. */
	  if (!iftSetHasElement(R, dt->root->val[q])){
	    iftUnionSetElem(&F,q);
	  }
	}
      }
    }
  }
  
  return(F);
}

void iftDiffDynSubtreeUpdate(iftDynTrees *dt, int q)
{
  iftSet *T = NULL; /* Set to visit the nodes of the subtree rooted in
		       q */
  /* If the subtree contains a single node, its root q, then update
     its cost, label and root information. */
  /* if (dt->Q->L.elem[q].color == IFT_GRAY){  */
  /*   iftRemoveGQueueElem(dt->Q, q); */
  /*   int p      = dt->pred->val[q]; */
  /*   int r      = dt->root->val[p]; */
  /*   float arcw = 0; */
  /*   for (int b=0; b < dt->img->m; b++){ */
  /*     arcw += (dt->img->val[q][b] - dt->cumfeat->val[r][b]/dt->treesize[r])* */
  /* 	      (dt->img->val[q][b] - dt->cumfeat->val[r][b]/dt->treesize[r]); */
  /*   } */
  /*   arcw              = CMAX * (arcw / dt->maxfeatdist);  */
  /*   dt->cost->val[q]  = iftMax(dt->cost->val[p], iftMin(arcw,CMAX)); */
  /*   dt->label->val[q] = dt->label->val[p]; */
  /*   dt->root->val[q]  = dt->root->val[p]; */
  /* } else { */
  /* the subtree contains one or more elements from a previous execution */   
  iftInsertSet(&T,q);
  while (T != NULL) {
    int q = iftRemoveSet(&T);
    
    /* update properties of the previous tree of q */ 
    /* int r = dt->root->val[q]; */
    /* for (int b=0; b < dt->img->m; b++) */
    /* 	dt->cumfeat->val[r][b] -= dt->img->val[q][b]; */
    /* dt->treesize[r] -= 1; */
    
    /* update properties of the new tree of q */
    int p      = dt->pred->val[q];
    int r      = dt->root->val[p];
    for (int b=0; b < dt->img->m; b++)
    	dt->cumfeat->val[r][b] += dt->img->val[q][b];
    dt->treesize[r] += 1;
    float arcw = 0;
    for (int b=0; b < dt->img->m; b++){
      arcw += (dt->img->val[q][b] - dt->cumfeat->val[r][b]/dt->treesize[r])*
	(dt->img->val[q][b] - dt->cumfeat->val[r][b]/dt->treesize[r]);
    }
    arcw              = CMAX * (arcw / dt->maxfeatdist); 
    dt->cost->val[q]  = iftMax(dt->cost->val[p], iftMin(arcw,CMAX));
    dt->label->val[q] = dt->label->val[p];
    dt->root->val[q]  = dt->root->val[p];
    
    /* Insert the childree of q in T to pursue the tree update
       process */
    p          = q;
    iftVoxel u = iftMGetVoxelCoord(dt->img,p);
    for (int i=1; i < dt->A->n; i++) {
      iftVoxel v = iftGetAdjacentVoxel(dt->A,u,i);
      if (iftMValidVoxel(dt->img,v)){
	int q = iftMGetVoxelIndex(dt->img, v);
	if (dt->pred->val[q]==p){ 
	  iftInsertSet(&T,q);
	}
      }
    }
  }
/* } */
}

/* Extracts the basename of an image file */

char *Basename(char *path)
{
  char *basename     = iftBasename(path);
  iftSList *slist    = iftSplitString(basename,"/");
  strcpy(basename,slist->tail->elem);
  iftDestroySList(&slist);
  return(basename);
}


int main(int argc, char *argv[]) {
    timer *tstart;
    
    if (argc!=4)
      iftError("Usage: diffdynamictrees <...>\n"
	       "[1] input image .png \n"
	       "[2] input sequence of seed addition/removal operations .txt \n"
	       "[3] output folder\n",
	       "main");

    tstart = iftTic();

    iftImage  *img      = iftReadImageByExt(argv[1]); 
    iftMImage *mimg     = NULL;
    iftAdjRel *A        = iftCircular(1.0);

    mimg = iftImageToMImage(img, LABNorm2_CSPACE); 
    
    iftMakeDir(argv[3]);

    iftDynTrees   *dt = iftCreateDynTrees(mimg, A);
    iftLabeledSet *S  = NULL;
    iftSet        *M  = NULL;
    iftCSV *seq_operations = iftReadCSV(argv[2],';');
    for (int row=0; row < seq_operations->nrows; row++){
      S=iftReadSeeds(img,seq_operations->data[row][1]);
      if (strcmp(seq_operations->data[row][0],"r")==0){
	while(S != NULL) {
	  int trash;
	  int p = iftRemoveLabeledSet(&S, &trash);
	  iftInsertSet(&M,p);
	}
	iftExecDiffDynTrees(dt,&S,&M);
      }	else {
	iftExecDiffDynTrees(dt,&S,&M);
      }
      iftImage *label = iftCopyImage(dt->label);
      for (int p=0; p < label->n; p++)
	label->val[p] = 127*label->val[p];
      iftColor RGB, YCbCr;
      RGB.val[0] = 255;
      RGB.val[1] = 0;
      RGB.val[2] = 255;
      YCbCr      = iftRGBtoYCbCr(RGB, 255);
      iftImage *cimg     = iftCopyImage(img);
      char     *basename = Basename(seq_operations->data[row][1]);
      char     filename[200];
      iftDrawBorders(cimg,label,A,YCbCr,A);
      sprintf(filename,"%s/%s_label.png",argv[3],basename);
      iftWriteImageByExt(label,filename);
      sprintf(filename,"%s/%s_segm.png",argv[3],basename);
      iftWriteImageByExt(cimg,filename);
      iftDestroyImage(&cimg);
      iftDestroyImage(&label);
      iftFree(basename);      
    }
    
    iftDestroyDynTrees(&dt);
    iftDestroyCSV(&seq_operations);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyAdjRel(&A);

    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
    return (0);
}

