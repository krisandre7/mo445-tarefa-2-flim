#include "ift.h"

void UpdateTreeAttributes(iftMImage *M, int *N, iftMImage *mimg, iftImage *root, int p)
{
  int r = root->val[p];
  for (int b=0; b < mimg->m; b++){
     M->val[r][b] += mimg->val[p][b];
   }
   N[r] += 1;
}

float TreeAttributeDist(iftMImage *mimg,int q,int r,iftMImage *M, int *N)
{
    float dist = 0.0;

    for (int b=0; b < mimg->m; b++){
        dist += powf(mimg->val[q][b]-M->val[r][b]/N[r],2.0);
    }
    return(sqrtf(dist));
}


void MyDynamicTrees(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *S, \
                    iftImage **cost, iftImage **label)
{
 iftMImage *M    = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);
 int *N          = iftAllocIntArray(mimg->n);
 iftImage *root  = iftCreateImage(mimg->xsize,mimg->ysize,mimg->zsize);
          *cost  = iftCreateImage(mimg->xsize,mimg->ysize,mimg->zsize);
          *label = iftCreateImage(mimg->xsize,mimg->ysize,mimg->zsize);
 iftGQueue *Q    = iftCreateGQueue(255*sqrtf(mimg->m), mimg->n, (*cost)->val);
 int        tmp  = 0;

 for (int p = 0; p < (*cost)->n; p++){
    (*cost)->val[p]=IFT_INFINITY_INT;
 }
 iftLabeledSet *seed=S;
 while (seed != NULL){
    int p = seed->elem;
    int l = seed->label;
    (*cost)->val[p]  = 0;
    (*label)->val[p] = l;
    root->val[p]     = p;
    iftInsertGQueue(&Q,p);
    seed = seed->next;
 }
 while (!iftEmptyGQueue(Q)){
    int p      = iftRemoveGQueue(Q);
    iftVoxel u = iftMGetVoxelCoord(mimg, p);
    int r      = root->val[p];
    UpdateTreeAttributes(M,N,mimg,root,p);
    for (int i=1; i < A->n; i++){
       iftVoxel v = iftGetAdjacentVoxel(A,u,i);
       if (iftMValidVoxel(mimg,v)){
         int q = iftMGetVoxelIndex(mimg,v);
         if ((*cost)->val[q] > (*cost)->val[p]){
            float arcw = TreeAttributeDist(mimg,q,r,M,N);
            tmp  = iftMax((*cost)->val[p], (int)(arcw));
            if (tmp < (*cost)->val[q]){
              if ((*cost)->val[q] != IFT_INFINITY_INT){
                 iftRemoveGQueueElem(Q, q);
              }
              (*cost)->val[q] = tmp; (*label)->val[q]=(*label)->val[p];
              root->val[q] = root->val[p]; iftInsertGQueue(&Q,q);
            }
         }
       }
    }
 }

 iftDestroyMImage(&M);
 iftFree(N);
 iftDestroyImage(&root);
 iftDestroyGQueue(&Q);
}

int main(int argc, char *argv[])
{
  if (argc!=4)
    iftError("Usage: iftDynamicTrees <image.*> <seeds.txt> <result.*>","main");

  iftImage  *img  = iftReadImageByExt(argv[1]);
  iftAdjRel *A    = NULL;
  iftMImage *mimg = iftImageToMImage(img, LAB_CSPACE);

  if (iftIs3DImage(img))
    A = iftSpheric(1.0);
  else
    A = iftCircular(1.0);

 iftLabeledSet *S = iftReadSeeds(img, argv[2]);
 iftImage *cost=NULL, *label=NULL;
 MyDynamicTrees(mimg, A, S, &cost, &label);
 if (iftNumberOfLabels(S)>=2){
   for (int p=0; p < label->n; p++)
      label->val[p] = label->val[p]*255;
   iftWriteImageByExt(label,argv[3]);
 }else{
    if (iftMaximumValue(label)>=1){
       iftImage *aux = iftComplement(cost);
       iftDestroyImage(&cost);
       cost = aux;
    }
    iftImage *salie = iftNormalize(cost, 0, 255);
    iftWriteImageByExt(salie,argv[3]);
    iftDestroyImage(&salie);
 }

 iftDestroyImage(&cost);
 iftDestroyImage(&label);
 iftDestroyImage(&img);
 iftDestroyMImage(&mimg);
 iftDestroyAdjRel(&A);
 iftDestroyLabeledSet(&S);

  return(0);
}

