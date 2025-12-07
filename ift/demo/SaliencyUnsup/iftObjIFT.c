#include "ift.h"


void usage();

void iftWriteOverlay(iftImage* orig, iftImage *label, const char *filename);
iftImage *iftObjectSaliencyIFT(iftImage *img, iftImage *objsm, iftImage *old_seeds, int num_seeds, float obj_perc, float gamma, iftImage **new_seeds);

//-------------------------------------------------------------

int main(int argc, const char *argv[]) 
{
  int num_seeds;
  float obj_perc, gamma;
  iftImage *img, *objsm, *old_seeds, *label_img, *new_seeds;

  if(argc < 7 || argc > 8) usage();

  img = iftReadImageByExt(argv[1]);
  objsm = iftReadImageByExt(argv[2]);
  num_seeds = atoi(argv[3]);
  obj_perc = atof(argv[4]);
  gamma = atof(argv[5]);

  new_seeds = NULL;

  if(argc == 8) old_seeds = iftReadImageByExt(argv[7]);
  else old_seeds = NULL;

  label_img = iftObjectSaliencyIFT(img, objsm, old_seeds, num_seeds, obj_perc, gamma, &new_seeds);

  iftWriteOverlay(img, label_img, argv[6]);
  iftWriteImageByExt(new_seeds,"new_seeds_%s", argv[6]);
  
  iftDestroyImage(&img);
  iftDestroyImage(&objsm);
  iftDestroyImage(&label_img);
  iftDestroyImage(&new_seeds);
  if(old_seeds != NULL) iftDestroyImage(&old_seeds);

  return (0);
}

//-------------------------------------------------------------

void usage()
{
  printf("Usage: iftObjIFT [1] [2] [3] [4] [5] [6] {7}\n");
  printf("---------------------------------------------------------\n");
  printf("[1] - Input image\n");
  printf("[2] - Gray-level object saliency map\n");
  printf("[3] - Number of superpixels\n");
  printf("[4] - Object percentage\n");
  printf("[5] - Gamma value\n");
  printf("[6] - Output image\n");
  printf("{7} - OPTIONAL: Old seed image\n\n");

  iftError("Too many/few args!", "main");
}

void iftWriteOverlay
(iftImage* orig, iftImage *label, const char *filename)
{
  int normvalue;
  iftImage *overlay, *copy;
  iftAdjRel *A;
  iftColor RGB, YCbCr;

  normvalue = iftNormalizationValue(iftMaximumValue(orig)); 

  copy = iftCopyImage(orig);
  A = iftCircular(0.0);
  
  overlay = iftBorderImage(label,0);

  RGB = iftRGBColor(0, normvalue, normvalue);
  YCbCr = iftRGBtoYCbCr(RGB, normvalue);
  
  iftDrawBorders(copy,overlay,A,YCbCr,A);

  iftWriteImageByExt(copy, filename);

  iftDestroyImage(&overlay);
  iftDestroyImage(&copy);
  iftDestroyAdjRel(&A);
}

iftImage *iftObjectSaliencyIFT(iftImage *img, iftImage *objsm, iftImage *old_seeds, int num_seeds, float obj_perc, float gamma, iftImage **new_seeds) {
  int K = IFT_QSIZE - 1, MAX_ARC_COST = K;
  int label_count;
  iftMImage *mimg, *obj_mimg;
  iftImage *mask, *seed_img, *C, *L, *P, *R; // Cost, Label, Predecessor and Root maps
  iftLabeledSet *S, *set_aux;
  iftGQueue *Q;
  iftAdjRel *A;

  label_count = 0;

  A = iftCircular(1.0);
  L = iftCreateImage(img->xsize, img->ysize, img->zsize);
  P = iftCreateImage(img->xsize, img->ysize, img->zsize);
  C = iftCreateImage(img->xsize, img->ysize, img->zsize);
  R = iftCreateImage(img->xsize, img->ysize, img->zsize);
  Q = iftCreateGQueue(MAX_ARC_COST + 1, img->n, C->val);

  if(*new_seeds != NULL) iftDestroyImage(new_seeds);
  *new_seeds = iftCreateImage(img->xsize, img->ysize, img->zsize);

  S = NULL;
  set_aux = NULL;

  if (iftIsColorImage(img)) mimg   = iftImageToMImage(img,LABNorm2_CSPACE);
  else mimg = iftImageToMImage(img,GRAYNorm_CSPACE);

  if (iftIsColorImage(objsm)) obj_mimg   = iftImageToMImage(objsm,LABNorm2_CSPACE);
  else obj_mimg = iftImageToMImage(objsm,GRAYNorm_CSPACE);

  mask = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);
  
  // Sampling procedure -------------------------------------------------------------------------//
  if(old_seeds != NULL){
    int adap_thresh, mean_saliency,max_label, remaining_num_seeds;
    iftIntArray *label_lut;
    iftImage *relabel_img;
    iftAdjRel *B;

    B = iftCircular(1.0);
    
    mean_saliency = 0;
    #pragma omp parallel for reduction(+:mean_saliency)
    for(int p = 0; p < objsm->n; p++) mean_saliency += objsm->val[p];
    
    adap_thresh = iftRound(2.0*mean_saliency/(float)objsm->n);

    #pragma omp parallel for
    for(int p = 0; p < mask->n; p++) {
      if(objsm->val[p] >= adap_thresh) mask->val[p] = 255;
      else mask->val[p] = 0;
    }

    relabel_img = iftFastLabelComp(mask,B);
    
    #pragma omp parallel for
    for(int p = 0; p < mask->n; p++) {
      if(objsm->val[p] >= adap_thresh) mask->val[p] = 0;
      else mask->val[p] = 255;
    }
    
    max_label = iftMaximumValue(relabel_img);
    label_lut = iftCreateIntArray(max_label + 1);

    #pragma omp parallel for
    for(int l = 0; l <= max_label; l++) label_lut->val[l] = IFT_NIL;
    

    for(int p = 0; p < old_seeds->n; p++) {
      if(old_seeds->val[p] != 0 && mask->val[p] == 0) {
        int label, tmp_label;

        tmp_label = relabel_img->val[p];

        if(label_lut->val[tmp_label] == IFT_NIL) label_lut->val[tmp_label] = ++label_count;
        
        label = label_lut->val[tmp_label];

        iftInsertLabeledSet(&S, p, label);
        
        (*new_seeds)->val[p] = 255;
      }
    }
    
    remaining_num_seeds = iftMax(num_seeds - label_count, 1);
    seed_img = iftSamplingByOSMOX(objsm, mask, remaining_num_seeds, 0.0);
    
    iftDestroyIntArray(&label_lut);
    iftDestroyImage(&relabel_img);
    iftDestroyAdjRel(&B);
  } else {
    seed_img = iftSamplingByOSMOX(objsm, mask, num_seeds, obj_perc);
  }
  

  for(int p = 0; p < seed_img->n; p++) {
    if(seed_img->val[p] != 0) { 
      iftInsertLabeledSet(&S, p, ++label_count); 
      (*new_seeds)->val[p] = 128;
    }
  }

  iftDestroyImage(&mask);
  iftDestroyImage(&seed_img);
  // Initialization procedure -------------------------------------------------------------------//
  #pragma omp parallel for
  for(int p = 0; p < mimg->n; p++) {
    C->val[p] = IFT_INFINITY_INT;
    L->val[p] = IFT_NIL;
    P->val[p] = IFT_NIL;
    R->val[p] = p;
  }

  set_aux = S;
  while(set_aux != NULL) {
    int index, label;

    index = set_aux->elem;
    label = set_aux->label;

    C->val[index] = 0;
    L->val[index] = label;

    iftInsertGQueue(&Q,index);

    set_aux = set_aux->next;
  }
  // IFT procedure ------------------------------------------------------------------------------//
  while (!iftEmptyGQueue(Q)){
    int p;
    iftVoxel u;

    p = iftRemoveGQueue(Q);
    u = iftMGetVoxelCoord(mimg, p);

    for (int i = 1; i < A->n; i++) {
      iftVoxel v;

      v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v)) {
        int q, tmp;
        float arc_cost, color_dist, obj_dist;

        q = iftMGetVoxelIndex(mimg, v);

        color_dist = (1.0 - gamma) * (iftMImageDist(mimg, p, q) / sqrtf(mimg->m)); // Normalize to [0,1]
        obj_dist = gamma * (iftMImageDist(obj_mimg, p, q) / sqrtf(obj_mimg->m)); // Normalize to [0,1]
        arc_cost = K * (color_dist + obj_dist);

        tmp = iftMax(C->val[p], iftRound(arc_cost));
        
        if (tmp < C->val[q]) {
          if(Q->L.elem[q].color == IFT_GRAY) iftRemoveGQueueElem(Q,q);
          L->val[q] = L->val[p];
          R->val[q] = R->val[p];
          P->val[q] = p;
          C->val[q] = tmp;
          iftInsertGQueue(&Q, q);
        }
      }
    }
  }
  // Debugging procedure ------------------------------------------------------------------------//
  iftDestroyMImage(&mimg);
  iftDestroyMImage(&obj_mimg);
  iftDestroyImage(&C);
  iftDestroyImage(&R);
  iftDestroyImage(&P);
  iftDestroyLabeledSet(&S);
  iftDestroyLabeledSet(&set_aux);
  iftDestroyGQueue(&Q);
  iftDestroyAdjRel(&A);
  
  return L;
}