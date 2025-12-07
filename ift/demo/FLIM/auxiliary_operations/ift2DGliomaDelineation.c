#include "ift.h"

#define BORDER_DIST 1
#define MIN_OBJ_AREA 100
#define MAX_OBJ_AREA 16200

/*
Author: Felipe Crispim da Rocha Salvagnini (February 07th, 2023)

  Description: Delineates Glioama tumor tissues from the
  decoded saliency maps.

Execution sample on BraTS FLAIR images for WT Detection:

iftGliomaDelineation <standardized_tf2> <saliency_maps> <modality> <n_layer> 
  <output_folder> <erosion_seed> <dilation_seed>

Example:

  ift2DGliomaDelineation ~/msc/baseline_to_read/brats_2021_2d/orig/ \
  flim_output/split1/train/saliency/ delineation_out 3 20
*/

int iftMaxIndexArray(int *array, int length) {
  int max = array[0];
  int max_index = 0;

  int i;
  for (i = 1; i < length; ++i) {
      if (array[i] > max) {
        max = array[i];
        max_index = i;
      }
  }

  return max_index;
}

iftSet *iftGetSetOfCoreComponents(iftImage *label, iftImage *sal, int mode) {
  int nlabels = iftMaximumValue(label);
  int *ctresh = iftAllocIntArray(nlabels);
  int *csalie = iftAllocIntArray(nlabels);
  int *nelems = iftAllocIntArray(nlabels);
  iftSet *SelectedComps = NULL;

  for (size_t l = 0; l <  nlabels; l++) {
    iftInsertSet(&SelectedComps, l + 1);
    iftImage *cmask = iftCreateImageFromImage(sal);

    for (size_t p = 0; p < label->n; p++) {
      if (label->val[p] == l + 1) {
        cmask->val[p] = 255;
      }
    }

    ctresh[l] = iftOtsuInRegion(sal, cmask);
    iftDestroyImage(&cmask);
  }

  for (size_t p = 0; p < label->n; p++) {
    if (label->val[p] > 0) {
      if (sal->val[p] < ctresh[label->val[p] - 1]) 
      {
        label->val[p] = 0;
      } else {
        // Increment elements cound (For area)
        nelems[label->val[p] - 1]++;
        // Updates the maximum saliency value
        if (sal->val[p] > csalie[label->val[p] - 1]) {
          csalie[label->val[p] - 1] = sal->val[p];
        }
      }
    }
  }

  // Mode to update label maps
  if (mode == 1) { // Will keep the component with the largest area
    int i_max = iftMaxIndexArray(nelems, nlabels);
    for (size_t p = 0; p < label->n; p++) {
      if (label->val[p] != i_max + 1) {
        label->val[p] = 0;
      }
    }
  }
  else if (mode == 2) { // Will keep the component with the largest saliency value
    int i_max = iftMaxIndexArray(csalie, nlabels);
    for (size_t p = 0; p < label->n; p++) {
      if (label->val[p] != i_max + 1) {
        label->val[p] = 0;
      }
    }
  }

  iftFree(ctresh);
  iftFree(csalie);
  iftFree(nelems);

  return SelectedComps;
}

iftImage *GetBrainMask(iftImage *orig) {
  iftImage *brain_mask = iftCreateImageFromImage(orig);
  for (size_t idx = 0; idx < orig->n; idx++) {
    if (orig->val[idx] > 0) {
      brain_mask->val[idx] = 255;
    }
  }

  return brain_mask;
}

iftImage *DynamicTrees(iftImage *orig, iftImage *seeds_in, iftImage *seeds_out)
{
  iftMImage *mimg = iftImageToMImage(orig, LAB_CSPACE);
  iftImage *pathval = NULL, *label = NULL, *root = NULL;
  float *tree_L = NULL;
  float *tree_A = NULL;
  float *tree_B = NULL;
  int *nnodes = NULL;
  int Imax = iftRound(sqrtf(3.0) *
                      iftMax(iftMax(iftMMaximumValue(mimg, 0),
                                    iftMMaximumValue(mimg, 1)),
                             iftMMaximumValue(mimg, 2)));
  iftGQueue *Q = NULL;
  iftAdjRel *A = iftCircular(1.0);
  int i, p, q, r, tmp;
  iftVoxel u, v;

  pathval = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
  label = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
  root = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
  tree_L = iftAllocFloatArray(orig->n);
  tree_A = iftAllocFloatArray(orig->n);
  tree_B = iftAllocFloatArray(orig->n);
  nnodes = iftAllocIntArray(orig->n);
  Q = iftCreateGQueue(2 * (Imax + 1), orig->n, pathval->val);

  for (p = 0; p < orig->n; p++)
  {
    pathval->val[p] = IFT_INFINITY_INT;
    if (seeds_in->val[p] != 0)
    {
      root->val[p] = p;
      label->val[p] = 255;
      pathval->val[p] = 0;
    }
    else
    {
      if (seeds_out->val[p] != 0)
      {
        root->val[p] = p;
        label->val[p] = 0;
        pathval->val[p] = 0;
      }
    }
    iftInsertGQueue(&Q, p);
  }

  // Image Foresting Transform

  while (!iftEmptyGQueue(Q))
  {
    p = iftRemoveGQueue(Q);
    r = root->val[p];
    tree_L[r] += mimg->val[p][0];
    tree_A[r] += mimg->val[p][1];
    tree_B[r] += mimg->val[p][2];
    nnodes[r] += 1;
    u = iftGetVoxelCoord(orig, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);

      if (iftValidVoxel(orig, v))
      {
        q = iftGetVoxelIndex(orig, v);

        if (Q->L.elem[q].color != IFT_BLACK)
        {

          int Wi = iftRound(
              sqrtf(powf((mimg->val[q][0] - tree_L[r] / nnodes[r]), 2.0) +
                    powf((mimg->val[q][1] - tree_A[r] / nnodes[r]), 2.0) +
                    powf((mimg->val[q][2] - tree_B[r] / nnodes[r]), 2.0)));

          tmp = iftMax(pathval->val[p], Wi);

          if (tmp < pathval->val[q])
          {
            if (Q->L.elem[q].color == IFT_GRAY)
              iftRemoveGQueueElem(Q, q);
            label->val[q] = label->val[p];
            root->val[q] = root->val[p];
            pathval->val[q] = tmp;
            iftInsertGQueue(&Q, q);
          }
        }
      }
    }
  }

  iftDestroyAdjRel(&A);
  iftDestroyGQueue(&Q);
  iftDestroyImage(&pathval);
  iftDestroyImage(&root);
  iftDestroyMImage(&mimg);
  iftFree(tree_L);
  iftFree(tree_A);
  iftFree(tree_B);
  iftFree(nnodes);

  return (label);
}

iftImage *Watershed(iftImage *gradI, iftImage *seeds_in, iftImage *seeds_out)
{
  iftImage *pathval = NULL, *label = NULL;
  iftGQueue *Q = NULL;
  int i, p, q, tmp;
  iftVoxel u, v;
  iftAdjRel *A = iftCircular(1.0);

  pathval = iftCreateImage(gradI->xsize, gradI->ysize, gradI->zsize);
  label = iftCreateImage(gradI->xsize, gradI->ysize, gradI->zsize);
  Q = iftCreateGQueue(iftMaximumValue(gradI) + 1, gradI->n, pathval->val);

  for (p = 0; p < gradI->n; p++)
  {
    pathval->val[p] = IFT_INFINITY_INT;
    if (seeds_in->val[p] != 0)
    {
      label->val[p] = 255;
      pathval->val[p] = 0;
    }
    else
    {
      if (seeds_out->val[p] != 0)
      {
        label->val[p] = 0;
        pathval->val[p] = 0;
      }
    }
    iftInsertGQueue(&Q, p);
  }

  while (!iftEmptyGQueue(Q))
  {
    p = iftRemoveGQueue(Q);
    u = iftGetVoxelCoord(gradI, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);

      if (iftValidVoxel(gradI, v))
      {
        q = iftGetVoxelIndex(gradI, v);

        if (Q->L.elem[q].color != IFT_BLACK)
        {

          tmp = iftMax(pathval->val[p], gradI->val[q]);

          if (tmp < pathval->val[q])
          {
            iftRemoveGQueueElem(Q, q);
            label->val[q] = label->val[p];

            pathval->val[q] = tmp;
            iftInsertGQueue(&Q, q);
          }
        }
      }
    }
  }

  iftDestroyAdjRel(&A);
  iftDestroyGQueue(&Q);
  iftDestroyImage(&pathval);

  return (label);
}

/* op specifies the type of operation employed to delineate:
- 1: DynamicTrees
- 2: Watershed
*/
iftImage *DelineateComps(
  iftImage *comps, iftSet *SelectedComps, iftImage *orig,
  int seed_erosion, int seed_dilation, int op
)
{
  iftImage *seeds_in = iftCreateImage(comps->xsize,comps->ysize,comps->zsize);
  for (int p=0; p < comps->n; p++){
    if (iftSetHasElement(SelectedComps, comps->val[p])){
      seeds_in->val[p] = 1;       
    }
  }
  iftSet   *S   = NULL;
  iftImage *bin = iftErodeBin(seeds_in,&S,seed_erosion);
  iftDestroyImage(&seeds_in);
  seeds_in      = bin;
  bin           = iftDilateBin(seeds_in,&S,seed_dilation);
  iftImage *seeds_out = iftComplement(bin);
  iftDestroyImage(&bin);
  iftDestroySet(&S);

  iftImage *label = NULL;

  if (op == 1) {
    label = DynamicTrees(orig,seeds_in,seeds_out);
  }
  else if (op == 2) {
    iftAdjRel *A = iftCircular(sqrtf(2.0));
    iftImage *grad_img = iftImageGradientMagnitude(orig, A);
    iftFImage *grad_smoothed = iftSmoothWeightImage(grad_img,0.5);
    iftDestroyImage(&grad_img);
    grad_img = iftFImageToImage(grad_smoothed, 255);
    label = Watershed(grad_img, seeds_in, seeds_out);
    iftDestroyImage(&grad_img);
    iftDestroyFImage(&grad_smoothed);
    iftDestroyAdjRel(&A);
  }
  iftDestroyImage(&seeds_in);
  iftDestroyImage(&seeds_out);

  return(label);
}

int main(int argc, char *argv[]) {
  timer *t_start = NULL;
  int mem_start, mem_end;

  if (argc != 6) {
    iftError(
      "Usage: iftGliomaDelineation <p1> <p2> <p3> <p4> <p5>\n"
      "[1] folder with input images\n"
      "[2] folder with the salience maps\n"
      "[3] output folder with the resulting images\n"
      "[4] erosion seed\n"
      "[5] dilation seed\n",
      "main"
    );
  }

  mem_start = iftMemoryUsed();
  t_start = iftTic();

  char *orig_path = iftAllocCharArray(512);
  char *save_path = iftAllocCharArray(512);
  iftFileSet *fs = iftLoadFileSetFromDir(argv[2], 0);
  int seed_erosion = atoi(argv[4]);
  int seed_dilation = atoi(argv[5]);

  for (size_t idx = 0; idx < fs->n; idx++) {
    char *filename = iftFilename(
      fs->files[idx]->path, fs->files[idx]->suffix
    );
    char *imagename = iftSplitStringAt(filename, ".", 0);
    sprintf(
      orig_path, "%s/orig/%s", argv[1], filename
    );
    printf("\r[INFO] (%ld/%ld) Delineating image %s", idx+1, fs->n, filename);

    iftImage *orig = iftReadImageByExt(orig_path);
    iftImage *sal = iftReadImageByExt(fs->files[idx]->path);
    iftImage *brain_mask = GetBrainMask(orig);
    iftImage *aux1 = NULL, *aux2 = NULL;

    // Applies Thresholding, using brain mask
    int otsu_brain_region = iftOtsuInRegion(sal, brain_mask);
    iftImage  *bin_sal = iftThreshold(
      sal, otsu_brain_region, IFT_INFINITY_INT, 255
    );

    aux1 = iftSelectCompInAreaInterval(
      bin_sal, NULL, MIN_OBJ_AREA, MAX_OBJ_AREA
    );

    /*
      Here we will label components and select the candadites. Before
    delineating using the components as a seed, we will:

    1. Compute a new otsu threshold, using each component mask;
    2. Updates the the components, using the new Otsu threshold.
    */
    aux2 = iftFastLabelComp(aux1, NULL);
    iftDestroyImage(&aux1);
    iftSet *SelectedComps = iftGetSetOfCoreComponents(
      aux2, sal, 2
    );
    iftImage *bin = NULL;

    if (SelectedComps != NULL) {
      iftImage *label = DelineateComps(
        aux2, SelectedComps, orig, seed_erosion, seed_dilation, 1
      );
      aux1 = iftSelectCompInAreaInterval(
        label, NULL, MIN_OBJ_AREA, MAX_OBJ_AREA
      );
      bin = iftThreshold(label, 1, 255, 255);
      iftDestroyImage(&label);
    } else {
      bin = iftCreateImageFromImage(sal);
    }

    sprintf(save_path, "%s/%s/%s", argv[3], "saliency", filename);
    iftWriteImageByExt(bin, save_path);

    iftDestroyImage(&orig);
    iftDestroyImage(&bin);
    iftDestroyImage(&bin_sal);
    iftDestroyImage(&sal);
    iftDestroyImage(&brain_mask);
    iftDestroyImage(&aux1);
    iftDestroyImage(&aux2);
    iftDestroySet(&SelectedComps);
    iftFree(filename);
    iftFree(imagename);
  }
  
  iftDestroyFileSet(&fs);
  iftFree(orig_path);
  iftFree(save_path);

  puts("\nDone ...");
  puts(iftFormattedTime(iftCompTime(t_start, iftToc())));
  mem_end = iftMemoryUsed();
  iftVerifyMemory(mem_start, mem_end);
}
