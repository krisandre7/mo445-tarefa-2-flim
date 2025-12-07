#include "ift.h"

#define YELLOW 1
#define BLUE 2
#define GREEN 3
#define RED 4

iftColor *root_color = NULL;

void saveRootMap(iftImageForest *fst, char *filename);
void savePredecessorMap(iftImageForest *fst, char *filename);
void bitmapToImage(iftBMap *bitmap, iftImage *img, char *filename);
void saveImage2d(iftImage *original, iftImage *seeds_image, iftImage *label_image, char *filename);
void frontierToImage(iftImage *img, iftBMap *frontier, char *filename);
int FindRelaxationRoot(iftImage *pred, iftImage *new_label, iftImage *old_label, int p, char *valid)
{
  int  q = pred->val[p], r = p;

  while ((q != NIL) && (new_label->val[q] != old_label->val[q]))
  {
    r     = q;
    q     = pred->val[q];
  }

  /* verify if the predecessor of the root candidate is connected to
     the root by a path that did not change the label */

  if (iftIsLabelRootConnected(pred, new_label, q))
  {
    *valid = 1;
  }
  else
  {
    *valid = 0;
  }

  return (r);
}

void PropagateLabelToSubtree(iftImageForest *fst, iftAdjRel *A, int new_label, int r, iftImage *seed)
{
  iftImage *pred = fst->pred, *label = fst->label;
  int       p, q, i;
  iftVoxel  u, v;
  iftSet   *T = NULL;

  iftInsertSet(&T, r);
  fst->root->val[r] = r;

  while (T != NULL)
  {
    p             = iftRemoveSet(&T);
    u             = iftGetVoxelCoord(pred, p);
    label->val[p] = new_label;
    if (seed->val[p] == 0)
      seed->val[p] = YELLOW;
    for (i = 1; i < A->n; i++)
    {
      v   = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(pred, v))
      {
        q = iftGetVoxelIndex(pred, v);
        if (pred->val[q] == p)
        {
          fst->root->val[q] = fst->root->val[p];
          iftInsertSet(&T, q);
        }
      }
    }
  }
}
void RelaxObjects(iftImageForest *fst, iftSmoothBorder *smooth, iftImage *img, int number)
{
  iftImage  *prev_label = smooth->prev_label, *next_label = smooth->next_label, *new_label = NULL, *old_label = NULL;
  iftFImage *prev_weight = smooth->prev_weight, *next_weight = smooth->next_weight;
  float      *sum, max_membership;
  int        l, i, p, q, r;
  int        borderSize = 0, max_label, iter;
  iftBMap   *processed = fst->processed, *inBorder = iftCreateBMap(fst->img->n);
  iftSet    *prev_border = NULL, *next_border = NULL, *dilated_border = NULL;
  iftVoxel   u, v;
  iftAdjRel *A = fst->A;
  char       valid;
  iftSet    *new_roots = NULL;


  iftImage *seed_image = iftCreateImage(img->xsize, img->ysize, img->zsize);
  char buffer[128];


  /* Initialization */


  sum         = iftAllocFloatArray(iftMaximumValue(fst->label) + 1);

  /* Detect border of the processed regions */


  for (p = 0; p < processed->n; p++)
  {
    /* Reset data structures for object boundary relaxation */

    prev_weight->val[p] = next_weight->val[p] = 1.0;
    prev_label->val[p]  = next_label->val[p]  = fst->label->val[p];

    u  = iftGetVoxelCoord(fst->img, p);
    if (iftBMapValue(processed, p))
    {
      for (i = 1; i < A->n; i++)
      {
        v = iftGetAdjacentVoxel(A, u, i);
        if (iftValidVoxel(prev_label, v))
        {
          q = iftGetVoxelIndex(fst->img, v);
          if (fst->label->val[p] != fst->label->val[q])
          {
            if (iftBMapValue(inBorder, p) == 0)
            {
              iftBMapSet1(inBorder, p);
              iftInsertSet(&prev_border, p);
              borderSize++;
            }
            if (iftBMapValue(inBorder, q) == 0)
            {
              iftBMapSet1(inBorder, q);
              iftInsertSet(&prev_border, q);
              borderSize++;
            }
          }
        }
      }
    }

  }

  sprintf(buffer, "frontier%d.ppm", number);
  frontierToImage(img, inBorder, buffer);

  prev_label->maxval = next_label->maxval = fst->label->maxval;


  if (borderSize == 0)
  {
    free(sum);
    iftDestroyBMap(&inBorder);
    iftWarning("There is no need for boundary smoothing", "iftRelaxObjects");
    return;
  }

  /* Relax object boundaries in border region while it is dilated */

  for (iter = 0; iter < smooth->smooth_iterations; iter++)
  {

    next_border = NULL;

    while (prev_border != NULL)
    {
      p = iftRemoveSet(&prev_border);
      iftInsertSet(&next_border, p);
      u   = iftGetVoxelCoord(prev_label, p);

      for (l = 0; l <= prev_label->maxval; l++)
        sum[l] = 0.0;

      for (i = 1; i < A->n; i++)
      {
        v = iftGetAdjacentVoxel(A, u, i);
        if (iftValidVoxel(prev_label, v))
        {
          q = iftGetVoxelIndex(prev_label, v);
          sum[prev_label->val[q]] += prev_weight->val[q] * smooth->border_weight->val[q];

          if (iftBMapValue(inBorder, q) == 0) /* expand border */
          {
            iftInsertSet(&next_border, q);
            iftBMapSet1(inBorder, q);
          }
        }
      }

      for (l = 0; l <= prev_label->maxval; l++)
        sum[l]  = sum[l] / smooth->norm_factor->val[p];

      max_membership = -INFINITY_FLT; max_label = NIL;
      for (l = 0; l <= prev_label->maxval; l++)
      {
        if (sum[l] > max_membership)
        {
          max_membership = sum[l];
          max_label      = l;
        }
      }
      next_label->val[p]  = max_label;
      next_weight->val[p] = sum[max_label];
    }

    prev_border = next_border;

    for (r = 0; r < prev_label->n; r++)
    {
      prev_weight->val[r] = next_weight->val[r];
      prev_label->val[r]  = next_label->val[r];
    }
  }

  free(sum);
  iftDestroyBMap(&inBorder);
  new_label      = next_label;
  old_label      = fst->label;
  dilated_border = next_border;

  sprintf(buffer, "label%d0.ppm", number);
  saveImage2d(img, NULL, next_label, buffer);

  /* Fix possible segmentation inconsistencies */

  while (dilated_border != NULL)
  {
    p = iftRemoveSet(&dilated_border);
    if (new_label->val[p] != old_label->val[p])
    {
      /* the label of p has changed */
      // seed_image->val[p] = BLUE;

      /* Find the new root defined by relaxation */
      r = FindRelaxationRoot(fst->pred, new_label, old_label, p, &valid);
      if (valid)
      {
        iftInsertSet(&new_roots, r);
        PropagateLabelToSubtree(fst, A, new_label->val[r], r, seed_image);
      }
    }
  }

  while (new_roots != NULL)
  {
    p = iftRemoveSet(&new_roots);
    seed_image->val[p] = RED;
    fst->pred->val[p] = NIL;
  }

  sprintf(buffer, "predcessor%d.ppm", number);
  savePredecessorMap(fst, buffer);

  sprintf(buffer, "fix%d.ppm", number);
  saveImage2d(img, NULL, seed_image, buffer);

}


int main(int argc, char *argv[])
{
  iftImage        *img               = NULL, *basins = NULL;
  iftImageForest  *fst               = NULL;
  iftLabeledSet   *seed1             = NULL;
  iftLabeledSet   *seed2             = NULL;
  iftAdjRel       *A                 = NULL;
  iftSmoothBorder *smooth            = NULL;
  timer *t1, *t2;
  char             ext[10], *pos;

  if (argc != 6)
    iftError("Usage: testDiffWatershedAndFrontier <image.ppm> <markers 1> <markers 2> <adjacency> <smooth_iterations> ", "main");

  pos = strrchr(argv[1], '.') + 1;
  sscanf(pos, "%s", ext);
  if (strcmp(ext, "ppm") == 0)
    img   = iftReadImageP6(argv[1]);
  else
  {
    printf("Invalid image format: %s\n", ext);
    exit(-1);
  }

  seed1             = iftReadSeeds2D(argv[2], img);
  seed2             = iftReadSeeds2D(argv[3], img);
  A                 = iftCircular(atof(argv[4]));
  basins            = iftImageBasins(img, A);
  fst               = iftCreateImageForest(basins, A);
  smooth            = iftCreateSmoothBorder(fst->img, fst->A, atoi(argv[5]), 0.5);

  root_color = (iftColor *) malloc(img->n * sizeof(iftColor *));


  t1 = iftTic();
  iftDiffWatershed(fst, seed1);
  t2 = iftToc();
  fprintf(stdout, "Segmentation time: %.2f s\n", iftCompTime(t1, t2) / 1000);

  iftIsSegmentationConsistent(fst);
  saveImage2d(img, NULL, fst->label, "label1.ppm");
  saveRootMap(fst, "root0.ppm");

  t1 = iftTic();
  RelaxObjects(fst, smooth, img, 1);
  t2 = iftToc();
  fprintf(stdout, "Smoothing time: %.2f s\n", iftCompTime(t1, t2) / 1000);

  iftIsSegmentationConsistent(fst);
  saveImage2d(img, NULL, fst->label, "label2.ppm");

  saveRootMap(fst, "root1.ppm");

  t1 = iftTic();
  iftDiffWatershed(fst, seed2);
  t2 = iftToc();
  fprintf(stdout, "Segmentation time [2]: %.2f s\n", iftCompTime(t1, t2) / 1000);

  iftIsSegmentationConsistent(fst);
  saveImage2d(img, NULL, fst->label, "label3.ppm");

  t1 = iftTic();
  RelaxObjects(fst, smooth, img, 2);
  t2 = iftToc();
  fprintf(stdout, "Smoothing time [2]: %.2f s\n", iftCompTime(t1, t2) / 1000);

  saveImage2d(img, NULL, fst->label, "label4.ppm");
  iftIsSegmentationConsistent(fst);

  iftDestroyImageForest(&fst);
  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);
  iftDestroyImage(&basins);

  free(root_color);

  return (0);
}
void bitmapToImage(iftBMap *bitmap, iftImage *img, char *filename)
{
  iftImage *output = iftCreateImage(img->xsize, img->ysize, img->zsize);

  for (int p = 0; p < bitmap->n; ++p)
  {
    if (iftBMapValue(bitmap, p))
      output->val[p] = 255;
  }
  iftWriteImageP2(output, filename);
  iftDestroyImage(&output);
}

void saveImage2d(iftImage *original, iftImage *seeds_image, iftImage *label_image, char *filename)
{
  int p = 0;
  iftColor yellow;
  iftColor red;
  iftColor green;
  iftColor blue;

  yellow.val[0] = 255;
  yellow.val[1] = 1;
  yellow.val[2] = 148;

  red.val[0] = 76;
  red.val[1] = 85;
  red.val[2] = 255;

  blue.val[0] = 41;
  blue.val[1] = 240;
  blue.val[2] = 110;

  green.val[0] = 137;
  green.val[1] = 95;
  green.val[2] = 79;

  iftImage *markers_image = iftCopyImage(original);
  if (seeds_image != NULL)
  {
    for (p = 0; p < markers_image->n; p++)
    {
      //Background marker colour
      if (seeds_image->val[p] == 0)
      {
        markers_image->val[p] = red.val[0];
        markers_image->Cb[p] = red.val[1];
        markers_image->Cr[p] = red.val[2];
      }
      //Object marker colour
      else if (seeds_image->val[p] == 1)
      {
        markers_image->val[p] = yellow.val[0];
        markers_image->Cb[p] = yellow.val[1];
        markers_image->Cr[p] = yellow.val[2];
      }
      else if (seeds_image->val[p] == 2)
      {
        markers_image->val[p] = blue.val[0];
        markers_image->Cb[p] = blue.val[1];
        markers_image->Cr[p] = blue.val[2];
      }
      //Available to conquest
      else if (seeds_image->val[p] == 3)
      {
        markers_image->val[p] = green.val[0];
        markers_image->Cb[p] = green.val[1];
        markers_image->Cr[p] = green.val[2];
      }
    }
  }

  //iftWriteImageP6(markers_image, "markesrimage.ppm");
  iftImage *output = iftCopyImage(markers_image);
  if (label_image != NULL)
  {
    for (p = 0; p < markers_image->n; p++)
    {
      //Background colour
      // if (label_image->val[p] == 0)
      // {
      //     output->Cb[p] = red.val[1];
      //     output->Cr[p] = red.val[2];
      // }
      //Object colour
      if (label_image->val[p] == 1)
      {
        output->Cb[p] = yellow.val[1];
        output->Cr[p] = yellow.val[2];
      }
      //Object colour
      if (label_image->val[p] == 2)
      {
        output->Cb[p] = blue.val[1];
        output->Cr[p] = blue.val[2];
      }
      if (label_image->val[p] == 3)
      {
        output->Cb[p] = green.val[1];
        output->Cr[p] = green.val[2];
      }
      if (label_image->val[p] == 4)
      {
        output->Cb[p] = red.val[1];
        output->Cr[p] = red.val[2];
      }
    }
  }
  //iftWriteImageP6(markers_image, path);
  iftWriteImageP6(output, filename);

  iftDestroyImage(&markers_image);
  iftDestroyImage(&output);
}
void frontierToImage(iftImage *img, iftBMap *frontier, char *filename)
{
  iftImage *output = iftCopyImage(img);
  int i = 0;
  for (i = 0; i < frontier->n; i++)
  {
    if (iftBMapValue(frontier, i))
    {
      output->val[i] = 0;
      output->Cb[i] = 50;
      output->Cr[i] = 250;
    }
  }
  iftWriteImageP6(output, filename);
}
void savePredecessorMap(iftImageForest *fst, char *filename)
{
  int p, q, length;
  iftImage *output = iftCreateImage(fst->img->xsize, fst->img->ysize, fst->img->zsize);
  iftImage *norm = NULL;

  for (p = 0; p < fst->pred->n; p++)
  {
    q = p;
    length = 0;
    while (fst->pred->val[q] != NIL)
    {
      length++;
      q = fst->pred->val[q];
    }
    output->val[p] = length;
  }
  norm = iftNormalize(output, 0, 127);
  iftWriteImageP2(norm, filename);
  iftDestroyImage(&output);
  iftDestroyImage(&norm);

}
void saveRootMap(iftImageForest *fst, char *filename)
{
  int p;
  iftColor c;
  iftImage *output = iftCreateColorImage(fst->img->xsize, fst->img->ysize, fst->img->zsize);

  c.val[0] = 0;
  c.val[1] = 0;
  c.val[2] = 0;
  for (p = 0; p < fst->root->n; p++)
    root_color[p] = c;

  for (p = 0; p < fst->root->n; p++)
  {
    if (fst->pred->val[p] == NIL)
    {
      c.val[0] = rand() % 235 + 16;
      c.val[1] = rand() % 240 + 16;
      c.val[2] = rand() % 240 + 16;
      root_color[p] = c;
    }
  }
  for (p = 0; p < fst->root->n; p++)
  {
    output->val[p] = root_color[fst->root->val[p]].val[0];
    output->Cb[p] = root_color[fst->root->val[p]].val[1];
    output->Cr[p] = root_color[fst->root->val[p]].val[2];
    if (fst->pred->val[p] == NIL)
    {
      output->val[p] = 0;
      output->Cb[p] = 0;
      output->Cr[p] = 0;
    }
  }
  iftWriteImageP6(output, filename);
  iftDestroyImage(&output);

}