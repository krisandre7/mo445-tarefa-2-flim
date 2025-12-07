#include "ift.h"

/*
  Author: Felipe Crispim da Rocha Salvagnini (July 31/07/2024)

  Given a Sorted Bag of Feature Points as input, it creates a convolutional
  layer with the most discriminative and descriptive layers (we use the
  handicap value for this purpose). The user must provide the number of kernels
  he wants to use for each class.
  
  !!!!Currently, it supports foreground and background only!!!!

  Arguments:

  - [1] Input folder with sorted feature points
  - [2] Json etwork architecture
  - [3] Layer to create the filters
  - [4] Folder to save output model
  - [5] Number of filters for background
  - [6] Number of filters for foreground

  Execution
  iftCreateSortedLayerModel 0_data/bofp_sorted_few/ 0_data/arch.json 1 model_sorted 2 2

*/

iftVoxel GetVoxelCoord(int xsize, int ysize, int zsize, int p)
{
  iftVoxel u;
  div_t res1 = div(p, xsize * ysize);
  div_t res2 = div(res1.rem, xsize);

  u.x = res2.rem;
  u.y = res2.quot;
  u.z = res1.quot;
  u.t = 0;
  return (u);
}

iftAdjRel *GetPatchAdjacency(iftMImage *mimg, iftFLIMLayer layer)
{
  iftAdjRel *A;

  if (iftIs3DMImage(mimg))
  {
    A = iftCuboidWithDilationForConv(layer.kernel_size[0],
                                     layer.kernel_size[1],
                                     layer.kernel_size[2],
                                     layer.dilation_rate[0],
                                     layer.dilation_rate[1],
                                     layer.dilation_rate[2]);
  }
  else
  {
    A = iftRectangularWithDilationForConv(layer.kernel_size[0],
                                          layer.kernel_size[1],
                                          layer.dilation_rate[0],
                                          layer.dilation_rate[1]);
  }

  return (A);
}

iftDataSet *ExtractSortedImagePatches(
  iftMImage *mimg, iftLabeledSet *S, iftFLIMLayer layer,
  float *scale, int n_bg_filters, int n_fg_filters
) {
  iftAdjRel *A = GetPatchAdjacency(mimg, layer);

  int nfeats = 0, nsamples = iftLabeledSetSize(S);
  nfeats = A->n * mimg->m;
  iftDataSet *Z = iftCreateDataSet(nsamples, 0);

  iftLabeledSet *Saux = S;
  int s = 0;
  // Reads S into a dataset
  while (Saux != NULL)
  {
    Z->sample[s].id = Saux->elem;
    Z->sample[s].weight = (float)Saux->handicap;
    // Trualabels vary from 1 to c
    Z->sample[s].truelabel = Saux->label + 1;
    if (Z->sample[s].truelabel > Z->nclasses) {
      Z->nclasses = Z->sample[s].truelabel;
    }
    s++;
    Saux = Saux->next;
  }

  int *bg_idx = calloc(n_bg_filters, sizeof(int));
  int *fg_idx = calloc(n_fg_filters, sizeof(int));

  // Stores sorted indexes
  for(size_t i = 0; i < Z->nsamples; i++) {
    // Background
    if (Z->sample[i].truelabel == 1) {
      if (Z->sample[i].weight <= n_bg_filters) {
        bg_idx[(int)Z->sample[i].weight - 1] = i;
      }
    } 
    // Foreground
    else if (Z->sample[i].truelabel == 2) {
      if (Z->sample[i].weight <= n_fg_filters) {
        fg_idx[(int)Z->sample[i].weight - 1] = i;
      }
    }
  }
  
  // Reads data into final dataset, also reading patches
  size_t z_idx = 0;
  iftDataSet *Z_sorted = iftCreateDataSet(
    n_bg_filters + n_fg_filters, nfeats
  );
  for (size_t i = 0; i < n_bg_filters; i++) {
    Z_sorted->sample[z_idx].id = Z->sample[bg_idx[i]].id;
    Z_sorted->sample[z_idx].truelabel = Z->sample[bg_idx[i]].truelabel;
    Z_sorted->sample[z_idx].weight = Z->sample[bg_idx[i]].weight;
    Z_sorted->nclasses = Z->nclasses;
    z_idx++;
  }
  for (size_t i = 0; i < n_fg_filters; i++) {
    Z_sorted->sample[z_idx].id = Z->sample[fg_idx[i]].id;
    Z_sorted->sample[z_idx].truelabel = Z->sample[fg_idx[i]].truelabel;
    Z_sorted->sample[z_idx].weight = Z->sample[fg_idx[i]].weight;
    Z_sorted->nclasses = Z->nclasses;
    z_idx++;
  }

  // Effective iterates over sorted dataset and read features
  for (size_t i = 0; i < Z_sorted->nsamples; i++) {
    iftVoxel u;
    if (iftIs3DMImage(mimg)) {
      u = GetVoxelCoord(
        (int)(mimg->xsize * scale[0]),
        (int)(mimg->ysize * scale[1]),
        (int)(mimg->zsize * scale[2]),
        Z_sorted->sample[i].id
      );
    }
    else {
      u = GetVoxelCoord(
        (int)(mimg->xsize * scale[0]),
        (int)(mimg->ysize * scale[1]),
        1,
        Z_sorted->sample[i].id
      );
    }
    u.x = u.x / scale[0];
    u.y = u.y / scale[1];
    u.z = u.z / scale[2];

    int j = 0;
    for (int k = 0; k < A->n; k++) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, k);
      if (iftMValidVoxel(mimg, v))
      {
        int q = iftMGetVoxelIndex(mimg, v);
        for (int b = 0; b < mimg->m; b++) {
          Z_sorted->sample[i].feat[j] = mimg->val[q][b];
          j++;
        }
      }
      else {
        for (int b = 0; b < mimg->m; b++) {
          Z_sorted->sample[i].feat[j] = 0;
          j++;
        }
      }
    }
  }

  free(bg_idx);
  free(fg_idx);
  iftSetStatus(Z_sorted, IFT_TRAIN);
  iftAddStatus(Z_sorted, IFT_SUPERVISED);
  iftDestroyDataSet(&Z);
  return Z_sorted;
}

void SaveBias(char *basepath, float *bias, int number_of_kernels)
{
  char filename[200];
  FILE *fp;

  sprintf(filename, "%s-bias.txt", basepath);
  fp = fopen(filename, "w");
  fprintf(fp, "%d\n", number_of_kernels);
  for (int k = 0; k < number_of_kernels; k++)
  {
    fprintf(fp, "%f ", bias[k]);
  }
  fclose(fp);
}

void SaveKernelWeights(char *basepath, int *truelabel, int number_of_kernels)
{
  char filename[200];
  FILE *fp;
  float pw = 1, nw = -1;

  sprintf(filename, "%s-weights.txt", basepath);
  fp = fopen(filename, "w");
  fprintf(fp, "%d\n", number_of_kernels);
  for (int k = 0; k < number_of_kernels; k++)
  {
    if (truelabel[k] == 2)
      fprintf(fp, "%f ", pw);
    else
      fprintf(fp, "%f ", nw);
  }
  fclose(fp);
}

int main(int argc, char **argv) {
  timer *tstart;
  int memory_start, memory_end;
  memory_start = iftMemoryUsed();
  tstart = iftTic();

  if (argc != 7) {
    iftError(
      "Usage: iftCreateLayerModel P1 P2 P3 P4\n"
      "P1: input folder with feature points (-fpts.txt)\n"
      "P2: input network architecture (.json)\n"
      "P3: input layer for patch definition (1, 2, 3, etc)\n"
      "P4: output folder with the models\n"
      "P5: number of filters for background\n"
      "P6: number of filters for foreground\n",
      "main"
    );
  }

  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(
    argv[1], "-fpts.txt", 1
  );
  iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
  int layer = atoi(argv[3]);

  char *filename = iftAllocCharArray(512);
  char *model_dir = argv[4];
  iftMakeDir(model_dir);

  /* Number of filters for each class (Only background and foreground) classes
  are supported. Background filters always come first.*/
  int n_bg_filters = atoi(argv[5]);
  int n_fg_filters = atoi(argv[6]);

  /*
    Extract patches from feature points of each image, compute filters, 
    biases, truelabel of the filters.
  */
  for (int i = 0; i < fs->n; i++) {
    char *basename = iftFilename(fs->files[i]->path, "-fpts.txt");
    sprintf(filename, "./layer%d/%s%s", layer - 1, basename, ".mimg");
    iftMImage *mimg = iftReadMImage(filename);
    iftLabeledSet *S = NULL;
    if (iftIs3DMImage(mimg)) {
      S = iftReadLabeledSet(fs->files[i]->path, 3);
    }
    else {
      S = iftReadLabeledSet(fs->files[i]->path, 2);
    }

    // Compute the scale factor w.r.t the input of the network
    float scale[3];
    sprintf(filename, "./layer0/%s.mimg", basename);
    iftMImage *input = iftReadMImage(filename);
    scale[0] = (float)input->xsize / (float)mimg->xsize;
    scale[1] = (float)input->ysize / (float)mimg->ysize;
    scale[2] = (float)input->zsize / (float)mimg->zsize;

    iftDestroyMImage(&input);

    // Extract Sorted Image Patches
    iftDataSet *Z = ExtractSortedImagePatches(
      mimg, S, arch->layer[layer - 1], scale, 
      n_bg_filters, n_fg_filters
    );
    iftNormalizeDataSetByZScoreInPlace(
      Z, NULL, arch->stdev_factor
    );
    iftMatrix *kernels = iftCreateMatrix(
      Z->nsamples, Z->nfeats
    );
    int *truelabel = iftAllocIntArray(Z->nsamples);

    // Compute filters and their truelabels
    for (int s = 0, col = 0; s < Z->nsamples; s++, col++) {
      iftUnitNorm(Z->sample[s].feat, Z->nfeats);
      truelabel[s] = Z->sample[s].truelabel;
      for (int row = 0; row < Z->nfeats; row++) {
        iftMatrixElem(kernels, col, row) = Z->sample[s].feat[row];
      }
    }

    // Compute biases and update filters
    float *bias = NULL;
    bias = iftAllocFloatArray(kernels->ncols);
    for (int col = 0; col < kernels->ncols; col++)
    {
      for (int row = 0; row < Z->nfeats; row++) {
        iftMatrixElem(kernels, col, row) =
          iftMatrixElem(kernels, col, row) / Z->fsp.stdev[row];
        bias[col] -= (
          Z->fsp.mean[row] * iftMatrixElem(kernels, col, row)
        );
      }
    }

    // Save kernels, biases, and truelabels of the kernels
    sprintf(filename, "%s/%s-conv%d-kernels.npy", model_dir, basename, layer);
    iftWriteMatrix(kernels, filename);
    sprintf(filename, "%s/%s-conv%d", model_dir, basename, layer);
    SaveBias(filename, bias, kernels->ncols);
    SaveKernelWeights(filename, truelabel, kernels->ncols);

    iftFree(bias);
    iftFree(truelabel);
    iftFree(basename);
    iftDestroyMImage(&mimg);
    iftDestroyMatrix(&kernels);
    iftDestroyLabeledSet(&S);
    iftDestroyDataSet(&Z);
  }

  // Releases resources
  iftDestroyFileSet(&fs);
  iftDestroyFLIMArch(&arch);
  iftFree(filename);

  puts("\nDone ...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
  memory_end = iftMemoryUsed();
  iftVerifyMemory(memory_start, memory_end);
}