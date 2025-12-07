#include "ift.h"

/*
  Author: Felipe Crispim da Rocha Salvagnini (August 08/05/2024)
  
  Given an input folder with Bag of Feature Points files and a json
  specifying the network architecture, it creates a layer. The user
  can select the number of foreground and background filters for
  the layer. Features are sorted using Cosine Similarity, and
  the best features are chosen. If the network has stride 
  (current layer input is smaller than the network input),
  it correctly scales the seed's coordinates.

  Arguments:

  - [1] Input folder with unsorted feature points
  - [2] Json network architecture
  - [3] Layer to create the filters
  - [4] Folder to save output model
  - [5] Number of filters for background
  - [6] Number of filters for foreground

  Execution
  iftCreateSortedLayerModelFromBoFP bofp/ ../arch.json 2 model_bofp_sorted 2 2
*/

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
    if (truelabel[k] == 1)
      fprintf(fp, "%f ", pw);
    else
      fprintf(fp, "%f ", nw);
  }
  fclose(fp);
}

/*
  Given a path to a Bag of Feature Points (BoFP) file and two M Images—layer 0
and layer N (target layer)—corresponding to the image described by the BoFP
file, this method reads the seeds and rescales them to the current layer domain
to guide the feature extraction.
*/
iftLabeledSet *iftReadAndRescaleLabeledSet(
  iftMImage *layer0, iftMImage *target_layer, char *bofp_path
) {
  iftLabeledSet *S = NULL;
  if (iftIs3DMImage(layer0)) {
    S = iftReadLabeledSet(bofp_path, 3);
  }
  else {
    S = iftReadLabeledSet(bofp_path, 2);
  }

  float scale[3];
  scale[0] = (float)target_layer->xsize / (float)layer0->xsize;
  scale[1] = (float)target_layer->ysize / (float)layer0->ysize;
  scale[2] = (float)target_layer->zsize / (float)layer0->zsize;

  if (scale[0] == 1 && scale[1] == 1 && scale[2] == 1) {
    return S;
  } else {
    iftLabeledSet *Saux = S, *Sunion=NULL;
    while (Saux != NULL)
    {
      int p = Saux->elem;
      iftVoxel v = iftMGetVoxelCoord(layer0, p);
      v.x = (int)((float)v.x * scale[0]);
      v.y = (int)((float)v.y * scale[1]);
      if (v.z != 1) {
        v.z = (int)((float)v.z * scale[2]);
      }
      int q = iftMGetVoxelIndex(target_layer, v);
      iftUnionLabeledSetElemMarkerAndHandicap(
        &Sunion, q, Saux->label, Saux->marker, Saux->handicap
      );
      Saux = Saux->next;
    }
  
    iftDestroyLabeledSet(&S);

    return Sunion;
  }
}

iftAdjRel *GetPatchAdjacency(iftMImage *mimg, iftFLIMLayer layer)
{
  iftAdjRel *A;

  if (iftIs3DMImage(mimg))
  {
    A = iftCuboidWithDilationForConv(
      layer.kernel_size[0],
      layer.kernel_size[1],
      layer.kernel_size[2],
      layer.dilation_rate[0],
      layer.dilation_rate[1],
      layer.dilation_rate[2]
    );
  }
  else
  {
    A = iftRectangularWithDilationForConv(
      layer.kernel_size[0],
      layer.kernel_size[1],
      layer.dilation_rate[0],
      layer.dilation_rate[1]
    );
  }

  return (A);
}

iftDataSet *LabeledSet2DataSet(
  iftLabeledSet *S, iftMImage *mimg, iftFLIMLayer layer
) {
  iftAdjRel *A = GetPatchAdjacency(mimg, layer);
  int n_feats = A->n * mimg->m;
  int n_samples = iftLabeledSetSize(S);
  int s = 0;
  iftLabeledSet *Saux = S;
  iftDataSet *Z = iftCreateDataSet(n_samples, n_feats);
  Z->ngroups = 0;
  Z->nclasses = 1;

  while (Saux != NULL) {
    Z->sample[s].id = Saux->elem;
    Z->sample[s].label = Saux->label;
    Z->sample[s].weight = (float)Saux->handicap;
    // Marker Label
    Z->sample[s].truelabel = Saux->marker;
    // N Groups (Markers)
    if (Z->sample[s].truelabel > Z->ngroups) {
      Z->ngroups = Z->sample[s].truelabel;
    }
    // N problem classes
    if (Z->sample[s].label >= Z->nclasses) {
      Z->nclasses++;
    }

    // Reads patch feature
    iftVoxel u = iftMGetVoxelCoord(mimg, Z->sample[s].id);
    int j = 0;
    for (int k = 0; k < A->n; k++) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, k);
      if (iftMValidVoxel(mimg, v)) {
        int q = iftMGetVoxelIndex(mimg, v);
        for (int b = 0; b < mimg->m; b++) {
          Z->sample[s].feat[j] = mimg->val[q][b];
          j++;
        }
      }
      else {
        for (int b = 0; b < mimg->m; b++) {
          Z->sample[s].feat[j] = 0;
          j++;
        }
      }
    }

    s++;
    Saux = Saux->next;
  }

  iftDestroyAdjRel(&A);
  iftSetStatus(Z, IFT_TRAIN);
  iftAddStatus(Z, IFT_SUPERVISED);

  return Z;
}

void SetPatchesWeight(iftDataSet *patches) {
  iftDoubleMatrix *M_cosine = iftCreateDoubleMatrix(
    patches->nsamples, patches->nsamples
  );

  for (size_t row = 0; row < patches->nsamples; row++) {
    for (size_t column = 0; column < patches->nsamples; column++) {
      float cosine = iftCosineDistance(
        patches->sample[row].feat,
        patches->sample[column].feat,
        patches->nfeats
      );
      iftMatrixElem(M_cosine, column, row) = cosine;
    }
  }

  float *intra_dist = (float *) calloc(patches->nsamples, sizeof(float));
  float *inter_dist = (float *) calloc(patches->nsamples, sizeof(float));
  float *combined_dist = (float *) calloc(patches->nsamples, sizeof(float));
  int *n_samples_by_class = (int *) calloc(patches->nclasses, sizeof(int));

  // Iterates over each patch computing its distance to other patches (intra and inter)
  for (size_t i=0; i < patches->nsamples; i++) {
    // Increases sample counting for each class, for sorting
    n_samples_by_class[patches->sample[i].label]++;
    for (size_t j=0; j < patches->nsamples; j++) {
      if (i == j) {
        continue;
      }
      // Currently 
      float distance = iftMatrixElem(M_cosine, j, i);
      if (patches->sample[i].label != patches->sample[j].label) {
        inter_dist[i] += distance;
      } else {
        intra_dist[i] += distance;
      }
    }
  }

  // Merge distance, with more weight to inter class distance
  for (size_t i = 0; i < patches->nsamples; i++) {
    combined_dist[i] = (2 * inter_dist[i] + intra_dist[i]) / 3;
  }

  // For each class, set patches weight
  for (size_t c = 0; c < patches->nclasses; c++) 
  {
    int *sorted_indexes = (int *)malloc(n_samples_by_class[c] * sizeof(int));
    memset(sorted_indexes, -1, n_samples_by_class[c] * sizeof(int));
    for (size_t i = 0; i < patches->nsamples; i++) 
    {
      if (patches->sample[i].label != c) {
        continue;
      }
      if (sorted_indexes[0] == -1) {
        sorted_indexes[0] = i;
      } else {
        int aux = 0, tmp;
        float dist_to_insert = combined_dist[i];
        int index_to_insert = i;
        while (aux < n_samples_by_class[c])
        {
          if (dist_to_insert > combined_dist[sorted_indexes[aux]]) 
          {
            dist_to_insert = combined_dist[sorted_indexes[aux]];
            tmp = sorted_indexes[aux];
            sorted_indexes[aux] = index_to_insert;
            index_to_insert = tmp;
            aux++;
            if (sorted_indexes[aux] == -1) 
            {
              sorted_indexes[aux] = index_to_insert;
              break;
            }
          }
          else 
          {
            aux++;
          }
        }
      } 
    }
    // Iterate through indexes, seting the weights
    for (size_t score = 0; score < n_samples_by_class[c]; score++) {
      patches->sample[sorted_indexes[score]].weight = (float) (score + 1);
    }
    free(sorted_indexes);
  }

  free(intra_dist);
  free(inter_dist);
  free(combined_dist);
  free(n_samples_by_class);
  iftDestroyDoubleMatrix(&M_cosine);
}

iftDataSet *GetSortedImagePatches(
  // Change better names
  iftDataSet *patches,
  iftDataSet *patches_orig, int n_bg_filters, int n_fg_filters
) {
  int *bg_idx = calloc(n_bg_filters, sizeof(int));
  int *fg_idx = calloc(n_bg_filters, sizeof(int));

  // Stores sorted indexes
  for (size_t i = 0; i < patches->nsamples; i++) {
    // Background
    if (patches->sample[i].label == 0) {
      if (patches->sample[i].weight <= n_bg_filters) {
        bg_idx[(int)patches->sample[i].weight - 1] = i;
      }
    }
    // Foreground
    else if (patches->sample[i].label == 1) {
      if (patches->sample[i].weight <= n_fg_filters) {
        fg_idx[(int)patches->sample[i].weight - 1] = i;
      }
    }
  }

  // Reads data into final dataset, also reading patches
  size_t idx = 0;
  iftDataSet *sorted = iftCreateDataSet(
    n_bg_filters + n_fg_filters, patches->nfeats
  );
  sorted->ngroups = patches->ngroups;
  sorted->nclasses = patches->nclasses;
  for (size_t i = 0; i < n_bg_filters; i++) {
    sorted->sample[idx].id = patches->sample[bg_idx[i]].id;
    sorted->sample[idx].group = patches->sample[bg_idx[i]].group;
    sorted->sample[idx].label = patches->sample[bg_idx[i]].label;
    sorted->sample[idx].truelabel = patches->sample[bg_idx[i]].truelabel;
    sorted->sample[idx].weight = patches->sample[bg_idx[i]].weight;
    memcpy(
      sorted->sample[idx].feat,
      patches_orig->sample[bg_idx[i]].feat,
      patches->nfeats * sizeof(float)
    );
    idx++;
  }
  for (size_t i = 0; i < n_fg_filters; i++) {
    sorted->sample[idx].id = patches->sample[fg_idx[i]].id;
    sorted->sample[idx].group = patches->sample[fg_idx[i]].group;
    sorted->sample[idx].label = patches->sample[fg_idx[i]].label;
    sorted->sample[idx].truelabel = patches->sample[fg_idx[i]].truelabel;
    sorted->sample[idx].weight = patches->sample[fg_idx[i]].weight;
    memcpy(
      sorted->sample[idx].feat,
      patches_orig->sample[fg_idx[i]].feat,
      patches->nfeats * sizeof(float)
    );
    idx++;
  }

  // // Computes mean and std-dev of features
  // sorted->fsp.mean   = iftAllocFloatArray(sorted->nfeats);
  // sorted->fsp.stdev  = iftAllocFloatArray(sorted->nfeats);
  // sorted->fsp.nfeats = sorted->nfeats;
  // for (size_t s_idx = 0; s_idx < sorted->nsamples; s_idx++)
  // {
  //   for (size_t f_idx = 0; f_idx < sorted->nfeats; f_idx++)
  //   {
  //     sorted->fsp.mean[f_idx] += sorted->sample[s_idx].feat[f_idx];
  //   }
  // }
  // // Computes mean
  // for (size_t f_idx = 0; f_idx < sorted->nfeats; f_idx++) 
  // {
  //   sorted->fsp.mean[f_idx] /= sorted->nsamples;
  // }
  // for (size_t s_idx = 0; s_idx < sorted->nsamples; s_idx++) 
  // {
  //   for (size_t f_idx = 0; f_idx < sorted->nfeats; f_idx++) 
  //   {
  //     sorted->fsp.stdev[f_idx] += powf(
  //       sorted->sample[s_idx].feat[f_idx] - sorted->fsp.mean[f_idx],
  //       2
  //     );
  //   }
  // }
  // // Computes std-dev
  // for (size_t f_idx = 0; f_idx < sorted->nfeats; f_idx++) 
  // {
  //   sorted->fsp.stdev[f_idx] = sqrtf(
  //     sorted->fsp.stdev[f_idx] / sorted->nsamples
  //   ) + stdev_factor;
  // }
  
  iftSetStatus(sorted, IFT_TRAIN);
  iftAddStatus(sorted, IFT_SUPERVISED);

  return sorted;
}

int main(int argc, char **argv) {
  timer *tstart;
  int memory_start, memory_end;
  memory_start = iftMemoryUsed();
  tstart = iftTic();

  if (argc != 7) {
    iftError(
      "Usage: iftCreateLayerModel P1 P2 P3 P4\n"
      "P1: input folder with unsorted feature points (-fpts.txt)\n"
      "P2: input network architecture (.json)\n"
      "P3: input layer for patch definition (1, 2, 3, etc) - It uses the previous layer\n"
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

  /* Extract patches from feature points of each image, compute filters, 
  biases, truelabel of the filters. */
  for (int i = 0; i < fs->n; i++) {
    char *basename = iftFilename(fs->files[i]->path, "-fpts.txt");
    printf("[INFO] Creating Sorted Layer for %s\n", basename);
    // m image of previous layer will be used for patch definition
    sprintf(filename, "./layer%d/%s%s", layer - 1, basename, ".mimg");
    iftMImage *mimg = iftReadMImage(filename);
    // Avoids reading layer0 twice
    iftMImage *layer0;
    if (layer != 1) {
      sprintf(filename, "./layer0/%s.mimg", basename);
      layer0 = iftReadMImage(filename);
    } else {
      layer0 = mimg;
    }
    
    /* The Labeled Set must be read and re-scaled (If necessary, .e.g.,
    the previous layers had stride)
    */
    iftLabeledSet *S = iftReadAndRescaleLabeledSet(
      layer0, mimg, fs->files[i]->path
    );
    if (layer != 1) {
      iftDestroyMImage(&layer0);
    }

    // 1. Get Image Patches
    iftDataSet *Z = LabeledSet2DataSet(
      S, mimg, arch->layer[layer - 1]
    );
    
    /* 2. Normalize image patches (iftNormalizeDataSetByZScoreInPlace) and/or
    iftNormOneDataSet.
    Experiment with/without using iftNormOneDataSet before sort */
    iftDataSet *Z_aux = iftCopyDataSet(Z, true);
    iftNormalizeDataSetByZScoreInPlace(
      Z_aux, NULL, arch->stdev_factor
    );
    iftDataSet *Z_norm = iftNormOneDataSet(Z_aux);
    
    // 3. Compute Weights for Image Patches
    SetPatchesWeight(Z_norm);

    // 4. Select sorted image patches
    iftDataSet *Z_sorted = GetSortedImagePatches(
      Z_norm, Z, n_bg_filters, n_fg_filters
    );
    iftNormalizeDataSetByZScoreInPlace(
      Z_sorted, NULL, arch->stdev_factor
    );

    // 5. Save Layer Parameters (Kernels, Biases, and Labels)
    iftMatrix *kernels = iftCreateMatrix(
      Z_sorted->nsamples, Z_sorted->nfeats
    );
    int *label = iftAllocIntArray(Z_sorted->nsamples);

    // Compute filters and their labels
    for (size_t s = 0, col = 0; s < Z_sorted->nsamples; s++, col++) {
      iftUnitNorm(Z_sorted->sample[s].feat, Z_sorted->nfeats);
      label[s] = Z_sorted->sample[s].label;
      for (size_t row = 0; row < Z_sorted->nfeats; row++) {
        iftMatrixElem(kernels, col, row) = Z_sorted->sample[s].feat[row];
      }
    }

    float *bias = NULL;
    bias = iftAllocFloatArray(kernels->ncols);
    for (int col = 0; col < kernels->ncols; col++) {
      for (int row = 0; row < Z_sorted->nfeats; row++) {
        iftMatrixElem(kernels, col, row) =
          iftMatrixElem(kernels, col, row) / Z_sorted->fsp.stdev[row];
        bias[col] -= (
          Z_sorted->fsp.mean[row] * iftMatrixElem(kernels, col, row)
        );
      }
    }
    sprintf(filename, "%s/%s-conv%d-kernels.npy", model_dir, basename, layer);
    iftWriteMatrix(kernels, filename);
    sprintf(filename, "%s/%s-conv%d", model_dir, basename, layer);
    SaveBias(filename, bias, kernels->ncols);
    SaveKernelWeights(filename, label, kernels->ncols);

    // 6. Releases resources
    iftDestroyMImage(&mimg);
    iftDestroyLabeledSet(&S);
    iftDestroyDataSet(&Z_norm);
    iftDestroyDataSet(&Z_aux);
    iftDestroyDataSet(&Z);
    iftDestroyDataSet(&Z_sorted);
    iftDestroyMatrix(&kernels);
    iftFree(basename);
    iftFree(label);
    iftFree(bias);
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