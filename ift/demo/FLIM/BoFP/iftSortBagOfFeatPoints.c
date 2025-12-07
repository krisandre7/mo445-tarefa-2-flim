#include "ift.h"

#define COLOR_SPACE GRAY_CSPACE
// Change to 1 to enable debugging across the program
#define IFT_DEBUG 0

/*
  Author: Felipe Crispim da Rocha Salvagnini (July 31/07/2024)

  Given an Input Bag of Feature points, we extract image patches and sort each
  for each class. Given Euclidean and cosine metrics, patches are sorted by
  considering inter- and intra-class distance (Currently, using only cosine
  provides better results). Patches are sorted for each class separately,
  from 1 to N (N is the number of feature points for a given class), where
  the sorted value is saved as the seed handicap.

  Arguments:

  - [1] Input Images Folder
  - [2] Input folder with Bag of Feature Points (Unsorted)
  - [3] Output folder to save Bag of Feature Points (Sorted)
  - [4] Adjacency radius of patches (Spheric or Circular)

  Sample of execution:

  iftSortBagOfFeatPoints \
  ~/msc/3_repos/msc_repo/experiments/8_flim_bofp_sorting/0_data/flair_std/ \
  ~/msc/3_repos/msc_repo/experiments/8_flim_bofp_sorting/0_data/bofp_unsorted_few/ \
  ~/msc/3_repos/msc_repo/experiments/8_flim_bofp_sorting/0_data/bofp_sorted_few \
  1
*/

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

// [Corrigir isso aqui, ler arch json e criar filtros a partir da arquitetura] Validar
// Automatizar a geração dos reports, para rodar para todos os dados (configurações de filtros) e atualizar planilhas
// Deixar certinho para rodar para parasitas também
// Deixar certinho para rodar para os dados do pulmao
// Comitar
// Avisar Gilsons
iftDataSet *GetPatches(
  iftImage *img, iftLabeledSet *S, iftFLIMArch *arch
) {
  iftMImage *m_img = iftImageToMImage(img, COLOR_SPACE);
  iftAdjRel *A = GetPatchAdjacency(m_img, arch->layer[0]);
  int n_samples = iftLabeledSetSize(S);
  int patches_size = m_img->m * A->n;
  iftDataSet *patches = iftCreateDataSet(n_samples, patches_size);

  patches->ngroups = 0;
  patches->nclasses = 1;
  int seed_idx = 0;
  iftLabeledSet *seed = S;
  while (seed != NULL)
  {
    int p = seed->elem;
    patches->sample[seed_idx].id = seed->elem;
    patches->sample[seed_idx].label = seed->label;
    // Marker label
    patches->sample[seed_idx].truelabel = seed->marker;
    // N groups (Markers)
    if (patches->sample[seed_idx].truelabel > patches->ngroups) {
      patches->ngroups = patches->sample[seed_idx].truelabel;
    }
    // N problem classes
    if (patches->sample[seed_idx].label >= patches->nclasses) {
      patches->nclasses++;
    }
    iftVoxel u = iftMGetVoxelCoord(m_img, p);
    int j = 0;
    for (size_t adj_idx = 0; adj_idx < A->n; adj_idx++) {
      iftVoxel v = iftGetAdjacentVoxel(A, u, adj_idx);
      if (iftMValidVoxel(m_img, v)) {
        int q = iftMGetVoxelIndex(m_img, v);
        for (int b = 0; b < m_img->m; b++) {
          patches->sample[seed_idx].feat[j] = m_img->val[q][b];
          j++;
        }
      }
      else {
        for (int b = 0; b < m_img->m; b++) {
          patches->sample[seed_idx].feat[j] = 0;
          j++;
        }
      }
    }
    seed_idx++;
    seed = seed->next;
  }

  iftDestroyAdjRel(&A);
  iftDestroyMImage(&m_img);

  iftSetStatus(patches, IFT_TRAIN);
  iftAddStatus(patches, IFT_SUPERVISED);

  return patches;
}

void WriteDistanceMatrix(
  iftDoubleMatrix *M, char *m_path, iftDataSet *patches
) {
  // Writes header
  FILE *fp = fopen(m_path, "w");
  fprintf(fp, "col/row elem;");
  for (size_t column = 0; column < patches->nsamples; column++) {
    fprintf(fp, "%d;", patches->sample[column].id);
  }
  fprintf(fp, "\n");
  for (size_t row = 0; row < patches->nsamples; row++) {
    for (size_t column = 0; column <= patches->nsamples; column++) {
      if (column == 0) {
        fprintf(fp, "%d;", patches->sample[row].id);
      } else {
        fprintf(fp, "%f;", iftMatrixElem(M, column - 1, row));
      }
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void GenerateSortedBoFP(
  iftDoubleMatrix *M_euclidean, iftDoubleMatrix *M_cosine, iftDataSet *patches,
  char *file_path, iftImage *img
) {
  float *intra_dist = (float *) calloc(patches->nsamples, sizeof(float));
  float *inter_dist = (float *) calloc(patches->nsamples, sizeof(float));
  float *combined_dist = (float *) calloc(patches->nsamples, sizeof(float));
  int *n_samples_by_class = (int *) calloc(patches->nclasses, sizeof(int));

  for (size_t i=0; i < patches->nsamples; i++) {
    // Increases sample couting for each class, for sorting
    n_samples_by_class[patches->sample[i].label]++;
    for (size_t j=0; j < patches->nsamples; j++) {
      if (i == j) {
        continue;
      }
      // Currently, using 
      float weighted_distance = iftMatrixElem(M_cosine, j, i) ;
                                // * iftMatrixElem(M_euclidean, j, i);
      if (patches->sample[i].label != patches->sample[j].label) {
        inter_dist[i] += weighted_distance;
      } else {
        intra_dist[i] += weighted_distance;
      }
    }
  }

  // Merge distance, with more weight to inter classes distance
  for (size_t i = 0; i < patches->nsamples; i++) {
    combined_dist[i] = (2 * inter_dist[i] + intra_dist[i]) / 3;
  }
  
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
      patches->sample[sorted_indexes[score]].weight = score + 1;
    }
    free(sorted_indexes);
  }

  iftLabeledSet *S = NULL;
  // Generates labeled set to save it
  for (size_t i = 0; i < patches->nsamples; i++) {
    iftInsertLabeledSetMarkerAndHandicap(
      &S, patches->sample[i].id, patches->sample[i].label,
      patches->sample[i].truelabel,
      patches->sample[i].weight
    );
  }
  
  iftWriteLabeledSet(S, img->xsize, img->ysize, img->zsize, file_path);

  iftDestroyLabeledSet(&S);
  free(intra_dist);
  free(inter_dist);
  free(combined_dist);
  free(n_samples_by_class);
}

void ComputesDistanceAndSort(
  iftDataSet *patches, char *file_basename, iftImage *img
) {
  iftDoubleMatrix *M_euclidean = iftCreateDoubleMatrix(
    patches->nsamples, patches->nsamples
  );
  iftDoubleMatrix *M_cosine = iftCreateDoubleMatrix(
    patches->nsamples, patches->nsamples
  );
  iftDataSet *patches_norm = iftNormOneDataSet(patches);

  for (size_t row = 0; row < patches->nsamples; row++) {
    for (size_t column = 0; column < patches->nsamples; column++) {
      float euclidean = iftEuclideanDistance(
        patches_norm->sample[row].feat,
        patches_norm->sample[column].feat,
        patches_norm->nfeats
      );
      float cosine = iftCosineDistance(
        patches_norm->sample[row].feat,
        patches_norm->sample[column].feat,
        patches_norm->nfeats
      );

      iftMatrixElem(M_euclidean, column, row) = euclidean;
      iftMatrixElem(M_cosine, column, row) = cosine;
    }
  }

  // Writes matrices
  char file_path[512];
  if (IFT_DEBUG) {
    // If debugging writes down the matrices as CSV
    sprintf(file_path, "%s-%s.csv", file_basename, "euclidean");
    WriteDistanceMatrix(M_euclidean, file_path, patches);
    sprintf(file_path, "%s-%s.csv", file_basename, "cosine");
    WriteDistanceMatrix(M_cosine, file_path, patches);              
  }
  sprintf(file_path, "%s-fpts.txt", file_basename);

  GenerateSortedBoFP(M_euclidean, M_cosine, patches_norm, file_path, img);

  iftDestroyDoubleMatrix(&M_euclidean);
  iftDestroyDoubleMatrix(&M_cosine);
  iftDestroyDataSet(&patches_norm);
}

int main(int argc, char **argv) {
  timer *tstart;
  int memory_start, memory_end;
  memory_start = iftMemoryUsed();
  tstart = iftTic();
  
  if (argc != 5) {
    iftError(
      "Usage: iftSortBagOfFeatPoints <P1> <P2> <P3>\n"
      "P1: image folder \n"
      "P2: Bag Of Feature Points Folder \n"
      "P3: OutPut Folder (For Sorted Feature Points) \n"
      // The first layer configurations are used to extract and sort patches
      "P4: input network architecture (.json)\n",
      "main");
  }

  iftFileSet *fs;
  fs = iftLoadFileSetFromDir(argv[1], 1);
  char extension[10];
  sprintf(extension, "%s", iftFileExt(fs->files[0]->path));
  iftDestroyFileSet(&fs);

  // Loads feature points
  fs = iftLoadFileSetFromDirBySuffix(
    argv[2], "-fpts.txt", 1
  );
  iftFLIMArch *arch = iftReadFLIMArch(argv[4]);
  char *filename = iftAllocCharArray(512);
  for (size_t i=0; i < fs->n; i++) {
    char *basename = iftFilename(
      fs->files[i]->path, "-fpts.txt"
    );
    printf(
      "[%ld/%ld] Processing file %s\n",
      i+1, fs->n, basename
    );
    sprintf(filename, "%s/%s%s", argv[1], basename, extension);
    iftImage *img = iftReadImageByExt(filename);
    iftLabeledSet *S = iftReadSeeds(img, fs->files[i]->path);
    iftLabeledSet *S_aux = S;
    if (IFT_DEBUG) {
      iftVoxel v;
      while (S_aux != NULL)
      {
        v = iftGetVoxelCoord(img, S_aux->elem);
        printf(
          "[SEED DATA] coord %d %d %d | marker %d | label %d | handicap %d\n",
          v.x, v.y, v.z, S_aux->marker, S_aux->label, S_aux->handicap
        );
        S_aux = S_aux->next;
      }
    }

    // Extracts patches dataset
    iftDataSet *patches = GetPatches(img, S, arch);
    iftNormalizeDataSetByZScoreInPlace(patches, NULL, 0);
    iftMakeDir(argv[3]);
    sprintf(filename, "%s/%s", argv[3], basename);
    ComputesDistanceAndSort(patches, filename, img);

    iftFree(basename);
    iftDestroyImage(&img);
    iftDestroyLabeledSet(&S);
    iftDestroyDataSet(&patches);
  }
  iftDestroyFileSet(&fs);
  iftDestroyFLIMArch(&arch);
  iftFree(filename);

  puts("\nDone ...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
  memory_end = iftMemoryUsed();
  iftVerifyMemory(memory_start, memory_end);
}