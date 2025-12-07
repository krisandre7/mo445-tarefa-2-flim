#include "ift.h"

iftImage *iftGetMajorityClassmap(iftImage *classmap, iftImage *regions) {
  int nregions, nobjs;
  int **mat_regions_objs, *region_labels;
  iftImage *output;
  output = iftCreateImage(classmap->xsize, classmap->ysize, classmap->zsize);
  nregions = iftMaximumValue(regions);
  nobjs = iftMaximumValue(classmap);

  mat_regions_objs = (int **) calloc(nregions, sizeof(int *));
  for (int i = 0; i < nregions; ++i)
    mat_regions_objs[i] =  iftAllocIntArray(nobjs);

  // fill matrix regions vs objs
  for (int p = 0; p < regions->n; ++p) {
    int index_region = regions->val[p] - 1;
    int index_obj = classmap->val[p] - 1;
    mat_regions_objs[index_region][index_obj]++;
  }

  // assign majority label
  region_labels = iftAllocIntArray(nregions);
  for (int i = 0; i < nregions; ++i) {
    int index_best_obj = 0;
    int best_npixels = -1;
    for (int j = 0; j < nobjs; ++j) {
      if (mat_regions_objs[i][j] > best_npixels) {
        best_npixels = mat_regions_objs[i][j];
        index_best_obj = j;
      }
    }
    region_labels[i] = index_best_obj + 1;
  }

  for (int p = 0; p < output->n; ++p) {
    int index_region = regions->val[p] - 1;
    output->val[p] = region_labels[index_region];
  }

  // free
  for (int i = 0; i < nregions; ++i)
    free(mat_regions_objs[i]);

  free(mat_regions_objs);
  free(region_labels);
  return output;
}

int main(int argc, char *argv[])
{
  iftImage *classmap, *regions, *output;
  
  if (argc != 4)
    iftError("Usage: iftPrintImageInfo <classification_pixel_level.pgm> <regions.pgm> <output.pgm>", "main");  
  
  classmap = iftReadImageByExt(argv[1]);
  regions = iftReadImageByExt(argv[2]);
  assert(classmap->n == regions->n);

  output = iftGetMajorityClassmap(classmap, regions);
  iftWriteImageP2(output, argv[3]);

  iftDestroyImage(&classmap);
  iftDestroyImage(&regions);
  iftDestroyImage(&output);
}
