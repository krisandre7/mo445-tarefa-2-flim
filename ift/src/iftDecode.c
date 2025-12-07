#include "ift.h"
#include "iftDecode.h"

// Private methods -------------------------------------------------------------

int *LoadKernelLabels(char *filename)
{
  int number_of_kernels;
  FILE *fp;
  int *truelabels;

  fp = fopen(filename, "r");
  fscanf(fp, "%d", &number_of_kernels);
  truelabels = iftAllocIntArray(number_of_kernels);
  for (int k = 0; k < number_of_kernels; k++)
  {
    fscanf(fp, "%d ", &truelabels[k]);
  }
  fclose(fp);

  return (truelabels);
}

int NumberOfTruelabels(int *truelabels, int n)
{
  int nlabels = 0;
  for (int i = 0; i < n; i++)
    if (truelabels[i] > nlabels)
      nlabels = truelabels[i];
  return (nlabels);
}

float *AdaptiveWeightsOtsu(iftMImage *mimg)
{
  float *weight = iftAllocFloatArray(mimg->m);
  float *mean = iftAllocFloatArray(mimg->m);
  float *maxval = iftAllocFloatArray(mimg->m);

  /* Normalize activations */
  for (int b = 0; b < mimg->m; b++)
  {
    maxval[b] = iftMMaximumValue(mimg, b);
  }

  // #pragma omp parallel for shared(mimg, maxval)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      if (maxval[b] > 0.0)
        mimg->val[p][b] /= maxval[b];
      else
        mimg->val[p][b] = 0.0;
    }
  }
  iftFree(maxval);

  /* Estimate the mean activation per channel */

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      mean[b] += mimg->val[p][b];
    }
  }
// Private methods
  // Just so it is easier to compute the otsu
  iftImage *means_image = iftCreateImage(mimg->m, 1, 1);
  float max_mean = 0.0;
  for (int b = 0; b < mimg->m; b++)
  {
    mean[b] /= mimg->n;
    means_image->val[b] = mean[b] * 255;
    if (means_image->val[b] > max_mean)
      max_mean = means_image->val[b];
  }
  if (max_mean > 0.0)
  {

    /* Estimate mean and standard deviation of the above mean activations */

    float mu = 0.0, stdev = 0.0;
    for (int b = 0; b < mimg->m; b++)
    {
      mu += mean[b];
    }
    mu /= mimg->m;

    int means_otsu = iftOtsu(means_image);

    for (int b = 0; b < mimg->m; b++)
    {
      stdev += (mean[b] - mu) * (mean[b] - mu);
    }
    stdev = sqrtf(stdev / mimg->m) * sqrtf(stdev / mimg->m);

    float low_end, high_end;
    if (means_otsu > 0.1)
      low_end = means_otsu - stdev;
    else
      low_end = means_otsu;
    high_end = means_otsu + stdev;

    /* Estimate the weight of each channel */

    for (int b = 0; b < mimg->m; b++)
    {
      iftImage *band = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
      for (int p = 0; p < band->n; p++)
        band->val[p] = mimg->val[p][b] * 255;
      float fg_proportion = 0;
      int max_band = 0.0;
      for (int p = 0; p < band->n; p++)
      {
        if (band->val[p] > max_band)
          max_band = band->val[p];
      }
      if (max_band > 0.0)
      {
        int otsu_band = iftOtsu(band);

        for (int p = 0; p < band->n; p++)
        {
          if (band->val[p] >= otsu_band)
            fg_proportion += 1;
        }
        fg_proportion /= band->n;
        if (mean[b] >= high_end && fg_proportion > 0.2)
        {
          weight[b] = -1.0;
        }
        else if (mean[b] <= low_end && fg_proportion < 0.1)
        {
          weight[b] = +1.0;
        }
        else
        {
          weight[b] = 0.0;
        }
      }
      else
      {
        weight[b] = 0.0;
      }
    }
  }
  else
  {
    for (int b = 0; b < mimg->m; b++)
      weight[b] = 0.0;
  }

  iftFree(mean);
  iftDestroyImage(&means_image);

  return (weight);
}

iftFImage *AdaptiveDecoder2(iftMImage *mimg, float adj_radius, float *channel_label)
{
  iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
  float *weight = iftAllocFloatArray(mimg->m);
  float *maxval = iftAllocFloatArray(mimg->m);
  int nobj = 0, nbkg = 0;

  /* Normalize activations per channel */

  for (int b = 0; b < mimg->m; b++)
  {
    maxval[b] = iftMMaximumValue(mimg, b);
    if (channel_label[b] > 0)
    { /* object channel */
      nobj++;
    }
    else
    { /* background channel */
      nbkg++;
    }
  }

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      if (maxval[b] > IFT_EPSILON)
        mimg->val[p][b] = mimg->val[p][b] / maxval[b];
    }
  }
  iftFree(maxval);

  /* Estimate saliency of each pixel */

  iftAdjRel *A = NULL;
  if (iftIs3DMImage(mimg))
    A = iftSpheric(adj_radius);
  else
    A = iftCircular(adj_radius);

  for (int p = 0; p < mimg->n; p++)
  {

    /* estimate mean and stdev of activations across channels */

    float mean[2] = {0.0, 0.0}, var[2] = {0.0, 0.0};
    iftVoxel u = iftMGetVoxelCoord(mimg, p);

    for (int i = 0; i < A->n; i++)
    {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v))
      {
        int q = iftMGetVoxelIndex(mimg, v);

        for (int b = 0; b < mimg->m; b++)
        {
          if (channel_label[b] > 0)
          { /* object channel */
            mean[0] += mimg->val[q][b];
          }
          else
          { /* background channel */
            mean[1] += mimg->val[q][b];
          }
        }
      }
    }
    mean[0] /= (nobj * A->n);
    mean[1] /= (nbkg * A->n);

    for (int i = 0; i < A->n; i++)
    {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v))
      {
        int q = iftMGetVoxelIndex(mimg, v);

        for (int b = 0; b < mimg->m; b++)
        {
          if (channel_label[b] > 0)
          { /* object channel */
            var[0] += powf(mimg->val[q][b] - mean[0], 2);
          }
          else
          { /* background channel */
            var[1] += powf(mimg->val[q][b] - mean[1], 2);
          }
        }
      }
    }
    var[0] = var[0] / (nobj * A->n);
    var[1] = var[1] / (nbkg * A->n);

    /* It is expected two Gaussian distributions Po and Pb of
       activations around mean[0] and mean[1]. We may expect that
       object pixels generate Po >> Pb and the other way around for
       background pixels. */

    float Po, Pb;
    for (int b = 0; b < mimg->m; b++)
    {
      if (var[0] > IFT_EPSILON)
        Po = exp(-powf(mimg->val[p][b] - mean[0], 2) / (2.0 * var[0]));
      else
      {
        var[0] = 1.0;
        Po = exp(-powf(mimg->val[p][b] - mean[0], 2) / (2.0 * var[0]));
      }
      if (var[1] > IFT_EPSILON)
        Pb = exp(-powf(mimg->val[p][b] - mean[1], 2) / (2.0 * var[1]));
      else
      {
        var[1] = 1.0;
        Pb = exp(-powf(mimg->val[p][b] - mean[1], 2) / (2.0 * var[1]));
      }

      if ((channel_label[b] > 0)&&(Po > Pb))
      {
        weight[b] = 1; // Po;
      }
      else
      {
        if ((channel_label[b] < 0)&&(Po < Pb))
        {
          weight[b] = -1; // Pb
        }
        else
        {
          weight[b] = 0;
        }
      }
    }

    /* Estimate the salience map using one neuron per pixel -- it is
       no longer a point-wise convolution */

    for (int b = 0; b < mimg->m; b++)
    {
      salie->val[p] += mimg->val[p][b] * weight[b];
    }
    if (salie->val[p] < 0) /* RELU (Sigmoid?) */
      salie->val[p] = 0;
  }
  iftDestroyAdjRel(&A);
  iftFree(weight);

  return (salie);
}

iftFImage *AdaptiveDecoder3(iftMImage *mimg, float adj_radius, float *channel_label)
{
  iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);
  float *weight = iftAllocFloatArray(mimg->m);
  float *maxval = iftAllocFloatArray(mimg->m);
  int nobj = 0, nbkg = 0;

  /* Normalize activations per channel */

  for (int b = 0; b < mimg->m; b++)
  {
    maxval[b] = iftMMaximumValue(mimg, b);
    if (channel_label[b] > 0)
    { /* object channel */
      nobj++;
    }
    else
    { /* background channel */
      nbkg++;
    }
  }

  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      if (maxval[b] > IFT_EPSILON)
        mimg->val[p][b] = mimg->val[p][b] / maxval[b];
    }
  }
  iftFree(maxval);

  /* Estimate saliency of each pixel */

  iftAdjRel *A = NULL;
  if (iftIs3DMImage(mimg))
    A = iftSpheric(adj_radius);
  else
    A = iftCircular(adj_radius);

  for (int p = 0; p < mimg->n; p++)
  {

    /* estimate the means of activations across channels */

    float mean[2] = {0.0, 0.0};
    iftVoxel u = iftMGetVoxelCoord(mimg, p);

    for (int i = 0; i < A->n; i++)
    {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);

      if (iftMValidVoxel(mimg, v))
      {
        int q = iftMGetVoxelIndex(mimg, v);

        for (int b = 0; b < mimg->m; b++)
        {
          if (channel_label[b] > 0)
          { /* object channel */
            mean[0] += mimg->val[q][b];
          }
          else
          { /* background channel */
            mean[1] += mimg->val[q][b];
          }
        }
      }
    }
    mean[0] /= (nobj * A->n);
    mean[1] /= (nbkg * A->n);

    /* It is expected that object pixels generate mean[0] > mean[1]
       and the other way around for background pixels. */

    for (int b = 0; b < mimg->m; b++)
    {
      if ((channel_label[b] > 0) && (mean[0] > mean[1]))
      {
        weight[b] = 1;
      }
      else
      {
        if ((channel_label[b] < 0) && (mean[0] < mean[1]))
        {
          weight[b] = -1;
        }
        else
        {
          weight[b] = 0;
        }
      }
    }

    /* Estimate the salience map using one neuron per pixel -- it is
       no longer a point-wise convolution */

    for (int b = 0; b < mimg->m; b++)
    {
      salie->val[p] += mimg->val[p][b] * weight[b];
    }
    if (salie->val[p] < 0) /* RELU (Sigmoid?) */
      salie->val[p] = 0;
  }

  iftDestroyAdjRel(&A);
  iftFree(weight);

  return (salie);
}

float *getKernelWeights(int *truelabels, int nkernels, int nlabels, int l)
{
  float *weight = iftAllocFloatArray(nkernels);
  /* set weights */
  for (int k = 0; k < nkernels; k++)
  {
    if ((l + 1) == truelabels[k])
      weight[k] = +1;
    else
      weight[k] = -1;
  }

  return weight;
}

int *GetLabels(
  char *model_dir, char *img_basename, int layer, int nkernels, int *nlabels
) {
  int *truelabels = NULL;
  char *filename = iftAllocCharArray(512);
  // Run for merged model
  sprintf(filename, "%s/conv%d-labels.txt", model_dir, layer);
  if (iftFileExists(filename))
  {
    truelabels = LoadKernelLabels(filename);
    *nlabels = NumberOfTruelabels(truelabels, nkernels);
  } // Run for image model
  else {
    sprintf(filename, "%s/%s-conv%d-labels.txt", model_dir, img_basename, layer);
    truelabels = LoadKernelLabels(filename);
    *nlabels = NumberOfTruelabels(truelabels, nkernels);
  }
  iftFree(filename);

  return truelabels;
}

// Public methods -------------------------------------------------------------

iftMImage *iftSimpleAdaptiveDecoder(iftMImage *mimg) // Decoder 1
{
  iftMImage *saliency = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize,1);
  float *weight = AdaptiveWeightsOtsu(mimg);

  /* decode layer */
  // #pragma omp parallel for shared(saliency, mimg, weight)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      saliency->val[p][0] += mimg->val[p][b] * weight[b];
    }
    if (saliency->val[p][0] < 0)
      saliency->val[p][0] = 0; /* ReLU (Sigmoid?) */
  }

  iftFree(weight);

  return (saliency);
}

iftMImage *iftProbabilityBasedAdaptiveDecoder( // Decoder 2
  iftMImage *mimg, char *model_dir, char *img_basename, int layer
)
{
  int nkernels = mimg->m;
  int nlabels = 0;
  int *truelabels = GetLabels(
    model_dir, img_basename, layer, nkernels, &nlabels
  );
  float radius;

  if (iftIs3DMImage(mimg))
    radius = 1.74;
  else
    radius = 1.5;
  
  // Estimate one salience map per object (class)
  iftMImage *saliencies = iftCreateMImage(
    mimg->xsize, mimg->ysize, mimg->zsize, nlabels
  );
  for (int l = 0; l < nlabels; l++)
  {
    float *weight = getKernelWeights(truelabels, nkernels, nlabels, l);
    iftFImage *saliency = AdaptiveDecoder2(mimg, radius, weight);
    for (size_t p=0; p < saliency->n; p++) {
      saliencies->val[p][l] = saliency->val[p];
    }
    iftFree(weight);
    iftDestroyFImage(&saliency);
  }
  
  // Releases alocated resources
  iftFree(truelabels);

  return saliencies;
}

iftMImage *iftMeanBasedAdaptiveDecoder( // Decoder3
  iftMImage *mimg, char *model_dir, char *img_basename, int layer
)
{
  int nkernels = mimg->m;
  int nlabels = 0;
  int *truelabels = GetLabels(
    model_dir, img_basename, layer, nkernels, &nlabels
  );  

  float radius;
  
  if (iftIs3DMImage(mimg))
    radius = 1.74;
  else
    radius = 1.5;
  
  // Etimate one salience map per object (class)
  iftMImage *saliencies = iftCreateMImage(
    mimg->xsize, mimg->ysize, mimg->zsize, nlabels
  );
  for (int l = 0; l < nlabels; l++)
  {
    float *weight = getKernelWeights(truelabels, nkernels, nlabels, l);
    iftFImage *saliency = AdaptiveDecoder3(mimg, radius, weight);
    for (size_t p=0; p < saliency->n; p++) {
      saliencies->val[p][l] = saliency->val[p];
    }
    iftFree(weight);
    iftDestroyFImage(&saliency);
  }
  
  // Releases alocated resources
  iftFree(truelabels);

  return saliencies;
}
