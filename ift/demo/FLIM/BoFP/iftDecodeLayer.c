#include "ift.h"

/* Author: Alexandre Xavier Falcão (September, 10th 2023)

   Description: Decodes the output of a given layer into an object
   saliency map. It may use the model of each training image with the
   saved kernel labels, or the consolidated model with its kernel
   labels.

*/

iftAdjRel *GetDiskAdjacency(iftImage *img, iftFLIMLayer layer)
{
  iftAdjRel *A;
  float radius = 0.0;

  if (iftIs3DImage(img))
  {
    for (int i = 0; i < 3; i++)
    {
      radius += powf(layer.kernel_size[i], 2);
    }
    radius = sqrtf(radius) / 2.0;
    A = iftSpheric(radius);
  }
  else
  {
    for (int i = 0; i < 2; i++)
    {
      radius += powf(layer.kernel_size[i], 2);
    }
    radius = sqrtf(radius) / 2.0;
    A = iftCircular(radius);
  }
  return (A);
}

iftAdjRel *MGetDiskAdjacency(iftMImage *mimg, iftFLIMLayer layer)
{
  iftAdjRel *A;
  float radius = 0.0;

  if (iftIs3DMImage(mimg))
  {
    for (int i = 0; i < 3; i++)
    {
      radius += powf(layer.kernel_size[i], 2);
    }
    radius = sqrtf(radius) / 2.0;
    A = iftSpheric(radius);
  }
  else
  {
    for (int i = 0; i < 2; i++)
    {
      radius += powf(layer.kernel_size[i], 2);
    }
    radius = sqrtf(radius) / 2.0;
    A = iftCircular(radius);
  }
  return (A);
}

void EraseFrameActivations(iftMImage *mimg, iftAdjRel *A)
{
  for (int p = 0; p < mimg->n; p++)
  {
    iftVoxel u = iftMGetVoxelCoord(mimg, p);
    for (int i = 1; i < A->n; i++)
    {
      iftVoxel v = iftGetAdjacentVoxel(A, u, i);
      if (!iftMValidVoxel(mimg, v))
      {
        for (int b = 0; b < mimg->m; b++)
        {
          mimg->val[p][b] = 0;
        }
        break;
      }
    }
  }
}

iftFImage *FixedWeightDecoder(iftMImage *mimg, float *weight)
{
  iftFImage *salie = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);

  /* decode layer */

#pragma omp parallel for shared(salie, mimg, weight)
  for (int p = 0; p < mimg->n; p++)
  {
    for (int b = 0; b < mimg->m; b++)
    {
      salie->val[p] += mimg->val[p][b] * weight[b];
    }
    if (salie->val[p] < 0)
      salie->val[p] = 0; /* ReLU (Sigmoid?) */
  }
  return (salie);
}

int main(int argc, char *argv[])
{
  /* Example: iftDecodeLayer 1 arch.json flim_models salie decoder_id [masks] */

  if ((argc != 7) && (argc != 6))
  {
    iftError("Usage: iftDecodeLayer <P1> <P2> <P3> <P4> <P5>\n"
             "[1] layer number (1, 2,..) \n"
             "[2] architecture of the network\n"
             "[3] folder with the models\n"
             "[4] output folder with the salience maps\n"
             "[5] decoder id (e.g., 1, 2, 3)\n"
             "[6] optional folder with regions of interest (.nii.gz or .png)\n",
             "main");
  }

  timer *tstart = iftTic();

  int layer = atoi(argv[1]);
  char *filename = iftAllocCharArray(512);
  sprintf(filename, "layer%d", layer);
  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(filename, ".mimg", true);
  iftFLIMArch *arch = iftReadFLIMArch(argv[2]);
  char *model_dir = argv[3];
  char *output_dir = argv[4];
  iftMakeDir(output_dir);
  int decoder_id = atoi(argv[5]);
  char *roi_dir = NULL;
  char label_dir[200];

  if (argc == 7)
    roi_dir = argv[6];

  int Imax;

  for (int i = 0; i < fs->n; i++)
  {
    printf("Processing image %d of %ld\r", i + 1, fs->n);
    char *basename = iftFilename(fs->files[i]->path, ".mimg");
    iftMImage *mimg = iftReadMImage(fs->files[i]->path);
    iftImage *mask = NULL;

    /* Compute the scale factor w.r.t. the input of the network */
    float scale[3];
    sprintf(filename, "./layer0/%s.mimg", basename);
    iftMImage *input = iftReadMImage(filename);

    scale[0] = (float)input->xsize / (float)mimg->xsize;
    scale[1] = (float)input->ysize / (float)mimg->ysize;
    scale[2] = (float)input->zsize / (float)mimg->zsize;
    iftDestroyMImage(&input);

    iftMImage *saliencies;
    switch (decoder_id)
    {
    case 1:
      saliencies = iftSimpleAdaptiveDecoder(mimg);
      break;
    case 2:
      saliencies = iftProbabilityBasedAdaptiveDecoder(
        mimg, model_dir, basename, layer
      );
      break;
    case 3:
      saliencies = iftMeanBasedAdaptiveDecoder(
        mimg, model_dir, basename, layer
      );
      break;
    default:
      iftError("Invalid Decoder ID", "IftDecodeLayer");
    }

    iftMImage *smap_temp = NULL;
    /* Interpolate the salience maps */
    if (iftIs3DMImage(mimg))
    {
      smap_temp = iftMInterp(saliencies, scale[0], scale[1], scale[2]);
      Imax = 4095;
    }
    else
    {
      smap_temp = iftMInterp2D(saliencies, scale[0], scale[1]);
      Imax = 255;
    }
    for (int b = 0; b < saliencies->m; b++)
    {
      /* Normalize the salience maps */
      iftImage *smap = iftMImageToImage(smap_temp, Imax, b);

      /* post-process and save the salience map */
      if (argc == 7)
      {
        if (iftIs3DMImage(mimg))
        {
          sprintf(filename, "%s/%s.nii.gz", roi_dir, basename);
        }
        else
        {
          sprintf(filename, "%s/%s.png", roi_dir, basename);
        }
        mask = iftReadImageByExt(filename);
      }

      sprintf(label_dir, "%s/label%d", output_dir, b + 1);
      if (!iftDirExists(label_dir))
        iftMakeDir(label_dir);

      if (iftIs3DMImage(mimg))
      {
        sprintf(filename, "%s/%s_layer%d.nii.gz", label_dir, basename, layer);
      }
      else
      {
        sprintf(filename, "%s/%s_layer%d.png", label_dir, basename, layer);
      }

      if (mask != NULL)
      {
        iftImage *aux = iftMask(smap, mask);
        iftDestroyImage(&smap);
        smap = aux;
        iftWriteImageByExt(smap, filename);
        iftDestroyImage(&mask);
      }
      else
      {
        iftWriteImageByExt(smap, filename);
      }
      iftDestroyImage(&smap);
    }

    iftDestroyMImage(&mimg);
    iftDestroyMImage(&smap_temp);
    iftDestroyMImage(&saliencies);
    iftFree(basename);
  }

  iftFree(filename);
  iftDestroyFileSet(&fs);
  iftDestroyFLIMArch(&arch);

  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

  return (0);
}
