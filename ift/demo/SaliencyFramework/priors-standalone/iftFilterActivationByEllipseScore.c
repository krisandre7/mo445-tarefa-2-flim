#include <ift.h>
typedef struct _output
{
  iftImage *image;
  float ellipseScoreSum;
} Output;

void DestroyOutput(Output **output)
{
  if (output != NULL)
  {
    Output *aux = *output;
    if (aux != NULL)
    {
      if (aux->image != NULL)
        iftDestroyImage(&aux->image);
      iftFree(aux);
      *output = NULL;
    }
  }
}
Output *iftFilterActivationByEllipseMatching(iftImage *salie_map, int tensor_npairs, int min_size, int max_size, float score_threshold)
{
  Output *output = (Output *)calloc(1, sizeof(Output));
  int o, q;
  iftImage *salie_map_filt = iftCreateImageFromImage(salie_map);
  iftImage *salie_map_bin = iftBinarize(salie_map);
  // label its components from 1 to n
  iftImage *salie_map_bin_label = iftFastLabelComp(salie_map_bin, NULL);
  iftDestroyImage(&salie_map_bin);

  for (q = 0; q < salie_map_bin_label->n; q++)
  {
    salie_map_bin_label->val[q] = salie_map_bin_label->val[q] + 1;
  }
  int nobjs = iftMaximumValue(salie_map_bin_label);
  iftTensorScale *tensor_scale = iftSuperpixelToTensorScale(salie_map_bin_label, tensor_npairs, min_size, max_size);
  float ellipseScoreSum = 0;
  for (o = 2; o <= nobjs; o++)
  {
    if (tensor_scale->area->val[o - 1] <= 0)
    {
      continue;
    }

    // extract the component from the label image
    iftImage *obj = iftExtractObject(salie_map_bin_label, o);

    // estimate binary area
    int object_area = 0;
    for (int a = 0; a < obj->n; a++)
    {
      if (obj->val[a] > 0)
      {
        object_area = object_area + 1;
      }
    }

    if ((object_area >= min_size) && (object_area <= max_size))
    {
      float ellipseScore = 0;

      for (int p = 0; p < obj->n; p++)
      {
        if (obj->val[p] > 0)
        {
          iftVoxel u = iftGetVoxelCoord(obj, p);
          float ellipse_eq = sqrtf(square((u.x - tensor_scale->pos_focus->val[o - 1].x)) + (square((u.y - tensor_scale->pos_focus->val[o - 1].y)))) + sqrtf(square((u.x - tensor_scale->neg_focus->val[o - 1].x)) + (square((u.y - tensor_scale->neg_focus->val[o - 1].y))));
          if (ellipse_eq <= 2 * tensor_scale->major_axis->val[o - 1])
          {
            ellipseScore += 1;
          }
        }
      }

      if ((object_area != 0)) // avoid division by zero
      {
        ellipseScore /= object_area;
      }
      else if ((object_area == 0) && (ellipseScore == 0))
      {
        ellipseScore = 0.0;
      }
      else
      {
        //  printf("Warning: Division by zero");
        ellipseScore = 0.0;
      }
      ellipseScoreSum += ellipseScore;
      if (ellipseScore >= score_threshold)
      {
        for (q = 0; q < obj->n; q++)
        {
          if (obj->val[q] > 0)
          {
            salie_map_filt->val[q] = salie_map->val[q];
          }
        }
      }
    }

    iftDestroyImage(&obj);
  }

  iftDestroyTensorScale(&tensor_scale);
  iftDestroyImage(&salie_map_bin_label);

  output->image = salie_map_filt;
  output->ellipseScoreSum = ellipseScoreSum;
  return (output);
}

iftMImage *iftFilterActivationsByEllipseScore(iftMImage *input, int tensor_npairs, int min_size, int max_size, float score_threshold)
{
  iftMImage *result = iftCreateMImage(input->xsize, input->ysize, input->zsize, input->m);

  for (int band = 0; band < input->m; band++)
  {
    iftImage *activ = iftMImageToImage(input, 255, band);
    iftImage *bin = iftBinarize(activ);
    iftImage *comp_activ = iftComplement(bin);
    iftDestroyImage(&bin);

    Output *o1 = iftFilterActivationByEllipseMatching(activ, tensor_npairs, min_size, max_size, score_threshold);
    Output *o2 = iftFilterActivationByEllipseMatching(comp_activ, tensor_npairs, min_size, max_size, score_threshold);
    iftDestroyImage(&comp_activ);
    iftDestroyImage(&activ);

    if (o1->ellipseScoreSum > o2->ellipseScoreSum) // is foreground activation
    {
      for (int p = 0; p < input->n; p++)
      {
        result->val[p][band] = o1->image->val[p];
      }
      DestroyOutput(&o1);
    }
    else
    {
      iftImage *comp_ellipse_background = iftComplement(o2->image);
      for (int p = 0; p < input->n; p++)
      {
        result->val[p][band] = comp_ellipse_background->val[p];
      }
      iftDestroyImage(&comp_ellipse_background);
      DestroyOutput(&o2);
    }
  }

  return (result);
}

int main(int argc, char *argv[])
{
  /* Example: ./iftFilterActivationByEllipseScore "1" "./layer_output"   */

  if ((argc != 3))
  {
    iftError("Usage: iftFilterActivationByEllipseMatching <P1> <P2> \n"
             "[1] layer number (1, 2,..) \n"
             "[2] output folder \n",
             "main");
  }

  timer *tstart = iftTic();

  int layer = atoi(argv[1]);
  char *filename = iftAllocCharArray(512);
  sprintf(filename, "/home/marcelo/LIBIFT/trunk/bin/layer%d_test", layer);
  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(filename, ".mimg", true);
  char *output_dir = argv[2];
  iftMakeDir(output_dir);

  for (int i = 0; i < fs->n; i++)
  {
    printf("Processing feature map %d of %ld\r", i + 1, fs->n);
    char *file = fs->files[i]->path;
    char *basename = iftFilename(file, ".mimg");
    iftMImage *layer_mimg = iftReadMImage(file);
    iftMImage *result;

    result = iftFilterActivationsByEllipseScore(layer_mimg, 32, 5, 6000, 0.5);

    printf("%s\n", file);
    char *name = iftFilename(file, ".mimg");
    sprintf(filename, "%s/%s.mimg", output_dir, name);
    /*
        for (int b = 0; b < result->m; b++)
        {
          iftImage *img = iftMImageToImage(result, 255, b);
          iftWriteImageByExt(img, "%s/%s/%03d.png", output_dir, name, b);
          iftDestroyImage(&img);
        }
    */
    iftWriteMImage(result, filename);
    iftDestroyMImage(&layer_mimg);
  }

  iftFree(filename);
  iftDestroyFileSet(&fs);

  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

  return (0);
}

