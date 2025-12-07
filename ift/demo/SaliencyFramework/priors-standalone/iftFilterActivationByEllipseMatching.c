#include <ift.h>

iftImage *iftFilterSaliencyByEllipseMatchingV2(iftImage *salie_map, int tensor_npairs, int min_size, int max_size, float score_threshold)
{
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
      ellipseScore /= object_area;
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

  return (salie_map_filt);
}

iftMImage *iftFilterByEllipseMatchingBatch(iftMImage *input, int tensor_npairs, int min_size, int max_size, float score_threshold, float *channel_label)
{
  iftMImage *output = iftCreateMImage(input->xsize, input->ysize, input->zsize, input->m);
  
  for (int band = 0; band < input->m; band++)
  {
    iftImage *activ      = iftMImageToImage(input, 255, band);
    iftImage *activ_filt = NULL;
    if (channel_label[band] > 0){
      activ_filt = iftFilterSaliencyByEllipseMatchingV2(activ, tensor_npairs, min_size, max_size, score_threshold);
    } else {
      iftImage *comp_activ = iftComplement(activ);
      iftImage *aux        = iftFilterSaliencyByEllipseMatchingV2(comp_activ, tensor_npairs, min_size, max_size, score_threshold);
      activ_filt           = iftComplement(aux);
      iftDestroyImage(&comp_activ);
      iftDestroyImage(&aux);
    }
    iftDestroyImage(&activ);    
    for (int p = 0; p < input->n; p++)
    {
      output->val[p][band] = activ_filt->val[p];
    }
    iftDestroyImage(&activ_filt);
  }

  return(output);
}

float *LoadKernelWeights(char *filename)
{
  int number_of_kernels;
  FILE *fp;
  float *weight;

  fp = fopen(filename, "r");
  fscanf(fp, "%d", &number_of_kernels);
  weight = iftAllocFloatArray(number_of_kernels);
  for (int k = 0; k < number_of_kernels; k++)
  {
    fscanf(fp, "%f ", &weight[k]);
  }
  fclose(fp);

  return (weight);
}



iftMImage *iftFilterByEllipseMatchingBatchV2(iftMImage *input, int tensor_npairs, int min_size, int max_size, float score_threshold, float *channel_label)
{
  iftMImage *output = iftCreateMImage(input->xsize, input->ysize, input->zsize, input->m);

  for (int band = 0; band < input->m; band++)
  {
    printf("\nIndex: %d\n", band);

    iftImage *activ = iftMImageToImage(input, 255, band);
    iftImage *activ_orig_filt, *complement_filt;
    iftImage *comp_activ = iftComplement(activ);

    complement_filt = iftFilterSaliencyByEllipseMatchingV2(comp_activ, tensor_npairs, min_size, max_size, score_threshold);
    activ_orig_filt = iftFilterSaliencyByEllipseMatchingV2(activ, tensor_npairs, min_size, max_size, score_threshold);

    for (int p = 0; p < input->n; p++)
    {
      if ((complement_filt->val[p] > 0))
      {

        output->val[p][band] = complement_filt->val[p];
      }
      else if ((activ_orig_filt->val[p] > 0))
      {
        output->val[p][band] = activ_orig_filt->val[p];
      }
    }

    iftDestroyImage(&activ);
    iftDestroyImage(&activ_orig_filt);
    iftDestroyImage(&complement_filt);
    iftDestroyImage(&comp_activ);
    printf("\nEnd Index: %d\n", band);
  }

  return (output);
}

int main(int argc, char *argv[])
{
  /* Example: ./iftFilterActivationByEllipseMatching "1" "./flim" "./layer_test_ellipse_output"   */

  if ((argc != 4))
  {
    iftError("Usage: iftFilterActivationByEllipseMatching <P1> <P2> <P3>\n"
             "[1] layer number (1, 2,..) \n"
             "[2] folder with the FLIM BagOfFeat model\n"
             "[3] output folder \n",
             "main");
  }

  timer *tstart = iftTic();

  int layer = atoi(argv[1]);
  char *filename = iftAllocCharArray(512);
  sprintf(filename, "./layer_test_ellipse%d", layer);
  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(filename, ".mimg", true);
  char *model_dir = argv[2];
  char *output_dir = argv[3];
  iftMakeDir(output_dir);

  for (int i = 0; i < fs->n; i++)
  {
    printf("Processing feature map %d of %ld\r", i + 1, fs->n);
    char *file = fs->files[i]->path;
    char *basename = iftFilename(file, ".mimg");
    iftMImage *layer_mimg = iftReadMImage(file);
    iftMImage *result;
    // load weights indicating object and background activation channels
    float *weight = NULL;
    sprintf(filename, "%s/conv%d-weights.txt", model_dir, layer);
    if (iftFileExists(filename))
    { // load merged model
      weight = LoadKernelWeights(filename);
    }
    else
    {
      sprintf(filename, "%s/%s-conv%d-weights.txt",
              model_dir, basename, layer);
      if (iftFileExists(filename))
      { // load image model
        weight = LoadKernelWeights(filename);
      }
    }

    /*
        int len = 14; // pois tenho 14 ativações
        for (int i = 0; i < len; ++i)
        {
          printf("%.2f ", *(weight + i));
        }
        printf("\n");
    */
    result = iftFilterByEllipseMatchingBatchV2(layer_mimg, 32, 1000, 9000, 0.7, weight);

    printf("%s\n", file);
    char *name = iftFilename(file, ".mimg");
    sprintf(filename, "%s/%s.mimg", output_dir, name);

    for (int b = 0; b < result->m; b++)
    {
      iftImage *img = iftMImageToImage(result, 255, b);
      iftWriteImageByExt(img, "%s/%s/%03d.png", output_dir, name, b);
      iftDestroyImage(&img);
    }

    iftDestroyMImage(&layer_mimg);
    iftFree(weight);
  }

  iftFree(filename);
  iftDestroyFileSet(&fs);

  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

  return (0);
}
