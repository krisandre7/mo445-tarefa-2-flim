//This inclusion must precede ift.h because the latter defines the macros MAX and MIN.
#include "graph.h"

#define class class_
extern "C" {
#include "ift.h"
#include "common.h"
}
#undef class

typedef Graph<double, double, double> GraphType;

GraphType *create_graph(iftImage *img, iftAdjRel *A, float beta)
{
  GraphType *graph = new GraphType(img->n, img->n);

  float max_dist = iftMaximumValue(img);
  float min_dist = iftMinimumValue(img);

  int p, q, i;
  iftVoxel v, u;
  for (p = 0; p < img->n; p++)
    graph->add_node();

  for (p = 0; p < img->n; p++)
  {
    u = iftGetVoxelCoord(img, p);
    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);
      if (iftValidVoxel(img, v))
      {
        q = iftGetVoxelIndex(img, v);
        if (p < q)
        {
          float dist = (img->val[q] - min_dist) / (max_dist - min_dist);
          float similarity = exp(-dist / 0.5);
          double edge_weight = pow(similarity, beta);
          graph->add_edge(p, q, edge_weight, edge_weight);
        }
      }
    }
  }
  return graph;
}

iftImage *graph_cut_diff_segmentation(GraphType *graph, iftImage *basins, iftLabeledSet *new_seeds)
{
  iftLabeledSet *s = new_seeds;
  int p, label;
  while (s)
  {
    p = s->elem;
    label = s->label;

    if (label == 1)
      graph->edit_tweights(p, INFINITY_FLT, 0);
    else
      graph->edit_tweights(p, 0, INFINITY_FLT);

    s = s->next;
  }

  graph->maxflow();

  iftImage *segmentation = iftCreateImage(basins->xsize, basins->ysize, basins->zsize);

  for (p = 0; p < segmentation->n; p++)
  {
    if (graph->what_segment(p) == GraphType::SOURCE)
      segmentation->val[p] = 1;
  }

  return segmentation;
}

// int main(int argc, char **argv)
// {
//     if (argc != 7)
//         iftError("Usage: gc_super [IMAGE_PATH] [SEEDS_PATH1] [SEEDS_PATH2] [OUTPUT_PATH1] [OUTPUT_PATH2] [BETA]", "gc_super");

//     float beta = atof(argv[6]);

//     iftImage *image = iftReadImageP6(argv[1]);
//     iftLabeledSet *seeds1 = iftReadSeeds2D(argv[2], image);
//     iftLabeledSet *seeds2 = iftReadSeeds2D(argv[3], image);

//     iftAdjRel *adj = iftCircular(sqrtf(2));

//     iftImage *basins = iftImageBasins(image, adj);
//     GraphType *graph = create_graph(basins, adj, beta);

//     iftImage *result = graph_cut_diff_segmentation(graph, basins, seeds1);

//     //First result
//     iftWriteImageP5(iftNormalize(result, 0, 255), argv[4]);

//     iftDestroyImage(&result);

//     result = graph_cut_diff_segmentation(graph, basins, seeds2);

//     //Second result
//     iftWriteImageP5(iftNormalize(result, 0, 255), argv[5]);

//     iftDestroyImage(&result);

//     return 0;
// }

/* =================== Private Functions ====================== */
void pvtOpenImage(ExperimentParameters *params, char *image_name, char *gt_name, iftImage **img, iftImage **gt)
{
  char path[256];
  char *database;
  database = iftSplitStringOld(params->image_database, "/", 2);
  iftImage *raw_image = NULL;
  iftAdjRel *spheric3 = iftSpheric(3);
  if (params->verbose > 0)
    fprintf(stdout, "Processing: %s and %s\n", image_name, gt_name);

  strcpy(path, params->image_database);
  strcat(path, image_name);
  if (strcmp(params->file_extension, "scn") == 0)
    raw_image = iftReadImage(path);
  else
    raw_image = iftReadImageP6(path);

  if (strcmp(database, "foot") == 0 && params->smooth_foot == 1)
    *img = iftSmoothImage(raw_image, spheric3, 15);
  else
    *img = iftCopyImage(raw_image);

  strcpy(path, params->label_database);
  strcat(path, gt_name);
  if (strcmp(params->file_extension, "scn") == 0)
    *gt = iftReadImage(path);
  else
    *gt = iftReadImageP5(path);

  params->curr_number_of_objects = iftMaximumValue(*gt);

  iftDestroyImage(&raw_image);
  iftDestroyAdjRel(&spheric3);
}
iftImage *pvtCreateGradient(ExperimentParameters *params, iftImage *img)
{
  iftAdjRel  *basins_adjacency  = iftSpheric(params->basins_adjacency);
  iftImage *basins = NULL;
  if (params->basins_type == 0)
    basins = iftImageBasins(img, basins_adjacency);
  else if (params->basins_type == 1)
    basins = iftSobelGradientMagnitude(img);
  else
    iftError("Wrong basins type, please select 0 or 1", "demo/relaxedIFT/MainRelax.c@RunRobot");

  iftDestroyAdjRel(&basins_adjacency);
  return basins;
}
iftFImage **pvtCreateObjectEDTFromGT(ExperimentParameters *params, iftImage *gt_image)
{
  int p, o;
  iftImage *temp_image;
  iftAdjRel *sphericsqrt3 = iftSpheric(sqrtf(3.0));
  iftFImage **object_edt_gt = (iftFImage **) malloc(sizeof(iftFImage *) * (params->curr_number_of_objects + 2));

  object_edt_gt[0] = NULL;
  printf("Generating individual GTs for ASD... ");
  for (o = 1; o <= params->curr_number_of_objects; o++)
  {
    temp_image = iftCreateImage(gt_image->xsize, gt_image->ysize, gt_image->zsize);
    for (p = 0; p < gt_image->n; p++)
    {
      if (gt_image->val[p] == o)
        temp_image->val[p] = 1;
    }
    object_edt_gt[o] = iftSignedDistTrans(temp_image, sphericsqrt3);
    iftDestroyImage(&temp_image);
  }
  iftDestroyAdjRel(&sphericsqrt3);
  printf("[DONE]\n");
  return object_edt_gt;
}
void pvtDestroyObjectEDTFromGT(ExperimentParameters *params, iftFImage **object_edt_gt)
{
  int o;
  if (object_edt_gt != NULL)
  {
    for (o = 1; o <= params->curr_number_of_objects; o++)
      iftDestroyFImage(&object_edt_gt[o]);
    free(object_edt_gt);
    object_edt_gt = NULL;
  }

}
iftLabeledSet *pvtRunGeodesicDGC(ExperimentParameters *params, iftImage *gt_image, iftImage *current_segmentation, iftImage *seeds_image, iftBMap *seeds_bmap, int *seeds_added)
{
  iftLabeledSet  *available_seeds = NULL, *current_seeds = NULL;
  iftImage *seeds_image_copy     = NULL;
  int p;

  available_seeds = iftGeodesicMarkersForSegmentation(gt_image, current_segmentation);

  seeds_image_copy = iftCopyImage(seeds_image);
  *seeds_added = iftMarkersFromMisclassifiedSeeds(seeds_image, available_seeds, seeds_bmap, params->seeds_per_iteration, params->curr_number_of_objects + 1, gt_image, current_segmentation, params->min_distance_border, params->max_marker_size, params->min_marker_size);

  //This produces only the new seeds added this iteration
  for (p = 0; p < seeds_image_copy->n; p++)
  {
    if (seeds_image_copy->val[p] == seeds_image->val[p])
      seeds_image_copy->val[p] = -1;
    else
      seeds_image_copy->val[p] = seeds_image->val[p];
  }
  current_seeds = iftLabeledSetFromSeedImage(seeds_image_copy);

  iftDestroyImage(&seeds_image_copy);
  iftDestroyLabeledSet(&available_seeds);

  return current_seeds;
}
void pvtComputeErrorsDGC(float *distance_error, float *dice_error, ExperimentParameters *params, iftImage *current_segmentation, iftImage *gt_image, iftFImage **object_edt_gt, int object)
{
  int i, o;
  float *asd_array = NULL, dice = 0.0;
  float sum_distance =0, sum_dice = 0;

  asd_array = iftASDError(current_segmentation, object_edt_gt, 1);
  // Compute the error in (mm)
  distance_error[object] = (asd_array[1] * gt_image->dx);
  free(asd_array);

  dice = iftDiceSimilarity(current_segmentation, gt_image);
  dice_error[object] = dice;

  for(i=1; i<=object; i++)
  {
    sum_distance += distance_error[i];
    sum_dice += dice_error[i];
  }
  distance_error[0] = sum_distance / object;
  dice_error[0] = sum_dice / object;
}

void runRobot (ExperimentParameters *params, iftImageNames *images, iftImageNames *gtruths)
{
  /* =================== Declarations ======================= */
  int i = 0, j = 0, k = 0, p = 0, o = 0, seeds_added = 0, stop = 0, final_iteration = 0;
  float iteration_time = 0, curr_distance_error = 0, curr_dice_error = 0;
  float sum_distance=0, sum_dice=0, sum_final_iteration = 0;
  char buffer[2048];
  
  iftImage *original_image       = NULL, *gt_image = NULL, *current_segmentation = NULL;
  iftImage *seeds_image          = NULL;
  iftImage *gt_image_kobj        = NULL;
  iftImage *basins               = NULL;
  iftFImage **object_edt_gt      = NULL;
  
  iftSmoothBorder *smooth        = NULL;
  iftBMap         *seeds_bmap    = NULL;

  iftAdjRel       *spheric15     = iftSpheric(1.5);
  iftAdjRel       *radiussqrt3   = iftSpheric(sqrtf(3.0));

  timer *t1                      = NULL, *t2 = NULL;
  iftLabeledSet  *current_seeds  = NULL, *tmp;

  ExperimentResults *exp_results = NULL;

  /* =================== Allocations ======================= */
  exp_results           = createExperimentResults(params);

  printHeader(params, "file", "DGC");
  printHeader(params, "screen", "DGC");

  /* =================== Loop through all images =================== */
  for (i = 0; i < params->number_of_images; i++)
  {
    pvtOpenImage(params, images[i].image_name, gtruths[i].image_name, &original_image, &gt_image);
    exp_results[i].seeds_added           = 0;
    exp_results[i].number_of_supervoxels = 0.0;
    exp_results[i].sum_final_iteration = 0;
    t1 = iftTic();
    basins                               = pvtCreateGradient(params, original_image);
    t2 = iftToc();
    exp_results[i].setup_time = iftCompTime(t1, t2) / SECONDS;

    for (k = 1; k <= params->curr_number_of_objects; k++)
    {
      printf("Processing object: %d\n", k);
      gt_image_kobj       = iftCreateImage(gt_image->xsize, gt_image->ysize, gt_image->zsize);
      for (p = 0; p < gt_image->n; p++)
      {
        if (gt_image->val[p] == k)
          gt_image_kobj->val[p] = 1;
      }

      current_segmentation = NULL;
      seeds_bmap           = iftCreateBMap(original_image->n);
      seeds_image          = iftCreateImage(original_image->xsize, original_image->ysize, original_image->zsize);
      iftSetImage(seeds_image, -1);

      t1 = iftTic();
      GraphType *graph = create_graph(basins, spheric15, params->gc_beta);
      t2 = iftToc();
      exp_results[i].setup_time += iftCompTime(t1, t2) / SECONDS;

      //This cannot be removed, because iftASSDError expects label from 0 to N.
      object_edt_gt = (iftFImage **) malloc(sizeof(iftFImage *) * 2);
      object_edt_gt[0] = NULL;
      object_edt_gt[1] = iftSignedDistTrans(gt_image_kobj, radiussqrt3);

      /* =================== Each Segmentation Iteration =================== */
      for (j = 0; j < params->n_segm_iterations; j++)
      {
        /* =================== Geodesic Robot =================== */

        iftDestroyLabeledSet(&current_seeds);
        current_seeds = pvtRunGeodesicDGC(params, gt_image_kobj, current_segmentation, seeds_image, seeds_bmap, &seeds_added);

        t1 = iftTic();
        current_segmentation = graph_cut_diff_segmentation(graph, basins, current_seeds);
        t2 = iftToc();

        iteration_time = iftCompTime(t1, t2);
        exp_results[i].total_time += iteration_time;
        exp_results[i].seeds_added += seeds_added;
        exp_results[i].sum_final_iteration++;


        /* =================== Error measure =================== */
        pvtComputeErrorsDGC(exp_results[i].distance_error_object, exp_results[i].dice_error_object, params, current_segmentation, gt_image_kobj, object_edt_gt, k);
        curr_distance_error = exp_results[i].distance_error_object[k];
        curr_dice_error = exp_results[i].dice_error_object[k];

        /* =================== Prints and Destroys =================== */
        if (params->verbose > 1)  fprintf(stdout, "j= %4d k=%4d ASD= %10f seeds_added=%2d dice=%f time=%f s\n", j, k, curr_distance_error, seeds_added, curr_dice_error, iteration_time / SECONDS);
        if (params->verbose == 3) for (o = 0; o <= params->curr_number_of_objects; o++) fprintf(stdout, "Obj %d, ASD=%f dice=%f\n", o, exp_results[i].distance_error_object[o], exp_results[i].dice_error_object[o]);
        if (params->save_ouput && strcmp(params->file_extension, "scn") == 0) saveImage3d(current_segmentation, images[i].image_name, j, params);

        if (seeds_added == 0)
          break;
      }
      exp_results[i].arr_final_iteration[k] = (j + 1); //Starts in 0
      iftDestroyImage(&seeds_image);
      iftDestroyImage(&gt_image_kobj);
      iftDestroyBMap(&seeds_bmap);
      iftDestroyFImage(&object_edt_gt[1]);
      free(object_edt_gt);
      delete graph;
    }
    /* =================== Save the results =================== */
    if (j >= params->n_segm_iterations) j = params->n_segm_iterations - 1;


    sum_distance=0, sum_dice=0;
    for (k = 1; k <= params->curr_number_of_objects; k++)
    {
        sum_distance += exp_results[i].distance_error_object[k];
        sum_dice += exp_results[i].dice_error_object[k];
    }   
    exp_results[i].distance_error_object[0] = sum_distance / params->curr_number_of_objects;
    exp_results[i].dice_error_object[0] = sum_dice / params->curr_number_of_objects;
    exp_results[i].final_iteration = exp_results[i].sum_final_iteration / params->curr_number_of_objects;

    strcpy(exp_results[i].image_name, images[i].image_name);
    exp_results[i].distance_error  = exp_results[i].distance_error_object[0];
    exp_results[i].dice_error      = exp_results[i].dice_error_object[0];

    if (params->verbose > 0)
      fprintf(stdout, "FinalASDError: %f mm\t Dice: %f\tExecutions: %f\tSeeds added: %.0f\tTot. Time: %f s\n\n",
              exp_results[i].distance_error, exp_results[i].dice_error, exp_results[i].final_iteration, exp_results[i].seeds_added, exp_results[i].total_time / SECONDS);

    //Destroy all for next image
    iftDestroyImage(&original_image);
    iftDestroyImage(&gt_image);
  }
  /* =================== Report to File =================== */
  printReport(params, exp_results, "screen", "DGC");
  printReport(params, exp_results, "file", "DGC");

  /* ================== Destroy Everything ================ */
  iftDestroyAdjRel(&spheric15);
  destroyExperimentResults(exp_results, params);
}

int main(int argc, char **argv)
{
  if (argc != 5)
    iftError("<configuration_file> <images_directory> <gt_directory> <gc_beta>", "Robot - main");
  //int i = 0;
  char *configuration_file = argv[1];
  iftImageNames *img_names, *gt_names;
  ExperimentParameters *all_parameters = createExperimentParameters();
  ExperimentParameters *current_parameters = createExperimentParameters();

  loadParameters(all_parameters, configuration_file);
  loadParameters(current_parameters, configuration_file);

  // Set the image and label directory
  setDirectories(all_parameters, argv[2], argv[3]);
  setDirectories(current_parameters, argv[2], argv[3]);

  int number_of_images = iftCountImageNames(all_parameters->image_database, all_parameters->file_extension);

  all_parameters->number_of_images = number_of_images;
  current_parameters->number_of_images = number_of_images;

  img_names = iftCreateAndLoadImageNames(number_of_images, all_parameters->image_database, all_parameters->file_extension);
  gt_names = iftCreateAndLoadImageNames(number_of_images, all_parameters->label_database, "scn");

  all_parameters->max_number_of_objects = getMaximumNumberOfObjects(all_parameters, gt_names);
  current_parameters->max_number_of_objects = all_parameters->max_number_of_objects;
  if (all_parameters->max_number_of_objects == 256)   iftWarning("Your groundtruth might be 0-255 instead of 0,1,2,3,...", "demo/relaxedIFT/Main");
  else if (all_parameters->max_number_of_objects == 0) iftError("Zero objects to segment, GT error probably", "demo/relaxedIFT/Main");

  fprintf(stdout, "===================================================================\n");
  fprintf(stdout, "WARNING: Beta parameter overwritten by program call parameter\n");
  fprintf(stdout, "Config file: %.0f  --->  Running with: %.0f\n", current_parameters->gc_beta, atof(argv[4]));
  fprintf(stdout, "===================================================================\n");

  current_parameters->gc_beta = atof(argv[4]);
  runRobot(current_parameters, img_names, gt_names);

  /* End Main code */
  destroyExperimentParameters(all_parameters);
  destroyExperimentParameters(current_parameters);
  iftDestroyImageNames(img_names);
  iftDestroyImageNames(gt_names);

  return 0;
}

