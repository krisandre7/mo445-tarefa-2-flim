#include "ift.h"
#include "common.h"

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
iftLabeledSet *pvtRunGeodesic(ExperimentParameters *params, iftImage *gt_image, iftImage *current_segmentation, iftImage *seeds_image, iftBMap *seeds_bmap, int *seeds_added)
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
  if (params->ift_differential)
    current_seeds = iftLabeledSetFromSeedImage(seeds_image_copy);
  else
    current_seeds = iftLabeledSetFromSeedImage(seeds_image);

  iftDestroyImage(&seeds_image_copy);
  iftDestroyLabeledSet(&available_seeds);

  return current_seeds;
}
void pvtComputeErrors(float *distance_error, float *dice_error, ExperimentParameters *params, iftImage *current_segmentation, iftImage *gt_image, iftFImage **object_edt_gt)
{
  int i, o;
  float *asd_array = NULL;

  asd_array = iftASDError(current_segmentation, object_edt_gt, params->curr_number_of_objects);
  // Compute the error in (mm)
  for (o = 0; o <= params->curr_number_of_objects; o++)
    distance_error[o] = (asd_array[o] * gt_image->dx);

  free(asd_array);

  iftDblArray *dices = iftDiceSimilarityMultiLabel(current_segmentation, gt_image, params->curr_number_of_objects);
  for (o = 0; o <= params->curr_number_of_objects; o++)
    dice_error[o] = dices->val[o];
  iftDestroyDblArray(&dices);

  free(asd_array);
}

void runRobot (ExperimentParameters *params, iftImageNames *images, iftImageNames *gtruths)
{
  /* =================== Declarations ======================= */
  int i = 0, j = 0, k = 0, p = 0, o = 0, seeds_added = 0, stop = 0, final_iteration = 0;
  float iteration_time = 0;
  char buffer[2048];

  iftImage *original_image  = NULL, *gt_image = NULL, *current_segmentation = NULL;
  iftImage *seeds_image     = NULL;
  iftFImage **object_edt_gt = NULL;

  iftImageForest  *forest       = NULL;
  iftSmoothBorder *smooth       = NULL;
  iftBMap         *seeds_bmap   = NULL;
  iftSet          *removal_markers = NULL;

  iftAdjRel       *spheric15    = NULL;

  timer *t1 = NULL, *t2 = NULL;
  iftLabeledSet  *current_seeds = NULL, *tmp;

  ExperimentResults *exp_results  = NULL;

  /* =================== Allocations ======================= */
  exp_results           = createExperimentResults(params);

  printHeader(params, "file", "IFT");
  printHeader(params, "screen", "IFT");

  /* =================== Loop through all images =================== */
  for (i = 0; i < params->number_of_images; i++)
  {
    pvtOpenImage(params, images[i].image_name, gtruths[i].image_name, &original_image, &gt_image);
    exp_results[i].prev_distance_error = -1;
    stop = 0;

    current_segmentation = NULL;
    spheric15            = iftSpheric(1.5);
    seeds_bmap           = iftCreateBMap(original_image->n);
    seeds_image          = iftCreateImage(original_image->xsize, original_image->ysize, original_image->zsize);
    iftSetImage(seeds_image, -1);

    t1 = iftTic();
    forest = iftCreateImageForest(pvtCreateGradient(params, original_image), spheric15);
    t2 = iftToc();

    object_edt_gt = pvtCreateObjectEDTFromGT(params, gt_image);
    smooth = iftCreateSmoothBorder(forest->img, forest->A, params->smooth_iterations, 0.5);

    exp_results[i].number_of_supervoxels = 0.0;
    exp_results[i].setup_time = iftCompTime(t1, t2) / SECONDS;

    /* =================== Each Segmentation Iteration =================== */
    for (j = 0; j < params->n_segm_iterations; j++)
    {
      /* =================== Geodesic Robot =================== */

      iftDestroySet(&removal_markers);
      iftDestroyLabeledSet(&current_seeds);
      current_seeds = pvtRunGeodesic(params, gt_image, current_segmentation, seeds_image, seeds_bmap, &seeds_added);


      // if (j > 0)
      // {
      //   tmp = current_seeds;
      //   int root = 0;
      //   while (tmp != NULL)
      //   {
      //     root = forest->root->val[tmp->elem];
      //     if (root != NIL && forest->label->val[root] != gt_image->val[tmp->elem])
      //     {
      //       iftInsertSet(&removal_markers, forest->root->val[tmp->elem]);
      //     }
      //     tmp = tmp->next;
      //   }
      // }


      /* =================== Segmentation =================== */
      t1 = iftTic();
      iftDiffWatershed(forest, current_seeds, removal_markers);
      if (params->relaxation == 2)  // Relax each iteration
        iftRelaxObjects(forest, smooth);
      t2 = iftToc();

      current_segmentation = forest->label;
      iteration_time = iftCompTime(t1, t2);
      exp_results[i].total_time += iteration_time;
      exp_results[i].seeds_added += seeds_added;

      /* =================== Error measure =================== */
      pvtComputeErrors(exp_results[i].distance_error_object, exp_results[i].dice_error_object, params, current_segmentation, gt_image, object_edt_gt);


      /* =================== Prints and Destroys =================== */
      if (params->verbose > 1)  fprintf(stdout, "j= %4d ASD= %10f seeds_added=%2d dice=%f time=%f s\n", j, exp_results[i].distance_error_object[0], seeds_added, exp_results[i].dice_error_object[0], iteration_time / SECONDS);
      if (params->verbose == 3) for (o = 0; o <= params->curr_number_of_objects; o++) fprintf(stdout, "Obj %d, ASD=%f dice=%f\n", o, exp_results[i].distance_error_object[o], exp_results[i].dice_error_object[o]);
      if (params->save_ouput && strcmp(params->file_extension, "scn") == 0) saveImage3d(current_segmentation, images[i].image_name, j, params);

      if (params->relaxation == 1 && ((exp_results[i].prev_distance_error == exp_results[i].distance_error_object[0]) || (j == params->n_segm_iterations - 1)))
      {
        printf("Relaxing at last iteration %d\n", j);
        iftFillBMap(forest->processed, 1);
        t1 = iftTic();
        iftRelaxObjects(forest, smooth);
        t2 = iftToc();
        iteration_time = iftCompTime(t1, t2);
        exp_results[i].total_time += iteration_time;
        printf("%d\n", current_segmentation == forest->label);
        current_segmentation = forest->label;

        if (params->save_ouput && strcmp(params->file_extension, "scn") == 0) saveImage3d(current_segmentation, images[i].image_name, j, params);
        pvtComputeErrors(exp_results[i].distance_error_object, exp_results[i].dice_error_object, params, current_segmentation, gt_image, object_edt_gt);

        printf("Mean EDT relaxing last: %f Time: %.2f s\n", exp_results[i].distance_error_object[0], iteration_time / SECONDS);
        stop = 1;
      }
      iftIsSegmentationConsistent(forest);
      if (stop || exp_results[i].prev_distance_error == exp_results[i].distance_error_object[0])
        break;
      exp_results[i].prev_distance_error = exp_results[i].distance_error_object[0];
    }
    /* =================== Save the results =================== */
    if (j >= params->n_segm_iterations) j = params->n_segm_iterations - 1;



    iftImage *error_image = iftCreateImage(current_segmentation->xsize, current_segmentation->ysize, current_segmentation->zsize);
    for(p = 0; p< error_image->n; p++)
    {
      if (current_segmentation->val[p] == gt_image->val[p])
      {
        if (current_segmentation->val[p] == 0)
          error_image->val[p] = 0;
        else
          error_image->val[p] = 1;
      }
      else
        error_image->val[p] = 2;
    }
    saveImage3d(error_image, "ErrorImage.scn", 999, params);














    strcpy(exp_results[i].image_name, images[i].image_name);
    exp_results[i].distance_error  = exp_results[i].distance_error_object[0];
    exp_results[i].dice_error      = exp_results[i].dice_error_object[0];
    exp_results[i].final_iteration = (j + 1); //Starts in 0
    final_iteration = j;

    if (params->verbose > 0)
      fprintf(stdout, "FinalASDError: %f mm\t Dice: %f\tExecutions: %f\tSeeds added: %.0f\tTot. Time: %f s\n\n",
              exp_results[i].distance_error, exp_results[i].dice_error, exp_results[i].final_iteration, exp_results[i].seeds_added, exp_results[i].total_time / SECONDS);

    //Destroy all for next image
    iftDestroyImage(&original_image);
    iftDestroyImage(&gt_image);
    iftDestroyImage(&seeds_image);
    iftDestroyBMap(&seeds_bmap);
    iftDestroyAdjRel(&spheric15);
    iftDestroyImageForest(&forest);
    pvtDestroyObjectEDTFromGT(params, object_edt_gt);
    iftDestroySmoothBorder(&smooth);

  }
  /* =================== Report to File =================== */
  printReport(params, exp_results, "screen", "IFT");
  printReport(params, exp_results, "file", "IFT");

  /* ================== Destroy Everything ================ */
  destroyExperimentResults(exp_results, params);
}

int main(int argc, char **argv)
{
  if (argc != 6)
    iftError("<configuration_file> <images_directory> <gt_directory> <smooth strategy=0|1|2> <smooth iterations>", "Robot - main");
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
  fprintf(stdout, "WARNING: Relaxation parameter overwritten by program call parameter\n");
  fprintf(stdout, "Config file: %d  --->  Running with: %d\n", current_parameters->relaxation, atoi(argv[4]));
  fprintf(stdout, "===================================================================\n");

  current_parameters->relaxation = atoi(argv[4]);
  current_parameters->smooth_iterations = atoi(argv[5]);
  runRobot(current_parameters, img_names, gt_names);

  /* End Main code */
  destroyExperimentParameters(all_parameters);
  destroyExperimentParameters(current_parameters);
  iftDestroyImageNames(img_names);
  iftDestroyImageNames(gt_names);

  return 0;
}
