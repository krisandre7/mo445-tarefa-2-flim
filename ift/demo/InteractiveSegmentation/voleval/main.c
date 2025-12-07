#include "ift.h"
#include "common.h"

/* =================== Private Functions ====================== */
void pvtOpenImage(ExperimentParameters *params, char *image_name, char *gt_name, iftImage **img, iftImage **gt)
{
    char path[256];
    if (params->verbose > 0)
        fprintf(stdout, "Processing: %s and %s\n", image_name, gt_name);

    strcpy(path, params->image_database);
    strcat(path, image_name);
    if (strcmp(params->file_extension, "scn") == 0)
        *img = iftReadImage(path);
    else
        *img = iftReadImageP6(path);

    strcpy(path, params->label_database);
    strcat(path, gt_name);
    if (strcmp(params->file_extension, "scn") == 0)
        *gt = iftReadImage(path);
    else
        *gt = iftReadImageP5(path);
}


void runRobot (ExperimentParameters *params, iftImageNames *images, iftImageNames *gtruths)
{

    /* =================== Declarations ======================= */

    int i = 0, j = 0, o = 0, p = 0, seeds_added = 0, final_iteration;
    float iteration_time = 0;

    char buffer[2048];

    iftImage        *original_image          = NULL;
    iftImage        *gt_image                = NULL;
    iftImage        *label_image             = NULL;
    iftImage        *current_segmentation    = NULL;
    iftImage        *seeds_image             = NULL;
    iftImage        *seeds_image_copy        = NULL;
    iftImage        *basins                  = NULL;
    iftImage        *marker                  = NULL;
    iftImage        *temp_image              = NULL;
    iftFImage       **object_edt_gt          = NULL;

    iftImageForest  *forest                  = NULL;
    iftBMap         *seeds_bmap              = NULL;

    iftAdjRel       *radiussqrt3             = iftSpheric(sqrtf(3.0));
    iftAdjRel       *basins_adjacency        = iftSpheric(params->spatial_radius);
    iftAdjRel       *radius15                = iftSpheric(1.5);
    iftAdjRel       *spheric1                = iftSpheric(1.0);
    iftAdjRel       *superpixel_adj          = iftSpheric(params->superpixel_adj);

    iftLabeledSet   *available_seeds         = NULL;
    iftLabeledSet   *current_seeds           = NULL;

    iftDataSet      *dataset                 = NULL;
    iftRegionGraph  *region_graph            = NULL;

    timer *t1                                = NULL;
    timer *t2                                = NULL;


    ExpMetrics *exp_metrics = NULL;


    /* =================== Allocations ======================= */
    exp_metrics           = createExperimentMetrics(params);

    printHeader(params, "file", "IFT");
    printHeader(params, "screen", "IFT");

    /* =================== Loop through all images =================== */
    for (i = 0; i < params->number_of_images; i++)
    {
        pvtOpenImage(params, images[i].image_name, gtruths[i].image_name, &original_image, &gt_image);
        params->curr_number_of_objects = iftMaximumValue(gt_image);

        /* =================== Superpixel creation =================== */
        if (params->superpixel)
        {
            t1 = iftTic();
            if (params->supervoxel_method == 0)
            {
                basins = iftImageBasins(original_image, superpixel_adj);
                marker = iftVolumeClose(basins, params->superpixel_volume);
                label_image = iftWaterGray(basins, marker, superpixel_adj);
                iftDestroyImage(&basins);
                iftDestroyImage(&marker);
            }
            else 
            {
                label_image = iftCreateImage(original_image->xsize, original_image->ysize, original_image->zsize);
                iftWriteRawIntArray(images[i].image_name, original_image->val, original_image->n);

                sprintf(buffer, "python slic3d.py %d %d %d %s %s %d %.2f", original_image->xsize, original_image->ysize, original_image->zsize,
                                                                         images[i].image_name, images[i].image_name,
                                                                         params->slic_nregions, params->slic_compactness);
                puts(buffer);
                system(buffer);
                free(label_image->val);
                label_image->val = iftReadRawIntArray(images[i].image_name, original_image->n);
                sprintf(buffer, "rm %s", images[i].image_name);
                system(buffer);
            }

            //Generate paper images only
            // iftImage *drawn_labels = iftCopyImage(original_image);
            // iftAdjRel *adj1 = iftSpheric(1.0);
            // iftAdjRel *adj0 = iftSpheric(0.0);
            // iftColor blue;
            // blue.val[0] = 4095; blue.val[1] = 4095; blue.val[2] = 4095;
            // blue = iftRGBtoYCbCr(blue);
            // iftDrawBorders(drawn_labels, label_image,adj1,blue,adj0);
            // iftWriteImage(drawn_labels, "superpixels.scn");

            if (params->superpixel_descriptor == 0)
                dataset = iftSupervoxelsToDataSet(original_image, label_image);
            else if (params->superpixel_descriptor == 1)
                dataset =   iftSupervoxelsToHistogramDataSet(original_image, label_image, params->nbins, params->bpp);
            else if (params->superpixel_descriptor == 2)
                dataset = iftSupervoxelsToAvgMinMaxDataSet(original_image, label_image);

            region_graph = iftRegionGraphFromLabelImage(label_image, dataset, spheric1);

            t2 = iftToc();
            exp_metrics[i].setup_time = iftCompTime(t1, t2) / 1000;
            exp_metrics[i].number_of_superpixels = (int) iftMaximumValue(label_image);
            fprintf(stdout, "Number of superpixels: %.0f\n", exp_metrics[i].number_of_superpixels);

        }
        else
            exp_metrics[i].number_of_superpixels = 0.0;

        /* =================== Initialization =================== */

        current_segmentation = NULL;
        available_seeds = NULL;

        strcpy(exp_metrics[i].image_name, images[i].image_name);
        exp_metrics[i].seeds_added = 0;
        exp_metrics[i].total_time = 0.0;

        seeds_bmap = iftCreateBMap(original_image->n);
        seeds_image = iftCreateImage(original_image->xsize, original_image->ysize, original_image->zsize);
        iftSetImage(seeds_image, -1);

        t1 = iftTic();
        basins = iftImageBasins(original_image, basins_adjacency);
        t2 = iftToc();
        if (!params->superpixel)
        {
            exp_metrics[i].number_of_superpixels = 0;
            exp_metrics[i].setup_time = iftCompTime(t1, t2) / 1000;
        } 

        forest = iftCreateImageForest(basins, radius15);
        iftMaximumValue(forest->img);

        //This may be removed in case of iftASSDError
        printf("Generating individual GTs for ASD... ");
        object_edt_gt = (iftFImage **) malloc(sizeof(iftFImage *) * (params->curr_number_of_objects + 2));
        object_edt_gt[0] = NULL;
        for (o = 1; o <= params->curr_number_of_objects; o++)
        {
            //printf("Object %d\n", o);
            int flag = 0;
            temp_image = iftCreateImage(gt_image->xsize, gt_image->ysize, gt_image->zsize);
            for (p = 0; p < gt_image->n; p++)
            {
                if (gt_image->val[p] == o)
                    temp_image->val[p] = 1;
            }
            object_edt_gt[o] = iftSignedDistTrans(temp_image, radiussqrt3);
            iftDestroyImage(&temp_image);
        }
        printf("[DONE]\n");

        /* =================== Pixel Robot =================== */
        if (strcmp(params->robot_strategy, "PIXEL") == 0) available_seeds = iftBorderMarkersForPixelSegmentation(basins, gt_image, params->border_distance);

        for (j = 0; j < params->n_segm_iterations; j++)
        {
            /* =================== Geodesic Robot =================== */
            if (strcmp(params->robot_strategy, "GEODESIC") == 0)
            {
                iftDestroyLabeledSet(&available_seeds);
                available_seeds = iftGeodesicMarkersForSegmentation(gt_image, current_segmentation);
                if (params->superpixel)
                    iftDestroyImage(&current_segmentation);
            }

            seeds_image_copy = iftCopyImage(seeds_image);
            seeds_added = iftMarkersFromMisclassifiedSeeds(seeds_image, available_seeds, seeds_bmap, params->seeds_per_iteration, params->curr_number_of_objects + 1, gt_image, current_segmentation, params->min_distance_border, params->max_marker_size, params->min_marker_size);

            for (p = 0; p < seeds_image_copy->n; p++)
            {
                if (seeds_image_copy->val[p] == seeds_image->val[p])
                    seeds_image_copy->val[p] = -1;
                else
                    seeds_image_copy->val[p] = seeds_image->val[p];
            }
            //This produces only the new seeds added this iteration
            if (params->ift_differential)
                current_seeds = iftLabeledSetFromSeedImage(seeds_image_copy);
            else
                current_seeds = iftLabeledSetFromSeedImage(seeds_image);

            //Generate paper images only
            // iftImage *copy = iftCopyImage(seeds_image);
            // char buffer[512];
            // sprintf(buffer, "seeds%d.scn", j);
            // iftWriteImage(copy, buffer);
            // printf("Saved\n");

            /* =================== Segmentation =================== */
            if (params->superpixel)
            {
                t1 = iftTic();
                if (params->ift_differential)
                    iftDiffSuperpixelClassification(region_graph, label_image, current_seeds);
                else
                    iftSuperpixelClassification(region_graph, label_image, current_seeds);
                t2 = iftToc();
                current_segmentation = iftSuperpixelLabelImageFromDataset(label_image, region_graph->dataset);
            }
            else
            {
                t1 = iftTic();
                if (params->ift_differential)
                    iftDiffWatershedTree(forest, current_seeds);
                else
                {
                    if (forest->label != NULL) iftDestroyImage(&forest->label);
                    forest->label = iftWatershed(basins, forest->A, current_seeds, NULL);
                }
                t2 = iftToc();
                current_segmentation = forest->label;
            }
            iteration_time = iftCompTime(t1, t2);
            exp_metrics[i].total_time += iteration_time;


            /* =================== Error measure =================== */
            iftDblArray *asd_array = iftASDMultiLabel(current_segmentation, object_edt_gt, params->curr_number_of_objects);

            // Compute the error in (mm)
            for (o = 0; o <= params->curr_number_of_objects; o++)
                exp_metrics[i].distance_error_matrix[j][o] = (asd_array->val[o] * original_image->dx) ;

            iftDblArray *dices = iftDiceSimilarityMultiLabel(current_segmentation, gt_image, params->curr_number_of_objects);
            exp_metrics[i].dice_error_matrix[j] = dices->val;
            dices->val = NULL;
            iftDestroyDblArray(&dices);
            //EM->fscore_error_matrix[i][j]    = iftFScoreMultiLabel(current_segmentation, gt_image, params->curr_number_of_objects);
            exp_metrics[i].seeds_added += seeds_added;
            exp_metrics[i].prev_distance_error = asd_array->val[0];

            /* =================== Prints and Destroys =================== */
            if (params->verbose > 1)  fprintf(stdout, "j= %4d ASD= %10f seeds_added=%2d dice=%f time=%f s\n", j, exp_metrics[i].distance_error_matrix[j][0], seeds_added, exp_metrics[i].dice_error_matrix[j][0], iteration_time / 1000);
            if (params->verbose == 3) for (o = 0; o <= params->curr_number_of_objects; o++) fprintf(stdout, "Obj %d, ASD=%f dice=%f\n", o, exp_metrics[i].distance_error_matrix[j][o], exp_metrics[i].dice_error_matrix[j][o]);
            if (params->save_ouput && strcmp(params->file_extension, "scn") == 0) saveImage3d(current_segmentation, images[i].image_name, j, params);

            iftDestroyLabeledSet(&current_seeds);
            iftDestroyImage(&seeds_image_copy);
            free(asd_array);

            if (seeds_added == 0) break;
        }

        /* =================== Save the results =================== */
        if (j == params->n_segm_iterations) j--;

        exp_metrics[i].distance_error  = exp_metrics[i].distance_error_matrix[j][0];
        exp_metrics[i].dice_error      = exp_metrics[i].dice_error_matrix[j][0];
        exp_metrics[i].final_iteration = (j + 1); //Starts in 0

        final_iteration = j;
        //All objects finished at the same iteration
        for (int k=0; k<params->curr_number_of_objects; k++)
            exp_metrics[i].arr_final_iteration[k] = final_iteration;

        //Fill the matrices with the last iteration values.
        for (; j < params->n_segm_iterations; j++)
        {
            for (o = 0; o <= params->curr_number_of_objects; o++)
            {
                exp_metrics[i].distance_error_matrix[j][o] = exp_metrics[i].distance_error_matrix[final_iteration][o];
                exp_metrics[i].dice_error_matrix[j][o]     = exp_metrics[i].dice_error_matrix[final_iteration][o];
            }
        }
        if (params->verbose > 0)
            fprintf(stdout, "FinalASDError: %f mm\t Dice: %f\tExecutions: %f\tSeeds added: %.0f\tTot. Time: %f s\n\n", exp_metrics[i].distance_error, exp_metrics[i].dice_error, exp_metrics[i].final_iteration, exp_metrics[i].seeds_added, exp_metrics[i].total_time / 1000);
        iftDestroyImage(&original_image);
        iftDestroyImage(&gt_image);
        iftDestroyImage(&seeds_image);
        iftDestroyImage(&basins);
        iftDestroyBMap(&seeds_bmap);
        iftDestroyLabeledSet(&available_seeds);
        iftDestroyLabeledSet(&current_seeds);
        iftDestroyImageForest(&forest);

        iftDestroyImage(&label_image);
        iftDestroyDataSet(&dataset);
        iftDestroyRegionGraph(&region_graph);

        for (o = 1; o <= params->curr_number_of_objects; o++)
            iftDestroyFImage(&object_edt_gt[o]);

    }
    /* =================== Report to File =================== */
    printReport(params, exp_metrics, "screen", "IFT");
    printReport(params, exp_metrics, "file", "IFT");


    /* ================== Destroy Everything ================ */
    destroyExperimentMetrics(exp_metrics, params);
    iftDestroyAdjRel(&basins_adjacency);
    iftDestroyAdjRel(&radius15);
    iftDestroyAdjRel(&radiussqrt3);
    iftDestroyAdjRel(&spheric1);
    iftDestroyAdjRel(&superpixel_adj);
}

int main(int argc, char **argv)
{
    if (argc != 4)
        iftError("<configuration_file> <superpixel_descriptor 0=mean, 1=histo, 2=avg,min,max> <method 0=watergray 1=slic>", "Voleval - main");
    char *configuration_file = argv[1];
    iftImageNames *img_names, *gt_names;
    ExperimentParameters *all_parameters = createExperimentParameters();
    ExperimentParameters *current_parameters = createExperimentParameters();

    loadParameters(all_parameters, configuration_file);
    loadParameters(current_parameters, configuration_file);

    all_parameters->superpixel_descriptor = atoi(argv[2]);
    current_parameters->superpixel_descriptor = atoi(argv[2]);

    all_parameters->supervoxel_method = atoi(argv[3]);
    current_parameters->supervoxel_method = atoi(argv[3]);

    int number_of_images = iftCountImageNames(all_parameters->image_database, all_parameters->file_extension);

    all_parameters->number_of_images = number_of_images;
    current_parameters->number_of_images = number_of_images;

    img_names = iftCreateAndLoadImageNames(number_of_images, all_parameters->image_database, all_parameters->file_extension);
    gt_names = iftCreateAndLoadImageNames(number_of_images, all_parameters->label_database, "scn");


    all_parameters->max_number_of_objects = getMaximumNumberOfObjects(all_parameters, gt_names);
    current_parameters->max_number_of_objects = all_parameters->max_number_of_objects;
    if (all_parameters->max_number_of_objects == 256)   iftWarning("Your groundtruth might be 0-255 instead of 0,1,2,3,...", "demo/relaxedIFT/Main");
    else if (all_parameters->max_number_of_objects == 0) iftError("Zero objects to segment, GT error probably", "demo/relaxedIFT/Main");

    if (all_parameters->superpixel)
    {
        current_parameters->superpixel = 1;
        current_parameters->nonsuperpixel = 0;
        runRobot(current_parameters, img_names, gt_names);
    }
    if (all_parameters->nonsuperpixel)
    {
        current_parameters->superpixel = 0;
        current_parameters->nonsuperpixel = 1;
        runRobot(current_parameters, img_names, gt_names);
    }

    destroyExperimentParameters(all_parameters);
    destroyExperimentParameters(current_parameters);
    iftDestroyImageNames(img_names);
    iftDestroyImageNames(gt_names);

    return 0;
}
