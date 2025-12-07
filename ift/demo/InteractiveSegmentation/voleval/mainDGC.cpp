//This inclusion must precede ift.h because the latter defines the macros MAX and MIN.
#include "graph.h"

#define class class_
extern "C" {
#include "ift.h"
#include "common.h"
}
#undef class

typedef Graph<double, double, double> GraphType;

GraphType *create_graph(iftRegionGraph *region_graph, float beta)
{
    GraphType *graph = new GraphType(region_graph->nnodes, region_graph->nnodes);
    iftDataSet *dataset = region_graph->dataset;

    float max_dist = -INFINITY_FLT;
    float min_dist = INFINITY_FLT;
    int r;
    for (r = 0; r < region_graph->nnodes; r++)
    {
        graph->add_node();

        iftSet *adj = region_graph->node[r].adjacent;
        while (adj)
        {
            int v = adj->elem;
            float dist = dataset->iftArcWeight(dataset->sample[r].feat, dataset->sample[v].feat, dataset->alpha, dataset->nfeats);
            if (dist > max_dist)
                max_dist = dist;
            if (dist < min_dist)
                min_dist = dist;
            adj = adj->next;
        }
    }

    for (r = 0; r < region_graph->nnodes; r++)
    {
        iftSet *adj = region_graph->node[r].adjacent;
        while (adj)
        {
            int v = adj->elem;
            if (r < v)
            {
                float dist = dataset->iftArcWeight(dataset->sample[r].feat, dataset->sample[v].feat, dataset->alpha, dataset->nfeats);
                dist = (dist - min_dist) / (max_dist - min_dist);
                float similarity = exp(-dist / 0.5);

                double edge_weight = pow(similarity, beta);
                graph->add_edge(r, v, edge_weight, edge_weight);
            }

            adj = adj->next;
        }
    }

    return graph;
}

iftImage *graph_cut_diff_segmentation(GraphType *graph, int nnodes, iftImage *label_image, iftLabeledSet *new_seeds)
{
    iftBMap *labeled = iftCreateBMap(nnodes);
    int r;
    iftLabeledSet *s = new_seeds;
    while (s)
    {
        int p = s->elem;

        r = label_image->val[p] - 1;

        if (!iftBMapValue(labeled, r))
        {
            int label = s->label;

            if (label == 1)
                graph->edit_tweights(r, INFINITY_FLT, 0);
            else
                graph->edit_tweights(r, 0, INFINITY_FLT);

            iftBMapSet1(labeled, r);
        }

        s = s->next;
    }

    graph->maxflow();

    iftImage *segmentation = iftCreateImage(label_image->xsize, label_image->ysize, label_image->zsize);

    int p;
    for (p = 0; p < segmentation->n; p++)
    {
        r = label_image->val[p] - 1;

        if (graph->what_segment(r) == GraphType::SOURCE)
            segmentation->val[p] = 1;
    }

    iftDestroyBMap(&labeled);

    return segmentation;
}

//int main(int argc, char **argv) {
//  if(argc != 9)
//      iftError("Usage: gc_super [IMAGE_PATH] [SEEDS_PATH1] [SEEDS_PATH2] [OUTPUT_PATH1] [OUTPUT_PATH2] [SPATIAL_RADIUS] [VOLUME THRESHOLD] [BETA]", "gc_super");
//
//  float spatial_radius = atof(argv[6]);
//  int volume_threshold = atoi(argv[7]);
//  float beta = atof(argv[8]);
//
//  iftImage *image = iftReadImageP6(argv[1]);
//  iftLabeledSet *seeds1 = iftReadSeeds2D(argv[2], image);
//  iftLabeledSet *seeds2 = iftReadSeeds2D(argv[3], image);
//
//  iftAdjRel *adj = iftCircular(spatial_radius);
//  iftAdjRel *adj1 = iftCircular(1.0);
//
//  iftImage *basins = iftImageBasins(image, adj);
//  iftImage *marker = iftVolumeClose(basins, volume_threshold);
//  iftImage *label = iftWaterGray(basins, marker, adj);
//
//  iftDataSet *dataset = iftSupervoxelsToDataSet(image, label);
//  dataset->alpha[0] = 0.2;
//
//  iftRegionGraph *region_graph = iftRegionGraphFromLabelImage(label, dataset, adj1);
//
//  GraphType* graph = create_graph(region_graph, beta);
//
//  iftImage *result = graph_cut_diff_segmentation(graph, region_graph->nnodes, label, seeds1);
//
//  //First result
//  iftWriteImageP5(result, argv[4]);
//
//  iftDestroyImage(&result);
//
//  result = graph_cut_diff_segmentation(graph, region_graph->nnodes, label, seeds2);
//
//  //Second result
//  iftWriteImageP5(result, argv[5]);
//
//  iftDestroyImage(&result);
//
//  return 0;
//}
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

    int i = 0, j = 0, o = 0, p = 0, k = 0, seeds_added = 0, final_iteration;
    float *asd_array = NULL, iteration_time;
    float final_fscore = 0, final_dice = 0, final_distance = 0;
    char buffer[2048];

    iftImage        *original_image          = NULL;
    iftImage        *gt_image                = NULL;
    iftImage        *gt_image_kobj           = NULL;
    iftImage        *label_image             = NULL;
    iftImage        *current_segmentation    = NULL;
    iftImage        *seeds_image             = NULL;
    iftImage        *seeds_image_copy        = NULL;
    iftImage        *basins                  = NULL;
    iftImage        *marker                  = NULL;
    iftImage        *temp_image              = NULL;
    iftFImage       **object_edt_gt          = NULL;

    iftBMap         *seeds_bmap              = NULL;

    iftAdjRel       *spheric1                = iftSpheric(1.0);
    iftAdjRel       *radius15                = iftSpheric(1.5);
    iftAdjRel       *radiussqrt3             = iftSpheric(sqrtf(3.0));
    iftAdjRel       *basins_adjacency        = iftSpheric(params->spatial_radius);
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

    printHeader(params, "file", "DGC");
    printHeader(params, "screen", "DGC");

    /* =================== Loop through all images =================== */
    for (i = 0; i < params->number_of_images; i++)
    {
        pvtOpenImage(params, images[i].image_name, gtruths[i].image_name, &original_image, &gt_image);
        params->curr_number_of_objects = iftMaximumValue(gt_image);

        /* =================== Superpixel creation =================== */
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

        /* =================== Initialization =================== */
        exp_metrics[i].seeds_added = 0;
        strcpy(exp_metrics[i].image_name, images[i].image_name);
        basins = iftImageBasins(original_image, basins_adjacency);

        /* ========== Generate the GT of only one object ============= */
        for (k = 1; k <= params->curr_number_of_objects; k++)
        {
            printf("Processing object: %d\n", k, params->curr_number_of_objects);
            gt_image_kobj       = iftCreateImage(gt_image->xsize, gt_image->ysize, gt_image->zsize);
            for (p = 0; p < gt_image->n; p++)
            {
                if (gt_image->val[p] == k)
                    gt_image_kobj->val[p] = 1;
            }

            /* I may need to reinitialize these variables: seeds_bmap, seeds_image, current_segmentation, temp_time */
            current_segmentation = NULL;
            seeds_bmap = iftCreateBMap(original_image->n);
            seeds_image = iftCreateImage(original_image->xsize, original_image->ysize, original_image->zsize);
            iftSetImage(seeds_image, -1);

            t1 = iftTic();
            GraphType *graph = create_graph(region_graph, params->gc_beta);
            t2 = iftToc();
            exp_metrics[i].setup_time += iftCompTime(t1, t2) / 1000;

            //This cannot be removed, because iftASSDError expects label from 0 to N.
            object_edt_gt = (iftFImage **) malloc(sizeof(iftFImage *) * 2);
            object_edt_gt[0] = NULL;
            object_edt_gt[1] = iftSignedDistTrans(gt_image_kobj, radiussqrt3);

            /* =================== Pixel Robot =================== */
            //if (strcmp(params->robot_strategy, "PIXEL") == 0) available_seeds = iftBorderMarkersForPixelSegmentation(basins, gt_image_kobj, params->border_distance);

            for (j = 0; j < params->n_segm_iterations; j++)
            {
                /* =================== Geodesic Robot =================== */
                if (strcmp(params->robot_strategy, "GEODESIC") == 0)
                    available_seeds = iftGeodesicMarkersForSegmentation(gt_image_kobj, current_segmentation);

                seeds_image_copy = iftCopyImage(seeds_image);
                seeds_added = iftMarkersFromMisclassifiedSeeds(seeds_image, available_seeds, seeds_bmap, params->seeds_per_iteration, params->curr_number_of_objects + 1, gt_image_kobj, current_segmentation, params->min_distance_border, params->max_marker_size, params->min_marker_size);

                for (p = 0; p < seeds_image_copy->n; p++)
                {
                    if (seeds_image_copy->val[p] == seeds_image->val[p])
                        seeds_image_copy->val[p] = -1;
                    else
                        seeds_image_copy->val[p] = seeds_image->val[p];
                }

                current_seeds = iftLabeledSetFromSeedImage(seeds_image_copy);

                /* =================== Segmentation =================== */

                t1 = iftTic();
                current_segmentation = graph_cut_diff_segmentation(graph, region_graph->nnodes, label_image, current_seeds);
                t2 = iftToc();

                /* ====================  Measures ===================== */
                iteration_time = iftCompTime(t1, t2);
                exp_metrics[i].total_time += iteration_time;


                asd_array = NULL;
                asd_array = iftASDError(current_segmentation, object_edt_gt, 1);

                exp_metrics[i].distance_error_matrix[j][k] = asd_array[0] * original_image->dx;  //Copy already in mm
                exp_metrics[i].dice_error_matrix[j][k]     = iftDiceSimilarity(current_segmentation, gt_image_kobj); //iftAccuracyGivenErrors(&errors);

                exp_metrics[i].seeds_added        += seeds_added;

                /* =================== Prints and Destroys =================== */
                if (params->verbose > 1)  fprintf(stdout, "j= %4d ASD= %10f seeds_added=%2d dice=%f time=%f s\n", j, exp_metrics[i].distance_error_matrix[j][k], seeds_added, exp_metrics[i].dice_error_matrix[j][k], iteration_time / 1000);
                if (params->save_ouput && strcmp(params->file_extension, "scn") == 0) saveImage3d(current_segmentation, images[i].image_name, j, params);

                if (strcmp(params->robot_strategy, "GEODESIC") == 0) iftDestroyLabeledSet(&available_seeds);
                iftDestroyLabeledSet(&current_seeds);
                iftDestroyImage(&seeds_image_copy);
                free(asd_array);

                if (seeds_added == 0) break;
            } /* END For each segmentation iteration */
            if (j == params->n_segm_iterations) j--;
            exp_metrics[i].final_iteration += (j + 1); //Starts in 0
            exp_metrics[i].distance_error += exp_metrics[i].distance_error_matrix[j][k] / (params->curr_number_of_objects * 1.0);
            exp_metrics[i].dice_error     += exp_metrics[i].dice_error_matrix[j][k] / (params->curr_number_of_objects * 1.0);

            final_iteration = j;
            exp_metrics[i].arr_final_iteration[k] = final_iteration;
            //Fill the matrices with the last iteration values.
            for (; j < params->n_segm_iterations; j++)
            {
                exp_metrics[i].distance_error_matrix[j][k] = exp_metrics[i].distance_error_matrix[final_iteration][k];
                exp_metrics[i].dice_error_matrix[j][k]     = exp_metrics[i].dice_error_matrix[final_iteration][k];
            }
            if (params->verbose > 0)
                fprintf(stdout, "Object %d: FinalASDError: %f mm\t Dice: %f\tExecutions: %f\tSeeds added: %.0f\tTot. Time: %f s\n\n", k, exp_metrics[i].distance_error_matrix[final_iteration][k], exp_metrics[i].dice_error_matrix[final_iteration][k], exp_metrics[i].final_iteration, exp_metrics[i].seeds_added, exp_metrics[i].total_time / 1000);

            delete graph;
            iftDestroyImage(&current_segmentation);
            iftDestroyBMap(&seeds_bmap);
            iftDestroyImage(&gt_image_kobj);
            iftDestroyImage(&seeds_image);
            iftDestroyFImage(&object_edt_gt[1]);
            free(object_edt_gt);

        } /* END For each object in the image */
        /* =================== Before moving to the next image =================== */
        fprintf(stdout, "Distance error\n");
        printf("%d: %.2f\t", 0, exp_metrics[i].distance_error);
        for (k = 1; k <= params->curr_number_of_objects; k++)
            printf("%d: %.2f\t", k, exp_metrics[i].distance_error_matrix[(int)exp_metrics[i].arr_final_iteration[k]][k]);
        fprintf(stdout, "\n");

        iftDestroyImage(&original_image);
        iftDestroyImage(&gt_image);
        iftDestroyImage(&basins);
        iftDestroyLabeledSet(&available_seeds);
        iftDestroyLabeledSet(&current_seeds);

        iftDestroyImage(&label_image);
        iftDestroyDataSet(&dataset);
        iftDestroyRegionGraph(&region_graph);
    } /* END For each image */

    /* =================== Report to File =================== */
    printReport(params, exp_metrics, "screen", "DGC");
    printReport(params, exp_metrics, "file", "DGC");

    /* ================== Destroy Everything ================ */
    iftDestroyAdjRel(&basins_adjacency);
    iftDestroyAdjRel(&radius15);
    iftDestroyAdjRel(&radiussqrt3);
    iftDestroyAdjRel(&spheric1);
    iftDestroyAdjRel(&superpixel_adj);
    destroyExperimentMetrics(exp_metrics, params);
}


int main(int argc, char **argv)
{
    if (argc != 4)
        iftError("<configuration_file> <superpixel_descriptor 0=mean, 1=histo, 2=avg,min,max> <method 0=watergray 1=slic>", "Voleval - main");
    //int i = 0;
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

    /* Main code */

    current_parameters->superpixel = 1;
    current_parameters->nonsuperpixel = 0;
    runRobot(current_parameters, img_names, gt_names);


    /* End Main code */
    destroyExperimentParameters(all_parameters);
    destroyExperimentParameters(current_parameters);
    iftDestroyImageNames(img_names);
    iftDestroyImageNames(gt_names);

    return 0;
}
