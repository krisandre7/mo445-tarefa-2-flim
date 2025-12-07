#include "ift.h"
#include "common.h"

int CONVERGENCE_COUNTER = 0;
float PREVIOUS_EDT_ERROR = 0;

ExperimentParameters *createExperimentParameters (char *extension)
{
    ExperimentParameters *params = (ExperimentParameters *) malloc (sizeof(ExperimentParameters));
    return (params);
}

void destroyExperimentParameters (ExperimentParameters *params)
{
    if (params != NULL)
    {
        free(params);
        params = NULL;
    }
}

ExpMetrics *createExperimentMetrics (ExperimentParameters *params)
{
    ExpMetrics *expm = (ExpMetrics *) malloc(sizeof(ExpMetrics) * params->number_of_images);
    for (int i = 0; i < params->number_of_images; i++)
    {
        strcpy(expm[i].image_name, "");
        expm[i].prev_distance_error = 0.0;

        expm[i].distance_error = 0.0;
        expm[i].dice_error = 0.0;
        expm[i].final_iteration = 0.0;
        expm[i].total_time = 0.0;
        expm[i].seeds_added = 0.0;

        // expm[i].arr_distance_error = iftAllocFloatArray(params->max_number_of_objects + 1);
        // expm[i].arr_dice_error = iftAllocFloatArray(params->max_number_of_objects + 1);
        // expm[i].arr_total_time = iftAllocFloatArray(params->max_number_of_objects + 1);
        expm[i].arr_final_iteration = iftAllocFloatArray(params->max_number_of_objects + 1);
        // expm[i].arr_seeds_added = iftAllocFloatArray(params->max_number_of_objects + 1);

        expm[i].distance_error_matrix = (float **) malloc(sizeof(float *)*params->n_segm_iterations);
        expm[i].dice_error_matrix = (float **) malloc(sizeof(float *)*params->n_segm_iterations);

        for (int j = 0; j < params->n_segm_iterations; j++)
        {
            expm[i].distance_error_matrix[j] = iftAllocFloatArray(params->max_number_of_objects + 1);
            expm[i].dice_error_matrix[j] = iftAllocFloatArray(params->max_number_of_objects + 1);
            for (int k = 0; k <= params->max_number_of_objects; k++)
            {
                expm[i].distance_error_matrix[j][k] = 0.0;
                expm[i].dice_error_matrix[j][k] = 0.0;
            }
        }
    }
    return expm;
}
void destroyExperimentMetrics(ExpMetrics *expM, ExperimentParameters *params)
{
    int i, j;
    for (i = 0; i < params->number_of_images; i++)
    {
        for (j = 0; j < params->n_segm_iterations; j++)
        {
            free(expM[i].distance_error_matrix[j]);
            free(expM[i].dice_error_matrix[j]);
        }
        // free(expM[i].arr_distance_error);
        // free(expM[i].arr_dice_error);
        // free(expM[i].arr_total_time);
        free(expM[i].arr_final_iteration);
        // free(expM[i].arr_seeds_added);

        free(expM[i].distance_error_matrix[i]);
        free(expM[i].dice_error_matrix[i]);
    }
    free(expM);
}


//Parse the config file to create an convolutional network
void loadParameters (ExperimentParameters *params, char *config_filename)
{
    FILE *fp = fopen(config_filename, "r");
    if (!fp) iftError("Configuration file not found", "Relaxed IFT - loadParameters");
    char line[512];
    char *pch;

    while (!feof(fp))
    {
        fgets(line, sizeof(char) * 512, fp);

        if (line[0] == '#' || strlen(line) == 0)
            continue;
        pch = strtok (line, ", ");
        if (strcmp(pch, "IMAGE_DATABASE") == 0)
        {
            pch = strtok (NULL, ", ");
            strcpy(params->image_database, pch);
            params->image_database[strlen(params->image_database) - 1] = '\0';
        }
        if (strcmp(pch, "LABEL_DATABASE") == 0)
        {
            pch = strtok (NULL, ", ");
            strcpy(params->label_database, pch);
            params->label_database[strlen(params->label_database) - 1] = '\0';
        }
        if (strcmp(pch, "FILE_EXTENSION") == 0)
        {
            pch = strtok (NULL, ", ");
            strcpy(params->file_extension, pch);
            params->file_extension[strlen(params->file_extension) - 1] = '\0';
        }
        else if (strcmp(pch, "OUTPUT_DIRECTORY") == 0)
        {
            pch = strtok (NULL, ", ");
            strcpy(params->output_directory, pch);
            params->output_directory[strlen(params->output_directory) - 1] = '\0';
        }
        else if (strcmp(pch, "ROBOT_STRATEGY") == 0)
        {
            pch = strtok (NULL, ", ");
            strcpy(params->robot_strategy, pch);
            params->robot_strategy[strlen(params->robot_strategy) - 1] = '\0';
        }
        else if (strcmp(pch, "SAVE_OUTPUT") == 0)
        {
            pch = strtok (NULL, ", ");
            params->save_ouput = atoi(pch);
        }
        else if (strcmp(pch, "VERBOSE") == 0)
        {
            pch = strtok (NULL, ", ");
            params->verbose = atoi(pch);
        }
        else if (strcmp(pch, "IFT_DIFFERENTIAL") == 0)
        {
            pch = strtok (NULL, ", ");
            params->ift_differential = atoi(pch);
        }
        else if (strcmp(pch, "BASINS_TYPE") == 0)
        {
            pch = strtok (NULL, ", ");
            params->basins_type = atoi(pch);
        }
        else if (strcmp(pch, "N_SEGM_ITERATIONS") == 0)
        {
            pch = strtok (NULL, ", ");
            params->n_segm_iterations = atoi(pch);
        }
        else if (strcmp(pch, "SEEDS_PER_ITERATION") == 0)
        {
            pch = strtok (NULL, ", ");
            params->seeds_per_iteration = atoi(pch);
        }
        else if (strcmp(pch, "MIN_DISTANCE_BORDER") == 0)
        {
            pch = strtok (NULL, ", ");
            params->min_distance_border = atoi(pch);
        }
        else if (strcmp(pch, "MAX_MARKER_SIZE") == 0)
        {
            pch = strtok (NULL, ", ");
            params->max_marker_size = atoi(pch);
        }
        else if (strcmp(pch, "MIN_MARKER_SIZE") == 0)
        {
            pch = strtok (NULL, ", ");
            params->min_marker_size = atoi(pch);
        }
        else if (strcmp(pch, "BORDER_DISTANCE") == 0)
        {
            pch = strtok (NULL, ", ");
            params->border_distance = atoi(pch);
        }
        else if (strcmp(pch, "SPATIAL_RADIUS") == 0)
        {
            pch = strtok (NULL, ", ");
            params->spatial_radius = atof(pch);
        }
        else if (strcmp(pch, "SUPERPIXEL_VOLUME") == 0)
        {
            pch = strtok (NULL, ", ");
            params->superpixel_volume = atof(pch);
        }
        else if (strcmp(pch, "SUPERPIXEL_ADJ") == 0)
        {
            pch = strtok (NULL, ", ");
            params->superpixel_adj = atof(pch);
        }
        else if (strcmp(pch, "SUPERPIXEL") == 0)
        {
            pch = strtok (NULL, ", ");
            params->superpixel = atoi(pch);
        }
        else if (strcmp(pch, "NONSUPERPIXEL") == 0)
        {
            pch = strtok (NULL, ", ");
            params->nonsuperpixel = atoi(pch);
        }
        else if (strcmp(pch, "GC_BETA") == 0)
        {
            pch = strtok (NULL, ", ");
            params->gc_beta = atof(pch);
        }
        else if (strcmp(pch, "NBINS") == 0)
        {
            pch = strtok (NULL, ", ");
            params->nbins = atoi(pch);
        }
        else if (strcmp(pch, "BPP") == 0)
        {
            pch = strtok (NULL, ", ");
            params->bpp = atoi(pch);
        }
        else if (strcmp(pch, "N_REGIONS") == 0)
        {
            pch = strtok (NULL, ", ");
            params->slic_nregions = atoi(pch);
        }
        else if (strcmp(pch, "COMPACTNESS") == 0)
        {
            pch = strtok (NULL, ", ");
            params->slic_compactness = atof(pch);
        }
        line[0] = 0;
    }
    fclose(fp);
}

void printHeader(ExperimentParameters *params, char *target, char *algorithm)
{
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char full_path[256], *database;
    database = iftSplitStringOld(params->image_database, "/", 2);
    FILE *output = NULL;
    sprintf(full_path, "./results/%s_%s_sv=%d_svmethod=%d.txt", database, algorithm, params->superpixel, params->supervoxel_method);

    if (strcmp(target, "screen") == 0) output = stdout;
    else                               output = fopen(full_path, "a");

    fprintf(output, "--------------------------------\n");
    fprintf(output, "%d/%d/%d %d:%d\n", tm.tm_mday, tm.tm_mon + 1, tm.tm_year + 1900, tm.tm_hour, tm.tm_min);
    fprintf(output, "Image Database: %s\n", params->image_database);
    fprintf(output, "Number of Images: %d\n", params->number_of_images);
    if(params->ift_differential)
        fprintf(output, "IFT Algorithm: DIFFERENTIAL\n");
    else
        fprintf(output, "IFT Algorithm: REGULAR\n");

    if (params->superpixel)
    {
        fprintf(output, "SV segmentation: True\n");
        fprintf(output, "SV adj: %g\n", params->superpixel_adj);
        if (params->supervoxel_method == 0)
            fprintf(output, "SV Method: %s\n", "Watergray");
        else if(params->supervoxel_method == 1)
            fprintf(output, "SV Method: %s\n", "Slic");

        if (params->superpixel_descriptor == 0)
            fprintf(output, "SV Descriptor: %s\n", "Mean color");
        else if (params->superpixel_descriptor == 1)
            fprintf(output, "SV Descriptor: %s\n", "Histogram");
        else if (params->superpixel_descriptor == 2)
            fprintf(output, "SV Descriptor: %s\n", "(Avg, min, max)");
        fprintf(output, "Volume Threshold: %g\n", params->superpixel_volume);
        if (strcmp(algorithm, "DGC") == 0)
            fprintf(output, "Beta: %.2f\n", params->gc_beta);
    }
    else
        fprintf(output, "Supervoxel segmentation: False\n");

    fprintf(output, "Robot: %s\n", params->robot_strategy);
    fprintf(output, "Segmentation Iterations: %d\n", params->n_segm_iterations);
    fprintf(output, "Seeds per Iteration: %d\n", params->seeds_per_iteration);
    fprintf(output, "Max Number of objects: %d\n", params->max_number_of_objects);

    fprintf(output, "--------------------------------\n");
    fprintf(output, "\n");
    if (strcmp(target, "file") == 0)
        fclose(output);
    //free(database);
}
float computeCorrectLabel(iftImage *label_image, iftImage *gt_image)
{
    float correct_label = 0;
    int p;

    for (p = 0; p < label_image->n; p++)
        if (label_image->val[p] == gt_image->val[p])
            correct_label++;
    return (float) correct_label / (label_image->n * 1.0);
}
void saveFinalValues(char *filename, float *array, int length)
{
    FILE *FP = fopen(filename, "w");
    int i = 0;
    for (i = 0; i < length; i++)
        fprintf(FP, "%f ", array[i]);
    fclose(FP);
}

//If I need this, make it work again
// void printIterationProfress(ExperimentParameters *params, ExpMetrics *exp_metrics)
// {
//     fprintf(output, "--------------------------------\n");
//     fprintf(output, "Average ASD per iteration\n");
//     for (j = 0; j < params->n_segm_iterations; j++)
//     {
//         fprintf(output, "%d:\t", j);
//         memset(temp_array, 0, sizeof(float)*params->number_of_images);
//         for (i = 0; i < params->number_of_images; i++)
//             temp_array[i] = EM->distance_error_matrix[i][j][0];
//         average = mean(temp_array, params->number_of_images);
//         std_dev = standard_deviation(temp_array, params->number_of_images);
//         fprintf(output, "%f +- %.2f\n", average, std_dev);
//     }

//     fprintf(output, "--------------------------------\n");
//     fprintf(output, "Dice per iteration\n");
//     for (j = 0; j < params->n_segm_iterations; j++)
//     {
//         fprintf(output, "%d:\t", j);
//         memset(temp_array, 0, sizeof(float)*params->number_of_images);
//         for (i = 0; i < params->number_of_images; i++)
//             temp_array[i] = EM->dice_error_matrix[i][j][0];

//         average = mean(temp_array, params->number_of_images);
//         std_dev = standard_deviation(temp_array, params->number_of_images);
//         fprintf(output, "%f +- %.2f\n", average, std_dev);
//     }
// }
void printReport(ExperimentParameters *params, ExpMetrics *exp_metrics, char *target, char *algorithm)
{
    FILE *output = NULL;
    int i = 0, j = 0, k = 0;
    char full_path[256], *database;
    float std_dev = 0, average = 0;
    float *final_error      = iftAllocFloatArray(params->number_of_images);
    float *final_iteration  = iftAllocFloatArray(params->number_of_images);
    float *temp_array       = iftAllocFloatArray(params->number_of_images);

    float worst_error, best_error;
    float worst_iteration, best_iteration;
    float worst_time, best_time;
    int index_worst_iteration, index_best_iteration;
    int index_worst_error, index_best_error;
    int index_worst_time, index_best_time;

    database = iftSplitStringOld(params->image_database, "/", 2);
    sprintf(full_path, "./results/%s_%s_sv=%d_svmethod=%d.txt", database, algorithm, params->superpixel, params->supervoxel_method);

    if (strcmp(target, "file") == 0)
        output = fopen(full_path, "a");
    else
        output = stdout;


    //Print tables
    memset(final_error, 0, sizeof(float)*params->number_of_images);
    memset(final_iteration, 0, sizeof(float)*params->number_of_images);
    memset(temp_array, 0, sizeof(float)*params->number_of_images);

    fprintf(output, "\n\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    
    fprintf(output, "ASD Mean: ");
    for (i = 0; i < params->number_of_images; i++)
        temp_array[i] = exp_metrics[i].distance_error;
    sprintf(full_path, "./results/stats/finalasd_%s_sup=%d_%s_desc=%d_svmethod=%d.txt", database, params->superpixel, algorithm, params->superpixel_descriptor, params->supervoxel_method);
    saveFinalValues(full_path, temp_array, params->number_of_images);
    average = iftMeanFloatArrayDiffZero(temp_array, params->number_of_images);
    std_dev = iftStddevFloatArray(temp_array, params->number_of_images);
    fprintf(output, "%d: %f +- %.2f;\t", k, average, std_dev);
    
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for (k = 1; k <= params->max_number_of_objects; k++)
    {
        for (i = 0; i < params->number_of_images; i++)
            temp_array[i] = exp_metrics[i].distance_error_matrix[(int)exp_metrics[i].arr_final_iteration[k]][k];
        average = iftMeanFloatArrayDiffZero(temp_array, params->number_of_images);
        std_dev = iftStddevFloatArray(temp_array, params->number_of_images);
        fprintf(output, "%d: %f +- %.2f;\t", k, average, std_dev);
    }


    fprintf(output, "\n--------------------------------\n");
    fprintf(output, "Dice Mean: ");
    for (i = 0; i < params->number_of_images; i++)
        temp_array[i] = exp_metrics[i].dice_error;
    sprintf(full_path, "./results/stats/finalasd_%s_sup=%d_%s_desc=%d_svmethod=%d.txt", database, params->superpixel, algorithm, params->superpixel_descriptor, params->supervoxel_method);
    saveFinalValues(full_path, temp_array, params->number_of_images);
    average = iftMeanFloatArrayDiffZero(temp_array, params->number_of_images);
    std_dev = iftStddevFloatArray(temp_array, params->number_of_images);
    fprintf(output, "%d: %f +- %.2f;\t", k, average, std_dev);
    
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for (k = 1; k <= params->max_number_of_objects; k++)
    {
        for (i = 0; i < params->number_of_images; i++)
            temp_array[i] = exp_metrics[i].dice_error_matrix[(int)exp_metrics[i].arr_final_iteration[k]][k];
        average = iftMeanFloatArrayDiffZero(temp_array, params->number_of_images);
        std_dev = iftStddevFloatArray(temp_array, params->number_of_images);
        fprintf(output, "%d: %f +- %.2f;\t", k, average, std_dev);
    }


    fprintf(output, "\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for (i = 0; i < params->number_of_images; i++)
        temp_array[i] = exp_metrics[i].seeds_added;
    average = iftMeanFloatArrayDiffZero(temp_array, params->number_of_images);
    std_dev = iftStddevFloatArray(temp_array, params->number_of_images);
    fprintf(output, "Seeds Added: %f +- %.2f &\t", average, std_dev);
    sprintf(full_path, "./results/stats/seedsadded_%s_sup=%d_%s_desc=%d_svmethod=%d.txt", database, params->superpixel, algorithm, params->superpixel_descriptor, params->supervoxel_method);
    saveFinalValues(full_path, temp_array, params->number_of_images);



    fprintf(output, "\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for (i = 0; i < params->number_of_images; i++)
        temp_array[i] = exp_metrics[i].number_of_superpixels;
    average = iftMeanFloatArrayDiffZero(temp_array, params->number_of_images);
    std_dev = iftStddevFloatArray(temp_array, params->number_of_images);
    fprintf(output, "Number of superpixels: %.0f +- %.2f &\t", average, std_dev);
    sprintf(full_path, "./results/stats/nbrsuperpixels%s_sup=%d_%s_desc=%d_svmethod=%d.txt", database, params->superpixel, algorithm, params->superpixel_descriptor, params->supervoxel_method);
    saveFinalValues(full_path, temp_array, params->number_of_images);


    fprintf(output, "\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for (i = 0; i < params->number_of_images; i++)
        temp_array[i] = exp_metrics[i].setup_time;
    average = iftMeanFloatArrayDiffZero(temp_array, params->number_of_images);
    std_dev = iftStddevFloatArray(temp_array, params->number_of_images);
    fprintf(output, "Setup time: %f +- %.2f &\t", average, std_dev);
    sprintf(full_path, "./results/stats/setuptime_%s_sup=%d_%s_desc=%d_svmethod=%d.txt", database, params->superpixel, algorithm, params->superpixel_descriptor, params->supervoxel_method);
    saveFinalValues(full_path, temp_array, params->number_of_images);

    fprintf(output, "\n");



    index_worst_iteration = index_worst_time = index_worst_error = 0;
    index_best_iteration = index_best_time = index_best_error = 0;

    worst_time = best_time = exp_metrics[0].total_time;
    worst_error = best_error = exp_metrics[0].distance_error;
    worst_iteration = best_iteration = exp_metrics[0].final_iteration;

    for (i = 0; i < params->number_of_images; i++)
    {
        final_error[i] = exp_metrics[i].distance_error;
        final_iteration[i] = exp_metrics[i].final_iteration;

        if (exp_metrics[i].distance_error > worst_error)
        {
            worst_error = exp_metrics[i].distance_error;
            index_worst_error = i;
        }
        if (exp_metrics[i].distance_error < best_error)
        {
            best_error = exp_metrics[i].distance_error;
            index_best_error = i;
        }
        if (exp_metrics[i].final_iteration > worst_iteration)
        {
            worst_iteration = exp_metrics[i].final_iteration;
            index_worst_iteration = i;
        }
        if (exp_metrics[i].final_iteration < best_iteration)
        {
            best_iteration = exp_metrics[i].final_iteration;
            index_best_iteration = i;
        }
        if (exp_metrics[i].total_time > worst_time)
        {
            worst_time = exp_metrics[i].total_time;
            index_worst_time = i;
        }
        if (exp_metrics[i].total_time < best_time)
        {
            best_time = exp_metrics[i].total_time;
            index_best_time = i;
        }
    }
    // fprintf(output, "--------------------------------\n");
    // fprintf(output, "Mean final ASD error (mm): %.4f +- %.2f\n", mean(final_error, params->number_of_images), standard_deviation(final_error, params->number_of_images));
    fprintf(output, "--------------------------------\n");
    fprintf(output, "Mean final iteration: %.4f +- %.2f\n", iftMeanFloatArrayDiffZero(final_iteration, params->number_of_images), iftStddevFloatArray(final_iteration, params->number_of_images));
    fprintf(output, "--------------------------------\n\n");

    fprintf(output, "Worst final error was: %.2f with %s\n", worst_error, exp_metrics[index_worst_error].image_name);
    fprintf(output, "Best final error was %.2f with %s\n\n", best_error, exp_metrics[index_best_error].image_name);

    fprintf(output, "Worst final iteration was: %.2f with %s\n", worst_iteration, exp_metrics[index_worst_iteration].image_name);
    fprintf(output, "Best final iteration was %.2f with %s\n\n", best_iteration, exp_metrics[index_best_iteration].image_name);

    fprintf(output, "Worst iterative time was: %.2f with %s\n", worst_time, exp_metrics[index_worst_time].image_name);
    fprintf(output, "Best iterative time was %.2f with %s\n\n", best_time, exp_metrics[index_best_time].image_name);

    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for (i = 0; i < params->number_of_images; i++)
        temp_array[i] = exp_metrics[i].final_iteration;
    sprintf(full_path, "./results/stats/finaliteration_%s_sup=%d_%s_desc=%d_svmethod=%d.txt", database, params->superpixel, algorithm, params->superpixel_descriptor, params->supervoxel_method);
    saveFinalValues(full_path, temp_array, params->number_of_images);


    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for (i = 0; i < params->number_of_images; i++)
        temp_array[i] = exp_metrics[i].total_time; //Divide by 1000 if you want to save in seconds.
    sprintf(full_path, "./results/stats/totaltime_%s_sup=%d_%s_desc=%d_svmethod=%d.txt", database, params->superpixel, algorithm, params->superpixel_descriptor, params->supervoxel_method);
    saveFinalValues(full_path, temp_array, params->number_of_images);


    fprintf(output, "Total segmentation execution time: %.4f s\n", iftSumFloatArray(temp_array, params->number_of_images) / 1000);
    fprintf(output, "Mean segmentation time per image: %.4f +- %.4fs\n", iftMeanFloatArrayDiffZero(temp_array, params->number_of_images) / 1000, iftStddevFloatArray(temp_array, params->number_of_images) / 1000);

    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for (i = 0; i < params->number_of_images; i++)
    {
        if (exp_metrics[i].final_iteration != 0)
            temp_array[i] = exp_metrics[i].total_time / exp_metrics[i].final_iteration * 1.0;
        else
            temp_array[i] = 0;
    }

    fprintf(output, "Mean segmentation time per iteration: %.4f +- %.4f s\n", iftMeanFloatArrayDiffZero(temp_array, params->number_of_images) / 1000, iftStddevFloatArray(temp_array, params->number_of_images) / 1000);

    fprintf(output, "=======================================================================================================\n");
    if (strcmp(target, "file") == 0)
        fclose(output);

    free(final_error);
    free(final_iteration);
    free(temp_array);
    //free(database);

}
int getMaximumNumberOfObjects (ExperimentParameters *params, iftImageNames *gtruths)
{
    char path[256];
    iftImage *gt_image = NULL;
    int number_of_objects = 0;
    int i = 0;
    for (i = 0; i < params->number_of_images; i++)
    {
        strcpy(path, params->label_database);
        strcat(path, gtruths[i].image_name);

        if (strcmp(params->file_extension, "scn") == 0)     gt_image = iftReadImage(path);
        else                                                gt_image = iftReadImageP5(path);

        int val = iftMaximumValue(gt_image);
        if (val > number_of_objects)
            number_of_objects = val;
        iftDestroyImage(&gt_image);

    }
    return number_of_objects;
}
float checkConvergence(float average_edt, int seeds_added)
{
    //If the error did no change and no seeds were added.
    if (average_edt == PREVIOUS_EDT_ERROR && seeds_added == 0)
        return -1;
    else if (average_edt > PREVIOUS_EDT_ERROR && seeds_added == 0)
        return -1;
    PREVIOUS_EDT_ERROR = average_edt;
    return PREVIOUS_EDT_ERROR;
}
void saveImage3d(iftImage *current_segmentation, char *filename, int iteration, ExperimentParameters *params)
{
    int p = 0;
    char path[256];
    iftCreateDirectory(params->output_directory);
    char *clean_name = iftSplitStringOld(filename, ".", 0);
    char *database = iftSplitStringOld(params->image_database, "/", 2);
    sprintf(path, "%s%s_%s_%d_%s.scn",
            params->output_directory,
            clean_name,
            database,
            iteration, params->robot_strategy);

    free(clean_name);

    iftImage *rendered = iftCreateImage(current_segmentation->xsize, current_segmentation->ysize, current_segmentation->zsize);
    //Copy SCN image to the new image
    for (p = 0; p < current_segmentation->n; p++)
        rendered->val[p] = current_segmentation->val[p];
    iftWriteImage(rendered, path);
    iftDestroyImage(&rendered);
    //free(database);
}
