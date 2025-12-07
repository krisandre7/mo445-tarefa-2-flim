#define class class_
extern "C" {
    #include "ift.h"
}
#undef class
#include "graph.h"

typedef struct exp_parameters {
  char image_database[256];
  char file_extension[10];
  char label_database[256];
  char output_directory[256];
  char robot_strategy[32];
  int number_of_objects;
  int save_ouput;
  int verbose;
  int ift_regular;
  int ift_differential;
  int basins_type;
  int n_segm_iterations;
  int seeds_per_iteration;
  int min_distance_border;
  int max_marker_size;
  int min_marker_size;
  int border_distance;
  int superpixel;
  int nonsuperpixel;
  float spatial_radius;
  float superpixel_adj;
  float superpixel_volume;
  int smooth_foot;

  int number_of_images;
} ExperimentParameters;

typedef struct exp_convergence {
    float *final_error;
    float final_iteration;
    float total_time;
    int seeds_added;
    float number_of_superpixels;
    float setup_time;
    //float final_n_correct_label;
    char image_name[256];
} Convergence;


ExperimentParameters * createExperimentParameters ();
void destroyExperimentParameters (ExperimentParameters * params);
void destroyConvergenceArray(Convergence * conv, ExperimentParameters *params);
Convergence * createConvergenceArray (ExperimentParameters * params);

// ErrorAccumulator * createErrorAccumulator (int length);
// void destroyErrorAccumulator(ErrorAccumulator *erracc);
// void cleanErrorAccumulator(ErrorAccumulator *erracc, int length);
// void accumulateErrorAccumulator(ErrorAccumulator *erracc, iftErrorClassification *errors, int position, float average_edt);
//Parse the config file to create an experiment
void loadParameters (ExperimentParameters *params, char * config_filename);
void printHeader(ExperimentParameters *params, char *target);
int getNumberOfObjects (ExperimentParameters *params, char *gt_name);
float* computeAvgEDT(iftImage *current_segmentation, iftImage *gt_image, int number_of_objects, iftImage *edt_gt_dist_map, iftImage *edt_gt_root_map);
float computeCorrectLabel(iftImage * label_image, iftImage *gt_image);
void printReport(ExperimentParameters *params, Convergence *convergence, float ***distance_error_matrix, float **dice_error_matrix, float ** fscore_error_matrix, char * target);
float sum (float * array, int length);
float mean (float * array, int length);
float variance (float * array, int length);
float standard_deviation (float *array, int length);
void saveFinalValues(char *filename, float *array, int length);
iftImage * binarize (iftImage * img, int threshold);
float checkConvergence(float average_edt, int seeds_added);
//void saveImage2d(iftImage * ppm_image, iftImage *current_segmentation, iftImage * seeds_image, char *filename, int iteration, ExperimentParameters *params);
void saveImage3d(iftImage *current_segmentation, char *filename, int iteration, ExperimentParameters *params);





int CONVERGENCE_COUNTER=0;
float PREVIOUS_EDT_ERROR=0;

ExperimentParameters * createExperimentParameters ()
{
    ExperimentParameters * params = (ExperimentParameters *) malloc (sizeof(ExperimentParameters));
    return (params);
}

void destroyExperimentParameters (ExperimentParameters * params)
{
    if (params != NULL)
    {
        free(params);
        params = NULL;
    }
}
void destroyConvergenceArray(Convergence * conv, ExperimentParameters *params)
{
    if (conv != NULL)
    {
        for (int i = 0; i < params->number_of_images ; i++)
            free(conv[i].final_error);
        free(conv);
        conv = NULL;
    }
}
Convergence * createConvergenceArray (ExperimentParameters * params)
{
    Convergence *c = (Convergence*) malloc (sizeof(Convergence)*params->number_of_images);
    for (int i=0; i< params->number_of_images; i++)
        c[i].final_error = iftAllocFloatArray(params->number_of_objects);
    return c;
}
//Parse the config file to create an convolutional network
void loadParameters (ExperimentParameters *params, char * config_filename)
{
    FILE *fp = fopen(config_filename, "r");
    if (!fp) iftError("Configuration file not found", "Relaxed IFT - loadParameters");
    char line[512];
    char *pch;

    while (!feof(fp))
    {
        fgets(line, sizeof(char)*512, fp);
        if (line[0] == '#' || strlen(line) == 0)
            continue;
        pch = strtok (line,", ");
        if (strcmp(pch, "IMAGE_DATABASE")==0)
        {
            pch = strtok (NULL,", ");
            strcpy(params->image_database, pch);
            params->image_database[strlen(params->image_database)-1] = '\0';
        }
        if (strcmp(pch, "LABEL_DATABASE")==0)
        {
            pch = strtok (NULL,", ");
            strcpy(params->label_database, pch);
            params->label_database[strlen(params->label_database)-1] = '\0';
        }
        if (strcmp(pch, "FILE_EXTENSION")==0)
        {
            pch = strtok (NULL,", ");
            strcpy(params->file_extension, pch);
            params->file_extension[strlen(params->file_extension)-1] = '\0';
        }
        else if (strcmp(pch, "OUTPUT_DIRECTORY")==0)
        {
            pch = strtok (NULL,", ");
            strcpy(params->output_directory, pch);
            params->output_directory[strlen(params->output_directory)-1] = '\0';
        }
        else if (strcmp(pch, "ROBOT_STRATEGY")==0)
        {
            pch = strtok (NULL,", ");
            strcpy(params->robot_strategy, pch);
            params->robot_strategy[strlen(params->robot_strategy)-1] = '\0';
        }
        else if (strcmp(pch, "SAVE_OUTPUT")==0)
        {
            pch = strtok (NULL,", ");
            params->save_ouput = atoi(pch);
        }
        else if (strcmp(pch, "VERBOSE")==0)
        {
            pch = strtok (NULL,", ");
            params->verbose = atoi(pch);
        }
        else if (strcmp(pch, "IFT_REGULAR")==0)
        {
            pch = strtok (NULL,", ");
            params->ift_regular = atoi(pch);
        }
        else if (strcmp(pch, "IFT_DIFFERENTIAL")==0)
        {
            pch = strtok (NULL,", ");
            params->ift_differential = atoi(pch);
        }
        else if (strcmp(pch, "BASINS_TYPE")==0)
        {
            pch = strtok (NULL,", ");
            params->basins_type = atoi(pch);
        }
        else if (strcmp(pch, "N_SEGM_ITERATIONS")==0)
        {
            pch = strtok (NULL,", ");
            params->n_segm_iterations = atoi(pch);
        }
        else if (strcmp(pch, "SEEDS_PER_ITERATION")==0)
        {
            pch = strtok (NULL,", ");
            params->seeds_per_iteration = atoi(pch);
        }
        else if (strcmp(pch, "MIN_DISTANCE_BORDER")==0)
        {
            pch = strtok (NULL,", ");
            params->min_distance_border = atoi(pch);
        }
        else if (strcmp(pch, "MAX_MARKER_SIZE")==0)
        {
            pch = strtok (NULL,", ");
            params->max_marker_size = atoi(pch);
        }
        else if (strcmp(pch, "MIN_MARKER_SIZE")==0)
        {
            pch = strtok (NULL,", ");
            params->min_marker_size = atoi(pch);
        }
        else if (strcmp(pch, "BORDER_DISTANCE")==0)
        {
            pch = strtok (NULL,", ");
            params->border_distance = atoi(pch);
        }
        else if (strcmp(pch, "SPATIAL_RADIUS")==0)
        {
            pch = strtok (NULL,", ");
            params->spatial_radius = atof(pch);
        }
        else if (strcmp(pch, "SUPERPIXEL_VOLUME")==0)
        {
            pch = strtok (NULL,", ");
            params->superpixel_volume = atof(pch);
        }
        else if (strcmp(pch, "SUPERPIXEL_ADJ")==0)
        {
            pch = strtok (NULL,", ");
            params->superpixel_adj = atof(pch);
        }
        else if (strcmp(pch, "SUPERPIXEL")==0)
        {
            pch = strtok (NULL,", ");
            params->superpixel = atoi(pch);
        }
        else if (strcmp(pch, "NONSUPERPIXEL")==0)
        {
            pch = strtok (NULL,", ");
            params->nonsuperpixel = atoi(pch);
        }
        line[0] = 0;
    }
    fclose(fp);
}

void printHeader(ExperimentParameters *params, char *target)
{
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char full_path[256], *database;
    database = iftSplitStringOld(params->image_database, "/", 2);
    FILE *output = NULL;
    sprintf(full_path, "./results/%s_error_MF.txt", database);

    if (strcmp(target, "screen") == 0) output = stdout;
    else                               output = fopen(full_path, "a");

    fprintf(output, "--------------------------------\n");
    fprintf(output, "%d/%d/%d %d:%d\n", tm.tm_mday, tm.tm_mon + 1, tm.tm_year + 1900, tm.tm_hour, tm.tm_min);
    fprintf(output, "Image Database: %s\n", params->image_database);
    fprintf(output, "Number of Images: %d\n", params->number_of_images);
    if (params->ift_differential)
        fprintf(output, "IFT Algorithm: DIFFERENTIAL\n");
    else
        fprintf(output, "IFT Algorithm: REGULAR\n");

    if(params->superpixel)
        fprintf(output, "SUPERPIXEL: True\n");
    else
        fprintf(output, "SUPERPIXEL: False\n");
    
    fprintf(output, "Robot: %s\n", params->robot_strategy);
    fprintf(output, "Segmentation Iterations: %d\n", params->n_segm_iterations);
    fprintf(output, "Seeds per Iteration: %d\n", params->seeds_per_iteration);
    fprintf(output, "Volume Threshold: %g\n", params->superpixel_volume);
    fprintf(output, "Superpixel adj: %g\n", params->superpixel_adj);

    fprintf(output, "Number of objects: %d\n", params->number_of_objects);

    fprintf(output, "--------------------------------\n");
    fprintf(output, "\n");
    if (strcmp(target, "file") == 0)
        fclose(output);
}
float* computeAvgEDT(iftImage *current_segmentation, iftImage *gt_image, int number_of_objects, iftImage *edt_gt_dist_map, iftImage *edt_gt_root_map)
{
    int p = 0, correct_label=0;
    iftImage * border_current_segmentation = NULL;
    float *num_border_pixels=NULL, *sum_object=NULL, mean_error=0, *error_array=NULL;
    num_border_pixels = iftAllocFloatArray(number_of_objects);
    sum_object = iftAllocFloatArray(number_of_objects);
    error_array = iftAllocFloatArray(number_of_objects);
    iftAdjRel *radius=NULL;

    memset(num_border_pixels, 0, sizeof(float)*number_of_objects);
    memset(sum_object, 0, sizeof(float)*number_of_objects);
    memset(error_array, 0, sizeof(float)*number_of_objects);

    if (iftIs3DImage(gt_image))
        radius = iftSpheric(1);
    else
        radius = iftCircular(1);
    border_current_segmentation = iftObjectBorders(current_segmentation, radius, false, true);
    for (p = 0; p < border_current_segmentation->n; p++)
    {
        if (border_current_segmentation->val[p] == 255)
        {
            if ( edt_gt_root_map->val[p] > 0 && edt_gt_root_map->val[p] < gt_image->n)
            {   
                correct_label = gt_image->val[edt_gt_root_map->val[p]];
                if (correct_label != 0) //Ignoring the background
                {
                    sum_object[correct_label] += edt_gt_dist_map->val[p];
                    num_border_pixels[correct_label]++;
                }
            }
        }
    }
    //Computing the average error of each object
    for (p=1; p<number_of_objects; p++)
    {
        if (num_border_pixels[p] != 0)
            error_array[p] = (float) (((sum_object[p] / num_border_pixels[p])*gt_image->dx)* 1.0);   //Measured in mm
        else
            error_array[p] = sum_object[p]*gt_image->dx;
    }
    //Computing the image average. (error of each object / number of objects)
    error_array[0] = 0;
    mean_error = sum(error_array, number_of_objects) / (number_of_objects-1);    //BG does not count
    error_array[0] = mean_error;
    iftDestroyImage(&border_current_segmentation);
    iftDestroyAdjRel(&radius);
    free(num_border_pixels);
    free(sum_object);
    return error_array;
}
float computeCorrectLabel(iftImage * label_image, iftImage *gt_image)
{
    float correct_label = 0;
    int p;

    for (p=0; p< label_image->n; p++)
        if (label_image->val[p] == gt_image->val[p])
            correct_label++;
    return (float) correct_label / (label_image->n * 1.0);
}
void saveFinalValues(char *filename, float *array, int length)
{
    FILE *FP = fopen(filename, "w");
    int i=0; 
    for (i=0; i<length; i++)
        fprintf(FP, "%f ", array[i]);
    fclose(FP);
}
void printReport(ExperimentParameters *params, Convergence *convergence, float ***distance_error_matrix, float **dice_error_matrix, float **fscore_error_matrix, char * target)
{
    int i = 0, j=0;
    // int o=0;
    float *final_error = iftAllocFloatArray(params->number_of_images);
    float *final_iteration = iftAllocFloatArray(params->number_of_images);
    char full_path[256], *database;
    FILE *output = NULL;
    database = iftSplitStringOld(params->image_database, "/", 2);
    sprintf(full_path, "./results/%s_error_MF.txt", database);
    
    if (strcmp(target, "file") == 0)
        output = fopen(full_path, "a");
    else
        output = stdout;

    if (strcmp(database, "thorax135") == 0 && params->superpixel)
        strcat(database, "sup");
    else
        strcat(database, "reg");

    float worst_error, best_error;
    float worst_iteration, best_iteration;
    int index_worst_iteration, index_best_iteration;
    int index_worst_error, index_best_error;
    index_worst_error = 0;
    index_best_error = 0;

    //Print tables
    float *temp_array = NULL;
    temp_array = iftAllocFloatArray(params->number_of_images);
    float std_dev = 0;
    float average = 0;    



    // fprintf(output, "--------------------------------\n");
    // fprintf(output, "Error array per iteration\n");
    // temp_array = iftAllocFloatArray(params->number_of_images);
    // for(j=0; j<params->n_segm_iterations; j++)
    // {
    //     fprintf(output, "%d:\t", j);
    //     temp_array = iftAllocFloatArray(params->number_of_images);
    //     for (i=0; i< params->number_of_images; i++)
    //         temp_array[i] = distance_error_matrix[i][j][0];
    //     average = mean(temp_array, params->number_of_images);
    //     fprintf(output, "%f\n", average);
    //     free(temp_array);
    // }
    fprintf(output, "--------------------------------\n");
    fprintf(output, "Dice per iteration\n");
    for(j=0; j<params->n_segm_iterations; j++)
    {
        fprintf(output, "%d:\t", j);
        memset(temp_array, 0, sizeof(float)*params->number_of_images);
        for (i=0; i< params->number_of_images; i++)
            temp_array[i] = dice_error_matrix[i][j];
        average = mean(temp_array, params->number_of_images);
        fprintf(output, "%f\n", average);
    }    

    fprintf(output, "--------------------------------\n");
    fprintf(output, "FSCORE per iteration\n");
    for(j=0; j<params->n_segm_iterations; j++)
    {
        fprintf(output, "%d:\t", j);
        memset(temp_array, 0, sizeof(float)*params->number_of_images);
        for (i=0; i< params->number_of_images; i++)
            temp_array[i] = fscore_error_matrix[i][j];
        average = mean(temp_array, params->number_of_images);
        fprintf(output, "%f\n", average);
    }


    fprintf(output, "\n\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    average = std_dev = 0;
    for(i=0; i< params->number_of_images; i++)
    {
        if (params->n_segm_iterations == convergence[i].final_iteration)
            temp_array[i] = fscore_error_matrix[i][(int)convergence[i].final_iteration-1];
        else
            temp_array[i] = fscore_error_matrix[i][(int)convergence[i].final_iteration];
    }
    average = mean(temp_array, params->number_of_images);
    std_dev = standard_deviation(temp_array, params->number_of_images);
    fprintf(output, "Fscore Mean: %f +- %.2f &\t", average, std_dev);
    sprintf(full_path, "./results/stats/finalfscore_%s_MF.txt", database);
    saveFinalValues(full_path, temp_array, params->number_of_images);


    fprintf(output, "\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    average = std_dev = 0;
    for(i=0; i< params->number_of_images; i++)
        if (params->n_segm_iterations == convergence[i].final_iteration)
            temp_array[i] = dice_error_matrix[i][(int)convergence[i].final_iteration-1];
        else
            temp_array[i] = dice_error_matrix[i][(int)convergence[i].final_iteration];
    average = mean(temp_array, params->number_of_images);
    std_dev = standard_deviation(temp_array, params->number_of_images);
    fprintf(output, "Dice Mean: %f +- %.2f &\t", average, std_dev);
    sprintf(full_path, "./results/stats/finaldice_%s_MF.txt", database);
    saveFinalValues(full_path, temp_array, params->number_of_images);

    fprintf(output, "\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    average = std_dev = 0;
    for(i=0; i< params->number_of_images; i++)
        temp_array[i] = convergence[i].seeds_added;
    average = mean(temp_array, params->number_of_images);
    std_dev = standard_deviation(temp_array, params->number_of_images);
    fprintf(output, "Seeds Added: %f +- %.2f &\t", average, std_dev);
    sprintf(full_path, "./results/stats/seedsadded_%s_MF.txt", database);
    saveFinalValues(full_path, temp_array, params->number_of_images);

    
    fprintf(output, "\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    average = std_dev = 0;
    for(i=0; i< params->number_of_images; i++)
        temp_array[i] = convergence[i].number_of_superpixels;
    average = mean(temp_array, params->number_of_images);
    std_dev = standard_deviation(temp_array, params->number_of_images);
    fprintf(output, "Number of superpixels: %f +- %.2f &\t", average, std_dev);


    fprintf(output, "\n--------------------------------\n");
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    average = std_dev = 0;
    for(i=0; i< params->number_of_images; i++)
        temp_array[i] = convergence[i].setup_time;
    average = mean(temp_array, params->number_of_images);
    std_dev = standard_deviation(temp_array, params->number_of_images);
    fprintf(output, "Setup time: %f +- %.2f &\t", average, std_dev);

    fprintf(output, "\n\n");


    worst_error = convergence[0].final_error[0];
    best_error = convergence[0].final_error[0];

    index_worst_iteration = 0;
    index_best_iteration = 0;
    worst_iteration = convergence[0].final_iteration;
    best_iteration = convergence[0].final_iteration;
    for (i=0; i< params->number_of_images; i++)
    {
        final_error[i] = convergence[i].final_error[0];
        final_iteration[i] = convergence[i].final_iteration;

        if (convergence[i].final_error[0] > worst_error)
        {
            worst_error = convergence[i].final_error[0];
            index_worst_error = i;
        }
        if (convergence[i].final_error[0] < best_error)
        {
            best_error = convergence[i].final_error[0];
            index_best_error = i;
        }
        if (convergence[i].final_iteration > worst_iteration)
        {
            worst_iteration = convergence[i].final_iteration;
            index_worst_iteration = i;
        }
        if (convergence[i].final_iteration < best_iteration)
        {
            best_iteration = convergence[i].final_iteration;
            index_best_iteration = i;
        }
    }
    fprintf(output, "Mean final ASSD error: %.4f +- %.2f\n", mean(final_error, params->number_of_images), standard_deviation(final_error, params->number_of_images));
    fprintf(output, "Mean final iteration: %.4f +- %.2f\n\n", mean(final_iteration, params->number_of_images), standard_deviation(final_iteration, params->number_of_images));
    
    fprintf(output, "Variance final error: %.2f\n", variance(final_error, params->number_of_images));
    fprintf(output, "Variance final iteration: %.2f\n", variance(final_iteration, params->number_of_images));

    fprintf(output, "Worst final error was: %.2f with %s\n", worst_error, convergence[index_worst_error].image_name);
    fprintf(output, "Best final error was %.2f with %s\n", best_error, convergence[index_best_error].image_name);
    
    fprintf(output, "Worst final iteration was: %.2f with %s\n", worst_iteration, convergence[index_worst_iteration].image_name);
    fprintf(output, "Best final iteration was %.2f with %s\n\n\n", best_iteration, convergence[index_best_iteration].image_name);
    
    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for(i=0; i< params->number_of_images; i++)
        temp_array[i] = convergence[i].final_iteration;
    sprintf(full_path, "./results/stats/finaliteration_%s_MF.txt", database);
    saveFinalValues(full_path, temp_array, params->number_of_images);


    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for(i=0; i< params->number_of_images; i++)
        temp_array[i] = convergence[i].total_time;
    sprintf(full_path, "./results/stats/totaltime_%s_MF.txt", database);
    saveFinalValues(full_path, temp_array, params->number_of_images);


    fprintf(output, "Total segmentation execution time: %.4f s\n", sum(temp_array, params->number_of_images)/1000);
    fprintf(output, "Mean segmentation time per image: %.4f +- %.4fs\n", mean(temp_array, params->number_of_images)/1000, standard_deviation(temp_array, params->number_of_images)/1000);

    memset(temp_array, 0, sizeof(float)*params->number_of_images);
    for(i=0; i< params->number_of_images; i++)
    {
        if (convergence[i].final_iteration != 0)
            temp_array[i] = convergence[i].total_time/convergence[i].final_iteration*1.0;
        else
            temp_array[i] = 0;
    }

    fprintf(output, "Mean segmentation time per iteration: %.4f +- %.4f s\n", mean(temp_array, params->number_of_images)/1000, standard_deviation(temp_array, params->number_of_images)/1000);

    fprintf(output, "=======================================================================================================\n");
    if (strcmp(target, "file") == 0)
        fclose(output);

    free(final_error);
    free(final_iteration);
    free(temp_array);

}
float sum (float * array, int length)
{
    int i =0; 
    float sum = 0;
    for(i=0; i < length; i++)
        sum += array[i];
    return (float) sum;
}
float mean (float * array, int length)
{
    return (float) sum(array,length) * 1.0 / length * 1.0;
}
float variance (float * array, int length)
{
    float average = mean(array, length);
    float accumulator = 0;
    float variance = 0;
    int i=0;
    for (i=0; i<length; i++)
        accumulator += pow((array[i] - average), 2);
    variance = (float) (accumulator*1.0) / (length * 1.0);
    return variance;
}
float standard_deviation (float *array, int length)
{
    return sqrt(variance(array, length));
}

iftImage * binarize (iftImage * img, int threshold)
{
    iftImage * output = iftCreateImage(img->xsize, img->ysize, img->zsize);
    if (!output)
        iftError("Error creating image", "Robot - binarize");

    int p =0;
    for (p = 0; p < img->n; p++)
    {
        if (img->val[p] >= threshold)
            output->val[p] = 1;
        else
            output->val[p] = 0;
    }
    return output;
}
int getNumberOfObjects (ExperimentParameters *params, char *gt_name)
{
    char path[256];
    iftImage *gt_image = NULL;
    int number_of_objects = 0;
    strcpy(path, params->label_database);
    strcat(path, gt_name);
    if (strcmp(params->file_extension, "scn") == 0)     gt_image = iftReadImage(path);
    else                                                gt_image = iftReadImageP5(path);
    number_of_objects = iftMaximumValue(gt_image)+1;
    iftDestroyImage(&gt_image);
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
    for (p=0; p < current_segmentation->n; p++)
        rendered->val[p] = current_segmentation->val[p];
    iftWriteImage(rendered, path);
    iftDestroyImage(&rendered);
}

iftImage* graph_cut_superpixel_segmentation(iftRegionGraph* region_graph, iftImage* label_image, iftLabeledSet *seed, int beta, float *ref_time){
    typedef Graph<double,double,double> GraphType;
    GraphType *graph = new GraphType(region_graph->nnodes, region_graph->nnodes);
    iftDataSet *dataset = region_graph->dataset;

    float max_dist = -INFINITY_FLT;
    float min_dist = INFINITY_FLT;
    int r;
    for(r = 0; r < region_graph->nnodes; r++){
        graph->add_node();

        iftSet *adj = region_graph->node[r].adjacent;
        while(adj){
            int v = adj->elem;
            float dist = dataset->iftArcWeight(dataset->sample[r].feat,dataset->sample[v].feat, dataset->alpha,dataset->nfeats);
            if(dist > max_dist)
                max_dist = dist;
            if(dist < min_dist)
                min_dist = dist;
            adj = adj->next;
        }
    }
    for(r = 0; r < region_graph->nnodes; r++){
        iftSet *adj = region_graph->node[r].adjacent;
        while(adj){
            int v = adj->elem;
            if(r < v){
                float dist = dataset->iftArcWeight(dataset->sample[r].feat,dataset->sample[v].feat, dataset->alpha,dataset->nfeats);
                dist = (dist - min_dist)/(max_dist - min_dist);
                float similarity = exp(-dist/0.5);

                double edge_weight = pow(similarity, beta);
                graph->add_edge(r,v, edge_weight, edge_weight);
            }

            adj = adj->next;
        }
    }

    timer *t1 = iftTic();

    iftBMap *labeled = iftCreateBMap(region_graph->nnodes);

    iftLabeledSet *s = seed;
    while(s){
        int p = s->elem;

        r = label_image->val[p] - 1;

        if(!iftBMapValue(labeled, r)){
            int label = s->label;

            if(label == 1)
                graph->add_tweights(r, INFINITY_FLT, 0);
            else
                graph->add_tweights(r, 0, INFINITY_FLT);

            iftBMapSet1(labeled,r);
        }

        s = s->next;
    }

    graph->maxflow();

    iftImage *segmentation = iftCreateImage(label_image->xsize, label_image->ysize, label_image->zsize);

    int p;
    for(p = 0; p < segmentation->n; p++){
        r = label_image->val[p] - 1;

        if (graph->what_segment(r) == GraphType::SOURCE)
        {
            segmentation->val[p] = 1;
        }
    }
    delete graph;
    iftDestroyBMap(&labeled);

    timer *t2 = iftToc();
    ref_time[0] = iftCompTime(t1,t2);
    //printf("GC_TIME_BY_ITERATION: %f\n",iftCompTime(t1,t2)/1000);

    return segmentation;
}


void runRobot (ExperimentParameters *params, iftImageNames * images, iftImageNames *gtruths)
{
    int i=0,j=0, o=0, k=0, seeds_added=0;
    char path[256];
    float *average_edt = NULL;
    //Images
    iftImage        *original_image                 = NULL;
    iftImage        *gt_image                       = NULL;
    iftImage        *label_image                    = NULL;
    iftImage        *current_segmentation           = NULL;
    iftImage        *seeds_image                    = NULL;
    iftImage        *basins                         = NULL;
    iftImage        *edt_gt_dist_map                = NULL;
    iftImage        *edt_gt_root_map                = NULL;
    iftImage        *marker                         = NULL;
    iftImage *copy_seeds = NULL;

    iftImageForest  *forest                         = NULL;
    iftBMap         *seeds_bmap                     = NULL;
    
    //Adjacencies
    iftAdjRel *radiussqrt3                          = iftSpheric(sqrt(3));
    iftAdjRel *basins_adjacency                     = iftSpheric(params->spatial_radius);
    iftAdjRel *radius1                              = iftSpheric(1);
    iftAdjRel *radius15                             = iftSpheric(1.5);
    
    iftAdjRel *spheric1                             = iftSpheric(1);
    iftAdjRel *superpixel_adj                       = iftSpheric(params->superpixel_adj);

    //Labeled sets
    iftLabeledSet *available_seeds  = NULL;
    iftLabeledSet *current_seeds    = NULL;

    iftDataSet     *dataset         = NULL;
    iftRegionGraph *region_graph    = NULL;


    //Timers
    timer *t1                                       = NULL;
    timer *t2                                       = NULL;
    float ref_time[1];
    float *temp_time                                = NULL;
    temp_time = iftAllocFloatArray(params->n_segm_iterations);

    //Errors
    iftErrorClassification errors;
    Convergence      *convergence_array = createConvergenceArray(params);
    float            *** distance_error_matrix   = (float ***) malloc (sizeof(float**)*params->number_of_images);
    float            **  dice_error_matrix   = (float **) malloc (sizeof(float*)*params->number_of_images);
    float            **  fscore_error_matrix   = (float **) malloc (sizeof(float*)*params->number_of_images);
    //distance_error_matrix[0][0][1] means image 0 at iteration 0 with object 1.
    for (i=0; i<params->number_of_images; i++)
    {
        distance_error_matrix[i] = (float **) malloc (sizeof(float*)*params->n_segm_iterations);
        dice_error_matrix[i] = (float *) malloc (sizeof(float)*params->n_segm_iterations);
        fscore_error_matrix[i] = (float *) malloc (sizeof(float)*params->n_segm_iterations);
        for(j=0; j<params->n_segm_iterations; j++)
        {
            dice_error_matrix[i][j] = fscore_error_matrix[i][j] = 0;
            distance_error_matrix[i][j] = iftAllocFloatArray(params->number_of_objects);
            for (int k =0; k<params->number_of_objects; k++)
                distance_error_matrix[i][j][k] = 0;
        }
    }

    printHeader(params, "file"); //This prints in the disk
    printHeader(params, "screen"); //This prints on the screen

    for (i = 0; i < params->number_of_images; i++)
    {
        /*          OPEN THE IMAGES           */
        if (params->verbose > 0) fprintf(stdout, "Processing: %s and %s\n", images[i].image_name, gtruths[i].image_name);
        strcpy(path, params->image_database);
        strcat(path, images[i].image_name);
        original_image = iftReadImage(path);
        strcpy(path, params->label_database);
        strcat(path, gtruths[i].image_name);
        gt_image = iftReadImage(path);


        /*     GC IS ALWAYS SUPERPIXEL CASE   */
        t1 = iftTic();
        basins = iftImageBasins(original_image, superpixel_adj);
        marker = iftVolumeClose(basins, params->superpixel_volume);
        label_image = iftWaterGray(basins,marker, superpixel_adj);
        t2 = iftToc();
        convergence_array[i].setup_time = iftCompTime(t1, t2)/1000;
        convergence_array[i].number_of_superpixels = iftMaximumValue(label_image);
        printf("Number of superpixels: %d\n", iftMaximumValue(label_image));
        iftDestroyImage(&basins);
        iftDestroyImage(&marker);

        dataset = iftSupervoxelsToDataSet(original_image, label_image);
        region_graph = iftRegionGraphFromLabelImage(label_image, dataset, spheric1);



        /*          RESET VARIABLES           */
        edt_gt_dist_map = NULL;
        edt_gt_root_map = NULL;
        current_segmentation = NULL;
        memset(temp_time, 0, sizeof(float)*params->n_segm_iterations);

        seeds_bmap = iftCreateBMap(original_image->n);
        seeds_image = iftCreateImage(original_image->xsize, original_image->ysize, original_image->zsize);
        basins = iftImageBasins(original_image, basins_adjacency);
        forest = iftCreateImageForest(basins, radius15);
        iftSetImage(seeds_image, -1);
        convergence_array[i].seeds_added = 0;

        iftMaximumValue(forest->img);
        iftDistTransRootMap(gt_image, radiussqrt3, '2', &edt_gt_dist_map, &edt_gt_root_map);


        /*                PIXEL ROBOT CASE                 */
		if (strcmp(params->robot_strategy, "PIXEL") == 0)
            available_seeds = iftBorderMarkersForPixelSegmentation(basins, gt_image, params->border_distance);
        for (j = 0; j < params->n_segm_iterations; j++)
		{            

            /*                GEODESIC ROBOT CASE                 */
            if (strcmp(params->robot_strategy, "GEODESIC") == 0) 
                available_seeds = iftGeodesicMarkersForSegmentation(gt_image, current_segmentation);
            
            seeds_added = iftMarkersFromMisclassifiedSeeds(seeds_image, available_seeds, seeds_bmap, params->seeds_per_iteration, params->number_of_objects, gt_image, current_segmentation, params->min_distance_border, params->max_marker_size, params->min_marker_size);
            current_seeds = iftLabeledSetFromSeedImage(seeds_image);
        
            ref_time[0] = 0;
            current_segmentation = graph_cut_superpixel_segmentation(region_graph, label_image, current_seeds, 20, ref_time);
            temp_time[j] = ref_time[0];

            average_edt = computeAvgEDT(current_segmentation, gt_image, params->number_of_objects, edt_gt_dist_map, edt_gt_root_map);
            /*
                average_edt[0] = mean of all objects error
                average_edt[1] = error for object 1
                average_edt[n] = error for object n
            */
            for (o=0; o<params->number_of_objects; o++)
                distance_error_matrix[i][j][o] = average_edt[o];
            errors = iftSegmentationErrors(gt_image, current_segmentation);
            dice_error_matrix[i][j] = iftAccuracyGivenErrors(&errors);
            fscore_error_matrix[i][j] = iftFScoreGivenErrors(&errors);

            if (params->verbose > 1)  
                fprintf(stdout, "j= %4d edt= %10f seeds_added=%2d accuracy=%f fscore=%f time=%f\n", j, average_edt[0], seeds_added, iftAccuracyGivenErrors(&errors), fscore_error_matrix[i][j], temp_time[j]/1000);
            if (params->verbose == 3) 
                for (int o=0; o < params->number_of_objects; o++) 
                    fprintf(stdout, "Obj %d, error=%f accuracy=%f\n", o, distance_error_matrix[i][j][o], iftAccuracyGivenErrors(&errors));
			if (params->save_ouput && strcmp(params->file_extension, "scn") == 0)
                saveImage3d(current_segmentation, images[i].image_name, j, params);            


            if (strcmp(params->robot_strategy, "GEODESIC") == 0) iftDestroyLabeledSet(&available_seeds);
            iftDestroyLabeledSet(&current_seeds);
            free(average_edt);
            //iftDestroyImage(&current_segmentation);
            convergence_array[i].seeds_added += seeds_added;
            if (seeds_added == 0) break;
        }
        
        /*           LOAD THE CONVERGENCE ARRAY         */
        strcpy(convergence_array[i].image_name, images[i].image_name);
        if (j == params->n_segm_iterations)
        {
            convergence_array[i].final_iteration = j;
            j--;
        }
        else convergence_array[i].final_iteration = j+1;
        for (o=0; o <params->number_of_objects; o++)
            convergence_array[i].final_error[o] = distance_error_matrix[i][j][o];

        int final_iteration = j;
        //FIll the matrices with the last iteration values.
        float final_fscore = fscore_error_matrix[i][j];
        float final_dice = dice_error_matrix[i][j];
        for(j=final_iteration;j<params->n_segm_iterations; j++)
        {
            for(o=0; o<params->number_of_objects; o++)
            {
                distance_error_matrix[i][j][o] = convergence_array[i].final_error[o];
                fscore_error_matrix[i][j] = final_fscore;
                dice_error_matrix[i][j] = final_dice;
            }
        }
        for(o=0; o<=final_iteration; o++)
            convergence_array[i].total_time += temp_time[o];


        if (params->verbose > 0) fprintf(stdout, "FinalError: %f\tItert.Required: %f\tTime: %f s\n\n", convergence_array[i].final_error[0], convergence_array[i].final_iteration, convergence_array[i].total_time/1000);
        //Destroy all for next image
        iftDestroyImage(&original_image);
        iftDestroyImage(&gt_image);
        iftDestroyImage(&seeds_image);
        iftDestroyImage(&edt_gt_dist_map);
        iftDestroyImage(&edt_gt_root_map);
        iftDestroyImage(&basins);
        iftDestroyBMap(&seeds_bmap);
        iftDestroyLabeledSet(&available_seeds);
        iftDestroyLabeledSet(&current_seeds);
        iftDestroyImageForest(&forest);

        iftDestroyImage(&label_image);
        iftDestroyDataSet(&dataset);
        iftDestroyRegionGraph(&region_graph);
    }
    /*      DATABASE REPORT    */
    printReport(params, convergence_array, distance_error_matrix, dice_error_matrix, fscore_error_matrix, "file");
    printReport(params, convergence_array, distance_error_matrix, dice_error_matrix, fscore_error_matrix, "screen");
    /*      CLEAN UP           */
    for (i=0; i<params->number_of_images; i++)
    {
        for(j=0; j<params->n_segm_iterations; j++)
            free(distance_error_matrix[i][j]);
        free(distance_error_matrix[i]);
        free(dice_error_matrix[i]);
        free(fscore_error_matrix[i]);
    }
    free(distance_error_matrix);
    free(dice_error_matrix);
    free(fscore_error_matrix);
    free(temp_time);
    
    iftDestroyAdjRel(&basins_adjacency);
    iftDestroyAdjRel(&radius1);
    iftDestroyAdjRel(&radius15);
    iftDestroyAdjRel(&radiussqrt3);
    iftDestroyAdjRel(&spheric1);
    destroyConvergenceArray(convergence_array, params);
    //free(error_value_per_iteration);

}

int main(int argc, char **argv) 
{
    if (argc != 2)
        iftError("<configuration_file>", "Voleval - main");
    //int i = 0;
    char * configuration_file = argv[1];
    iftImageNames * img_names, *gt_names;
    ExperimentParameters *all_parameters = createExperimentParameters();
    ExperimentParameters *current_parameters = createExperimentParameters();

    loadParameters(all_parameters, configuration_file);
    loadParameters(current_parameters, configuration_file);

    int number_of_images = iftCountImageNames(all_parameters->image_database, all_parameters->file_extension);

    all_parameters->number_of_images = number_of_images;
    current_parameters->number_of_images = number_of_images;

    
    img_names = iftCreateAndLoadImageNames(number_of_images, all_parameters->image_database, all_parameters->file_extension);
    printf("Here: %s\n", all_parameters->label_database);
    gt_names = iftCreateAndLoadImageNames(number_of_images, all_parameters->label_database, "scn");
    

    all_parameters->number_of_objects = getNumberOfObjects(all_parameters, gt_names[0].image_name);
    current_parameters->number_of_objects = all_parameters->number_of_objects;
    if (all_parameters->number_of_objects == 256)   iftWarning("Your groundtruth might be 0-255 instead of 0,1,2,3,...", "demo/relaxedIFT/Main");
    else if (all_parameters->number_of_objects == 0) iftError("Zero objects to segment, GT error probably", "demo/relaxedIFT/Main");

    /* Main code */



    if (all_parameters->superpixel)
    {
        current_parameters->superpixel=1;
        current_parameters->nonsuperpixel=0;
        runRobot(current_parameters, img_names, gt_names);
    }
    if (all_parameters->nonsuperpixel)
    {
        current_parameters->superpixel=0;
        current_parameters->nonsuperpixel=1;
        runRobot(current_parameters, img_names, gt_names);
    }


    /* End Main code */
    destroyExperimentParameters(all_parameters);
    destroyExperimentParameters(current_parameters);
    iftDestroyImageNames(img_names);
    iftDestroyImageNames(gt_names);

    return 0;
}
