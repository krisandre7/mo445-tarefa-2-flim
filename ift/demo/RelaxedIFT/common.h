#include "ift.h"

#define SECONDS 1000

typedef struct exp_parameters
{
    /* Folders and files */
    char image_database[256];
    char label_database[256];
    char output_directory[256];
    char file_extension[10];

    /* Robot configuration */
    char robot_strategy[32];
    int seeds_per_iteration;
    int min_distance_border;
    int max_marker_size;
    int min_marker_size;
    int border_distance;

    /* General Configuration */
    int save_ouput;
    int verbose;
    int ift_differential;
    int relaxation;
    int basins_type;
    float basins_adjacency;
    int n_segm_iterations;
    int smooth_iterations;
    float smooth_factor;
    int smooth_foot;

    /* Not configurable */
    int max_number_of_objects;
    int curr_number_of_objects;
    int number_of_images;

    /* Supervoxel */
    int supervoxel;
    int supervoxel_method;
    int supervoxel_descriptor;
    float supervoxel_adj;
    float supervoxel_volume;

    /* Slic */
    int slic_nregions;
    float slic_compactness;
    /* Others */
    int nbins;
    int bpp;
    float gc_beta;

} ExperimentParameters;


typedef struct exp_metrics
{
    char image_name[256];

    float prev_distance_error;

    // Used in the IFT algorithm and DGC
    float distance_error;
    float *distance_error_object;
    float dice_error;
    float *dice_error_object;
    float final_iteration;
    float total_time;
    float seeds_added;

    //Used only in the DGC. The size is the max (number of objects + 1)
    // float *arr_distance_error;
    // float *arr_dice_error;
    // float *arr_seeds_added;
    // float *arr_total_time;
    float *arr_final_iteration;
    int sum_final_iteration;

    float number_of_supervoxels;
    float setup_time;

    //Matrix of size: n_segm_iterations x max_numb_objects
    //distance_error_matrix[0][1] iteration 0 with object 1.
    //float **distance_error_matrix;
    //float **dice_error_matrix;
} ExperimentResults;


ExperimentParameters *createExperimentParameters ();
void destroyExperimentParameters (ExperimentParameters *params);

ExperimentResults *createExperimentResults (ExperimentParameters *params);
void destroyExperimentResults(ExperimentResults *expM, ExperimentParameters *params);

//Parse the config file to create an experiment
void loadParameters (ExperimentParameters *params, char *config_filename);
void setDirectories(ExperimentParameters *params, char *image_database, char *label_database);

void printHeader(ExperimentParameters *params, char *target, char *algorithm);
int getMaximumNumberOfObjects (ExperimentParameters *params, iftImageNames *gtruths);
float computeCorrectLabel(iftImage *label_image, iftImage *gt_image);
void printReport(ExperimentParameters *params, ExperimentResults *exp_results, char *target, char *algorithm);
float checkConvergence(float average_edt, int seeds_added);
void saveFinalValues(char *filename, float *array, int length);
//void saveImage2d(iftImage * ppm_image, iftImage *current_segmentation, iftImage * seeds_image, char *filename, int iteration, ExperimentParameters *params);
void saveImage3d(iftImage *current_segmentation, char *filename, int iteration, ExperimentParameters *params);
