#include "ift.h"

typedef struct exp_parameters
{
    char image_database[256];
    char file_extension[10];
    char label_database[256];
    char output_directory[256];
    char robot_strategy[32];
    int max_number_of_objects;
    int curr_number_of_objects;
    int save_ouput;
    int verbose;
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
    int nbins;
    int bpp;
    int superpixel_descriptor;
    float spatial_radius;
    float superpixel_adj;
    float superpixel_volume;
    float gc_beta;
    int smooth_foot;

    int supervoxel_method;
    int slic_nregions;
    float slic_compactness;

    int number_of_images;
} ExperimentParameters;


typedef struct exp_metrics
{
    char image_name[256];

    float prev_distance_error;

    // Used in the IFT algorithm and DGC
    float distance_error;
    float dice_error;
    float final_iteration;
    float total_time;
    float seeds_added;

    //Used only in the DGC. The size is the max (number of objects + 1)
    // float *arr_distance_error;
    // float *arr_dice_error;
    // float *arr_seeds_added;
    // float *arr_total_time;
    float *arr_final_iteration;

    float number_of_superpixels;
    float setup_time;

    //Matrix of size: n_segm_iterations x max_numb_objects
    //distance_error_matrix[0][1] iteration 0 with object 1.
    float **distance_error_matrix;
    float **dice_error_matrix;
} ExpMetrics;


ExperimentParameters *createExperimentParameters ();
void destroyExperimentParameters (ExperimentParameters *params);

ExpMetrics *createExperimentMetrics (ExperimentParameters *params);
void destroyExperimentMetrics(ExpMetrics *expM, ExperimentParameters *params);

//Parse the config file to create an experiment
void loadParameters (ExperimentParameters *params, char *config_filename);
void printHeader(ExperimentParameters *params, char *target, char *algorithm);
int getMaximumNumberOfObjects (ExperimentParameters *params, iftImageNames *gtruths);
float computeCorrectLabel(iftImage *label_image, iftImage *gt_image);
void printReport(ExperimentParameters *params, ExpMetrics *exp_metrics, char *target, char *algorithm);
float checkConvergence(float average_edt, int seeds_added);
void saveFinalValues(char *filename, float *array, int length);
//void saveImage2d(iftImage * ppm_image, iftImage *current_segmentation, iftImage * seeds_image, char *filename, int iteration, ExperimentParameters *params);
void saveImage3d(iftImage *current_segmentation, char *filename, int iteration, ExperimentParameters *params);
