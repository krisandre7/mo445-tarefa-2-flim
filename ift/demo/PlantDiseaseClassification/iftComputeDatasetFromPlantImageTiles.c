#include "ift.h"

iftDict *iftGetArguments(int argc, const char **argv);

iftDataSet *iftComputeDatasetFromPlantTiles(  iftCSV *samples_info, const char *training_dir, iftJson *config);

int main(int argc, const char **argv) {
    iftJson *config = NULL;
    iftDict *args = NULL;
    iftCSV *samples_info = NULL;
    iftDataSet *dataset = NULL;

    /*--------------------------------------------------------*/
    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/


    args = iftGetArguments(argc, argv);

    char *training_dir          = iftGetStrValFromDict("--input-tiles-folder", args);
    char *config_filename       = iftGetStrValFromDict("--input-config", args);
    char *output_dataset        = iftGetStrValFromDict("--output-dataset", args);
    char *sample_info_filename  = iftJoinPathnames(2, training_dir, "samples_info.csv");

    if(!iftDirExists(training_dir))
        iftError("Tile image directory %s does not exist!", "main");

    if(!iftFileExists(sample_info_filename))
        iftError("Could not find file %s! It is required to compute the dataset from the tiles.", "main",
                 sample_info_filename);

    if(!iftFileExists(config_filename))
        iftError("Cound not find configuration file %s!", "main", config_filename);

    samples_info = iftReadCSV(sample_info_filename, ',');

    dataset = iftComputeDatasetFromPlantTiles(samples_info, training_dir, config);

    iftWriteOPFDataSet(dataset, output_dataset);

    iftDestroyCSV(&samples_info);
    iftDestroyJson(&config);
    iftDestroyDataSet(&dataset);
    iftDestroyDict(&args);
    free(training_dir);
    free(config_filename);
    free(output_dataset);
    free(sample_info_filename);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);

}

iftDataSet *iftComputeDatasetFromPlantTiles(  iftCSV *samples_info, const char *training_dir, iftJson *config) {
    int bins_per_band = 0;
    int distance_function_number = 10;

    iftDataSet *dataset = NULL;
    iftJson *BIC = iftGetJDict(config, "BIC");

    bins_per_band = iftGetJInt(BIC, "bins_per_band");
    distance_function_number = iftGetJInt(config, "distance_function_number");

    iftDestroyJson(&BIC);

    iftPrintJson(config);

    // Starts from 1 to ignore header
    for(size_t i = 1; i < samples_info->nrows; i++) {
        char *filename          = iftJoinPathnames(2, training_dir, samples_info->data[i][0]);
        const char *ext         = iftFileExt(filename); // ext has already '.' now
        char *bname_without_ext = NULL;

        iftFeatures *feats      = NULL;
        iftImage *img           = NULL;

        if(!iftFileExists(filename))
            iftError("File %s does noe exist %s!", "iftComputeDatasetFromPlantTiles", filename);

//        fprintf(stderr, "Processing image %s\n", filename);

        img = iftReadImageByExt(filename);

//        fprintf(stderr,"Extracting BIC for the whole tile image...\n");
        feats = iftExtractBIC(img, NULL, bins_per_band);

        // Initializing the dataset after the features have been extracted for the first image
        if(dataset == NULL) {
            dataset = iftCreateDataSet(samples_info->nrows-1, feats->n);
            iftSetDistanceFunction(dataset, distance_function_number); // L1 norm

            // Number of classes represented in the dataset (2 because a tile may be sick or healthy)
            // Sick tiles belong have label 2, healthy tiles have label 1
            dataset->nclasses = 2;
        }

        for(int j = 0; j < feats->n; j++)
            dataset->sample[i-1].feat[j] = feats->val[j];

        // Plant tiles with disease have been labeled as 1, or 0 otherwise. So we convert those labels to 1,2
        dataset->sample[i-1].truelabel = atoi(samples_info->data[i][samples_info->ncols-1])+1;

        iftDestroyFeatures(&feats);
        iftDestroyImage(&img);
        free(filename);
        free(bname_without_ext);
    }

    return dataset;
}


iftDict *iftGetArguments(int argc, const char **argv) {
    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-tiles-folder", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input dataset tiles folder"},
            {.short_name = "-c", .long_name = "--input-config", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input json file with configuration of the splitting method"},
            {.short_name = "-o", .long_name = "--output-dataset", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="The output dataset filename"}
    };

    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    char program_description[2048] = "This is program takes as input a folder with plant image tiles and computes a dataset\n" \
                                     "according to whether their masks present diseases or not.";

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);

    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

    iftDestroyCmdLineParser(&parser);

    return args;
}
