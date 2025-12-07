#include "ift.h"

iftDict *iftGetArguments(int argc, const char **argv);

/**
 * @brief Implicitly calculates a dataset from a sliding window over a test image, by computing tiles with overlap.
 *
 * @param tile_xsize The size of the window in the X dimension.
 * @param tile_ysize The size of the window in the Y dimension.
 * @param bins_per_band The number of bins per band for BIC.
 * @param xstride The stride in the X dimension.
 * @param ystride The stride in the Y dimension.
 * @param tiles Returns the overlapped image tiles.
 *
 * @return The computed dataset.
 */
iftDataSet *iftComputeDatasetWithBICFeaturesFromImageTiles(iftImage *img, int tile_xsize, int tile_ysize,
                                                           int bins_per_band, int xstride, int ystride,
                                                           iftImageTiles **tiles);

iftFImage *iftFuzzyClassifyPlantImage(iftImage *img, iftCplGraph *graph, iftJson *config, iftFImage **bin_likelihood);

void iftFilterAndThresholdLikelihoodAndSaveFiles(iftFImage *likelihood, iftFImage *bin_likelihood, int volume,
                                                 double threshold_perc, const char *output_folder,
                                                 const char *bname);

int main(int argc, const char **argv) {
    int volume;
    double threshold_perc;

    iftDict *args             = NULL;
    iftFileSet *test_files = NULL;
    iftJson *config           = NULL;
    iftCplGraph *graph        = NULL;
    char *test_folder         = NULL;
    char *classifier_filename = NULL;
    char *config_filename     = NULL;
    char *output_folder       = NULL;

    iftFImage *likelihood     = NULL;
    iftFImage *bin_likelihood     = NULL;
    iftImage  *likelihood_img = NULL;

    args = iftGetArguments(argc, argv);

    test_folder         = iftGetStrValFromDict("--input-test-folder", args);
    classifier_filename = iftGetStrValFromDict("--input-classifier", args);
    config_filename     = iftGetStrValFromDict("--input-config", args);
    output_folder       = iftGetStrValFromDict("--output-folder", args);;

    test_files          = iftLoadFileSetFromDir(test_folder, 0);
    config              = iftReadJson(config_filename);
    graph               = iftReadCplGraph(classifier_filename);

    iftMakeDir(output_folder);

    volume = iftGetJInt(config, "volume");
    threshold_perc = iftGetJDouble(config, "likelihood_threshold");

    for(int i = 0; i < test_files->n; i++) {
        char *bname = NULL;
        timer *tic = NULL, *toc = NULL;

        iftImage *img = iftReadImageByExt(test_files->files[i]->path);

        bname = iftFilename(test_files->files[i]->path, iftFileExt(test_files->files[i]->path));

        tic = iftTic();

        /* Fuzzy classifying image */
        likelihood = iftFuzzyClassifyPlantImage(img, graph, config, &bin_likelihood);

        toc = iftToc();

        iftFilterAndThresholdLikelihoodAndSaveFiles(likelihood, bin_likelihood, volume, threshold_perc, output_folder,
                                                    bname);

        fprintf(stderr, "Fuzzy plant image classification for file %s\n", bname);
        iftPrintFormattedTime(iftCompTime(tic, toc));

        iftDestroyImage(&img);
        iftDestroyFImage(&likelihood);
        iftDestroyFImage(&bin_likelihood);
        iftDestroyImage(&likelihood_img);
        free(bname);
    }

    iftDestroyDict(&args);
    iftDestroyJson(&config);
    iftDestroyFileSet(&test_files);
    iftDestroyDataSet(&graph->Z);
    iftDestroyCplGraph(&graph);
    free(test_folder);
    free(classifier_filename);
    free(config_filename);
    free(output_folder);

    return 0;
}


iftFImage *iftFuzzyClassifyPlantImage(iftImage *img, iftCplGraph *graph, iftJson *config, iftFImage **bin_likelihood) {
    int tile_xsize, tile_ysize, bins_per_band = 0, xstride, ystride;

    iftFImage *likelihood = NULL;
    iftDataSet *Z = NULL;
    iftImageTiles *tiles = NULL;
    iftJson *BIC = NULL;

    if (iftIs3DImage(img)) {
        iftError("Fuzzy plant image classification only supports 2D images!", "iftFuzzyClassifyPlantImage");
    }

    tile_xsize   = iftGetJInt(config, "tile_xsize");
    tile_ysize   = iftGetJInt(config, "tile_ysize");
    xstride      = iftGetJInt(config, "xstride");
    ystride      = iftGetJInt(config, "ystride");
    BIC          = iftGetJDict(config, "BIC");
    bins_per_band = iftGetJInt(BIC, "bins_per_band");
    iftDestroyJson(&BIC);

    likelihood = iftCreateFImage(img->xsize, img->ysize, img->zsize);
    *bin_likelihood = iftCreateFImage(img->xsize, img->ysize, img->zsize);

    Z = iftComputeDatasetWithBICFeaturesFromImageTiles(img, tile_xsize, tile_ysize, bins_per_band, xstride, ystride,
                                                       &tiles);
    iftSetDistanceFunction(Z, graph->Z->function_number);

    fprintf(stderr,"Performing classification of %d image tiles with an OPF classifier with %d nodes and %d features each\n",
            Z->nsamples, graph->nnodes, Z->nfeats);

    // Binary tile classification
    iftClassifyWithCertaintyValues(graph, Z);

    #pragma omp parallel for
    for(int p = 0; p < likelihood->n; p++) {
        double weight = 0.0, nintersections = 0.0, bin_weight = 0.0;
        iftVoxel v = iftGetVoxelCoord(img, p);
        iftSet *intersecting_tiles = iftGetIndicesFromAllTilesIntersectingVoxel(tiles, v);
        iftSet *S = NULL;

        for(S = intersecting_tiles; S != NULL; S = S->next) {
            int tile = S->elem;
            double w = Z->sample[tile].weight;

            /* If the sample's class is 2, then it has been classified as disease. The sample weight is given by
                wclass/(wclass + wotherclass) in iftBinClassify, where wclass and wotherclass are the values of the
                class of the sample and the other class in the dataset, respectively. Hence, this weight is w1/(w1+w2) if
                the sample was classified with label 1, or w2(w1+w2) if the sample was labeled as 2. Since we intend
                to compute the likelihood map as (w1/(w2+w1) for all samples, since we expect that value to be higher
                when the patch most resembles sick patches, we compute the complement weight for class w2.
            */
            if(Z->sample[tile].label == 2) {
                w = 1.0 - w;
            }

            weight += w;

            bin_weight += (Z->sample[tile].label == 2) ? 1 : 0;

            nintersections++;
        }

        weight /= iftMax(nintersections, 1.0);
        bin_weight /= iftMax(nintersections, 1.0);

        likelihood->val[p] = weight;
        (*bin_likelihood)->val[p] += bin_weight;

        iftDestroySet(&intersecting_tiles);
    }

    iftDestroyImageTiles(&tiles);
    iftDestroyDataSet(&Z);

    return likelihood;
}

iftDataSet *iftComputeDatasetWithBICFeaturesFromImageTiles(iftImage *img, int tile_xsize, int tile_ysize,
                                                           int bins_per_band, int xstride, int ystride,
                                                           iftImageTiles **tiles) {
    iftImage *tile = NULL;
    iftFeatures *feats = NULL;
    iftDataSet *Z = NULL;

    iftDestroyImageTiles(tiles);

    *tiles = iftImageTilesByStriding(img, tile_xsize, tile_ysize, 1, xstride, ystride, 1);

    tile = iftExtractTile(img, *tiles, 0);

    feats = iftExtractBIC(tile, NULL, bins_per_band);

    // Initializing the dataset after the features have been extracted for the first image
    Z = iftCreateDataSet((*tiles)->ntiles, feats->n);
    Z->nclasses = 2;

    // Copying data for the first tile
    Z->sample[0].id = 0;
    for (int j = 0; j < feats->n; j++)
        Z->sample[0].feat[j] = feats->val[j];

    iftDestroyImage(&tile);
    iftDestroyFeatures(&feats);

    #pragma omp parallel for shared(Z) private(tile, feats)
    for(int i = 1; i < (*tiles)->ntiles; i++) {
        iftImage *tile = NULL;
        iftFeatures *feats = NULL;

        tile = iftExtractTile(img, *tiles, i);

        feats = iftExtractBIC(tile, NULL, bins_per_band);

        Z->sample[i].id = i;
        for (int j = 0; j < feats->n; j++)
            Z->sample[i].feat[j] = feats->val[j];

        iftDestroyImage(&tile);
        iftDestroyFeatures(&feats);
    }

    return Z;
}



void iftFilterAndThresholdLikelihoodAndSaveFiles(iftFImage *likelihood, iftFImage *bin_likelihood, int volume,
                                                 double threshold_perc, const char *output_folder,
                                                 const char *bname) {
    iftImage *likelihood_img = NULL;
    iftImage *filtered = NULL;
    iftImage *thresholded = NULL;

    likelihood_img = iftFImageToImage(likelihood, 255);

    filtered = iftVolumeOpen(likelihood_img, volume);

    thresholded = iftThreshold(filtered, threshold_perc*iftMaximumValue(filtered), IFT_INFINITY_INT, 255);

    /* Saving likelihood image */
    char test_bname[IFT_STR_DEFAULT_SIZE];
    char *output_file = NULL;

    sprintf(test_bname, "%s_likelihood.png", bname);
    output_file = iftJoinPathnames(2, output_folder, test_bname);

    iftWriteImageByExt(likelihood_img, output_file);
    iftDestroyImage(&likelihood_img);
    free(output_file);

    /* Saving binary likelihood image */
    sprintf(test_bname, "%s_bin_likelihood.png", bname);
    output_file = iftJoinPathnames(2, output_folder, test_bname);

    likelihood_img = iftFImageToImage(bin_likelihood, 255);

    iftWriteImageByExt(likelihood_img, output_file);
    iftDestroyImage(&likelihood_img);
    free(output_file);

    /* Saving filtered likelihood image */
    sprintf(test_bname, "%s_likelihood_filtered.png", bname);
    output_file = iftJoinPathnames(2, output_folder, test_bname);

    iftWriteImageByExt(filtered, output_file);
    iftDestroyImage(&filtered);
    free(output_file);

    /* Saving thresholded filtered likelihood image */
    sprintf(test_bname, "%s_likelihood_filtered_thresholded.png", bname);
    output_file = iftJoinPathnames(2, output_folder, test_bname);

    iftWriteImageByExt(thresholded, output_file);
    iftDestroyImage(&thresholded);
    free(output_file);

}


iftDict *iftGetArguments(int argc, const char **argv) {
    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-test-folder", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input test dataset folder"},
            {.short_name = "", .long_name = "--input-classifier", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="The input classifier filename"},
            {.short_name = "-c", .long_name = "--input-config", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="The configuration file"},
            {.short_name = "-o", .long_name = "--output-folder", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="The output folderin which the likelihood images will be saved"},
    };

    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    char program_description[2048] = "This takes a pre-computed classifer and outputs a disease likelihood map for each image.";

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);

    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

    iftDestroyCmdLineParser(&parser);

    return args;
}