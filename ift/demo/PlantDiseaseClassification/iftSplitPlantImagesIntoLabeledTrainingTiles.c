#include "ift.h"

iftDict *iftGetArguments(int argc, const char **argv);

void iftSplitPlantImagesIntoLabeledTrainingTiles(  iftFileSet *files,   iftFileSet *mask_files, iftJson *config,
                                                 const char *output_image_tiles_dir);


int main(int argc, const char **argv) {
    iftJson *config = NULL;
    iftDict *args = NULL;
    iftFileSet *files = NULL;
    iftFileSet *mask_files = NULL;

    /*--------------------------------------------------------*/
    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/


    args = iftGetArguments(argc, argv);

    char *config_file  = iftGetStrValFromDict("--input-config", args);
    char *training_dir = iftGetStrValFromDict("--input-dataset-folder", args);
    char *mask_dir     = iftGetStrValFromDict("--input-dataset-mask-folder", args);
    char *output_image_tiles_dir = iftGetStrValFromDict("--ouptut-image-tiles-folder", args);

    if(!iftFileExists(config_file))
        iftError("Json configuration file %s does not exist!", argv[0], config_file);

    if(!iftDirExists(training_dir))
        iftError("Directory %s with training images does not exist!", argv[0]);

    config = iftReadJson(config_file);

    files = iftLoadFileSetFromDirBySuffix(training_dir, "png");

    if(mask_dir != NULL)
        mask_files = iftLoadFileSetFromDirBySuffix(mask_dir, "png");

    iftSplitPlantImagesIntoLabeledTrainingTiles(files, mask_files, config, output_image_tiles_dir);

    iftDestroyFileSet(&files);
    iftDestroyFileSet(&mask_files);
    iftDestroyDict(&args);
    iftDestroyJson(&config);

    free(config_file);
    free(training_dir);
    free(mask_dir);
    free(output_image_tiles_dir);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);
}


void iftSplitPlantImagesIntoLabeledTrainingTiles(  iftFileSet *files,   iftFileSet *mask_files, iftJson *config,
                                                 const char *output_image_tiles_dir) {
    int tile_xsize, tile_ysize;
    iftCSV *csv_samples = NULL;
    double perc_of_pixels_labeled_as_disease_threshold = 0.0;

    // Getting the tile size
    tile_xsize = iftGetJInt(config, "tile_xsize");
    tile_ysize = iftGetJInt(config, "tile_ysize");
    perc_of_pixels_labeled_as_disease_threshold = iftGetJDouble(config, "perc_of_pixels_labeled_as_disease_threshold");

    if(!iftDirExists(output_image_tiles_dir))
        iftMakeDir(output_image_tiles_dir);

    for(size_t i = 0; i < files->n; i++) {
        iftImage *img = NULL, *mask = NULL;
        iftImageTiles *tiles = NULL;
        char *file_basename_without_ext = NULL, *file_basename = NULL;
        const char *ext_with_dot = NULL;
        int ntiles_x, ntiles_y, ntiles_z = 1;

        if(iftIsImageFile(files->files[i]->path)) {
            iftImage **img_tiles = NULL;
            iftImage **mask_tiles = NULL;

            fprintf(stderr, "%s\n", files->files[i]->path);

            img = iftReadImageByExt(files->files[i]->path);

            // Determining the number of tiles that should be used to partition the image, given the desired
            // tile size
            ntiles_x = iftMax(1, img->xsize/tile_xsize);
            ntiles_y = iftMax(1, img->ysize/tile_ysize);

            file_basename_without_ext = iftFilename(files->files[i]->path, iftFileExt(files->files[i]->path));
            file_basename = iftConcatStrings(2, file_basename_without_ext, "_M.png");


            if (mask_files != NULL && iftFileExists(mask_files->files[i]->path)) {
                mask = iftReadImageByExt(mask_files->files[i]->path);

                iftVerifyImageDomains(img, mask, "iftSplitPlantImagesIntoLabeledTrainingTiles");
            }

            int xpadding = (img->xsize - ntiles_x*tile_xsize);
            int ypadding = (img->ysize - ntiles_y*tile_ysize);

            // Removing the undesired padding at the end of the image to ensure that tiles will have the same size
            iftImage *img_roi = iftRemPadding(img, xpadding, ypadding, 0);

            // Tiling image
            img_tiles  = iftSplitImageIntoTilesByEquallyDividingAxes(img_roi, ntiles_x, ntiles_y, ntiles_z, &tiles);

            iftDestroyImage(&img_roi);

            // Tiling mask if it exists
            if(mask != NULL) {
                iftImageTiles *tiles_foo = NULL;
                // Removing the undesired padding at the end of the image to ensure that tiles will have the same size
                iftImage *mask_roi = iftRemPadding(mask, xpadding, ypadding, 0);

                // Tiling the mask's ROI after removing
                mask_tiles = iftSplitImageIntoTilesByEquallyDividingAxes(mask_roi, ntiles_x, ntiles_y, ntiles_z, &tiles_foo);

                iftDestroyImage(&mask_roi);

                // We only write the CSV file if there exist masks in the directory to determine whether
                // an image tile is healthy or not
                if(csv_samples == NULL) {
                    csv_samples = iftCreateCSV(files->n*tiles->ntiles+1, 4);
                    sprintf(csv_samples->data[0][0], "filename");
                    sprintf(csv_samples->data[0][1], "x");
                    sprintf(csv_samples->data[0][2], "y");
                    sprintf(csv_samples->data[0][3], "doente?");
                }

                iftDestroyImageTiles(&tiles_foo);

            }

            // Writing the image tiles and to disk and creating a CSV file with the description and labels of the tiles
            for(int j = 0; j < tiles->ntiles; j++) {
                char *out_img_tile_file  = NULL;
                char *out_mask_tile_file = NULL;
                char *format = NULL, tmp[IFT_STR_DEFAULT_SIZE];

                format               = iftConcatStrings(3, file_basename_without_ext, "_%05d", ext_with_dot);
                sprintf(tmp, format, j);
                out_img_tile_file    = iftJoinPathnames(2, output_image_tiles_dir, tmp);
                free(format);

                // Writing mask image to disk by determining the image type from the file name's extension
                iftWriteImageByExt(img_tiles[j], out_img_tile_file);

                // We write the mask tiles if the mask image was set. The same reasoning is valid for the CSV, since
                // we can only know whether a tile has disease if it has been labeled in the mask.
                if(mask != NULL) {
                    int tile_has_disease = 0, ndisease_pixels = 0;
                    char *out_img_tile_file_basename = iftFilename(out_img_tile_file, "");

                    format                = iftConcatStrings(3, file_basename_without_ext, "_%05d_M", ext_with_dot);
                    sprintf(tmp, format, j);
                    out_mask_tile_file    = iftJoinPathnames(2, output_image_tiles_dir, tmp);
                    free(format);

                    // Writing mask image to disk by determining the image type from the file name's extension
                    iftWriteImageByExt(mask_tiles[j], out_mask_tile_file);
                    free(out_mask_tile_file);

                    for(int p = 0; p < mask_tiles[j]->n; p++) {
                        ndisease_pixels += (mask_tiles[j]->val[p] > 0) ? 1 : 0;
                    }

                    tile_has_disease = ndisease_pixels >= iftMax(1,perc_of_pixels_labeled_as_disease_threshold*mask_tiles[j]->n);

                    sprintf(csv_samples->data[j+1+i*tiles->ntiles][0], "%s", out_img_tile_file_basename);
                    sprintf(csv_samples->data[j+1+i*tiles->ntiles][1], "%d", tiles->tile_coords[j].begin.x);
                    sprintf(csv_samples->data[j+1+i*tiles->ntiles][2], "%d", tiles->tile_coords[j].begin.y);
                    sprintf(csv_samples->data[j+1+i*tiles->ntiles][3], "%d", tile_has_disease);

                    free(out_img_tile_file_basename);
                }
                free(out_img_tile_file);
            }

            for(int j = 0; j < tiles->ntiles; j++) {
                iftDestroyImage(&img_tiles[j]);
                if(mask_tiles != NULL)
                    iftDestroyImage(&mask_tiles[j]);
            }

            free(img_tiles);
            if(mask_tiles != NULL) free(mask_tiles);

            iftDestroyImageTiles(&tiles);

            iftDestroyImage(&mask);
            iftDestroyImage(&img);

            free(file_basename);
            free(file_basename_without_ext);
        }
    }

    if(csv_samples != NULL) {
        char *output_csv_filename = NULL, output_csv_basename[512] = "samples_info.csv";
        output_csv_filename = iftJoinPathnames(2, output_image_tiles_dir, output_csv_basename);

        iftWriteCSV(csv_samples, output_csv_filename, ';');
        iftDestroyCSV(&csv_samples);
        free(output_csv_filename);
    }
}


iftDict *iftGetArguments(int argc, const char **argv) {
    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-dataset-folder", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input dataset folder"},
            {.short_name = "-m", .long_name = "--input-dataset-mask-folder", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input dataset folder with masks"},
            {.short_name = "-c", .long_name = "--input-config", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input json file with configuration of the splitting method"},
            {.short_name = "-o", .long_name = "--ouptut-image-tiles-folder", .has_arg=true, .arg_type = IFT_STR_TYPE, .required=true, .help="Output folder wher the tiles must be saved"},

    };

    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    char program_description[2048] = "This program splits the plant images in a directory into tiles and labels them\n" \
                                     "according to whether their masks present diseases or not.";

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);

    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

    iftDestroyCmdLineParser(&parser);

    return args;
}
