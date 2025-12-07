//
// Created by tvspina on 3/27/17.
//

#include "ift.h"

iftDict *iftGetArguments(int argc, const char **argv);

int main(int argc, const char *argv[]) {
    iftImageTiles *tiles;
    iftDict *args = NULL;

    args = iftGetArguments(argc, argv);

    const char *input_img   = iftGetConstStrValFromDict("--input-img", args);
    double overlap          = iftGetDblValFromDict("--overlap", args);
    ulong tile_size         = iftGetLongValFromDict("--tile-size", args);

    if(!iftFileExists(input_img)) {
        iftError("Please provide an existing image file", argv[0]);
    }

    if(!iftIsPowerOf2(tile_size)) {
        iftError("Please provide a power of 2 tile size. %lu given", argv[0], tile_size);
    }

    if(overlap < 0.0 || overlap >= 1.0) {
        iftError("The overlap between tiles must be between [0.0, 1.0). %lf given", argv[0], overlap);
    }

    ulong stride = iftMax(1, (1.0-overlap)*tile_size);

//    stride = iftPowerOf2GtN(stride_aux, &exp);
//
//    if(iftMax(0, stride_aux - stride / 2) < stride - stride_aux)
//        stride = tile_size - stride / 2;
//    else
//        stride = tile_size - stride;

    fprintf(stderr,"Using stride of: %lu", stride);

//    imgtile = iftSplitImageIntoTilesByStriding(img, tiles_xsize, tiles_ysize, tiles_zsize, stride, stride, stride, true, &tiles);


    iftDestroyDict(&args);

    return 0;
}



iftDict *iftGetArguments(int argc, const char *argv[]) {
    // this array of options, n_opts, and the program description could be defined in the main or as a global variables
    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-i", .long_name = "--input-img", .has_arg=true, .required=true, .arg_type = IFT_STR_TYPE, .help="Input image"},
            {.short_name = "", .long_name = "--tile-size", .has_arg=true, .required=true, .arg_type = IFT_LONG_TYPE, .help="Tile size"},
            {.short_name = "", .long_name = "--overlap", .has_arg=true, .required=true, .arg_type = IFT_DBL_TYPE, .help="Overlap percentage between tiles [0.0, 1.0)"}
    };

    // we could simply assign the number 5 in this case
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    char program_description[IFT_STR_DEFAULT_SIZE] = "This program splits a 2D/3D image into tiles. There are two types of tiling:\n" \
                                                        "** EQUAL partitions the image by equally dividing the axes into a grid of --ntiles_x \\times --ntiles_y \\times --ntiles_z tiles. \n" \
                                                        "** STRIDE partitions the image into tiles of the same given --tile_x/y/zsize, while also allowing the tiles to overlap depending on --x/y/zstride.\n" \
                                                        "   If the tiles must not overlap then use --x/y/z/stride at least equal to --tile_x/y/zsize for type STRIDE.\n"
            "   Examples:  1) iftImageToTiles -i image.ppm -t STRIDE --tile_xsize 400 --tile_ysize 300 --xstride 400 --ystride 300\n" \
                                                        "                 Partitions an image into tiles of size 400x300 without overlap.\n" \
                                                        "              2) iftImageToTiles -i image.ppm -t STRIDE --tile_xsize 400 --tile_ysize 300 --xstride 300 --ystride 200\n" \
                                                        "                 Partitions an image into tiles of size 400x300 with overlap of 100 pixels in x and y.\n" \
                                                        "              3) iftImageToTiles -i image.ppm -t STRIDE --tile_xsize 400 --tile_ysize 300 --xstride 500 --ystride 400\n" \
                                                        "                 Partitions an image into tiles of size 400x300 without overlap and separated by 100 pixels in x and y.\n" \
                                                        "              4) iftImageToTiles -i image.ppm -t EQUAL --ntiles_x 4 --ntiles_y 30\n" \
                                                        "                 Partitions an image into tiles of approximately equal size using a 4 by 30 grid.\n"\
                                                        "                 If the image has size of 4000x3000 pixels, then the tiles will have dimension of 1000x100 pixels.";

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);

    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments

    iftDestroyCmdLineParser(&parser);

    return args;
}



float iftImageVariance(iftImage *img) {
    iftFloatArray *arr = NULL;
    float variance, max_val;

    max_val = iftMaximumValue(img);
    arr = iftCreateFloatArray(img->n);

    for(int p = 0; p < img->n; p++) {
        arr->val[p] = img->val[p] / iftMax(1.0, max_val);
    }

    variance = iftVarianceFloatArray(arr->val, arr->n);

    iftDestroyFloatArray(&arr);

    return variance;
}