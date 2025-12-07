#include "ift.h"
#include "iftIterativeOPF.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(iftDict *args, char **input_image, int *num_superpixels, int *max_iterations, char **output_image);
/*************************************************************/

int main(int argc, const char* argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    // It is mandatory to include the following line of code in order to select random initial centers.
    iftRandomSeed(time(NULL));

    // mandatory args
    char *input_image = NULL, *output_image = NULL; 
    int num_superpixels, max_iterations;
    timer *tic = NULL, *toc = NULL;

    iftGetRequiredArgs(args, &input_image, &num_superpixels, &max_iterations, &output_image);

    iftIGraph *igraph;
    iftImage *img = iftReadImageByExt(input_image);
    iftImage *label_img;
    iftMImage *mimg = iftImageToMImage(img, LABNorm_CSPACE);
    
    float r = sqrt(2);
    iftAdjRel *A = iftCircular(r);

    tic = iftTic();

    // IDT
    igraph = iftSuperpixelSegmentationByIDT(mimg, NULL, A, num_superpixels, max_iterations);
    
    toc = iftToc();

    label_img = iftIGraphLabel(igraph);
    A = iftCircular(1.5);
    iftColorTable *cmap = iftCreateColorTable(num_superpixels + 1);
    iftDrawLabels(img, label_img, cmap, A, true, 0.25);

    iftWriteImageByExt(img, output_image);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tic, toc)));

    return 0;
}

/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Segments an input image into superpixels using the IDT algorithm.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--input-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Input image."},
        {.short_name = "-n", .long_name = "--num-superpixels", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=true, .help="Number of superpixels."},
        {.short_name = "-m", .long_name = "--max-iterations", .has_arg=true, .arg_type=IFT_LONG_TYPE,
                .required=true, .help="Maximum number of iterations (e.g., 20)."},
        {.short_name = "-o", .long_name = "--output-image", .has_arg=true, .arg_type=IFT_STR_TYPE,
                .required=true, .help="Output file."}
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetRequiredArgs(iftDict *args, char **input_file, int *num_superpixels, int *max_iterations, char **output_file) {
    *input_file      = iftGetStrValFromDict("--input-image", args);
    *num_superpixels = (int) iftGetLongValFromDict("--num-superpixels", args);
    *max_iterations  = (int) iftGetLongValFromDict("--max-iterations", args);
    *output_file     = iftGetStrValFromDict("--output-image", args);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

    puts("-----------------------");
    printf("- Input image: \"%s\"\n", *input_file);
    printf("- # of superpixels: \"%d\"\n", *num_superpixels);
    printf("- Maximum # of iterations: \"%d\"\n", *max_iterations);
    printf("- Output image: \"%s\"\n", *output_file);
    puts("-----------------------\n");
}
/*************************************************************/
