#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, char **img_path, iftFileSet **train_set,
                  char **out_path, char **mask_path, char **bias_path);
/*************************************************************/





int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path = NULL;
    iftFileSet *train_set = NULL;
    char *out_path = NULL;
    char *mask_path = NULL;
    char *bias_path = NULL;

    iftGetInputs(args, &img_path, &train_set, &out_path, &mask_path, &bias_path);

    timer *t1 = iftTic();

    iftImage *img = iftReadImageByExt(img_path);
    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;
    iftImage *bias = (bias_path) ? iftReadImageByExt(bias_path) : NULL;
    
    iftFImage *mean_map = iftCreateFImage(img->xsize, img->ysize, img->zsize);
    char *abs_img_path = iftAbsPathname(img_path);

    float n_train_imgs = 0.0;

    for (long f = 0; f < train_set->n; f++) {
        char *abs_train_img_path = iftAbsPathname(train_set->files[f]->path);
        n_train_imgs += (!iftCompareStrings(abs_img_path, abs_train_img_path));
        iftFree(abs_train_img_path);
    }

    printf("n_train_imgs: %f\n\n", n_train_imgs);

    for (long f = 0; f < train_set->n; f++) {
        printf("[%ld/%ld] %s\n", f, train_set->n - 1, train_set->files[f]->path);
        char *abs_train_img_path = iftAbsPathname(train_set->files[f]->path);

        if (iftCompareStrings(abs_img_path, abs_train_img_path)) {
            printf("    ===> SAME IMAGE (IGNORE IT)\n");
        }
        else {
            iftImage *train_img = iftReadImageByExt(train_set->files[f]->path);

            #pragma omp parallel for
            for (int p = 0; p < img->n; p++) {
                // mean_map->val[p] += (abs(img->val[p] - train_img->val[p]) / norm_val);
                mean_map->val[p] += (abs(img->val[p] - train_img->val[p]) / n_train_imgs);
            }
            
            iftDestroyImage(&train_img);
        }

        iftFree(abs_train_img_path);
    }


    iftImage *mean_map_int = iftRoundFImage(mean_map);

    if (bias) {
        puts("- Applying Bias");
        iftImage *aux = mean_map_int;
        mean_map_int = iftSubReLU(mean_map_int, bias);
        iftDestroyImage(&aux);
    }

    if (mask) {
        puts("- Masking");
        iftImage *aux = mean_map_int;
        mean_map_int = iftMask(mean_map_int, mask);
        iftDestroyImage(&aux);
    }

    puts("- Saving Map");
    printf("- max: %d\n", iftMaximumValue(mean_map_int));
    iftWriteImageByExt(mean_map_int, out_path);
    // iftWriteFImage(mean_map, out_path);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyImage(&img);
    iftDestroyImage(&mask);
    iftDestroyImage(&bias);
    iftDestroyFImage(&mean_map);
    iftDestroyImage(&mean_map_int);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Given a target image, compute its mean absolute difference with respect to a " \
        "a training set.\n";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--image", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Target Image."},
        {.short_name = "-t", .long_name = "--train-image-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="CSV or Dir with the Train Images."},
        {.short_name = "-o", .long_name = "--output-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Output Difference Map."},
        {.short_name = "-m", .long_name = "--label-image-path", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname with the Label Image of the target objects."}, 
        {.short_name = "-b", .long_name = "--bias", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname from the bias (e.g. normal reg error magnitude) used to attenuate the " \
                                "resulting registration error magnitude."} 
    };

    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetInputs(  iftDict *args, char **img_path, iftFileSet **train_set,
                  char **out_path, char **mask_path, char **bias_path) {
    *img_path = iftGetStrValFromDict("--image", args);
    const char *train_entry = iftGetStrValFromDict("--train-image-entry", args);
    *train_set = iftLoadFileSetFromDirOrCSV(train_entry, 0, true);
    *out_path = iftGetStrValFromDict("--output-map", args);

    // if (!iftEndsWith(*out_path, ".npy")) {
    //     iftError("Invalid Extension for Output Filename. Try *.npy", "main");
    // }

    if (iftDictContainKey("--label-image-path", args, NULL)) {
        *mask_path = iftGetStrValFromDict("--label-image-path", args);
    }
    else { *mask_path = NULL; }

    if (iftDictContainKey("--bias", args, NULL)) {
        *bias_path = iftGetStrValFromDict("--bias", args);
    }
    else { *bias_path = NULL; }


    puts("---------------------------");
    printf("- Image: %s\n", *img_path);
    printf("- Train Set Entry: %s\n", train_entry);
    printf("- Output Map: %s\n", *out_path);
    puts("---------------------------");
    if (*mask_path) {
        printf("- Mask: %s\n", *mask_path);
    }
    if (*bias_path) {
        printf("- Bias: %s\n", *bias_path);
    }
    puts("---------------------------\n");
}
