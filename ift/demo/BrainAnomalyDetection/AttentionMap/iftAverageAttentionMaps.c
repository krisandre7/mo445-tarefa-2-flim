#include "ift.h"


/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetInputs(  iftDict *args, iftFileSet **att_map_set, char **out_path,
                  char **mask_path, bool *use_stdev);

iftImage *iftAverageAttentionMaps(  iftFileSet *att_map_set, bool use_stdev);
/*************************************************************/



int main(int argc, const char *argv[]) {
    iftDict *args = iftGetArgs(argc, argv);

    iftFileSet *att_map_set = NULL;
    char *out_path = NULL;
    char *mask_path = NULL;
    bool use_stdev = false;

    iftGetInputs(args, &att_map_set, &out_path, &mask_path, &use_stdev);
    
    iftImage *mask = (mask_path) ? iftReadImageByExt(mask_path) : NULL;

    timer *t1 = iftTic();

    puts("- Computing Mean Registration Error Magnitude");
    iftImage *mean_att_map = iftAverageAttentionMaps(att_map_set, use_stdev);


    if (mask) {
        puts("****************************************");
        iftImage *aux = mean_att_map;
        mean_att_map = iftMask(mean_att_map, mask);
        iftDestroyImage(&aux);
    }

    puts("- Writing Mean Registration Error");
    iftWriteImageByExt(mean_att_map, out_path);


    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));


    // DESTROYERS
    iftDestroyDict(&args);
    iftDestroyFileSet(&att_map_set);
    iftDestroyImage(&mask);
    iftDestroyImage(&mean_att_map);

    return 0;
}


/************************** SOURCES **************************/
iftDict *iftGetArgs(int argc, const char *argv[]) {
    char program_description[2048] = \
        "- Average of a set of Attention Maps.";

    iftCmdLineOpt cmd_line_opts[] = {
        {.short_name = "-i", .long_name = "--attention-maps-entry", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Directory of CSV with the attention maps."},
        {.short_name = "-o", .long_name = "--out-mean-attention-map", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=true, .help="Pathname for the resulting mean attention map."},
        {.short_name = "-m", .long_name = "--mask", .has_arg=true, .arg_type=IFT_STR_TYPE,
         .required=false, .help="Pathname with the mask of the target objects."},
        {.short_name = "", .long_name = "--use-stdev", .has_arg=false,
         .required=false, .help="Add/Use the Standard Deviation Attention Maps into the output map."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}


void iftGetInputs(  iftDict *args, iftFileSet **att_map_set, char **out_path,
                  char **mask_path, bool *use_stdev) {
    const char *reg_error_entry = iftGetConstStrValFromDict("--attention-maps-entry", args);
    *att_map_set = iftLoadFileSetFromDirOrCSV(reg_error_entry, 0, true);
    *out_path = iftGetStrValFromDict("--out-mean-attention-map", args);

    if (iftDictContainKey("--mask", args, NULL))
        *mask_path = iftGetStrValFromDict("--mask", args);

    *use_stdev = iftDictContainKey("--use-stdev", args, NULL);

    puts("-----------------------");
    printf("- Attention Maps Entry: %s\n", reg_error_entry);
    printf("- Output Mean Attention Map: %s\n", *out_path);
    puts("-----------------------");
    if (*mask_path)
        printf("- Mask: %s\n", *mask_path);
    printf("- Add/Use Standard Deviation: %s\n", iftBoolAsString(*use_stdev));
    puts("-----------------------\n");
}


iftImage *iftAverageAttentionMaps(  iftFileSet *att_map_set, bool use_stdev) {
    iftImage *att_map_0 = iftReadImageByExt(att_map_set->files[0]->path);
    iftFImage *mean_att_map = iftCreateFImage(att_map_0->xsize, att_map_0->ysize, att_map_0->zsize);
    iftDestroyImage(&att_map_0);

    
    for (int i = 0; i < att_map_set->n; i++) {
        iftImage *att_map_mag = iftReadImageByExt(att_map_set->files[i]->path);

        #pragma omp parallel for
        for (int p = 0; p < att_map_mag->n; p++)
            mean_att_map->val[p] += (att_map_mag->val[p] / ((float) att_map_set->n));
        
        iftDestroyImage(&att_map_mag);
    }

    if (use_stdev) {
        iftFImage *stdev_att_map_mag = iftCreateFImage(mean_att_map->xsize, mean_att_map->ysize,
                                                       mean_att_map->zsize);
        
        for (int i = 0; i < att_map_set->n; i++) {
            iftImage *att_map_mag = iftReadImageByExt(att_map_set->files[i]->path);

            #pragma omp parallel for
            for (int p = 0; p < att_map_mag->n; p++)
                stdev_att_map_mag->val[p] += powf(att_map_mag->val[p] - mean_att_map->val[p], 2);
            
            iftDestroyImage(&att_map_mag);
        }
        
        // adding stdev asymmetries to the mean asymmetries
        #pragma omp parallel for
        for (int p = 0; p < stdev_att_map_mag->n; p++)
            mean_att_map->val[p] = mean_att_map->val[p] + (sqrtf(stdev_att_map_mag->val[p] / att_map_set->n));

        iftDestroyFImage(&stdev_att_map_mag);
    }

    iftImage *rounded_mean_att_map = iftRoundFImage(mean_att_map);
    iftDestroyFImage(&mean_att_map);

    return rounded_mean_att_map;
}
/*************************************************************/


