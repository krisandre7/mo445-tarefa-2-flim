#include "ift.h"



int main(int argc, char *argv[]) {
    iftImage *label = NULL;
    iftImage *gt = NULL;
    int number_of_objects = 0;
    int i = 0;
    int type = 0;
    char *ext;
//    timer             *t1=NULL,*t2=NULL;


    if (argc != 4)
        iftError(
                "Usage: iftSegmentationErrors <result_label_map.[scn,pgm]> <ground_truth.[scn,pgm]> <DICE - 0, ASSD - 1, BOTH - 2>",
                "main");


    //Open label image
    ext = iftSplitStringAt(argv[1], ".", -1);

    if (strcmp(ext, "pgm") == 0) {
        label = iftReadImageP5(argv[1]);
        //A   = iftCircular(sqrtf(2.0));
    }
    else if (strcmp(ext, "scn") == 0) {
        label = iftReadImage(argv[1]);
        //A    = iftSpheric(sqrtf(3.0));
    }
    else {
        iftError("Invalid image format\n", "main");
        exit(1);
    }

    //Open gt image
    if (strcmp(ext, "pgm") == 0) {
        gt = iftReadImageP5(argv[2]);
    }
    else if (strcmp(ext, "scn") == 0) {
        gt = iftReadImage(argv[2]);
    }
    else {
        iftError("Invalid image format\n", "main");
        exit(1);
    }

    type = atoi(argv[3]);

    if(type < 0 || type > 2)
        iftError("Accuracy type %d not valid. It must be: DICE - 0, ASSD - 1, BOTH - 2!", "main", type);

    number_of_objects = iftMaximumValue(gt);

    if(type == 0 || type == 2) {
        printf("Dice:\n");
    //    t1 = iftTic();
        iftDblArray *dices = iftDiceSimilarityMultiLabel(label, gt, number_of_objects);
    //    t2 = iftToc();
    //    printf("Time spent: %f ms\n", iftCompTime(t1, t2));
        for (i = 0; i <= number_of_objects; i++) {
            if (i == 0)
                printf("Mean: %f\n", dices->val[i]);
            else
                printf("Object %02d: %f\n", i, dices->val[i]);
        }
        iftDestroyDblArray(&dices);
    }

    if(type == 1 || type == 2) {
        printf("ASSD:\n");
        //    t1 = iftTic();
        iftDblArray *error_dbl_array = iftASSDMultiLabel(label, gt, number_of_objects);
        //    t2 = iftToc();
        //    printf("Time spent: %f ms\n", iftCompTime(t1, t2));
        for (i = 0; i <= number_of_objects; i++) {
            if (iftIs3DImage(label)) {
                if (i == 0)
                    printf("Mean: %f\n", error_dbl_array->val[i]);
                else
                    printf("Object %02d: %f\n", i, error_dbl_array->val[i]);
            } else {
                if (i == 0)
                    printf("Mean: %f\n", error_dbl_array->val[i]);
                else
                    printf("Object %02d: %f\n", i, error_dbl_array->val[i]);
            }
        }
        iftDestroyDblArray(&error_dbl_array);
    }

    free(ext);
    iftDestroyImage(&label);
    iftDestroyImage(&gt);
    

    return(0);
}

