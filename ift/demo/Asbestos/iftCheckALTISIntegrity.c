//
// Created by azaelmsousa on 12/04/21.
//

#include "ift.h"

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path);
/*************************************************************/

int iftExamineRelativeVolumeDifference(iftImage *segmentation, double *volumeLeftLung, double *volumeRightLung, double *volumeTrachea)
#define LEFT_LUNG 1
#define RIGHT_LUNG 2
#define TRACHEA 3
#define PERCENTAGE_THRESHOLD 30
#define HUNDREAD_PERC 100
{
    double maxVol, minVol;
    double perc;

    *volumeLeftLung = *volumeRightLung = *volumeTrachea = 0;
    for (int i = 0; i < segmentation->n; i++){
        if (segmentation->val[i] == LEFT_LUNG){
            (*volumeLeftLung)++;
        } else if (segmentation->val[i] == RIGHT_LUNG){
            (*volumeRightLung)++;
        } else if (segmentation->val[i] == TRACHEA){
            (*volumeTrachea)++;
        }
    }

    if (*volumeLeftLung > *volumeRightLung) {
        maxVol = *volumeLeftLung;
        minVol = *volumeRightLung;
    } else {
        maxVol = *volumeRightLung;
        minVol = *volumeLeftLung;
    }

    perc = (minVol * HUNDREAD_PERC) / maxVol;

    if ((HUNDREAD_PERC - perc) <= PERCENTAGE_THRESHOLD)
        return 1;
    else
        return 0;
}

int iftExamineComponentsPosition(iftImage *segmentation)
{
    iftLabeledSet *S;
    iftVoxel centers[3];
    int p, label, maxX, minX;

    S = iftGeometricCenters(segmentation);

    while (S != NULL){
        p = iftRemoveLabeledSet(&S, &label);
        centers[label-1] = iftGetVoxelCoord(segmentation,p);
    }

    maxX = (centers[0].x >= centers[1].x)?centers[0].x:centers[1].x;
    minX = (centers[0].x <= centers[1].x)?centers[0].x:centers[1].x;

    if (maxX == minX)
        return 0;

    if ((centers[2].x < minX) || (centers[2].x > maxX)){
        return 0;
    }
    return 1;
}

int iftScanAnomaliesOnLungSegmentation(iftImage *segmentation)
#define VOLUME_THRESHOLD 60
{
    double volumeLeftLung, volumeRightLung, volumeTrachea;

    puts("");
    puts("- Checking Segmentation Integrity");

    printf("--- Component's Position Check.....................");
    fflush(stdout);
    if (!iftExamineComponentsPosition(segmentation)){
        puts("Failed");
        return 0;
    } else {
        puts("Approved");
    }

    printf("--- Relative Lungs Volume Comparison...............");
    fflush(stdout);
    if (!iftExamineRelativeVolumeDifference(segmentation, &volumeLeftLung, &volumeRightLung, &volumeTrachea)){
        puts("Failed");
        return 0;
    } else {
        puts("Approved");
    }

    return 1;
}

int main(int argc, const char *argv[])
{
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path          = NULL;

    iftGetRequiredArgs(args,&img_path);

    iftImage *label = iftReadImageByExt(img_path);

    // Checking Segmentation Result
    int segmentationIntegrity = 0;
    if (!iftScanAnomaliesOnLungSegmentation(label))
        segmentationIntegrity = 1;

    return segmentationIntegrity;
}

iftDict *iftGetArgs(int argc, const char *argv[])
{
    char program_description[2048] = \
        "- This program applies mathematical morphology operators to detect pleural anomaly according to the paper:\n" \
        "- AM e Sousa, AX Falcão, E Bagatin, GS Meirelles, \"A computational method to aid the detection and annotation of pleural lesions in CT images of the thorax\",\n"
        "published at the SPIE Medical Imaging, 2019.\n";

    iftCmdLineOpt cmd_line_opts[] = {
            {.short_name = "-l", .long_name = "--label-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Input Scene Image."},
            {.short_name = "-o", .long_name = "--output-img", .has_arg=true, .arg_type=IFT_STR_TYPE,
                    .required=true, .help="Output Scene Image."},
    };
    int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

    // Parser Setup
    iftCmdLineParser *parser = iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
    iftDict *args            = iftParseCmdLine(argc, argv, parser); // getting the passed options/arguments
    iftDestroyCmdLineParser(&parser);

    return args;
}

void iftGetRequiredArgs(  iftDict *args, char **img_path) {
    *img_path = iftGetStrValFromDict("--input-img", args);

    if (!iftFileExists(*img_path))
        iftError("Input image %s does not exists", "iftGetRequiredArgs", *img_path);

    puts("-----------------------");
    printf("- Input Image: %s\n", *img_path);
    puts("-----------------------");
}