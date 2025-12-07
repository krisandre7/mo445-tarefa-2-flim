/*
Created by azaelmsousa on 12/04/21.

   This program is associated with the paper "A computational method to aid the detection and annotation of pleural lesions in CT images of the thorax",
published at the SPIE Medical Imaging, 2019.

   It applies morphology operations to include pleural anomalies into the original ALTIS segmentation.
*/

#include "ift.h"

#define LEFT_LUNG 1
#define RIGHT_LUNG 2

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **img_out);

/*    Morphology-based Pleural Anomaly Detection    */
iftImage *iftPleuralAnomalyDetectionByMathMorphology(iftImage *segm);
/*************************************************************/

int main(int argc, const char *argv[])
{
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path          = NULL;
    char *out_path          = NULL;

    iftGetRequiredArgs(args,&img_path,&out_path);

    iftImage *label = iftReadImageByExt(img_path);

    puts("- Applying pleural anomaly detection");
    iftImage *out = iftPleuralAnomalyDetectionByMathMorphology(label);
    iftDestroyImage(&label);

    puts("- Writing output image");
    iftWriteImageByExt(out,out_path);

    puts("- Deallocating memory");
    iftDestroyImage(&out);

    return 0;
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

void iftGetRequiredArgs(  iftDict *args, char **img_path, char **out_img) {
    *img_path = iftGetStrValFromDict("--label-img", args);
    *out_img = iftGetStrValFromDict("--output-img", args);

    if (!iftFileExists(*img_path))
        iftError("Input image %s does not exists", "iftGetRequiredArgs", *img_path);

    puts("-----------------------");
    printf("- Label Image: %s\n", *img_path);
    printf("- Output Image: %s\n", *out_img);
    puts("-----------------------");
}

iftImage *iftRestrictedCloseBin(iftImage *bin, iftImage *forbidden, float radius){

    iftImage *bin_frame = iftAddFrame(bin,iftRound(radius)+5,0);
    iftImage *forbidden_frame = iftAddFrame(forbidden,iftRound(radius)+5,0);

    // IFT variables
    iftImage  *dist=NULL,*root=NULL,*dil=NULL;
    iftGQueue *Q=NULL;
    int        i,p,q,tmp;
    iftVoxel   u,v,r;
    iftAdjRel *A=iftSpheric(sqrt(3.)),*B=iftSpheric(1.);
    float      maxdist=radius*radius; // the EDT computes squared
    iftSet    *seed=NULL;

    // IFT Dilation limited by mediastinal pleura
    seed = iftObjectBorderSet(bin_frame,B);
    iftDestroyAdjRel(&B);

    // Initialization

    dil   = iftCopyImage(bin_frame);
    dist  = iftCreateImage(bin_frame->xsize,bin_frame->ysize,bin_frame->zsize);
    root  = iftCreateImage(bin_frame->xsize,bin_frame->ysize,bin_frame->zsize);
    Q     = iftCreateGQueue(IFT_QSIZE,bin_frame->n,dist->val);

    for (p=0; p < bin_frame->n; p++) {
        if (bin_frame->val[p] == 0)
            dist->val[p]= IFT_INFINITY_INT;
    }

    while (seed != NULL) {
        p = iftRemoveSet(&seed);
        dist->val[p]=0;
        root->val[p]=p;
        iftInsertGQueue(&Q,p);
    }

    // Image Foresting Transform

    while(!iftEmptyGQueue(Q)) {
        p=iftRemoveGQueue(Q);

        if (forbidden_frame->val[p] > 0)
            continue;

        if (dist->val[p] <= maxdist){

            dil->val[p] = bin_frame->val[root->val[p]]; // dilation
            u = iftGetVoxelCoord(bin_frame,p);
            r = iftGetVoxelCoord(bin_frame,root->val[p]);

            for (i=1; i < A->n; i++){
                v = iftGetAdjacentVoxel(A,u,i);
                if (iftValidVoxel(bin_frame,v)){
                    q = iftGetVoxelIndex(bin_frame,v);
                    if (dist->val[q] > dist->val[p]){
                        tmp = (v.x-r.x)*(v.x-r.x) + (v.y-r.y)*(v.y-r.y) + (v.z-r.z)*(v.z-r.z);
                        if (tmp < dist->val[q]){
                            if (dist->val[q] != IFT_INFINITY_INT)
                                iftRemoveGQueueElem(Q, q);
                            dist->val[q]  = tmp;
                            root->val[q]  = root->val[p];
                            iftInsertGQueue(&Q, q);
                        }
                    }
                }
            }
        }else{ /* seeds for a possible subsequent erosion */
            iftInsertSet(&seed,p);
        }
    }

    iftDestroyImage(&bin_frame);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&root);
    iftDestroyImage(&dist);
    iftDestroyAdjRel(&A);

    // Performing erosion
    iftImage *er   = iftErodeBin(dil,&seed,radius);
    iftImage *close = iftRemFrame(er,iftRound(radius)+5);
    iftDestroyImage(&er);
    iftDestroyImage(&dil);
    iftDestroySet(&seed);

    return close;
}

/*
 * An adaptation of the work of SOUSA et al., 2019. In this work the authors
 * explored morphological operations to segment pleural anomalies. Here we are
 * including these segmentations into the lungs mask.
 *
 * e Sousa, A. D. M., Falcão, A. X., Bagatin, E., & Meirelles, G. S. (2019, March).
 * A computational method to aid the detection and annotation of pleural lesions in
 * CT images of the thorax. In International Society for Optics and Photonics Medical
 * Imaging 2019: Image Processing. Vol. 10949, p. 109490Z.
 *
 */
iftImage *iftPleuralAnomalyDetectionByMathMorphology(iftImage *segm){

    iftImage *mediastinal_voxels = iftCreateImageFromImage(segm);
    iftAdjRel *A = iftSpheric(sqrt(3.));

    // 1) Selecting mediastinal voxels adjacent to the trachea
    for (int p = 0; p < segm->n; p++){
        if ((segm->val[p] > 0) && (segm->val[p] < 3)) {
            iftVoxel u = iftGetVoxelCoord(segm, p);
            for (int a = 1; a < A->n; a++) {
                iftVoxel v = iftGetAdjacentVoxel(A, u, a);
                if (iftValidVoxel(segm, v)) {
                    int q = iftGetVoxelIndex(segm, v);
                    if (segm->val[q] == 3) { //trachea
                        mediastinal_voxels->val[p] = segm->val[p];
                        break;
                    }
                }
            }
        }
    }

    // 2) Dilating voxels of (1) in a radius r1
    float r1 = 15.;
    iftSet *seed = NULL;
    iftImage *mediastinum_dil = iftDilateBin(mediastinal_voxels,&seed,r1);
    for (int p = 0; p < segm->n; p++){
        if (segm->val[p] > 0)
            mediastinum_dil->val[p] = 0;
    }
    iftDestroyImage(&mediastinal_voxels);
    iftDestroySet(&seed);

    // 3) Morphological closing with respect to the dilated voxels of (2)
    float r2 = 25.;
    iftImage *LL = iftExtractObject(segm,LEFT_LUNG);
    iftImage *LL_forbidden = iftExtractObject(mediastinum_dil,LEFT_LUNG);
    iftImage *LL_closed = iftRestrictedCloseBin(LL,LL_forbidden,r2);
    iftDestroyImage(&LL);
    iftDestroyImage(&LL_forbidden);

    iftImage *RL = iftExtractObject(segm,RIGHT_LUNG);
    iftImage *RL_forbidden = iftExtractObject(mediastinum_dil,RIGHT_LUNG);
    iftImage *RL_closed = iftRestrictedCloseBin(RL,RL_forbidden,r2);
    iftDestroyImage(&RL);
    iftDestroyImage(&RL_forbidden);

    // 4) Residue form the pleura anomalies
    iftImage *closed_segm = iftOr(LL_closed,RL_closed);
    iftDestroyImage(&LL_closed);
    iftDestroyImage(&RL_closed);

    for (int p = 0; p < segm->n; p++)
        if (segm->val[p] == 3)
            closed_segm->val[p] = 3;

    return closed_segm;
}