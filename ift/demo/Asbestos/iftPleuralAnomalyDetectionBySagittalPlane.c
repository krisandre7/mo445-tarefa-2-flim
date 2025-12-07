//
// Created by azaelmsousa on 25/08/21.
//

#include "ift.h"

#define LEFT_LUNG 1
#define RIGHT_LUNG 2

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **img_out);


iftImage *iftImproveALTIS(iftImage *segm);

/*    Skeleton-based Pleural Anomaly Detection    */
iftImage *iftSeparatePleuraBySkel(iftImage *lung, int lung_label);
iftImage *iftDetectPleuralAnomaly(iftImage *lung,iftImage *pleura);

/*    Compute Sagittal Plane    */
iftPlane *iftSagittalPlaneByPCA(iftImage *cloud_points);
/*************************************************************/

int main(int argc, const char *argv[])
{
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path          = NULL;
    char *out_path          = NULL;

    iftGetRequiredArgs(args,&img_path,&out_path);

    iftImage *label = iftReadImageByExt(img_path);

    puts("- Applying pleural anomaly detection");
    iftImage *out = iftImproveALTIS(label);
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
        "- This program applies 2D skeleton operator to separate the pleura costal and mediastinal." \
        "From this point, the anomalies located at the costal pleura are detected by morphological operations\n";

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

iftImage *iftImproveALTIS(iftImage *segm)
{
    //closing left lung
    puts("--- Left Lung");
    iftImage *LL = iftExtractObject(segm,LEFT_LUNG);
    iftImage *LL_border = iftSeparatePleuraBySkel(LL,LEFT_LUNG);
    iftDestroyImage(&LL);
    iftWriteImageByExt(LL_border,"LL_border.nii.gz");
    iftDestroyImage(&LL_border);

    return iftCopyImage(segm);
}

iftImage *iftSeparatePleuraBySkel(iftImage *lung, int lung_label)
{
    /* Distance to sagittal plane in 2D */
    float radius = 20.;

    iftSet *seeds = NULL;
    iftImage *aux = iftAddFrame(lung, iftRound(radius + 5), 0);
    iftImage *aux_dil = iftDilateBin(aux, &seeds, radius);
    iftImage *lung_dil = iftRemFrame(aux_dil, iftRound(radius + 5));
    iftDestroyImage(&aux);
    iftDestroyImage(&aux_dil);
    iftImage *pleura = iftCreateImageFromImage(lung);
    iftImage *border = iftBorderImage(lung, FALSE);
    iftWriteImageByExt(border,"border.nii.gz");

    iftImage *skel = iftCreateImageFromImage(lung);
    // for each slice containing a lung mask
    for (int s = 0; s < pleura->zsize; s++) {
        printf("----- Processing slice %3d/%3d\r", s, pleura->zsize - 1);
        fflush(stdout);

        iftImage *slice = iftGetXYSlice(lung_dil, s);
        if (iftMaximumValue(slice) == 0) {
            iftDestroyImage(&slice);
            continue;
        }

        // slice skeleton
        iftAdjRel *A = iftCircular(sqrt(2.));
        iftFImage *geo_skel = iftMSSkel2D(slice, A, IFT_INTERIOR, NULL, NULL);
        float geo_max = iftFMaximumValue(geo_skel);
        float th_rate = 0.1;
        iftImage *lung_skel = iftFThreshold(geo_skel, th_rate * geo_max, geo_max, 1);
        iftDestroyFImage(&geo_skel);
        iftDestroyAdjRel(&A);
        iftPutXYSlice(skel,lung_skel,s);

        //object's border vector to closest skeleton point
        iftImage *slice_border = iftGetXYSlice(border, s);
        iftSet *S = NULL;
        int n_spels = 0;
        for (int i = 0; i < slice_border->n; i++) {
            if (slice_border->val[i] > 0) {
                n_spels++;
                iftInsertSet(&S, i);
            }
        }
        iftVoxelArray *arr_voxel = iftCreateVoxelArray(n_spels);
        int p = 0;
        while (S != NULL) {
            int b = iftRemoveSet(&S);
            arr_voxel->val[p] = iftGetVoxelCoord(border, b);
            p++;
        }
        iftVoxelArray *closest_obj_voxels = iftFindClosestObjectVoxels(lung_skel, arr_voxel);

        //vector to skeleton + inner product between them and the skeleton's center vector
        iftImage *new_border = iftCreateImageFromImage(slice_border);
        for (int i = 0; i < arr_voxel->n; i++) {
            int dist_border = (arr_voxel->val[i].x-lung->xsize)*(arr_voxel->val[i].x-lung->xsize);
            int dist_skel   = (closest_obj_voxels->val[i].x-lung->xsize)*(closest_obj_voxels->val[i].x-lung->xsize);
            p = iftGetVoxelIndex(slice_border, arr_voxel->val[i]);
            if (dist_border > dist_skel){
                new_border->val[p] = 4; // costal pleura
            } else {
                new_border->val[p] = 5; // mediastinal pleura
            }
        }
        iftDestroyVoxelArray(&arr_voxel);
        iftDestroyVoxelArray(&closest_obj_voxels);
        iftDestroyImage(&lung_skel);
        iftDestroyImage(&slice_border);

        iftPutXYSlice(pleura, new_border, s);
        iftDestroyImage(&new_border);
    }
    puts("\n----- Done");
    iftDestroyImage(&border);

    iftWriteImageByExt(skel,"skel.nii.gz");
    iftDestroyImage(&skel);

    // Clean pleura label
    iftAdjRel *A = iftSpheric(sqrt(2.));
    iftImage *costal = iftExtractObject(pleura, 4);
    iftImage *mediastinal = iftExtractObject(pleura, 5);
    iftImage *largest_costal = iftSelectLargestComp(costal, A);
    iftImage *largest_mediastinal = iftSelectLargestComp(mediastinal, A);
    iftDestroyAdjRel(&A);
    //--- removing small disconnected objects
    for (int i = 0; i < pleura->n; i++) {
        if ((costal->val[i] > 0) && (largest_costal->val[i] == 0)) {
            pleura->val[i] = 5;
        } else if ((mediastinal->val[i] > 0) && (largest_mediastinal->val[i] == 0)) {
            pleura->val[i] = 4;
        }
    }
    iftDestroyImage(&costal);
    iftDestroyImage(&largest_costal);
    iftDestroyImage(&mediastinal);
    iftDestroyImage(&largest_mediastinal);

    return pleura;


    /* Distance to sagittal plane in 3D */
    /*

    float radius = 20.;

    iftSet *seeds = NULL; //
    iftImage *aux = iftAddFrame(lung, iftRound(radius + 5), 0); //
    iftImage *aux_dil = iftDilateBin(aux, &seeds, radius); //
    iftImage *lung_dil = iftRemFrame(aux_dil, iftRound(radius + 5)); //
    iftDestroyImage(&aux); //
    iftDestroyImage(&aux_dil); //
    iftDestroySet(&seeds); //
    iftImage *pleura = iftCreateImageFromImage(lung); // lung_dil -> lung
    iftImage *border = iftBorderImage(lung, FALSE); // lung_dil -> lung

    iftFImage *geo_skel = iftMSSkel(lung_dil);
    float threshold = 0.1;
    iftImage *lung_skel = iftSurfaceSkeleton(geo_skel,threshold*iftFMaximumValue(geo_skel),1);
    iftDestroyFImage(&geo_skel);
    iftDestroyImage(&lung_dil);
    iftWriteImageByExt(lung_skel,"lung_skel.nii.gz");

    iftVoxelArray *arr = iftCreateVoxelArray(iftCountObjectSpels(border,lung_label));
    int pos = 0;
    for (int i = 0; i < border->n; i++){
        if (border->val[i] > 0){
            arr->val[pos++] = iftGetVoxelCoord(border,i);
        }
    }
    iftDestroyImage(&border);

    iftVoxelArray *closest = iftFindClosestObjectVoxels(lung_skel,arr);
    iftDestroyImage(&lung_skel);

    for (int i = 0; i < closest->n; i++){
        int dist_border = (arr->val[i].x-lung->xsize/2)*(arr->val[i].x-lung->xsize/2);
        int dist_skel   = (closest->val[i].x-lung->xsize/2)*(closest->val[i].x-lung->xsize/2);
        int p = iftGetVoxelIndex(lung,arr->val[i]);
        if (dist_border-dist_skel > 0){
            pleura->val[p] = 1;
        } else {
            pleura->val[p] = 2;
        }
    }
    iftDestroyVoxelArray(&arr);
    iftDestroyVoxelArray(&closest);

    return pleura;*/
}

iftImage *iftDetectPleuralAnomaly(iftImage *lung,iftImage *pleura){

    // Global variables
    float radius = 20.;
    iftImage *costal = NULL, *costal_frame = NULL, *er = NULL, *close = NULL;
    iftImage *mediastinum = NULL, *mediastinum_frame = NULL, *mediastinum_dil = NULL;
    costal        = iftExtractObject(pleura,4);
    costal_frame  = iftAddFrame(costal,iftRound(radius)+5,0);
    iftDestroyImage(&costal);

    // IFT variables
    iftImage  *dist=NULL,*root=NULL,*dil=NULL;
    iftGQueue *Q=NULL;
    int        i,p,q,tmp;
    iftVoxel   u,v,r;
    iftAdjRel *A=iftSpheric(sqrt(3.)),*B=iftSpheric(1.);
    float      maxdist=radius*radius; // the EDT computes squared
    iftSet    *seed = NULL;

    mediastinum = iftExtractObject(pleura,5);
    mediastinum_frame = iftAddFrame(mediastinum,iftRound(radius)+5,0);
    mediastinum_dil = iftDilateBin(mediastinum_frame,&seed,0.);
    iftDestroyImage(&mediastinum);
    iftDestroyImage(&mediastinum_frame);
    iftDestroySet(&seed);

    // IFT Dilation limited by mediastinal pleura
    seed = iftObjectBorderSet(costal_frame,B);
    iftDestroyAdjRel(&B);

    // Initialization

    dil   = iftCopyImage(costal_frame);
    dist  = iftCreateImage(costal_frame->xsize,costal_frame->ysize,costal_frame->zsize);
    root  = iftCreateImage(costal_frame->xsize,costal_frame->ysize,costal_frame->zsize);
    Q     = iftCreateGQueue(IFT_QSIZE,costal_frame->n,dist->val);

    for (p=0; p < costal_frame->n; p++) {
        if (costal_frame->val[p] == 0)
            dist->val[p]= IFT_INFINITY_INT;
    }

    while (seed != NULL) {
        p = iftRemoveSet(&seed);
        dist->val[p]=0;
        root->val[p]=p;
        iftInsertGQueue(&Q,p);
    }

    // Image Foresting Transform: Distance transform

    while(!iftEmptyGQueue(Q)) {
        p=iftRemoveGQueue(Q);

        if (mediastinum_dil->val[p] == 5) // mediastinal pleura
            continue;

        if (dist->val[p] <= maxdist){

            dil->val[p] = costal_frame->val[root->val[p]]; // dilation
            u = iftGetVoxelCoord(costal_frame,p);
            r = iftGetVoxelCoord(costal_frame,root->val[p]);

            for (i=1; i < A->n; i++){
                v = iftGetAdjacentVoxel(A,u,i);
                if (iftValidVoxel(costal_frame,v)){
                    q = iftGetVoxelIndex(costal_frame,v);
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

    iftDestroyGQueue(&Q);
    iftDestroyImage(&costal_frame);
    iftDestroyImage(&root);
    iftDestroyImage(&dist);
    iftDestroyAdjRel(&A);

    // Performing erosion
    er   = iftErodeBin(dil,&seed,radius);
    close = iftRemFrame(er,iftRound(radius)+5);
    iftDestroyImage(&er);
    iftDestroyImage(&dil);
    iftDestroySet(&seed);

    return close;
}