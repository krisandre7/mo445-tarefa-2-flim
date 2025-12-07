//
// Created by azaelmsousa on 23/08/21.
//

#include "ift.h"

#define LEFT_LUNG 1
#define RIGHT_LUNG 2

/************************** HEADERS **************************/
iftDict *iftGetArgs(int argc, const char *argv[]);
void iftGetRequiredArgs(  iftDict *args, char **img_path, char **img_out);

/* Geodesic-based Pleural Anomaly Detection */
iftImage *iftPleuralAnomalyDetectionByGeodesicDistance(iftImage *segm, float radius, float geo_th);
iftImage *iftGeodesicDistanceInPleura(iftImage *mask, iftImage *lung, float geo_dist);
iftImage *iftRestrictedCloseBin(iftImage *bin, float radius, iftImage *forbidden);
iftImage *iftRemoveMediastinumForest(iftImage *mediastinum_bin, iftImage *closed, iftImage *close_root, iftSet *seeds);
/*************************************************************/

int main(int argc, const char *argv[])
{
    iftDict *args = iftGetArgs(argc, argv);

    char *img_path    = NULL;
    char *out_path    = NULL;
    float geo_th      = 0.15; //default
    float radius      = 20.;  //default
    timer *t1,*t2;

    iftGetRequiredArgs(args,&img_path,&out_path);
    if (iftDictContainKey("--geodesic-threshold",args,NULL)){
        geo_th = iftGetDblValFromDict("--geodesic-threshold",args);
    }
    if (iftDictContainKey("--radius",args,NULL)){
        radius = iftGetDblValFromDict("--radius",args);
    }

    iftImage *label = iftReadImageByExt(img_path);

    t1 = iftTic();

    puts("- Applying pleural anomaly detection");
    iftImage *out = iftPleuralAnomalyDetectionByGeodesicDistance(label, radius, geo_th);
    iftDestroyImage(&label);

    t2 = iftToc();

    puts("- Writing output image");
    iftWriteImageByExt(out,out_path);

    puts("- Deallocating memory");
    iftDestroyImage(&out);

    puts("\nDone...");
    printf("Total Pleural Anomaly Detection took ");
    char *formatted = iftFormattedTime(iftCompTime(t1, t2));
    puts(formatted);
    iftFree(formatted);

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
            {.short_name = "-g", .long_name = "--geodesic-threshold", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=false, .help="Threshold [0,1] to be used in the geodesic distance transform (Default 0.15)."},
            {.short_name = "-r", .long_name = "--radius", .has_arg=true, .arg_type=IFT_DBL_TYPE,
                    .required=false, .help="Morphological Closing radius (Default 20.)"},
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

/* ToDo: Depict method
 * Separate pleuras by geodesic distance starting from trachea-lung voxels followed by a morphological
 * closing with respect to the pleuras.
 */
iftImage *iftPleuralAnomalyDetectionByGeodesicDistance(iftImage *segm, float radius, float geo_th)
{

    timer *t1,*t2;

    // Extracting LEFT LUNG
    puts("--- Left Lung");
    t1 = iftTic();
    iftImage *LL                = iftExtractObject(segm,LEFT_LUNG);
    t2 = iftToc();
    printf("Extract Left Lung took ");
    puts(iftFormattedTime(iftCompTime(t1, t2)));

    t1 = iftTic();
    iftImage *LL_mediastinum    = iftGeodesicDistanceInPleura(segm, LL, geo_th);
    t2 = iftToc();
    printf("Geodesic Distance In Pleura took ");
    puts(iftFormattedTime(iftCompTime(t1, t2)));

    t1 = iftTic();
    iftImage *LL_closed         = iftRestrictedCloseBin(LL,radius,LL_mediastinum);
    t2 = iftToc();
    printf("Restricted Close Bin took ");
    puts(iftFormattedTime(iftCompTime(t1, t2)));
    iftDestroyImage(&LL_mediastinum);
    iftDestroyImage(&LL);

    // Extracting RIGHT LUNG
    puts("--- Right Lung");
    t1 = iftTic();
    iftImage *RL                = iftExtractObject(segm,RIGHT_LUNG);
    t2 = iftToc();
    printf("Extract Right Lung took ");
    puts(iftFormattedTime(iftCompTime(t1, t2)));

    t1 = iftTic();
    iftImage *RL_mediastinum    = iftGeodesicDistanceInPleura(segm, RL, geo_th);
    t2 = iftToc();
    printf("Geodesic Distance In Pleura took ");
    puts(iftFormattedTime(iftCompTime(t1, t2)));

    t1 = iftTic();
    iftImage *RL_closed         = iftRestrictedCloseBin(RL,radius,RL_mediastinum);
    t2 = iftToc();
    printf("Restricted Close Bin took ");
    puts(iftFormattedTime(iftCompTime(t1, t2)));
    iftDestroyImage(&RL_mediastinum);
    iftDestroyImage(&RL);

    //merging objects
    iftImage *closed_segm = iftOr(LL_closed,RL_closed);
    iftDestroyImage(&LL_closed);
    iftDestroyImage(&RL_closed);

    //adding trachea
    for (int p = 0; p < segm->n; p++)
        if (segm->val[p] == 3)
            closed_segm->val[p] = 3;

    return closed_segm;

}

iftImage *iftGeodesicDistanceInPleura(iftImage *mask, iftImage *lung, float geo_dist)
{
    iftImage *border      = iftBorderImage(lung,FALSE);
    iftImage *mediastinum = NULL;
    iftFImage *dist       = NULL;
    iftFHeap *Q           = NULL;
    iftSet *S             = NULL;
    int i,p,q;
    float tmp;
    iftVoxel u,v;
    iftSet *Saux;

    mediastinum = iftCreateImageFromImage(mask);

    // Computing Seeds
    iftAdjRel *A  = iftSpheric(sqrt(3.));
    for (int p = 0; p < border->n; p++)
    {
        if (border->val[p] == 0)
            continue;
        iftVoxel u = iftGetVoxelCoord(border,p);
        for (int a = 1; a < A->n; a++){
            iftVoxel v = iftGetAdjacentVoxel(A,u,a);
            if (iftValidVoxel(mask,v)){
                int q = iftGetVoxelIndex(border,v);
                if (mask->val[q] == 3){
                    mediastinum->val[q] = 1;
                    iftInsertSet(&S,p);
                    break;
                }
            }
        }
    }

    dist  = iftCreateFImage(border->xsize,border->ysize,border->zsize);
    Q     = iftCreateFHeap(dist->n,dist->val);

    for (p=0; p < border->n; p++) {
        if (border->val[p] != 0)
            dist->val[p]= IFT_INFINITY_FLT;
    }

    Saux = S;
    while (Saux != NULL) {
        p = Saux->elem;
        if (border->val[p] == 0)
            iftError("Seed set must be inside mask", "iftGeodesicDistTrans");
        dist->val[p]=0;
        iftInsertFHeap(Q,p);
        Saux = Saux->next;
    }

    // Image Foresting Transform

    while(!iftEmptyFHeap(Q)) {

        p=iftRemoveFHeap(Q);

        if (dist->val[p] > geo_dist)
            continue;

        //Gets the voxel.
        u = iftGetVoxelCoord(border,p);

        for (i=1; i < A->n; i++){
            v = iftGetAdjacentVoxel(A,u,i);
            if (iftValidVoxel(border,v)){
                q = iftGetVoxelIndex(border,v);
                if ((dist->val[q] > dist->val[p]) && (border->val[q] != 0)){
                    tmp = dist->val[p] + iftSmoothEuclideanDistance((float) iftSquaredVoxelDistance(u, v));
                    if (tmp < dist->val[q]){
                        mediastinum->val[q] = 1;
                        dist->val[q]  = tmp;
                        if(Q->color[q] == IFT_WHITE){
                            iftInsertFHeap(Q, q);
                        }else{
                            iftGoUpFHeap(Q, Q->pos[q]);
                        }
                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyImage(&border);
    iftDestroyFHeap(&Q);
    iftDestroyFImage(&dist);
    iftDestroySet(&S);

    return mediastinum;
}

iftImage *iftRestrictedCloseBin(iftImage *bin, float radius, iftImage *forbidden)
{
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
    iftDestroyImage(&dist);
    iftDestroyAdjRel(&A);

    // ToDo: Remove mediastinum forest from seeds
    iftImage *filtered_dil = iftRemoveMediastinumForest(forbidden_frame,dil,root,seed); //
    iftDestroyImage(&forbidden_frame);
    iftDestroyImage(&root);
    iftDestroyImage(&dil);

    // Performing erosion
    iftImage *er   = iftErodeBin(filtered_dil,&seed,radius);
    iftImage *close = iftRemFrame(er,iftRound(radius)+5);
    iftDestroyImage(&er);
    iftDestroyImage(&filtered_dil);
    iftDestroySet(&seed);

    return close;
}

iftImage *iftRemoveMediastinumForest(iftImage *mediastinum_bin, iftImage *closed, iftImage *closed_root, iftSet *seeds)
{
    iftImage *out = iftCopyImage(closed);
    for (int i = 0; i < closed->n; i++){
        if (mediastinum_bin->val[closed_root->val[i]] > 0){
            out->val[i] = 0;
        }
    }

    iftSet *aux = seeds;
    while (aux != NULL){
        int elem = aux->elem;
        if (mediastinum_bin->val[closed_root->val[elem]] > 0)
            iftRemoveSetElem(&seeds,elem);
        aux = aux->next;
    }

    return out;
}