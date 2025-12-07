#include "AllFunctions.c"

#define ALPHA 0.5
#define BETA 12
#define SEGM_ITERS 1
#define SMOOTH_ITERS 2

int main(int argc, const char *argv[]) 
{
    if(argc != 10) iftError("Usage: <image><objsm><num_seeds><obj_perc><map_thr><method><output_segm><output_seed><gt_img>", "main");

    int method_op, num_seeds;
    float obj_perc, map_thr;
    char *basename, *img_id, *csv_results;
    iftImage  *img, *mask, *label_img, *seed_img, *gt_img, *objsm_img, *blur_img;
    iftMImage *mimg;
    iftAdjRel *A;
    iftIGraph *igraph;
    iftKernel *gaussian;
    iftSList *split_name;
    
    basename = iftBasename(argv[1]);
    img = iftReadImageByExt(argv[1]);
    objsm_img = iftReadImageByExt(argv[2]);
    num_seeds = atoi(argv[3]);
    obj_perc = atof(argv[4]);
    map_thr = atof(argv[5]);
    method_op = atoi(argv[6]);
    gt_img = iftReadImageByExt(argv[9]);

    split_name = iftSplitString(basename,"/");
    img_id = split_name->tail->elem;

    mask = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);

    gaussian = iftGaussianKernel2D(sqrtf(2.0),5.0);

    blur_img = iftLinearFilter(img, gaussian);

    if (iftIs3DImage(img)) A = iftSpheric(1.0);
    else A = iftCircular(1.0);

    if (iftIsColorImage(img)) mimg = iftImageToMImage(blur_img,LABNorm_CSPACE);
    else mimg = iftImageToMImage(blur_img,GRAY_CSPACE);

    if( method_op == 0 ) seed_img = iftGridSampling(mimg, mask, num_seeds);
    else if( method_op == 1 ) seed_img = iftAltMixedSampling(mimg, mask, num_seeds);
    else if( method_op == 2 ) seed_img = iftGrayObjMapGridSamplOnMaskByArea(objsm_img, mask, num_seeds, map_thr, obj_perc);
    else if( method_op == 3 ) seed_img = OSMOXSampling(objsm_img, mask, num_seeds, obj_perc);
    else iftError("Non-existent sampling method!", "main");
    
    igraph = iftImplicitIGraph(mimg, mask, A);

    _iftIGraphISF_Root(igraph, seed_img, ALPHA, BETA, SEGM_ITERS);

    if (SMOOTH_ITERS > 0){
        iftIGraphSetWeightForRegionSmoothing(igraph, img);
        iftIGraphSmoothRegions(igraph, SMOOTH_ITERS);
    }
    label_img = iftIGraphLabel(igraph);

    csv_results = computeScores(img_id, label_img, seed_img, gt_img);
    printf("%s\n", csv_results);

    iftWriteImageByExt(label_img, argv[7]);
    iftWriteImageByExt(seed_img, argv[8]);
    
    free(basename);
    iftDestroySList(&split_name);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&img);
    iftDestroyImage(&mask);
    iftDestroyImage(&label_img);
    iftDestroyImage(&seed_img);
    iftDestroyImage(&gt_img);
    iftDestroyImage(&objsm_img);
    iftDestroyMImage(&mimg);
    iftDestroyImage(&blur_img);
    iftDestroyKernel(&gaussian);
    iftDestroyIGraph(&igraph);

    return 0;
}