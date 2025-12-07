///
/// INCLUDES
///
#include <ift.h>

///
/// HEADERS
///
void usage();

///
/// MAIN
///
int main(int argc, const char *argv[]) 
{
    int num_init_seeds, num_superpixels;
    iftImage *label_img, *img, *mask;
    iftMImage *mimg;
    iftAdjRel *A;    
    timer *t1, *t2;
    
    if(argc != 5 && argc != 6) usage();

    // Read inputs
    img = iftReadImageByExt(argv[1]);
    num_init_seeds = atoi(argv[2]);
    num_superpixels = atoi(argv[3]);

    mask = NULL;
    if(argc == 6) mask = iftReadImageByExt(argv[5]);

    // Validate inputs
    if(num_init_seeds < 0) iftError("Non-positive N_0 = %d", "main", num_init_seeds);
    if(num_superpixels < 0) iftError("Non-positive N_f = %d", "main", num_superpixels);
    if(num_superpixels >= num_init_seeds) iftError("N_f >= N_0", "main");
    if(mask != NULL) iftVerifyImageDomains(img, mask, "main");

    // Init other inputs
    if (iftIsColorImage(img)) mimg = iftImageToMImage(img,LABNorm_CSPACE);
    else mimg = iftImageToMImage(img,GRAYNorm_CSPACE);    

    // 4- and 6-adjacency
    // if(iftIs3DMImage(mimg)) A = iftSpheric(sqrtf(1.0));
    // else A = iftCircular(sqrtf(1.0));
    // 8- and 24-adjacency
    if(iftIs3DMImage(mimg)) A = iftSpheric(sqrtf(3.0));
    else A = iftCircular(sqrtf(2.0));


    // DISF
    t1 = iftTic();
    label_img = iftDISF(mimg, A, num_init_seeds, num_superpixels, mask);
    t2 = iftToc();
    
    // Finish
    iftPrintCompTime(t1, t2, "Execution time");
    iftWriteImageByExt(label_img, "%s", argv[4]);

    // Clear
    iftDestroyImage(&img);
    if(argc == 6) iftDestroyImage(&mask);
    iftDestroyImage(&label_img);
    iftDestroyAdjRel(&A);
    iftDestroyMImage(&mimg);
}

///
/// METHODS
///
void usage()
{
    printf("USAGE: iftDISF [1] ... [4] {5}\n");
    printf("REQUIRED:\n");
    printf("\t[1] - Input image (.png, .jpg, .pgm, .scn, .ppm)\n");
    printf("\t[2] - Initial number of seeds (N_0 > 0)\n");
    printf("\t[3] - Final number of superpixels (N_f << N_0)\n");
    printf("\t[4] - Output label image (.png, .jpg, .pgm, .scn)\n");
    printf("OPTIONAL:\n");
    printf("\t{5} - Mask image (x > 0 in ROIs)\n");
    iftError("Too many/few parameters", "main");
}