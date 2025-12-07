///
/// INCLUDES
///
#include <ift.h>

///
/// HEADERS
///
void usage();

void checkIfSeedsAreOutsideLabel(iftImage *label, iftLabeledSet *seeds) {
    iftLabeledSet *S = seeds;
    int count = 0;
    while (S) {
        if (label->val[S->elem] == 0) {
            count++;
        }
        S = S->next;
    }
    printf("seeds outside: %d\n", count);
}

///
/// MAIN
///
int main(int argc, const char *argv[])
{
    // DISF img1
    int num_init_seeds, num_superpixels;
    iftAdjRel *A, *A_next;
    //timer *t1, *t2;


    if(argc != 6)usage();

    // Read inputs
    const char *img_dir   = argv[1];
    const char *out_dir   = argv[4];
    const char *mask_dir  = argv[5];

    iftMakeDir(out_dir);

    iftFileSet *img_set = iftLoadFileSetFromDir(img_dir, 1);
    iftFileSet *mask_set = iftLoadFileSetFromDir(mask_dir, 1);

    num_init_seeds = atoi(argv[2]);
    num_superpixels = atoi(argv[3]);


    if (img_set->n != mask_set->n) {
        iftError("Mask images and original images directory have different"
                 "quantity of files","main");
    }


    // Read Imgs
    iftImage **imgs = (iftImage**) iftAlloc(img_set->n, sizeof(iftImage*));
    iftImage **masks = (iftImage**) iftAlloc(mask_set->n, sizeof(iftImage*));
    iftMImage **mimgs = (iftMImage**) iftAlloc(img_set->n, sizeof(iftMImage*));
    iftImage **label_imgs = (iftImage**) iftAlloc(img_set->n, sizeof(iftImage*));

    for (int i = 0; i < img_set->n; i++) {
        char *img_path = img_set->files[i]->path;
        char *mask_path   = mask_set->files[i]->path;
        masks[i] = iftReadImageByExt(mask_path);
        imgs[i] = iftReadImageByExt(img_path);
        iftImage *masked = iftMask(imgs[i], masks[i]);
        iftDestroyImage(&imgs[i]);
        imgs[i] = masked;
        mimgs[i] = iftImageToMImage(imgs[i],GRAYNorm_CSPACE);
    }

    // Validate inputs
    if(num_init_seeds < 0) iftError("Non-positive N_0 = %d", "main", num_init_seeds);
    if(num_superpixels < 0) iftError("Non-positive N_f = %d", "main", num_superpixels);
    if(num_superpixels >= num_init_seeds) iftError("N_f >= N_0", "main");

    for (int i = 0; i < img_set->n; i++)
        iftVerifyImageDomains(imgs[i], masks[i], "main");


    // 4- and 6-adjacency
    A = iftSpheric(sqrtf(1.0));
    // 8- and 24-adjacency
    A_next = iftSpheric(sqrtf(112.0));



    // DISF
    //t1 = iftTic();
    label_imgs[0] = iftDISF(mimgs[0], A, num_init_seeds, num_superpixels, masks[0]);
    //t2 = iftToc();

    // Finish
    //iftPrintCompTime(t1, t2, "Execution time DISF");
    //iftWriteImageByExt(label_imgs[1], "%s", argv[4]);


    // Superspels (Supervoxels no tempo)

    //Center
    iftLabeledSet **centers = (iftLabeledSet**) iftAlloc(img_set->n, sizeof(iftLabeledSet*));
    iftLabeledSet **seeds = (iftLabeledSet**) iftAlloc(img_set->n, sizeof(iftLabeledSet*));
    centers[0] = iftGeodesicCenters(label_imgs[0]);
    seeds[0] = iftCopyLabeledSet(centers[0]);




    for (int i = 1 ; i < img_set->n; i++){ //iterando nos centroides de cada imagem //4D
        iftLabeledSet *center = centers[i-1];
        iftLabeledSet *seed   = NULL;           // set of centers from the supervoxels of the i-1th image
        while (center != NULL) {  //3D

            int diff = IFT_INFINITY_INT;
            int q_temp=0;
            int val_temp=0;

            int p = center->elem;                           // get the index value stored in the set (it is the centroid from the i-1th image)
            iftVoxel u = iftGetVoxelCoord(masks[i], p); // coordenada do centroide

            for (int k = 0; k < A_next->n; k++) {      //iterando na adjacencia
                iftVoxel v = iftGetAdjacentVoxel(A_next, u, k);         // neighbor pixel of p in coordinate form
                if (iftValidVoxel(imgs[i], v)) {
                    int q = iftGetVoxelIndex(imgs[i], v);        // neighbor pixel of p in index form
                    int intensity_a = imgs[i]->val[q];                             // intensity value (scalar) associated with the neighbor pixel of p
                    int intensity_p = imgs[i -
                                           1]->val[p];                           // intensity value (scalar) associated with the pixel of p
                    //transformar v em indice q
                    //comparar os dois, o que tiver menor diferença, salvar
                    if ((diff > abs(intensity_a - intensity_p))) {
                        if ((masks[i]->val[q])) {
                            q_temp = q;
                            val_temp = label_imgs[i - 1]->val[p];
                            //if(intensity_a==intensity_p)
                            //break;
                            diff = abs(intensity_a - intensity_p);
                        }
                    }
                }
            }
            if (diff != IFT_INFINITY_INT)
                iftInsertLabeledSet(&seed, q_temp, val_temp);
            center = center->next;
        }
        seeds[i] = seed;
        label_imgs[i] = iftDynamicSetRootPolicyInMask(mimgs[i], A, seeds[i], masks[i]);
        printf("seed set size: %d\n", iftLabeledSetSize(seeds[i]));
        printf("n objects from label: %d\n", iftMaximumValue(label_imgs[i]));
        //checkIfSeedsAreOutsideLabel(masks[i], seeds[i]);
        iftWriteImageByExt(label_imgs[i], "debug%d.nii.gz", i+1);
        centers[i] = iftGeodesicCenters(label_imgs[i]);
        printf("seed set size before x after x after do after: %d %d %d\n", iftLabeledSetSize(centers[i-1]), iftLabeledSetSize(seed), iftLabeledSetSize(centers[i]));
        //checkIfSeedsAreOutsideLabel(masks[i], centers[i]);
        printf("seed set size before x after x after do after: %d %d %d\n", iftLabeledSetSize(centers[i-1]), iftLabeledSetSize(seed), iftLabeledSetSize(centers[i]));
    }

    for (int i = 0; i < img_set->n; i++) {
        char *img_path = img_set->files[i]->path;
        char *filename  = iftFilename(img_path, NULL);


        iftWriteSeeds(seeds[i],label_imgs[i], "%s/%s-seeds.txt", out_dir, filename);
        iftWriteSeeds(centers[i],label_imgs[i], "%s/%s-centers.txt", out_dir, filename);
        iftWriteImageByExt(label_imgs[i], "label-%s/%s", out_dir, filename);
    }
    for (int i = 0; i < img_set->n; i++) {
        iftDestroyImage(&imgs[i]);
        iftDestroyImage(&masks[i]);
        iftDestroyMImage(&mimgs[i]);
        iftDestroyImage(&label_imgs[i]);
        iftDestroyLabeledSet(&seeds[i]);
        iftDestroyLabeledSet(&centers[i]);
    }
    // Clear
    iftDestroyFileSet(&img_set);
    iftDestroyFileSet(&mask_set);
    iftFree(imgs);
    iftFree(masks);
    iftFree(mimgs);
    iftFree(seeds);
    iftFree(centers);
    iftFree(label_imgs);
    iftDestroyAdjRel(&A);

}

///
/// METHODS
///
void usage()
{
    printf("USAGE: iftDISF [1] ... [5]\n");
    printf("REQUIRED:\n");
    printf("\t[1] - Input image DIR\n");
    printf("\t[2] - Initial number of seeds (N_0 > 0)\n");
    printf("\t[3] - Final number of superpixels (N_f << N_0)\n");
    printf("\t[4] - Output DIR \n");
    printf("\t[5] - Mask image DIR (x > 0 in ROIs)\n");
    iftError("Too many/few parameters", "main");
}
