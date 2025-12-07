#include "ift.h"


int main(int argc, const char *argv[]) {
    if ((argc != 3) && (argc != 4))
        iftError("%s <input_image> <output_image> <(optional) Adjacency Radius: Default: 1.0>", "main", argv[0]);

    iftImage *img = iftReadImageByExt(argv[1]);

    float adj_radius = (argc == 4) ? atof(argv[3]) : 1.0;
    iftAdjRel *A = (iftIs3DImage(img)) ? iftSpheric(adj_radius) : iftCircular(adj_radius);


    printf("- Filtering image %s by Mean: adj. radius: %f\n", argv[1], adj_radius);
    iftImage *filt_img = iftMeanFilter(img, A);
    iftWriteImageByExt(filt_img, argv[2]);


    iftDestroyAdjRel(&A);
    iftDestroyImage(&img);

    return 0;
}






