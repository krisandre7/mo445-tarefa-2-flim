
#include "ift.h"

int main(int argc, char **argv) {

	if(argc != 6)
		iftError("Usage: gc_pixel [IMAGE_PATH] [SEEDS_PATH] [OUTPUT_PATH] [SPATIAL_RADIUS] [BETA]", "gc_pixel");

	float spatial_radius = atof(argv[4]);
	int beta = atoi(argv[5]);

	iftImage *image = iftReadImageByExt(argv[1]);
	iftLabeledSet *seeds = iftReadSeeds(image, argv[2]);

	iftAdjRel *adj = (iftIs3DImage(image) ? iftSpheric(spatial_radius) : iftCircular(spatial_radius));

	iftImage *basins = iftImageBasins(image, adj);

    /* Boykov Maxflow */
    puts("Boykov Maxflow");
    timer *t1 = iftTic();
	iftImage *result = iftGraphCut(basins, seeds, beta);
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    for(int p = 0; p < result->n; p++) {
		result->val[p] *= 255;
	}

	iftWriteImageByExt(result, argv[3]);

	iftDestroyImage(&result);

	/* My Maxflow */
    puts("My Maxflow");
    t1 = iftTic();
	result = iftOptimumPathGraphCut(basins, seeds, beta);
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

	for(int p = 0; p < result->n; p++) {
		result->val[p] *= 255;
	}

	iftWriteImageByExt(result, "jordao.png");

	iftDestroyImage(&image);
	iftDestroyImage(&result);
	iftDestroyLabeledSet(&seeds);
	iftDestroyAdjRel(&adj);
	iftDestroyImage(&basins);

	return 0;
}
