#include "ift.h"

int main(int argc, char *argv[])
{
	iftImage *img, *label, *gt_img;
	iftDataSet *output_dataset;
	int nbins_per_band, desc_type;
	int *superpixel_clslabel;
	int nclasses, nsuperpixels;
			
	if (argc != 7)
		iftError("Usage: iftComputeSuperpixelDescriptor <image.ppm> <superpixels.pgm> <gt_img.pgm> <num_bins_per_band> <type 1: ColorMoments, 2: BIC> <output_dataset>", "main");
	
	img = iftReadImageByExt(argv[1]);
	label = iftReadImageByExt(argv[2]);
	gt_img = iftReadImageByExt(argv[3]);
	nbins_per_band = atoi(argv[4]);
	desc_type = atoi(argv[5]);
	
	if (desc_type == 1) {
		output_dataset = iftSupervoxelsToLabColorMomentsDataset(img, label);
	} else {
		output_dataset = iftSupervoxelsToBICDataset(img, label, nbins_per_band, 0, NULL);
	}

	nclasses = iftMaximumValue(gt_img);
	nsuperpixels = iftMaximumValue(label);

	superpixel_clslabel = iftAllocIntArray(nsuperpixels);
	for (int p = 0; p < gt_img->n; ++p) {
		int index_sup = label->val[p] - 1;
		superpixel_clslabel[index_sup] = gt_img->val[p];
	}
	
	printf("Copying labels ...\n");
	// copy labels
	output_dataset->nclasses = nclasses;
	for (int i = 0; i < output_dataset->nsamples; ++i) {
		output_dataset->sample[i].truelabel = superpixel_clslabel[i];
	}
	printf("nfeats %d\n", output_dataset->nfeats);
	printf("nsamples %d\n", output_dataset->nsamples);
	
	// write dataset
	iftSetDistanceFunction(output_dataset, 1);
    iftWriteOPFDataSet(output_dataset, argv[6]);

    // Free
    free(superpixel_clslabel);
	iftDestroyDataSet(&output_dataset);
	iftDestroyImage(&img);
	iftDestroyImage(&label);
	iftDestroyImage(&gt_img);
	
}