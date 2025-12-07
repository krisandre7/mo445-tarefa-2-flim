#include "ift.h"

int main(int argc, char **argv) {
	if(argc != 13)
		iftError("Usage: iftMergeWatergray2D INPUT_IMAGE OUTPUT_IMAGE VOL_THRESHOLD AREA_THRESHOLD A1,..,A7 NREGIONS\n"
				"Alphas are weights for: mean b1, mean b2, mean b3, area, color_histogram, lbp_histogram, rectangularity","main");

	iftImage *image = iftReadImageP6(argv[1]);

	iftAdjRel *adj1 = iftCircular(sqrtf(5.0));

	iftAdjRel *adj2 = iftCircular(1.0);
	iftAdjRel *adj3 = iftCircular(0.0);

    iftImage *basins = iftImageBasins(image,adj1);
    iftImage *marker = iftVolumeClose(basins, atoi(argv[3]));

    iftImage *label_image = iftWaterGray(basins,marker,adj1);

    iftDataSet* dataset = iftSupervoxelsToSelectiveSearchDataset(image, label_image, 25, LAB_CSPACE);
    int i;
    for(i = 0; i < 7; i++)
    	dataset->alpha[i] = atof(argv[5 + i]);

    int nregions = atoi(argv[12]);
    iftRegionHierarchy *rh = iftCreateRegionHierarchy(dataset, adj2, iftMergeSelectiveSearchSupervoxel);

    iftImage *hlabel_image = iftFlattenRegionHierarchy(rh, nregions);

    iftDestroyDataSet(&dataset);

    dataset = iftSupervoxelsToSelectiveSearchDataset(image, hlabel_image, 25, LAB_CSPACE);

    iftImage *hlabel_filtered = iftEliminateRegionsByArea(dataset, adj2, atoi(argv[4]));
    iftWriteImageP2(hlabel_filtered, "label.pgm");

    iftColor rgb_color;
    rgb_color.val[0] = 255; rgb_color.val[1] = 0; rgb_color.val[2] = 0;
    iftColor ycbcr_color = iftRGBtoYCbCr(rgb_color,255);

    iftImage *hlabel_drawn = iftCopyImage(image);
    iftDrawBorders(hlabel_drawn,hlabel_filtered, adj2, ycbcr_color, adj3);

    iftWriteImageP6(hlabel_drawn,argv[2]);

    iftDestroyImage(&image);
    iftDestroyAdjRel(&adj1);
    iftDestroyAdjRel(&adj2);
    iftDestroyAdjRel(&adj3);
    iftDestroyImage(&basins);
    iftDestroyImage(&marker);
    iftDestroyImage(&label_image);
    iftDestroyImage(&hlabel_filtered);
    iftDestroyDataSet(&dataset);
    iftDestroyImage(&hlabel_image);
    iftDestroyRegionHierarchy(&rh);

    iftDestroyImage(&hlabel_drawn);
}
