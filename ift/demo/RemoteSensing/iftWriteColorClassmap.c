#include "ift.h"

int main(int argc, char *argv[]) {

	iftImage *label, *color_img;
	int n_colors;

	if (argc != 3) {
		iftError("Usage: iftWriteColorClassmap <label_image> <output_image>", "main");
	}

	label = iftReadImageByExt(argv[1]);
	n_colors = iftMaximumValue(label);

	iftColorTable *ctb = iftCreateColorTable(n_colors + 1);
	// reserve color 0 for background
	ctb->color[0].val[0] = 0;
	ctb->color[0].val[1] = 0;
	ctb->color[0].val[2] = 0;

	color_img = iftCreateColorImage(label->xsize, label->ysize, label->zsize);

	for (int p = 0; p < color_img->n; ++p) {
		iftColor YCbCr = iftRGBtoYCbCr(ctb->color[label->val[p]], 255);
		color_img->val[p] = YCbCr.val[0];
		color_img->Cb[p] = YCbCr.val[1];
		color_img->Cr[p] = YCbCr.val[2];
	}

	iftWriteImageP6(color_img, argv[2]);
	iftDestroyImage(&color_img);
	iftDestroyImage(&label);
	iftDestroyColorTable(&ctb);

	return 0;
}