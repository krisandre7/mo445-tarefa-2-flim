#include "ift.h"

int main(int argc, char **argv) 
{
	if (argc != 4)
		iftError("<SCN Image> <slic regions> <slic_compactness>", "Slic - main");

	char buffer[1024];
	iftImage *label_image = NULL;
	iftImage *drawn_labels = NULL;
	iftImage *image = iftReadImage(argv[1]);
	iftAdjRel *adj1 = iftSpheric(1.0);
    iftAdjRel *adj0 = iftSpheric(0.0);
	int slic_nregions = atoi(argv[2]);
	float slic_compactness = atof(argv[3]);
	char *image_name;
	image_name = iftSplitStringAt(argv[1], "/", -1);

	label_image = iftCreateImage(image->xsize, image->ysize, image->zsize);
    iftWriteRawIntArray(image_name, image->val, image->n);

    sprintf(buffer, "python slic3d.py %d %d %d %s %s %d %.2f", image->xsize, image->ysize, image->zsize, image_name, image_name, slic_nregions, slic_compactness);
	puts(buffer);
    system(buffer);
    free(label_image->val);
    label_image->val = iftReadRawIntArray(image_name, image->n);
    sprintf(buffer, "rm %s", image_name);
    system(buffer);

    //Generate paper images only
    drawn_labels = iftCopyImage(image);
    iftColor blue;
    blue.val[0] = 4095; blue.val[1] = 4095; blue.val[2] = 4095;
    blue = iftRGBtoYCbCr(blue);
    iftDrawBorders(drawn_labels, label_image, adj1, blue, adj0);
    iftWriteImage(drawn_labels, "superpixels.scn");
	
	return 0;
}