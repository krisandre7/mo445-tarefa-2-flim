#include "ift.h"




int main(int argc, char **argv) {
	if (argc != 3)
		iftError("iftTestRaniel <image.pgm> <out.pgm>", "main");

	char filename[100]; strcpy(filename, argv[1]);
	char out[100]; strcpy(out, argv[2]);
	
	iftImage *gimg = iftReadImageP5(filename);

	// // printing as a vector
	// for (int p = 0; p < gimg->n; p++)
	// 	printf("[%d] = %d\n", p, gimg->val[p]);

	// // printing as a matrix
	// for (int p = 0; p < gimg->n; p++) {
	// 	int x = iftGetXCoord(gimg, p); // get the x index from the pixel p
	// 	int y = iftGetYCoord(gimg, p); // get the x index from the pixel p
	// 	printf("[%d][%d] = %d\n", y, x, gimg->val[p]);
	// }

	// scanning the adjacents vector
	iftAdjRel *A = iftCircular(1.5); // 8-neighborhood

	for (int p = 500; p < 505; p++) {
		int x = iftGetXCoord(gimg, p); // get the x index from the pixel p
		int y = iftGetYCoord(gimg, p); // get the x index from the pixel p

		printf("Pixel [%d][%d]\n", y, x);
		for (int i = 0; i < A->n; i++) {
			int qx = x + A->dx[i]; // coordinate x from the adjacent pixel q from p
			int qy = y + A->dy[i]; // coordinate y from the adjacent pixel q from p
			printf("- [%d][%d]\n", qy, qx);
		}
		puts("");
	}

	iftWriteImageP5(gimg, out);


	iftDestroyImage(&gimg);

	return 0;
}