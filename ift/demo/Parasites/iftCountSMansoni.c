#include <ift.h>

iftImage* removeSmallComponents(iftImage* binimg, int minArea, int maxArea) {
	iftAdjRel* A = iftCircular(2.0);
	iftImage* regions = iftRelabelRegions(binimg, A);

	int val = iftMaximumValue(regions);
	iftIntArray* count = iftCreateIntArray(val+1);

	for (int i = 0; i < regions->n; ++i) {
		count->val[regions->val[i]]++;
	}

//	iftPrintIntArray(count->val, count->n);

	for (int i = 0; i < regions->n; ++i) {
		if(regions->val[i])
			regions->val[i] = (count->val[regions->val[i]]>=minArea && count->val[regions->val[i]]<=maxArea)*255;
	}

	iftDestroyIntArray(&count);
	iftDestroyAdjRel(&A);

	return regions;
}

int main(int argc, const char** argv) {

//	printf("%f\n", iftFPDF(1, 2, 3));

//	printf("%f\n", iftPearson4Constant(100, 20, 100));

	iftImage* img = iftReadImageByExt("/Users/peixinho/Downloads/parasite.png");

	iftWriteImageByExt(img, "/Users/peixinho/Downloads/out.ppm");

	float ll[] = { 1.5,        -3.35023485,  4.57307558,  5.1074573};

	float k = iftPearson4Constant(ll[0], ll[1], ll[2]);

	float x = 0.0044067886435;
	float fx = (float)iftPearson4PDF(x, ll[0], ll[1], ll[2], ll[3], k);

    const char* basepath = argv[1];
    const char* origpath = iftConcatStrings(2, basepath, "orig");
    const char* autopath = iftConcatStrings(2, basepath, "auto");


//    iftColor rgb;
//
//    rgb.val[0] = 0;
//    rgb.val[1] = 0;
//    rgb.val[2] = 255;
//
//    iftColor ycbcr = iftRGBtoYCbCr(rgb, 255);
//
//    iftFColor c = iftYCbCrtoLab(ycbcr, 255.0);


    iftFileSet* files = iftLoadFileSetFromDir(origpath, 0);

	int threshold = 0;

	iftAdjRel* A = iftCircular(3.0);

	#pragma omp parallel for
	for (int i = 0; i < files->n; ++i) {

		printf("%s\n", files->files[i]->path);
		iftImage* img = iftReadImageByExt(files->files[i]->path);
        iftImage* nimg = iftNormalize(img, 0, 255);
        iftDestroyImage(&img);
        img = nimg;

        iftFImage* pimg = iftCreateFImage(img->xsize, img->ysize, img->zsize);

        for (int p = 0; p < img->n; ++p) {
            iftColor ycbcr;

            int* val = ycbcr.val;
            val[0] = img->val[p];
            val[1] = img->Cb[p];
            val[2] = img->Cr[p];

            iftFColor lab = iftYCbCrtoLab(ycbcr, 255);

            float x = lab.val[0];
            float y = img->val[p];

//            printf("%f %f\n", (float)img->val[p], x);
            pimg->val[p] = iftPearson4PDF(x, ll[0], ll[1], ll[2], ll[3], k);
        }



////		printf("(%d,%d)\n", img->xsize, img->ysize);
//
//		iftImage* gray = iftImageGray(img);
//		iftDestroyImage(&img);
//
//		iftImage* quant = iftNormalize(gray, 0, 255);
//		iftDestroyImage(&gray);
//
//		if(threshold==0) {
//			threshold = (1.0/IFT_PI)*iftOtsu(quant);
//		}
//
//		iftImage* bin = iftThreshold(quant, 0, threshold, 255);
//		iftDestroyImage(&quant);
//
//		char* outfile = iftReplaceString(files->files[i]->path, origpath, autopath);
//		iftWriteImageByExt(bin, outfile);
//		iftFree(outfile);
//
//		iftImage* components = removeSmallComponents(bin, 30000, 100000);
//		iftDestroyImage(&bin);
//
//		iftImage* final = iftOpen(components, A);
//		iftDestroyImage(&components);
//
//
//		iftDestroyImage(&gray);
//		iftDestroyImage(&quant);
//		iftDestroyImage(&bin);
//
//		outfile = iftReplaceString(files->files[i]->path, origpath, autopath);
//        printf(" -> %s\n", outfile);
//        iftWriteImageByExt(final, outfile);
//		iftFree(outfile);
//		iftDestroyImage(&final);

        char* outfile = iftReplaceString(files->files[i]->path, origpath, autopath);

        iftImage* outimage = iftFImageToImage(pimg, 255.0);
        iftDestroyFImage(&pimg);
        printf("%lf\n", (double)iftSumIntArray(outimage->val, outimage->n));
        iftDestroyImage(&img);

        iftWriteImageByExt(outimage, "temp.pgm");
        iftWriteImageByExt(outimage, outfile);
        iftDestroyImage(&outimage);
    }

    iftFree(origpath);
    iftFree(autopath);

	return 0;
}