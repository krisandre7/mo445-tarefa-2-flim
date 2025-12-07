//
// Created by Cesar Castelo on Nov 24, 2017.
//

#include <ift.h>

int main(int argc, char** argv)  {
    if(argc != 5) {
        iftError("Usage: <input_image> <bins_per_band> <quantization_method [0-traditional, 1-OPF clustering (for 1D data only)]> <output_file>", "iftBICClusteringQuantization.c");
        return 1;
    }

    iftImage *img=NULL, *quant_img=NULL, *gray_img=NULL;
    int bins_per_band = atoi(argv[2]);
    int quantization_method = atoi(argv[3]);
    //iftFeatures *feat = NULL;

    img = iftReadImageByExt(argv[1]);

    timer *t1 = iftTic();
    switch (quantization_method) {
      case 0:
        quant_img = iftQuantize(img, bins_per_band, 255);
        //feat = iftExtractBIC(img, bins_per_band);
        break;
      case 1:
        quant_img = iftQuantizeByClustering1D(img, bins_per_band);
        //feat = iftExtractBICClustering1D(img, bins_per_band);
        break;
    }
    timer *t2 = iftToc();
    fprintf(stdout, "Processing time: %s\n", iftFormattedTime(iftCompTime(t1, t2)));

    iftWriteImageByExt(iftNormalize(quant_img, 0, 255), argv[4]);

    iftDestroyImage(&img);
    iftDestroyImage(&gray_img);
    iftDestroyImage(&quant_img);

    return 0;
}
