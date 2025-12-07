#include "ift.h"




int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("iftErodeLabelImage <label_image> <radius> <out_label_img>\n\n" \
                 "Erode a Label Image (multiple objects) by applying the erosion operation on each object independently", "main");

    iftImage *label_image = iftReadImageByExt(argv[1]);
    float radius = atof(argv[2]);

    puts("- Eroding Label Image");
    iftImage *out = iftErodeLabelImage(label_image, radius);

    iftWriteImageByExt(out, argv[3]);

    iftDestroyImage(&label_image);
    iftDestroyImage(&out);

    return 0;
}