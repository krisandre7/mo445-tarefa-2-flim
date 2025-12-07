
#include "iftqtapi.h"


QColor heatmap(double value) {
    QColor c;
    c.setHslF((1.0 - value) * 0.66666666, 1.0, 0.5);
    return c;
}


QImage iftImageToQImage(const iftImage *input)
{
    QImage output(input->xsize, input->ysize, QImage::Format_Grayscale8);

    for (int i = 0; i < output.height(); i++) {
        for (int j = 0; j < output.width(); j++) {
            iftVoxel v; v.x = j; v.y = i; v.z = 0;
            int gray = input->val[iftGetVoxelIndex(input, v)];
            output.setPixel(j, i, QColor(gray, gray, gray).rgb());
        }
    }
    return output;
}


iftImage *QImageToiftImage(const QImage &input)
{
    if (input.format() != QImage::Format_Grayscale8)
        throw std::invalid_argument("Input must be an indexed 8bits image");

    iftImage *output = iftCreateImage(input.width(), input.height(), 1);
    for (int i = 0; i < input.height(); i++) {
        for (int j = 0; j < input.width(); j++) {
            iftVoxel v; v.x = j; v.y = i; v.z = 0;
            output->val[iftGetVoxelIndex(output, v)] = qGray(input.pixel(j, i));
        }
    }

    return output;
}


QImage iftFImageToQImage(const iftFImage *input)
{
    QImage output(input->xsize, input->ysize, QImage::Format::Format_RGB32);
    for (int i = 0; i < output.height(); i++) {
        for (int j = 0; j < output.width(); j++) {
            iftVoxel v; v.x = j; v.y = i; v.z = 0;
            int p = iftGetVoxelIndex(input, v);
            if (input->val[p] <= 1.0f)
                output.setPixelColor(j, i, Qt::white);
            else if (input->val[p] > std::numeric_limits<float>::epsilon())
                output.setPixelColor(j, i, heatmap(static_cast<double>(input->val[p]) / 100.0));

        }
    }
    return output;
}


void drawBorders(QImage &image, const iftImage *bin_img)
{
    if (image.width() != bin_img->xsize || image.height() != bin_img->ysize)
        throw std::invalid_argument("image and bin_img must have the same dimensions");

    iftAdjRel *A = iftCircular(1.0f);
    for (int i = 1; i < bin_img->ysize - 1; i++) {
        for (int j = 1; j < bin_img->xsize - 1; j++) {
            iftVoxel v; v.x = j; v.y = i; v.z = 0;
            int p = iftGetVoxelIndex(bin_img, v);
            for (int a = 1; a < A->n; a++) {
                iftVoxel u = iftGetAdjacentVoxel(A, v, a);
                int q = iftGetVoxelIndex(bin_img, u);
                if (bin_img->val[p] != bin_img->val[q]) {
                    image.setPixel(j, i, Qt::black);
                }
            }
        }
    }
    iftDestroyAdjRel(&A);
}
