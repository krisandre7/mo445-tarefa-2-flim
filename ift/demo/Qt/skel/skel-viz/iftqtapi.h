
#ifndef IMAGO_IFTQTAPI_H
#define IMAGO_IFTQTAPI_H

#include <QImage>
extern "C" {
#include "ift.h"
}

QImage iftImageToQImage(const iftImage *input);

iftImage *QImageToiftImage(const QImage &input);

QImage iftFImageToQImage(const iftFImage *input);

void drawBorders(QImage &image, const iftImage *bin_img);

#endif //IMAGO_IFTQTAPI_H
