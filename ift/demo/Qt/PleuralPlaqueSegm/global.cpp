#include "global.h"

//Transforms iftImage into QImage
QImage iftImage2QImage(iftImage *img)
{

    QImage dst(img->xsize,img->ysize,QImage::Format_Indexed8);

    for (int i=0;i<img->n;i++){
        iftVoxel u;
        u = iftGetVoxelCoord(img,i);
        if (iftValidVoxel(img,u)){
            int q = iftGetVoxelIndex(img,u);
            dst.setPixel(u.x,u.y,img->val[q]);
        }
    }

    return dst;
}

//Transforms QImage into iftImage
iftImage* iftQImage2Image(QImage img)
{
    iftImage *dst=NULL;
    uchar *u;

    dst=iftCreateImage(img.height(),img.width(),1);

    for (int i=0; i<img.height();i++){
        u = img.scanLine(i);
        for (int j=0;j<img.width();j++,u++){
            if (img.valid(i,j))
            {
                iftVoxel v;
                v = iftGetVoxelCoord(dst,i*img.width()+j);
                if (iftValidVoxel(dst,v)){
                    int q = iftGetVoxelIndex(dst,v);
                    dst->val[q] = *u;
                }
            }
        }
    }

    return dst;
}

//Return min or max if value outrange them
int minmax(int value, int min, int max)
{
   if (value < min)
      return min;
   if (value > max)
      return max;
   return value;
}

//Calculate the constrast of a QImage
void qContrast(QImage &img, double factor)
{
    int x,y;

    for (y = 0; y<img.height();y++)
    {
        for (x = 0; x<img.width();x++)
        {
            QRgb color = img.pixel(QPoint(x,y));
            QColor colorTable(color);
            colorTable.setRed(minmax((int)colorTable.red()*factor,0,255));
            colorTable.setGreen(minmax((int)colorTable.green()*factor,0,255));
            colorTable.setBlue(minmax((int)colorTable.blue()*factor,0,255));
            img.setPixel(x,y,colorTable.rgb());
         }
     }
}

//Calculate the brightness of a QImage
void qBrightness(QImage &img, int shift)
{
    int x,y;

    for (y = 0; y<img.height();y++)
    {
        for (x = 0; x<img.width();x++)
        {
            QRgb color = img.pixel(QPoint(x,y));
            QColor colorTable(color);
            colorTable.setRed(minmax(colorTable.red()+shift,0,255));
            colorTable.setGreen(minmax(colorTable.green()+shift,0,255));
            colorTable.setBlue(minmax(colorTable.blue()+shift,0,255));
            img.setPixel(x,y,colorTable.rgb());
        }
    }
}

uint qMinimumValue(QImage &img)
{
    return 0;
}
