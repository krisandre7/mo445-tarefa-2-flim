
#ifndef CIMG_H
#define CIMG_H 1

typedef unsigned char component;

typedef struct _ColorImage {
  int W,H,rowlen;
  component *val;
} CImg;

CImg       *CImgNew(int width, int height);
void        CImgDestroy(CImg *x);
void        CImgFill(CImg *x, int color);
int         CImgCompatible(CImg *a, CImg *b);

void        CImgSet(CImg *z, int x,int y, int color);
int         CImgGet(CImg *z, int x,int y);
void        CImgFillRect(CImg *z, int x, int y, int w, int h, int color);
void        CImgDrawLine(CImg *z, int x1, int y1, int x2, int y2, int color);

void        CImgFullCopy(CImg *dest, CImg *src);

CImg       *CImgIntegerZoom(CImg *src, int zoom);
CImg       *CImgHalfScale(CImg *src);

int         CImgReadP6(CImg *src, char *path);
int         CImgWriteP6(CImg *src, char *path);

/* 180 deg rotation, in place */
void        CImgRot180(CImg *src);

void RgbSet(void *dest,int rgb, int count);
void RgbRect(void *dest,int x,int y,int w,int h,int rgb,int rowoffset);

int box_intersection(int x1,int y1,int w1,int h1,
		     int x2,int y2,int w2,int h2);
int inbox(int x,int y,int x0,int y0,int w,int h);

float FloatNormalize(float value,float omin,float omax,
		     float nmin, float nmax);
int   IntNormalize(int value,int omin,int omax,int nmin,int nmax);



#endif
