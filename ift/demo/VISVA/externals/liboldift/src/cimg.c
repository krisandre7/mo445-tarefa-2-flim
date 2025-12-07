#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cimg.h"
#include "color.h"

CImg *CImgNew(int width, int height) {
  CImg *x;
  x=(CImg *)malloc(sizeof(CImg));
  x->W = width;
  x->H = height;
  x->rowlen = 3 * width;
  x->val=(component *)malloc(3*width*height*sizeof(component));
  memset(x->val,0,x->H * x->rowlen);
  return x;
}

void  CImgDestroy(CImg *x) {
  if (x) {
    if (x->val)
      free(x->val);
    free(x);
  }
}

void CImgFill(CImg *x, int color) {

  RgbSet(x->val, color, x->W * x->H);

}

void CImgSet(CImg *z, int x,int y, int color) {
  component r,g,b, *p;

  r=t0(color);
  g=t1(color);
  b=t2(color);
  
  p = z->val + (z->rowlen * y + 3 * x);
  *(p++) = r;
  *(p++) = g;
  *p = b;
}

int CImgGet(CImg *z, int x,int y) {
  component r,g,b, *p;

  p = z->val + (z->rowlen * y + 3 * x);
  r = *(p++);
  g = *(p++);
  b = *p;

  return(triplet(r,g,b));
}

void CImgFullCopy(CImg *dest, CImg *src) {
  if (dest->W != src->W || dest->H != src->H)
    return;
  memcpy(dest->val,src->val,src->rowlen * src->H);
}


int CImgCompatible(CImg *a, CImg *b) {
  if (a==0 || b==0) return 0;
  return (a->W == b->W && a->H == b->H);
}

void CImgFillRect(CImg *z, int x, int y, int w, int h, int color) {
  if (x >= z->W) return;
  if (y >= z->H) return;

  if (x<0) { w+=x; x=0; }
  if (y<0) { h+=y; y=0; }
  if (x+w >= z->W) w=z->W-x-1;
  if (y+h >= z->H) h=z->H-y-1;
  if (w<0 || h<0) return;

  RgbRect(z->val, x, y, w, h, color, z->rowlen);
}

// http://trident.mcs.kent.edu/~rduckwor/SV/ASSGN/HW2/functions.txt
// (a versao do Foley nao trata todos os casos)
void CImgDrawLine(CImg *z, int x1, int y1, int x2, int y2, int color) 
{
  component triplet[3], *p=0;
  int x, y;
  int dy = y2 - y1;
  int dx = x2 - x1;
  int G, DeltaG1, DeltaG2;	
  int swap;
  int inc = 1;

  triplet[0] = t0(color);
  triplet[1] = t1(color);
  triplet[2] = t2(color);

#define SetPixel(a,b,c) { p=a->val+a->rowlen*c+3*b; *(p++)=triplet[0]; *(p++)=triplet[1]; *p=triplet[2]; }

  if (x1>=0 && y1>=0 && x1<z->W && y1<z->H)
    SetPixel(z, x1, y1);
  
  if (abs(dy) < abs(dx)) {
    /* -1 < slope < 1 */
    if (dx < 0) {
      dx = -dx;
      dy = -dy;
      
      swap = y2;
      y2 = y1;
      y1 = swap;
      
      swap = x2;
      x2 = x1;
      x1 = swap;
    }
      
    if (dy < 0) {
      dy = -dy;
      inc = -1;
    }
      
    y = y1;
    x = x1 + 1;
      
    G = 2 * dy - dx;
    DeltaG1 = 2 * (dy - dx);
    DeltaG2 = 2 * dy;
    
    while (x <= x2) {
      if (G > 0) {
	G += DeltaG1;
	y += inc;
      } else
	G += DeltaG2;

      if (x>=0 && y>=0 && x<z->W && y<z->H)
	SetPixel(z, x, y);
      x++;
    }
  } else {
    /* slope < -1 or slope > 1 */
    if (dy < 0) {
      dx = -dx;
      dy = -dy;
      
      swap = y2;
      y2 = y1;
      y1 = swap;
      
      swap = x2;
      x2 = x1;
      x1 = swap;
    }
      
    if (dx < 0) {
      dx = -dx;
      inc = -1;
    }
      
    x = x1;
    y = y1 + 1;
      
    G = 2 * dx - dy;
    DeltaG1 = 2 * (dx - dy);
    DeltaG2 = 2 * dx;
      
    while (y <= y2) {
      if (G > 0) {
	G += DeltaG1;
	x += inc;
      } else
	G += DeltaG2;
	  
      if (x>=0 && y>=0 && x<z->W && y<z->H)
	SetPixel(z, x, y);
      y++;
    }
  }
}
#undef SetPixel

int CImgReadP6(CImg *src, char *path) {
  FILE *f;
  int i,j;
  component *p;
  
  f = fopen(path,"rb");
  if (!f) return -1;

  fscanf(f,"P6\n%d %d\n255\n",&src->W,&src->H);

  j = 3 * src->W * src->H;
  p = src->val;
  for(i=0;i<j;i++,p++)
    *p = fgetc(f);

  fclose(f);
  return 0;
}

int CImgWriteP6(CImg *src, char *path) {
  FILE *f;
  int i,j;
  component *p;
  
  f = fopen(path,"w");
  if (!f) return -1;

  fprintf(f,"P6\n%d %d\n255\n",src->W,src->H);

  j = 3 * src->W * src->H;
  p = src->val;
  for(i=0;i<j;i++)
    fputc(*(p++),f);

  fclose(f);
  return 0;
}

CImg  *CImgIntegerZoom(CImg *src, int zoom) {
  CImg *dest;
  int i,j;

  dest = CImgNew(src->W * zoom, src->H * zoom);
  if (zoom == 1) {
    CImgFullCopy(dest,src);
    return dest;
  }

  for(i=0;i<src->H;i++)
    for(j=0;j<src->W;j++)
      RgbRect(dest->val, j*zoom, i*zoom, zoom, zoom, 
	      CImgGet(src,j,i), dest->rowlen);

  return dest;
}

CImg  *CImgHalfScale(CImg *src) {
  CImg *dest;
  int i,j,k,w,h;
  int r[4],g[4],b[4];

  w = src->W / 2;
  h = src->H / 2;

  if (w<1) w=1;
  if (h<1) h=1;

  dest = CImgNew(w,h);

  for(j=0;j<src->H-1;j+=2) 
    for(i=0;i<src->W-1;i+=2) {

      r[0] = CImgGet(src,i,  j);
      r[1] = CImgGet(src,i+1,j);
      r[2] = CImgGet(src,i,  j+1);
      r[3] = CImgGet(src,i+1,j+1);

      for(k=0;k<4;k++) {
	g[k] = t1(r[k]);
	b[k] = t2(r[k]);
	r[k] = t0(r[k]);	
      }
      for(k=1;k<4;k++) {
	r[0] += r[k];
	g[0] += g[k];
	b[0] += b[k];
      }
      CImgSet(dest,i/2,j/2,triplet( r[0]/4, g[0]/4, b[0]/4 ));
    }

  return dest;
}

void CImgRot180(CImg *src) {
  component *f, *b;
  component x,y,z;
  int i;

  f = src->val;
  b = src->val + (src->rowlen * (src->H-1) + 3 * (src->W-1));

  for(i = (src->W * src->H) / 2;i;i--) {
    x = *b;
    y = *(b+1);
    z = *(b+2);

    *b = *f;
    *(b+1) = *(f+1);
    *(b+2) = *(f+2);

    *f     = x;
    *(f+1) = y;
    *(f+2) = z;

    f+=3;
    b-=3;
  }
}

void RgbSet(void *dest,int rgb, int count) {
  int i,r,g,b;
  unsigned char *p;
  r=(rgb>>16);
  g=(rgb>>8)&0xff;
  b=rgb&0xff;
  p=(unsigned char *)dest;
  for(i=count;i;i--) {
    *(p++) = r;
    *(p++) = g;
    *(p++) = b;
  }
}

void RgbRect(void *dest,int x,int y,int w,int h,int rgb,int rowoffset) {
  unsigned char r,g,b, *p, *q;
  int i;

  r=(rgb >> 16) & 0xff;
  g=(rgb >> 8) & 0xff;
  b=rgb & 0xff;
  
  p = ((unsigned char *)dest) + y * rowoffset + 3 * x;
  for(;h;--h) {
    q = p;
    for(i=w;i;--i) {
      *(p++) = r;
      *(p++) = g;
      *(p++) = b;
    }
    p = q + rowoffset;
  }  
}

int box_intersection(int x1,int y1,int w1,int h1,
		     int x2,int y2,int w2,int h2)
{
  w1 += x1 - 1;
  w2 += x2 - 1;
  h1 += y1 - 1;
  h2 += y2 - 1;
  if (! ((x1 >= x2 && x1 <= w2) || (w1 >= x2 && w1 <= w2)) )
    return 0;
  if (! ((y1 >= y2 && y1 <= h2) || (h1 >= y2 && h1 <= h2)) )
    return 0;
  return 1;
}

float FloatNormalize(float value,float omin,float omax,
		     float nmin, float nmax)
{
  float tmp,i;
  if ( (omax - omin) < 0.00001) return 0.0;
  tmp = (value - omin) / (omax - omin);
  i = nmin + tmp * (nmax - nmin);
  return i;
}

int   IntNormalize(int value,int omin,int omax,int nmin,int nmax)
{
  float tmp;
  int   i;
  if ( (omax - omin) == 0) return 0;
  tmp = ((float)(value - omin)) / ((float)(omax - omin));
  i = nmin + (int)(tmp * ((float)(nmax - nmin)));
  return i;
}

int inbox(int x,int y,int x0,int y0,int w,int h) {
  return( (x>=x0)&&(y>=y0)&&(x<x0+w)&&(y<y0+h) );
}
