
#ifndef _VIEW_H_
#define _VIEW_H_

#include "oldift.h"
#include "shared.h"

Image  *WideHighlight(Image *img, Image *label, float radius, int value, bool fill);
CImage *CWideHighlight(CImage *cimg, Image *label, float radius, int color, bool fill);

CImage *ColourLabels(CImage *cimg, Image *label, float radius, bool fill);
CImage *CWideHighlightLabels(CImage *cimg, Image *label, float radius, int *colormap, bool fill);

bool    HStripedTexture(int x, int y, int w, int h);
bool    VStripedTexture(int x, int y, int w, int h);
bool    BackslashTexture(int x, int y, int w, int h);
bool    SlashTexture(int x, int y, int w, int h);
bool    GridTexture(int x, int y, int w, int h);
bool    RGridTexture(int x, int y, int w, int h);

Image  *TextureHighlight(Image *img, Image *label, float radius, int value, bool fill,
			 bool (*texture)(int,int,int,int), int w, int h);
CImage *CTextureHighlight(CImage *cimg, Image *label, float radius, int color, bool fill,
			  bool (*texture)(int,int,int,int), int w, int h);

#endif

