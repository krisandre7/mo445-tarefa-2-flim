
#ifndef _CURSOR_H_
#define _CURSOR_H_

#include "gui.h"


wxCursor *Image2Cursor(Image *img, Image *mask,
		       int hotSpotX, int hotSpotY, 
		       int zoom, int color1, int color0);

wxCursor *CrossCursor(int zoom);


#endif


