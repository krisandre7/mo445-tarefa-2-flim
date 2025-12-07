
#ifndef _GUIBASIC_H_
#define _GUIBASIC_H_

#include "wxgui.h"
extern "C" {
#include "oldift.h"
}

void SetColor(wxColour *wxcolor, int color);


/* Prints "label" at point (x,y), using the current label font size. 
   The parameter "xalign" (resp. "yalign)" specifies which point of the string's 
   bounding box will end up at (x,y): 0.0 means the left (resp. bottom) side,
   1.0 means the right (resp. top) side.  Default is (0.5, 0.5), meaning 
   the box will be centered at (x,y). */
void DrawAsciiString(wxMemoryDC *ibuf,
		     char *str,
		     int x, int y,
		     int color, int border_color,
		     double xalign, double yalign,
		     double angle);
void DrawWxString(wxMemoryDC *ibuf,
		  wxString *wxstr,
		  int x, int y,
		  int color, int border_color,
		  double xalign, double yalign,
		  double angle);

#include "gui.h"

#endif
