
#include "guibasic.h"


void SetColor(wxColour *wxcolor, int color){
#if wxMINOR_VERSION>=8
  wxcolor->Set(t0(color),
	       t1(color),
	       t2(color),
	       wxALPHA_OPAQUE);
#else
  wxcolor->Set(t0(color),
	       t1(color),
	       t2(color));
#endif
}


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
		     double angle){
  wxString wxstr(str, wxConvUTF8);
  DrawWxString(ibuf, &wxstr, x, y, color, border_color, 
	       xalign, yalign, angle);
}


void DrawWxString(wxMemoryDC *ibuf,
		  wxString *wxstr,
		  int x, int y,
		  int color, int border_color,
		  double xalign, double yalign,
		  double angle){
  int dhx,dhy;
  wxColour wxcolor;
  wxSize size;
  int w,h,dw,dh;
  double rad = angle*PI/180.0;

  yalign = 1.0 - yalign;
  ibuf->SetTextBackground(*wxWHITE);
#if wxMINOR_VERSION>=8
  size = ibuf->GetTextExtent(*wxstr);
#else
  wxCoord wc,hc;
  ibuf->GetTextExtent(*wxstr, &wc, &hc,
		      NULL, NULL, NULL);
  size.Set(wc, hc);
#endif

  w = size.GetWidth();
  h = size.GetHeight();
  if(angle != 0.0){
    dw =  ROUND(-sin(rad)*h*yalign - cos(rad)*w*xalign );
    dh = -ROUND( cos(rad)*h*yalign - sin(rad)*w*xalign );
  }
  else{
    dw = -ROUND(w*xalign);
    dh = -ROUND(h*yalign);
  }

  if(border_color>=0){
    SetColor(&wxcolor, border_color);
    ibuf->SetTextForeground(wxcolor);
    for(dhy = -1; dhy <= 1; dhy++)
      for(dhx = -1; dhx <= 1; dhx++)
	if(dhx!=0 || dhy!=0)
	  ibuf->DrawRotatedText(*wxstr, 
				ROUND(x + dw + dhx),
				ROUND(y + dh + dhy),
				angle);
  }

  if(color>=0){
    SetColor(&wxcolor, color);
    ibuf->SetTextForeground(wxcolor);
  }
  ibuf->DrawRotatedText(*wxstr, 
			ROUND(x + dw),
			ROUND(y + dh),
			angle);
}


