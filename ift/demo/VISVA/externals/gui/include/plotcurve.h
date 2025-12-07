
#ifndef _PLOTCURVE_H_
#define _PLOTCURVE_H_

#include "wxgui.h"
extern "C" {
#include "oldift.h"
}

typedef struct _plotRange {
  double begin;
  double end;
} PlotRange;


typedef struct _plotLandmark {
  double X;
  int color;
  char name[512];
} PlotLandmark;


CImage *PlotCurve(Curve *C, int w, int h,
		  PlotRange *rangex, PlotRange *rangey,
		  char *title, char *xlabel, char *ylabel,
		  int plotColor, PlotLandmark marks[], int nmarks);

#include "gui.h"

#endif


