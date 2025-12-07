
#include "plotcurve.h"



CImage *PlotCurve(Curve *C, int w, int h,
		  PlotRange *rangex, PlotRange *rangey,
		  char *title, char *xlabel, char *ylabel,
		  int plotColor, PlotLandmark *marks, int nmarks){
  double x_min,x_max;
  double y_min,y_max;
  CImage *cimg;
  int margin_left,margin_right;
  int margin_top,margin_bottom;
  int dx,dy,sizex,sizey;
  int i,j,p,q,k,n,m;
  unsigned char *data;
  char str[100];
  wxColour wxcolor;
  wxMemoryDC ibuf;

  margin_left   = 70;
  margin_right  = 40;
  margin_top    = 45;
  margin_bottom = 40;
  sizex = w-(margin_left+margin_right);
  sizey = h-(margin_top+margin_bottom);

  InvertXY(C);
  SortCurve(C, 0, C->n-1, INCREASING);
  InvertXY(C);

  if(rangex!=NULL){
    x_min = rangex->begin;
    x_max = rangex->end;
  }
  else{
    x_min = DBL_MAX;
    x_max = -DBL_MAX;
    for(k=0; k<C->n; k++){
      if(C->X[k]>x_max && C->X[k]!=DBL_MAX)
	x_max = C->X[k];
      if(C->X[k]<x_min && C->X[k]!=-DBL_MAX)
	x_min = C->X[k];
    }
  }

  if(rangey!=NULL){
    y_min = rangey->begin;
    y_max = rangey->end;
  }
  else{
    y_min = DBL_MAX;
    y_max = -DBL_MAX;
    for(k=0; k<C->n; k++){    
      if(C->X[k]<x_min || C->X[k]>x_max)
	continue;

      if(C->Y[k]>y_max && C->Y[k]!=DBL_MAX)
	y_max = C->Y[k];
      if(C->Y[k]<y_min && C->Y[k]!=-DBL_MAX)
	y_min = C->Y[k];
    }
  }

  cimg  = CreateCImage(w, h);

  /*
  p = margin_left + margin_top*w;
  q = margin_left + (h-margin_bottom)*w;
  DrawCImageLine(cdraw, alpha, p, q, 
		 0.0, 0xffffff, 255, false);
  p = margin_left + (h-margin_bottom)*w;
  q = (w-margin_right) + (h-margin_bottom)*w;
  DrawCImageLine(cdraw, alpha, p, q, 
		 0.0, 0xffffff, 255, false);
  */

  for(m=0; m<nmarks; m++){
    if(marks[m].X<x_min || marks[m].X>x_max)
      continue;

    dx = ROUND(sizex*(marks[m].X-x_min)/(x_max-x_min));
    j = margin_left + dx;
    DrawCImageLineDDA(cimg, j, margin_top, 
		      j, h-margin_bottom, 
		      marks[m].color);
  }

  q = NIL;
  for(k=0; k<C->n; k++){
    if(C->X[k]==-DBL_MAX || C->X[k]==DBL_MAX)
      continue;
    if(C->Y[k]==-DBL_MAX || C->Y[k]==DBL_MAX)
      continue;

    if(C->X[k]<x_min || C->X[k]>x_max)
      continue;
    if(C->Y[k]<y_min || C->Y[k]>y_max)
      continue;

    dx = ROUND(sizex*(C->X[k]-x_min)/(x_max-x_min));
    dy = ROUND(sizey*(C->Y[k]-y_min)/(y_max-y_min));
    j = margin_left + dx;
    i = margin_top + (sizey-dy);
    p = j + i*w;

    if(q!=NIL)
      DrawCImageLineDDA(cimg, j, i, 
			q%w, q/w, plotColor);
    q = p;
  }

  wxImage wximg(w, h, false);
  data = wximg.GetData();

  n = w*h;
  for(p=0,q=0;p<n;p++,q+=3){
    data[q]   = cimg->C[0]->val[p];
    data[q+1] = cimg->C[1]->val[p];
    data[q+2] = cimg->C[2]->val[p];
  }
  wxBitmap wxbit(wximg, -1);
  ibuf.SelectObject(wxbit);

  wxFont font(10, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, 
	      wxFONTWEIGHT_NORMAL, false, _T(""), 
	      wxFONTENCODING_DEFAULT);

  wxFont font2(12, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, 
	       wxFONTWEIGHT_NORMAL, false, _T(""), 
	       wxFONTENCODING_DEFAULT);
  wxFont font3(15, wxFONTFAMILY_SWISS, wxFONTSTYLE_NORMAL, 
	       wxFONTWEIGHT_NORMAL, false, _T(""), 
	       wxFONTENCODING_DEFAULT);

  SetColor(&wxcolor, 0xFFFFFF);
  ibuf.SetTextBackground(*wxBLACK);
  ibuf.SetTextForeground(wxcolor);
  ibuf.SetFont(font);

  sprintf(str,"%.1f",x_min);
  DrawAsciiString(&ibuf, str, 
		  margin_left, 
		  h-margin_bottom+10,
		  -1, -1, 0.5, 0.5, 0.0);
  sprintf(str,"%.1f",x_max);
  DrawAsciiString(&ibuf, str, 
		  w-margin_right, 
		  h-margin_bottom+10,
		  -1, -1, 0.5, 0.5, 0.0);
  sprintf(str,"%.1f",(x_min+x_max)/2.0);
  DrawAsciiString(&ibuf, str, 
		  margin_left+(sizex/2),
		  h-margin_bottom+10,
		  -1, -1, 0.5, 0.5, 0.0);

  sprintf(str,"%.1g",y_min);
  DrawAsciiString(&ibuf, str, 
		  margin_left-2,
		  h-margin_bottom,
		  -1, -1, 1.0, 0.5, 0.0);
  sprintf(str,"%.1g",y_max);
  DrawAsciiString(&ibuf, str, 
		  margin_left-2,
		  margin_top,
		  -1, -1, 1.0, 0.5, 0.0);
  sprintf(str,"%.1g",(y_min+y_max)/2.0);
  DrawAsciiString(&ibuf, str, 
		  margin_left-2,
		  margin_top+(sizey/2),
		  -1, -1, 1.0, 0.5, 0.0);

  ibuf.SetFont(font2);
  DrawAsciiString(&ibuf, xlabel,
		  w/2,
		  h-margin_bottom/2,
		  -1, -1, 0.5, 0.7, 0.0);
  DrawAsciiString(&ibuf, ylabel,
		  3, (h-margin_bottom)/2,
		  -1, -1, 0.5, 1.0, 90.0);

  SetColor(&wxcolor, plotColor);
  ibuf.SetTextBackground(*wxBLACK);
  ibuf.SetTextForeground(wxcolor);
  ibuf.SetFont(font3);
  DrawAsciiString(&ibuf, title,
		  w/2, 
		  margin_top/2,
		  -1, -1, 0.5, 0.2, 0.0);
  /*
  if(showmin){
    SetColor(&wxcolor, objColor);
    ibuf.SetTextBackground(*wxBLACK);
    ibuf.SetTextForeground(wxcolor);
    ibuf.SetFont(font2);
    k = index_min;
    dx = ROUND(sizex*(C->X[k]-x_min)/(x_max-x_min));
    j = margin_left + dx;
    i = margin_top-12;
    DrawAsciiString(&ibuf, (char *)"minimum", j, i,
		    -1, -1, 0.5, 0.5, 0.0);
  }
  */

  for(m=0; m<nmarks; m++){
    if(marks[m].X<x_min || marks[m].X>x_max)
      continue;
    SetColor(&wxcolor, marks[m].color);
    ibuf.SetTextBackground(*wxBLACK);
    ibuf.SetTextForeground(wxcolor);
    ibuf.SetFont(font);
    dx = ROUND(sizex*(marks[m].X-x_min)/(x_max-x_min));
    j = margin_left + dx;
    i = margin_top-12;
    DrawAsciiString(&ibuf, (char *)marks[m].name, j, i,
		    -1, -1, 0.5, 0.5, 0.0);
  }

  wxImage wximg2 = wxbit.ConvertToImage();
  data = wximg2.GetData();

  for(p=0,q=0;p<n;p++,q+=3){
    cimg->C[0]->val[p] = data[q];
    cimg->C[1]->val[p] = data[q+1];
    cimg->C[2]->val[p] = data[q+2];
  }

  return cimg;
}


