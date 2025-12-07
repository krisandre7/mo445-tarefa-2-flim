
#include "view.h"


CImage *CWideHighlight(CImage *cimg, Image *label, float radius, int color, bool fill){
  CImage *hcimg;
  int p,q,i;
  AdjRel *A=NULL;
  Pixel u,v;

  hcimg = CopyCImage(cimg);
  A    = Circular(radius);
  for (u.y=0; u.y < hcimg->C[0]->nrows; u.y++){
    for (u.x=0; u.x < hcimg->C[0]->ncols; u.x++){
      p = u.x + hcimg->C[0]->tbrow[u.y];
      if(fill){
	if(label->val[p] > 0){
	  hcimg->C[0]->val[p] = ROUND(0.3*t0(color) + 0.7*hcimg->C[0]->val[p]);
	  hcimg->C[1]->val[p] = ROUND(0.3*t1(color) + 0.7*hcimg->C[1]->val[p]);
	  hcimg->C[2]->val[p] = ROUND(0.3*t2(color) + 0.7*hcimg->C[2]->val[p]);
	}
      }

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(hcimg->C[0],v.x,v.y)){
	  q = v.x + hcimg->C[0]->tbrow[v.y];
	  if (label->val[p] > label->val[q]){
	    hcimg->C[0]->val[p] = t0(color);
	    hcimg->C[1]->val[p] = t1(color);
	    hcimg->C[2]->val[p] = t2(color);
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(hcimg);
}


Image *WideHighlight(Image *img, Image *label, float radius, int value, bool fill){
  Image *himg;
  int p,q,i;
  AdjRel *A=NULL;
  Pixel u,v;

  himg = ift_CopyImage(img);
  A    = Circular(radius);
  for (u.y=0; u.y < himg->nrows; u.y++){
    for (u.x=0; u.x < himg->ncols; u.x++){
      p = u.x + himg->tbrow[u.y];
      if(fill){
	if(label->val[p] > 0)
	  himg->val[p] = ROUND(0.3*value + 0.7*himg->val[p]);
      }

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(himg,v.x,v.y)){
	  q = v.x + himg->tbrow[v.y];
	  if (label->val[p] > label->val[q]){
	    himg->val[p] = value;
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(himg);
}


CImage *ColourLabels(CImage *cimg, Image *label, float radius, bool fill){
  int labelcolor[256];
  int i,p,q,n,ncols,nrows,color;
  CImage *hcimg;
  AdjRel *A=NULL;
  Pixel u,v;

  hcimg = CopyCImage(cimg);
  A     = Circular(radius);

  ncols = label->ncols;
  nrows = label->nrows;
  n = ncols*nrows;
  labelcolor[0] = 0x000000;
  labelcolor[1] = 0xffff00;
  labelcolor[2] = 0x00ff00;
  labelcolor[3] = 0x00ffff;
  labelcolor[4] = 0xff80ff;
  labelcolor[5] = 0xff8000;
  labelcolor[6] = 0x008000;
  labelcolor[7] = 0x0000ff;
  labelcolor[8] = 0x800000;

  for(i=9; i<256; i++)
    labelcolor[i] = rand()%(0xffffff+1);

  for(u.y=0; u.y<nrows; u.y++){
    for(u.x=0; u.x<ncols; u.x++){
      p = u.x + label->tbrow[u.y];
      
      if(label->val[p] == 0)
	continue;
      else if(label->val[p]<256)
	color = labelcolor[ label->val[p] ];
      else
	color = 0x000000;

      if(fill){
	hcimg->C[0]->val[p] = ROUND(0.3*t0(color) + 0.7*hcimg->C[0]->val[p]);
	hcimg->C[1]->val[p] = ROUND(0.3*t1(color) + 0.7*hcimg->C[1]->val[p]);
	hcimg->C[2]->val[p] = ROUND(0.3*t2(color) + 0.7*hcimg->C[2]->val[p]);
      }
      
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if(ValidPixel(label,v.x,v.y)){
	  q = v.x + label->tbrow[v.y];
	  if(label->val[p] != label->val[q]){
	    hcimg->C[0]->val[p] = t0(color);
	    hcimg->C[1]->val[p] = t1(color);
	    hcimg->C[2]->val[p] = t2(color);
	    break;
	  }
	}
      }
    }
  }

  DestroyAdjRel(&A);

  return(hcimg);
}

CImage *CWideHighlightLabels(CImage *cimg, Image *label, float radius, int *colormap, bool fill){
  CImage *hcimg;
  int p,q,i,lb;
  AdjRel *A=NULL;
  Pixel u,v;

  hcimg = CopyCImage(cimg);
  A    = Circular(radius);
  for (u.y=0; u.y < hcimg->C[0]->nrows; u.y++){
    for (u.x=0; u.x < hcimg->C[0]->ncols; u.x++){
      p = u.x + hcimg->C[0]->tbrow[u.y];
      if(fill){
        lb = label->val[p];
	if(lb> 0){
	  hcimg->C[0]->val[p] = ROUND(0.3*t0(colormap[lb]) + 0.7*hcimg->C[0]->val[p]);
	  hcimg->C[1]->val[p] = ROUND(0.3*t1(colormap[lb]) + 0.7*hcimg->C[1]->val[p]);
	  hcimg->C[2]->val[p] = ROUND(0.3*t2(colormap[lb]) + 0.7*hcimg->C[2]->val[p]);
	}
      }

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(hcimg->C[0],v.x,v.y)){
	  q = v.x + hcimg->C[0]->tbrow[v.y];
          lb = label->val[p];
	  if (lb> label->val[q]){
	    hcimg->C[0]->val[p] = t0(colormap[lb]);
	    hcimg->C[1]->val[p] = t1(colormap[lb]);
	    hcimg->C[2]->val[p] = t2(colormap[lb]);
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(hcimg);
}


bool    HStripedTexture(int x, int y, int w, int h){
  x %= w;
  y %= h;
  if(y==h/2) return 1;
  else       return 0;
}

bool    VStripedTexture(int x, int y, int w, int h){
  x %= w;
  y %= h;
  if(x==w/2) return 1;
  else       return 0;
}

bool    BackslashTexture(int x, int y, int w, int h){
  x %= w;
  y %= h;
  if(x==y) return 1;
  else     return 0;
}

bool    SlashTexture(int x, int y, int w, int h){
  x %= w;
  y %= h;
  if(y==-x+h-1) return 1;
  else          return 0;
}

bool    GridTexture(int x, int y, int w, int h){
  x %= w;
  y %= h;
  return(VStripedTexture(x,y,w,h) ||
	 HStripedTexture(x,y,w,h));
}

bool    RGridTexture(int x, int y, int w, int h){
  x %= w;
  y %= h;
  return(BackslashTexture(x,y,w,h) ||
	 SlashTexture(x,y,w,h));
}


Image  *TextureHighlight(Image *img, Image *label, 
			 float radius, int value, bool fill,
			 bool (*texture)(int,int,int,int), int w, int h){
  Image *himg;
  int p,q,i;
  AdjRel *A=NULL;
  Pixel u,v;
  bool b;

  himg = ift_CopyImage(img);
  A    = Circular(radius);
  for (u.y=0; u.y < himg->nrows; u.y++){
    for (u.x=0; u.x < himg->ncols; u.x++){
      p = u.x + himg->tbrow[u.y];

      if(label->val[p] > 0){
	b = (*texture)(u.x,u.y,w,h);
	if(b)
	  himg->val[p] = value;
	else if(fill)
	  himg->val[p] = ROUND(0.3*value + 0.7*himg->val[p]);
      }

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(himg,v.x,v.y)){
	  q = v.x + himg->tbrow[v.y];
	  if (label->val[p] > label->val[q]){
	    himg->val[p] = value;
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(himg);
}



CImage *CTextureHighlight(CImage *cimg, Image *label, 
			  float radius, int color, bool fill,
			  bool (*texture)(int,int,int,int), int w, int h){
  CImage *hcimg;
  int p,q,i;
  AdjRel *A=NULL;
  Pixel u,v;
  bool b;

  hcimg = CopyCImage(cimg);
  A    = Circular(radius);
  for (u.y=0; u.y < hcimg->C[0]->nrows; u.y++){
    for (u.x=0; u.x < hcimg->C[0]->ncols; u.x++){
      p = u.x + hcimg->C[0]->tbrow[u.y];

      if(label->val[p] > 0){
	b = (*texture)(u.x,u.y,w,h);
	if(b){
	  hcimg->C[0]->val[p] = t0(color);
	  hcimg->C[1]->val[p] = t1(color);
	  hcimg->C[2]->val[p] = t2(color);
	}
	else if(fill){
	  hcimg->C[0]->val[p] = ROUND(0.3*t0(color) + 0.7*hcimg->C[0]->val[p]);
	  hcimg->C[1]->val[p] = ROUND(0.3*t1(color) + 0.7*hcimg->C[1]->val[p]);
	  hcimg->C[2]->val[p] = ROUND(0.3*t2(color) + 0.7*hcimg->C[2]->val[p]);
	}
      }

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(hcimg->C[0],v.x,v.y)){
	  q = v.x + hcimg->C[0]->tbrow[v.y];
	  if (label->val[p] > label->val[q]){
	    hcimg->C[0]->val[p] = t0(color);
	    hcimg->C[1]->val[p] = t1(color);
	    hcimg->C[2]->val[p] = t2(color);
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(hcimg);
}


