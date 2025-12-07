#include "cimage.h"
#include "msfiltering.h"
#include "geometric2.h"

CImage *CreateCImage(int ncols, int nrows)
{
  CImage *cimg=NULL;
  int i;

  cimg = (CImage *) calloc(1, sizeof(CImage));
  for (i=0; i < 3; i++)
    cimg->C[i] = CreateImage(ncols,nrows);
  return(cimg);
}

void    DestroyCImage(CImage **cimg)
{
  CImage *tmp;
  int i;

  tmp = *cimg;
  if (tmp != NULL) {
    for (i=0; i < 3; i++)
      DestroyImage(&(tmp->C[i]));
    free(tmp);
    *cimg = NULL;
  }
}

CImage *ReadCImage(char *filename)
{
  CImage *cimg=NULL;
  FILE *fp=NULL;
  char type[10];
  int  t,i,ncols,nrows,n;
  char z[256];

  fp = fopen(filename,"rb");
  if (fp == NULL){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }
  fscanf(fp,"%s\n",type);
  if((strcmp(type,"P6")==0)){
    NCFgets(z,255,fp);
    t = sscanf(z,"%d %d\n",&ncols,&nrows);
    if(t == EOF || t < 2){
      NCFgets(z,255,fp);
      sscanf(z,"%d %d\n",&ncols,&nrows);
    }
    n = ncols*nrows;
    NCFgets(z,255,fp);
    sscanf(z,"%d\n",&i);
    cimg = CreateCImage(ncols,nrows);
    for (i=0; i < n; i++){
      cimg->C[0]->val[i] = fgetc(fp);
      cimg->C[1]->val[i] = fgetc(fp);
      cimg->C[2]->val[i] = fgetc(fp);
    }
    fclose(fp);
  }else{
    fprintf(stderr,"Input image must be P6\n");
    exit(-1);
  }

  return(cimg);
}

void    WriteCImage(CImage *cimg, char *filename)
{
  FILE *fp;
  int i,n;

  fp = fopen(filename,"w");
  fprintf(fp,"P6\n");
  fprintf(fp,"%d %d\n",cimg->C[0]->ncols,cimg->C[0]->nrows);
  fprintf(fp,"255\n");
  n = cimg->C[0]->ncols*cimg->C[0]->nrows;
  for (i=0; i < n; i++) {
    fputc(cimg->C[0]->val[i],fp);
    fputc(cimg->C[1]->val[i],fp);
    fputc(cimg->C[2]->val[i],fp);
  }
  fclose(fp);
}

CImage *CopyCImage(CImage *cimg)
{
  CImage *imgc;
  int i;

  imgc = (CImage *) calloc(1,sizeof(CImage));
  if (imgc == NULL){
    Error(MSG1,"CopyCImage");
  }
  for (i=0; i<3; i++)
    imgc->C[i] = ift_CopyImage(cimg->C[i]);
  return imgc;
}


void    SetCImage(CImage *cimg, int color){
  SetImage(cimg->C[0], t0(color));
  SetImage(cimg->C[1], t1(color));
  SetImage(cimg->C[2], t2(color));
}


CImage *Convert2CImage(Image *img)
{
  CImage *imgc;
  int i;

  imgc = (CImage *) calloc(1,sizeof(CImage));
  if (imgc == NULL){
    Error(MSG1,"CopyCImage");
  }
  for (i=0; i<3; i++)
    imgc->C[i] = ift_CopyImage(img);
  return imgc;
}

CImage *CROI(CImage *cimg, int xl, int yl, int xr, int yr)
{
  CImage *croi=NULL;
  int i;

  if (ValidPixel(cimg->C[0],xl,yl)&&ValidPixel(cimg->C[0],xr,yr)&&
      (xl <= xr)&&(yl <= yr)) {
    croi = (CImage *) calloc(1,sizeof(CImage));
    if (croi == NULL){
      Error(MSG1,"CreateCImage");
    }
    for (i=0; i<3; i++)
      croi->C[i] = ROI(cimg->C[i], xl, yl, xr, yr);
  }
  return (croi);
}


void PasteSubCImage(CImage *cimg, CImage *sub, int x, int y){
  PasteSubImage(cimg->C[0], sub->C[0], x, y);
  PasteSubImage(cimg->C[1], sub->C[1], x, y);
  PasteSubImage(cimg->C[2], sub->C[2], x, y);
}


CImage *CScale(CImage *cimg, float Sx, float Sy)
{
  CImage *scl=NULL;
  int i;

  scl = (CImage *) calloc(1,sizeof(CImage));
  if (scl == NULL){
    Error(MSG1,"CreateCImage");
  }
  for (i=0; i<3; i++)
    scl->C[i] = Scale(cimg->C[i], Sx, Sy);
  return (scl);
}


CImage *CZoom(CImage *cimg, float Sx, float Sy){
  CImage *scl=NULL;
  int i;

  scl = (CImage *) calloc(1,sizeof(CImage));
  if (scl == NULL){
    Error(MSG1,"CZoom");
  }
  for (i=0; i<3; i++)
    scl->C[i] = Zoom(cimg->C[i], Sx, Sy);
  return (scl);
}


CImage *COverlayAlphaImage (CImage *cimg, Image *alpha)
{
  CImage *out=NULL;
  int i;

  out = (CImage *) calloc(1,sizeof(CImage));
  if (out == NULL){
    Error(MSG1,"CreateCImage");
  }
  for (i=0; i<3; i++)
    out->C[i] = OverlayAlphaImage(cimg->C[i], alpha);
  return (out);
}


Image  *GetBand(CImage *cimg, char band)
{
  Image *img=ift_CopyImage(cimg->C[(int)band]);

  return(img);
}

CImage *ColorizeImage(Image *img, float R, float G, float B)
{
  CImage *cimg=CreateCImage(img->ncols,img->nrows);
  float Cb,Cr,Y;
  int Imax,p,n=img->ncols*img->nrows;

  Imax = MAX(MaximumValue(img),1);

  Cb  = -0.148*R-0.291*G+0.439*B+128.0/255.;
  Cr  = 0.439*R-0.368*G-0.071*B+128.0/255.;

  for (p=0; p < n; p++) {

    Y = ((float)img->val[p]/Imax);

    R=296.82*(Y-0.062745098039)+
      406.98*(Cr-0.50196078431);

    G=296.82*(Y-0.062745098039)-
      207.315*(Cr-0.50196078431)-
      99.96*(Cb-0.50196078431);

    B=296.82*(Y-0.062745098039)+
      514.335*(Cb-0.50196078431);

    if (R<0.0) R=0.0;
    if (G<0.0) G=0.0;
    if (B<0.0) B=0.0;
    if (R>255.0) R=255.0;
    if (G>255.0) G=255.0;
    if (B>255.0) B=255.0;
    cimg->C[0]->val[p]=(int)(R);
    cimg->C[1]->val[p]=(int)(G);
    cimg->C[2]->val[p]=(int)(B);
  }
  return(cimg);
}


CImage *BlendImages(CImage *cimg1, CImage *cimg2, float alpha1, float alpha2)
{
  CImage *cimg3=CreateCImage(cimg1->C[0]->ncols,cimg1->C[0]->nrows);
  int p,n=cimg1->C[0]->ncols*cimg1->C[0]->nrows;

  for (p=0; p < n; p++) {
    cimg3->C[0]->val[p] = alpha1*cimg1->C[0]->val[p] + alpha2*cimg2->C[0]->val[p];
    cimg3->C[1]->val[p] = alpha1*cimg1->C[1]->val[p] + alpha2*cimg2->C[1]->val[p];
    cimg3->C[2]->val[p] = alpha1*cimg1->C[2]->val[p] + alpha2*cimg2->C[2]->val[p];
  }
  return(cimg3);
}

CImage *CLinearStretch(CImage *cimg, int f1, int f2, int g1, int g2)
{
  CImage *simg=NULL;
  int p,n;
  float a,R,G,B,Y,Cb,Cr,yf1,yf2,yg1,yg2;

  simg = CreateCImage(cimg->C[0]->ncols,cimg->C[0]->nrows);
  n    = simg->C[0]->ncols*simg->C[0]->nrows;
  if (f1 != f2) {
    a = ((float)(g2-g1))/(f2-f1);
  }
  else {
    a = INT_MAX;
  }

  yf1 = (float)f1/255.;
  yf2 = (float)f2/255.;
  yg1 = (float)g1/255.;
  yg2 = (float)g2/255.;

  for (p=0; p < n; p++){

    R = (float)cimg->C[0]->val[p]/255.;
    G = (float)cimg->C[1]->val[p]/255.;
    B = (float)cimg->C[2]->val[p]/255.;

    Y   = 0.257*R+0.504*G+0.098*B+16.0/255.;
    Cb  =-0.148*R-0.291*G+0.439*B+128.0/255.;
    Cr  = 0.439*R-0.368*G-0.071*B+128.0/255.;

    if (Y < yf1)
      Y = yg1;
    else
      if (Y > yf2)
	Y = yg2;
      else {
	if (a != INT_MAX)
	  Y = (a*(Y-yf1)+yg1);
	else{
	  Y = yg2;
	}
      }

    R= 296.82*(Y-0.062745098039)+\
       406.98*(Cr-0.50196078431);

    G =296.82*(Y-0.062745098039)-\
       207.315*(Cr-0.50196078431)-\
       99.96*(Cb-0.50196078431);

    B= 296.82*(Y-0.062745098039)+\
       514.335*(Cb-0.50196078431);

    if (R<0.0) R=0.0;
    if (G<0.0) G=0.0;
    if (B<0.0) B=0.0;
    if (R>255.0) R=255.0;
    if (G>255.0) G=255.0;
    if (B>255.0) B=255.0;

    simg->C[0]->val[p]=(int)(R);
    simg->C[1]->val[p]=(int)(G);
    simg->C[2]->val[p]=(int)(B);
  }
  return(simg);
}

CImage *CImageRGBtoYCbCr(CImage *cimg)
{
  CImage *ncimg=NULL;
  int p,n,i;


  ncimg = CreateCImage(cimg->C[0]->ncols,cimg->C[0]->nrows);
  n    = ncimg->C[0]->ncols*ncimg->C[0]->nrows;

  for (p=0; p < n; p++){

    i = triplet(cimg->C[0]->val[p],cimg->C[1]->val[p],cimg->C[2]->val[p]);
    i = RGB2YCbCr(i);
    ncimg->C[0]->val[p]=t0(i);
    ncimg->C[1]->val[p]=t1(i);
    ncimg->C[2]->val[p]=t2(i);
  }

  return(ncimg);
}

CImage *CImageYCbCrtoRGB(CImage *cimg)
{
  CImage *ncimg=NULL;
  int p,n,i;


  ncimg = CreateCImage(cimg->C[0]->ncols,cimg->C[0]->nrows);
  n    = ncimg->C[0]->ncols*ncimg->C[0]->nrows;

  for (p=0; p < n; p++){

    i = triplet(cimg->C[0]->val[p],cimg->C[1]->val[p],cimg->C[2]->val[p]);
    i = YCbCr2RGB(i);
    ncimg->C[0]->val[p]=t0(i);
    ncimg->C[1]->val[p]=t1(i);
    ncimg->C[2]->val[p]=t2(i);
  }

  return(ncimg);
}



CImage *CImageRGBtoHSV(CImage *cimg)
{
  CImage *ncimg=NULL;
  int p,n,i;


  ncimg = CreateCImage(cimg->C[0]->ncols,cimg->C[0]->nrows);
  n    = ncimg->C[0]->ncols*ncimg->C[0]->nrows;

  for (p=0; p < n; p++){

    i = triplet(cimg->C[0]->val[p],cimg->C[1]->val[p],cimg->C[2]->val[p]);
    i = RGB2HSV(i);
    ncimg->C[0]->val[p]=t0(i);
    ncimg->C[1]->val[p]=t1(i);
    ncimg->C[2]->val[p]=t2(i);
  }

  return(ncimg);
}


CImage *CImageHSVtoRGB(CImage *cimg)
{
  CImage *ncimg=NULL;
  int p,n,i;


  ncimg = CreateCImage(cimg->C[0]->ncols,cimg->C[0]->nrows);
  n    = ncimg->C[0]->ncols*ncimg->C[0]->nrows;

  for (p=0; p < n; p++){

    i = triplet(cimg->C[0]->val[p],cimg->C[1]->val[p],cimg->C[2]->val[p]);
    i = HSV2RGB(i);
    ncimg->C[0]->val[p]=t0(i);
    ncimg->C[1]->val[p]=t1(i);
    ncimg->C[2]->val[p]=t2(i);
  }

  return(ncimg);
}

CImage *CImageRGBtoLUV(CImage *rgb)
{

  int i,n = rgb->C[0]->ncols * rgb->C[0]->nrows;
  CImage *luv = CreateCImage(rgb->C[0]->ncols, rgb->C[0]->nrows);
    double XYZ[3][3] = {	{  0.4125,  0.3576,  0.1804 },
				{  0.2125,  0.7154,  0.0721 },
				{  0.0193,  0.1192,  0.9502 }	};


  double	x, y, z, L0, u_prime, v_prime, constant;
    double Yn			= 1.00000;
    double Un_prime	= 0.19784977571475;
    double Vn_prime	= 0.46834507665248;
    double Lt	= 0.008856;

  //convert RGB to XYZ...
  for (i=0;i<n;i++) {
    x = XYZ[0][0]*rgb->C[0]->val[i] + XYZ[0][1]*rgb->C[1]->val[i] + XYZ[0][2]*rgb->C[2]->val[i];
    y = XYZ[1][0]*rgb->C[0]->val[i] + XYZ[1][1]*rgb->C[1]->val[i] + XYZ[1][2]*rgb->C[2]->val[i];
    z = XYZ[2][0]*rgb->C[0]->val[i] + XYZ[2][1]*rgb->C[1]->val[i] + XYZ[2][2]*rgb->C[2]->val[i];

    //convert XYZ to LUV...

    //compute L*
    L0		= y / (255.0 * Yn);
    if(L0 > Lt)
      L0	= (double)(116.0 * (pow(L0, 1.0/3.0)) - 16.0);
    else
      L0	= (double)(903.3 * L0);

    //compute u_prime and v_prime
    constant	= x + 15 * y + 3 * z;
    if(constant != 0)
      {
	u_prime = (4 * x) / constant;
	v_prime = (9 * y) / constant;
	}
    else
      {
	u_prime	= 4.0;
	v_prime	= 9.0/15.0;
      }

    //compute u* and v*
    luv->C[0]->val[i] = L0;
    luv->C[1]->val[i] = (13 * L0 * (u_prime - Un_prime));
    luv->C[2]->val[i] = (13 * L0 * (v_prime - Vn_prime));
  }

  //done.
  return luv;
}

Image *ColorGradient(CImage *cimg)
{
  Image  *grad=NULL,*tmp[3];
  CImage *ncimg=NULL;
  int p,n;
  //  AdjRel *A=Circular(1.5);

  ncimg   = CImageRGBtoYCbCr(cimg);
  tmp[0]  = SobelFilter(ncimg->C[1]);        // MorphGrad(ncimg->C[1],A);
  tmp[1]  = SQRT(tmp[0]);
  DestroyImage(&tmp[0]);
  tmp[0]  = SobelFilter(ncimg->C[2]); //  MorphGrad(ncimg->C[2],A);
  tmp[2]  = SQRT(tmp[0]);
  DestroyImage(&tmp[0]);
  DestroyCImage(&ncimg);

  grad = CreateImage(tmp[1]->ncols,tmp[1]->nrows);
  n    = tmp[1]->ncols*tmp[1]->nrows;

  for (p=0; p < n; p++){
    grad->val[p]=MAX(tmp[1]->val[p],tmp[2]->val[p]);
  }
  DestroyImage(&tmp[1]);
  DestroyImage(&tmp[2]);
  return(grad);
}

//
//Image *CFeatureGradient(CImage *cimg, int nfeats, int maxval)
//{
//  Image *band[3],*grad;
//  int p,n=cimg->C[0]->ncols*cimg->C[0]->nrows;
//
//
//  band[0] = FeatureGradient(cimg->C[0],5,maxval);
//  band[1] = FeatureGradient(cimg->C[1],5,maxval);
//  band[2] = FeatureGradient(cimg->C[2],5,maxval);
//  grad    = CreateImage(band[0]->ncols,band[0]->nrows);
//  for (p=0; p < n; p++)
//    grad->val[p]=MAX(band[0]->val[p],MAX(band[1]->val[p],band[2]->val[p]));
//  DestroyImage(&band[0]);
//  DestroyImage(&band[1]);
//  DestroyImage(&band[2]);
//
//  return(grad);
//}

CImage *DrawCBorder(Image *img, Image *label){
  CImage *border;
  int i,j,k,p,q,u,v;
  AdjRel *A;
  Image *img8;

  img8 = ConvertToNbits(img, 8);
  border = Convert2CImage(img8);
  DestroyImage(&img8);
  A = Circular(1.0);
  for(i=0;i<img->nrows;i++){
    for(j=0;j<img->ncols;j++){
      p = j + img->tbrow[i];
      for(k=1;k<A->n;k++){
	u = j+A->dx[k];
	v = i+A->dy[k];
	if(ValidPixel(img,u,v)){
	  q= u + img->tbrow[v];
	  if (label->val[p] > label->val[q]){
	  switch( label->val[p]){
	    case 0:
	      border->C[0]->val[p] = 255;
	      border->C[1]->val[p] = 255;
	      border->C[2]->val[p] = 255;
	      break;
	    case 1:
	      border->C[0]->val[p] = 255;
	      border->C[1]->val[p] = 0;
	      border->C[2]->val[p] = 0;
	      break;
	    case 2:
	      border->C[0]->val[p] = 0;
	      border->C[1]->val[p] = 255;
	      border->C[2]->val[p] = 0;
	      break;
	    case 3:
	      border->C[0]->val[p] = 0;
	      border->C[1]->val[p] = 0;
	      border->C[2]->val[p] = 255;
	      break;
	    default:
	      border->C[0]->val[p] = 255;
	      border->C[1]->val[p] = 0;
	      border->C[2]->val[p] = 0;
	  }
	  break;
	  } else { /* thicker lines */
	    if (label->val[p] < label->val[q]){
	      switch( label->val[q]){
	      case 0:
		border->C[0]->val[p] = 255;
		border->C[1]->val[p] = 255;
		border->C[2]->val[p] = 255;
		break;
	      case 1:
		border->C[0]->val[p] = 255;
		border->C[1]->val[p] = 0;
		border->C[2]->val[p] = 0;
		break;
	      case 2:
		border->C[0]->val[p] = 0;
		border->C[1]->val[p] = 255;
		border->C[2]->val[p] = 0;
		break;
	      case 3:
		border->C[0]->val[p] = 0;
		border->C[1]->val[p] = 0;
		border->C[2]->val[p] = 255;
		break;
	      default:
		border->C[0]->val[p] = 255;
		border->C[1]->val[p] = 0;
		border->C[2]->val[p] = 0;
	      }
	      break;
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return border;
}

void    DrawCPoint(CImage *cimg, int x, int y, float raio, int R, int G, int B)
{
  int p,i,j;

  for(i=-raio;i<=raio;i++){
    for(j=-raio;j<=raio;j++){
      if ((x+j >= 0) && (x+j < cimg->C[0]->ncols) &&
	  (y+i >= 0) && (y+i < cimg->C[0]->nrows)){
	if(pow(j,2.0)+pow(i,2.0)<=pow(raio,2.0)){
	  p = x+j + cimg->C[0]->tbrow[y+i];
	  cimg->C[0]->val[p] = R;
	  cimg->C[1]->val[p] = G;
	  cimg->C[2]->val[p] = B;
	}
      }
    }
  }
}


void    DrawCImageLineDDA(CImage *cimg, int x1, int y1, int xn, int yn, int color){

  DrawImageLineDDA(cimg->C[0], x1, y1, xn, yn, t0(color));
  DrawImageLineDDA(cimg->C[1], x1, y1, xn, yn, t1(color));
  DrawImageLineDDA(cimg->C[2], x1, y1, xn, yn, t2(color));
}

CImage *DrawLabeledRegions(Image *img, Image *label){
  CImage *border=CreateCImage(img->ncols,img->nrows);
  int x,y,k,p,q,u,v;
  AdjRel *A;

  A = Circular(1.0);
  for(y=0;y<img->nrows;y++){
    for(x=0;x<img->ncols;x++){
      p = x + img->tbrow[y];
      border->C[0]->val[p]=
	border->C[1]->val[p]=border->C[2]->val[p]=img->val[p];

      for(k=1;k<A->n;k++){
	u = x+A->dx[k];
	v = y+A->dy[k];
	if(ValidPixel(img,u,v)){
	  q= u + img->tbrow[v];
	  if (label->val[p] != label->val[q]){
	    border->C[0]->val[p]=255;
	    border->C[1]->val[p]=0;
	    border->C[2]->val[p]=0;
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(border);
}


/** Increases the size of CImage until it becomes power-of-2-sided.
 *  The resulting image IS squared.
 **/
CImage* CImagePower2Squared(CImage* img)
{
    int ncols,nrows;
    CImage* result;
    int x,y;

    if(img == NULL) Error("Img NULL", "CImagePower2");

    ncols = (int)ceil(log(img->C[0]->ncols)/log(2));
    nrows = (int)ceil(log(img->C[0]->nrows)/log(2));
    ncols = 1 << ncols;
    nrows = 1 << nrows;

    ncols = MAX(ncols,nrows);
    nrows = ncols;

    result = CreateCImage(ncols, nrows);

    for(y = 0; y < nrows; y++)
    {
        for(x = 0; x < ncols; x++)
        {
            if(y < img->C[0]->nrows && x < img->C[0]->ncols)
            {
                int i = img->C[0]->tbrow[y]+x;
                int j = result->C[0]->tbrow[y] + x;
                result->C[0]->val[j] = img->C[0]->val[i];
                result->C[1]->val[j] = img->C[1]->val[i];
                result->C[2]->val[j] = img->C[2]->val[i];
            }
        }
    }

    return result;
}


int IsCImageGray(CImage* cimg)
{
    int i;

    if(cimg == NULL) Error("Cimg NULL", "IsCImageGray");

    for(i = 0; i < cimg->C[0]->ncols*cimg->C[0]->nrows; i++)
        if(cimg->C[0]->val[i] != cimg->C[1]->val[i] ||
            cimg->C[1]->val[i] != cimg->C[2]->val[i]) return 0;

    return 1;
}

CImage* CConvertToNbits(CImage* cimg, int N)
{
    CImage* caux = (CImage*)calloc(1,sizeof(CImage));

    if(caux == NULL) Error("Could not allocate memory for CImage* ","CConvertToNbits");

    caux->C[0] = ConvertToNbits(cimg->C[0], N);
    caux->C[1] = ConvertToNbits(cimg->C[1], N);
    caux->C[2] = ConvertToNbits(cimg->C[2], N);

    return caux;
}

CImage* CImageRGBtoLab(CImage* cimg)
{
    int i,n;
    CImage* lab;
    if(cimg == NULL) return NULL;

    n = cimg->C[0]->ncols*cimg->C[0]->nrows;

    lab = CreateCImage(cimg->C[0]->ncols, cimg->C[0]->nrows);

    for(i = 0; i < n; i++)
    {
        float R,G,B;

        R = (float)cimg->C[0]->val[i];
        G = (float)cimg->C[1]->val[i];
        B = (float)cimg->C[2]->val[i];

        RGB2Lab(&R,&G,&B);

        lab->C[0]->val[i] = (int)R;
        lab->C[1]->val[i] = (int)G;
        lab->C[2]->val[i] = (int)B;
    }

    return lab;
}

CImage* CImageLabtoRGB(CImage* cimg)
{
    int n,i;
    CImage* rgb;
    if(cimg == NULL) return NULL;

    n = cimg->C[0]->ncols*cimg->C[0]->nrows;

    rgb = CreateCImage(cimg->C[0]->ncols, cimg->C[0]->nrows);

    for(i = 0; i < n; i++)
    {
        float L,a,b;

        L = (float)cimg->C[0]->val[i];
        a = (float)cimg->C[1]->val[i];
        b = (float)cimg->C[2]->val[i];

        Lab2RGB(&L,&a,&b);

        rgb->C[0]->val[i] = (int)L;
        rgb->C[1]->val[i] = (int)a;
        rgb->C[2]->val[i] = (int)b;
    }

    return rgb;
}


Image *CFeatureGradient(CImage *cimg, int maxval)
{
    real    dist,gx,gy;
    int     j,i,p,q,n=cimg->C[0]->ncols*cimg->C[0]->nrows;
    Pixel   u,v;
    AdjRel *A=Circular(1.5);
    real   *md=AllocRealArray(A->n);

    Features* l = LabCImageFeats(cimg);

    float max = -FLT_MAX;
    Features* mslp;
    Features* f;
    Image* grad;
	float x,y;

    for(i = 0; i < l->nfeats; i++)
    {
        float min = FLT_MAX;
        for(j = 0; j < l->nelems; j++)
            min = MIN(l->elem[j].feat[i], min);

        min = fabsf(min);

        for(j = 0; j < l->nelems; j++)
        {
            l->elem[j].feat[i] += min;
            max = MAX(l->elem[j].feat[i], max);
        }
    }

    for(i = 0; i < l->nfeats; i++)
        for(j = 0; j < l->nelems; j++)
            l->elem[j].feat[i] = l->elem[j].feat[i]*255/max;

    l->Imax = (int)max;

    mslp = MSLowPassFeats(l, 3);

    for(i = 0; i < l->nfeats; i++)
        for(j = 0; j < l->nelems; j++)
            l->elem[j].feat[i] = l->elem[j].feat[i]/255;

    f = ConcatFeatures(l,mslp);

    DestroyFeatures(&mslp);
    DestroyFeatures(&l);

    grad = CreateImage(f->ncols, f->nrows);

    for (i=1; i < A->n; i++)
        md[i]=sqrt(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]);

    for (p=0; p < n; p++)
    {
        u.x = p%f->ncols;
        u.y = p/f->ncols;

        gx = gy = 0.0;

        for (i=1; i < A->n; i++)
        {
            v.x = u.x + A->dx[i];
            v.y = u.y + A->dy[i];
            if ((v.x>=0 && v.x<f->ncols) && (v.y>=0 && v.y<f->nrows))
            {
                q    = v.x + v.y*f->ncols;

                for (j=0; j<f->nfeats; j++)
                {
                    dist = (f->elem[q].feat[j]-f->elem[p].feat[j]);
                    ///(float)grad->Imax;
                    gx  += dist*A->dx[i]/md[i];
                    gy  += dist*A->dy[i]/md[i];

                }
            }
        }
        x =(maxval*gx)/(float)f->nfeats;
        y =(maxval*gy)/(float)f->nfeats;

        grad->val[p] = (int)sqrtf(x*x + y*y);
    }

    free(md);
    DestroyAdjRel(&A);
    DestroyFeatures(&f);

    return(grad);
}

CImage* TransformCImage(CImage *img, float M[4][4], int ncols, int nrows)
{
    CImage* result = (CImage*)calloc(1,sizeof(CImage*));

    result->C[0] = TransformImage(img->C[0], M, ncols, nrows);
    result->C[1] = TransformImage(img->C[1], M, ncols, nrows);
    result->C[2] = TransformImage(img->C[2], M, ncols, nrows);

    return result;
}
