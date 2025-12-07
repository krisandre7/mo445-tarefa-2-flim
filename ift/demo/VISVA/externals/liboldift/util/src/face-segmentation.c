#include "ift.h"

Subgraph *MergeSamples(Subgraph *sg1, Subgraph *sg2)
{
  Subgraph *sg=CreateSubgraph(sg1->nnodes+sg2->nnodes);
  int i,j;
  
  j = 0;
  for (i=0; i < sg1->nnodes; i++) {
    sg->node[j].pixel=sg1->node[i].pixel; 
    j++;
  }
  for (i=0; i < sg2->nnodes; i++) {
    sg->node[j].pixel=sg2->node[i].pixel; 
    j++;
  }
  return(sg);
}

Subgraph *GetBorderSamples(Image *img, int size, int nsamples)
{
  int max_nnodes = size*img->ncols*2+(img->nrows-(2*size))*2*size;
  Subgraph *sg=CreateSubgraph(nsamples);
  Pixel u;
  int i, j, *border=AllocIntArray(max_nnodes), *used=AllocIntArray(max_nnodes);

  if (max_nnodes < nsamples) {
    printf("Error: the number of samples exceeded the number of available border pixels.\n");
    exit(1);
  }

  i=0;
  for (u.y=0; u.y < size; u.y++) 
    for (u.x=0; u.x < img->ncols; u.x++){
      border[i] = u.x + img->tbrow[u.y]; i++;
    }
  for (u.y=img->nrows-size; u.y < img->nrows; u.y++) 
    for (u.x=0; u.x < img->ncols; u.x++) {
      border[i] = u.x + img->tbrow[u.y]; i++;
    }
  for (u.x=0; u.x < size; u.x++) 
    for (u.y=size; u.y < img->nrows-size; u.y++) {
      border[i] = u.x + img->tbrow[u.y]; i++;
    }
  for (u.x=img->ncols-size; u.x < img->ncols; u.x++) 
    for (u.y=size; u.y < img->nrows-size; u.y++) {
      border[i] = u.x + img->tbrow[u.y]; i++;
    }
 
  i    = 0;
  while( i < nsamples ) {
    j = RandomInteger( 0, max_nnodes - 1 );
    if ( used[ j ] == 0 ) {
      sg->node[ i ].pixel    = border[j];
      used[ j ] = 1;
      i++;
    }
  }
  
  free( used );
  free( border );

  return( sg );
}

Subgraph *GetSkinSamples(CImage *img, int nsamples)
{
  int ncols=img->C[0]->ncols,nrows=img->C[0]->nrows;
  int xm=ncols/2,ym=nrows/2;
  int xo=xm-60,xf=xm+60;
  int yo=ym-5,yf=ym+5;
  int max_nnodes=(xf-xo+1)*(yf-yo+1);
  int x,y,p,i,j;
  Subgraph *sg=NULL;
  int *pixel=AllocIntArray(max_nnodes), *used=AllocIntArray(max_nnodes);

  if (max_nnodes < nsamples) {
    printf("Error: the number of samples exceeded the number of available region pixels in the face.\n");
    exit(1);
  }

  i=0;  
  for (y=yo; y <= yf; y++) 
    for (x=xo; x <= xf; x++){
      p = x + img->C[0]->tbrow[y];      
      pixel[i] = p;
      i++;
    }

  sg = CreateSubgraph(nsamples);
  i    = 0;
  while( i < nsamples ) {
    j = RandomInteger( 0, max_nnodes - 1 );
    if ( used[ j ] == 0 ) {
      sg->node[ i ].pixel    = pixel[j];
      used[ j ] = 1;
      i++;
    }
  }
  
  free( used );
  free( pixel );

  return(sg);
}

Features *GetCbCrFeatures(CImage *img)
{
  int ncols=img->C[0]->ncols,nrows=img->C[0]->nrows;
  int p,n=ncols*nrows;
  float Imax;
  Features *f=CreateFeatures(ncols,nrows,2);
  CImage *YCbCr=CImageRGBtoYCbCr(img);
  
  Imax=0.0;
  for (p=0; p < n; p++) {
    f->elem[p].feat[0]=YCbCr->C[1]->val[p];
    f->elem[p].feat[1]=YCbCr->C[2]->val[p];
    if (MAX(f->elem[p].feat[0],f->elem[p].feat[1])>Imax)
      Imax = MAX(f->elem[p].feat[0],f->elem[p].feat[1]);
  }
  f->Imax = (int)Imax;
  for (p=0; p < n; p++) {
    f->elem[p].feat[0]=f->elem[p].feat[0]/Imax;
    f->elem[p].feat[1]=f->elem[p].feat[1]/Imax;
  }
  DestroyCImage(&YCbCr);
  return(f);
}

Image *GetOutliers(Subgraph *sg, Features *f)
{
    int     i,p,k;
    Image  *outliers=CreateImage(f->ncols,f->nrows);
    float   dist;

    for (p=0; p < f->nelems; p++)
      outliers->val[p]=-1;

    for (p=0; p < f->nelems; p++)
    {
      for (i = 0; i < sg->nnodes; i++){
	k = sg->ordered_list_of_nodes[i];
	dist = opf_ArcWeight(sg->node[k].feat,f->elem[p].feat,sg->nfeats);
	if (dist <= sg->node[k].radius){
	  outliers->val[p]=(int)(sg->node[k].dens);
	  break;
	}
      }
    }

    for (p=0; p < f->nelems; p++)
      if (outliers->val[p]==-1)
	outliers->val[p]=1;
      else
	outliers->val[p]=0;

    return(outliers);
}

int main(int argc, char **argv)
{
  CImage   *orig=NULL;
  Subgraph *sg_skin=NULL,*sg_border=NULL,*sg=NULL;
  Features *f=NULL;
  Image    *outliers=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/
  
  if (argc != 2) {
    fprintf(stderr,"usage: face-recognition <input.ppm> \n");
    exit(-1);
  }

  orig       = ReadCImage(argv[1]);
  sg         = GetSkinSamples(orig,200);
  //sg_skin    = GetSkinSamples(orig,200);
  //sg_border  = GetBorderSamples(orig->C[0],5,200);
  // sg         = MergeSamples(sg_skin,sg_border);
  f          = LabCImageFeats(orig);//GetCbCrFeatures(orig);
  // DestroySubgraph(&sg_skin);
  // DestroySubgraph(&sg_border);

  SetSubgraphFeatures(sg,f);
  opf_BestkMinCut(sg,(int)(0.1*sg->nnodes),(int)(0.4*sg->nnodes));
  opf_OPFClustering(sg);

  outliers=GetOutliers(sg,f);
  

  Image *bin=OpenBin(outliers,3.0);
  Image *area=Area(bin);
  
  DestroyImage(&bin);
  bin = Threshold(area,500,MaximumValue(area)-1);

  
  WriteImage(bin,"outliers.pgm");


  DestroyImage(&bin);
  DestroyImage(&area);
  DestroyCImage(&orig);
  DestroyImage(&outliers);
  DestroySubgraph(&sg);
  DestroyFeatures(&f);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}
