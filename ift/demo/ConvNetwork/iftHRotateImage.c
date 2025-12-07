#include "ift.h"

iftImage  *iftCrop2DImage(iftImage *img, int x1, int y1, int x2, int y2)
{
    iftImage *imgc=iftCreateImage(x2-x1,y2-y1,1);
    int pcrop;

    pcrop = 0;
    for (int y = y1; y < y2; y++){
        for (int x = x1; x < x2; x++){
    	    int p = x + img->tby[y];
    	    imgc->val[pcrop]=img->val[p];
    	    pcrop++;
        }
    }

    if (img->Cb != NULL) {
        pcrop = 0;
		for (int y = y1; y < y2; y++)
			for (int x = x1; x < x2; x++){
			    int p = x + img->tby[y];
			    imgc->Cb[pcrop]=img->Cb[p];
			    imgc->Cr[pcrop]=img->Cr[p];
			    pcrop++;
			}
    }
    return(imgc);
}

int iftMaxRemFrame(iftImage *bin)
{
  int xmin,ymin,xmax,ymax,p;
  int padxl,padxr,padyt,padyb,pad;
  xmin = 999999999;
  ymin = 999999999;
  xmax = -1;
  ymax = -1;

  for(p=0;p<bin->n;p++){
    iftVoxel u   = iftGetVoxelCoord(bin ,p);
    if(bin->val[p] == 255){
      if(u.x < xmin)
        xmin = u.x;
      if(u.y < ymin)
        ymin = u.y;
      if(u.x > xmax)
        xmax = u.x;
      if(u.y > ymax)
        ymax = u.y;
    }
  }
  padxl = xmin - 1;
  padxr = (bin->xsize - xmax) - 1;
  padyt = ymin - 1;
  padyb = (bin->ysize - ymax) - 1;
  pad = padxl;
  if(padxr < pad)
    pad = padxr;
  if(padyt < pad)
    pad = padyt;
  if(padyb < pad)
    pad = padyb;
  return pad;
}

iftImage  *iftCenterImage(iftImage *img, iftImage *bin)
{
  iftImage *crop_img;
  int xmin,ymin,xmax,ymax,p;
  int x1,y1,x2,y2;
  xmin = 999999999;
  ymin = 999999999;
  xmax = -1;
  ymax = -1;
  for(p=0;p<bin->n;p++){
    iftVoxel u   = iftGetVoxelCoord(bin ,p);
    if(bin->val[p] == 255){
      if(u.x < xmin)
        xmin = u.x;
      if(u.y < ymin)
        ymin = u.y;
      if(u.x > xmax)
        xmax = u.x;
      if(u.y > ymax)
        ymax = u.y;
    }
  }
  int width = (xmax - xmin);
  int height = (ymax - ymin);
  if(width > height){
	  x1 = xmin;
	  x2 = xmax;
	  y1 = 0;
	  if((int)((width - height) / 2) < ymin){
		  y1 = ymin - (int)((width - height) / 2);
	  }
	  y2 = y1 + width;
  }else{
	  y1 = ymin;
	  y2 = ymax;
	  x1 = 0;
	  if((int)((height - width) / 2) < xmin){
		  x1 = xmin - (int)((height - width) / 2);
	  }
	  x2 = x1 + height;
  }
  printf("(%d,%d) - (%d,%d) \n",x1,y1,x2,y2);
  crop_img = iftCrop2DImage(img,x1,y1,x2,y2);
  return crop_img;
}

iftImage *iftHRotate2DImage(iftImage *img, iftImage *bin)
{
  iftMatrix *U,*S,*Vt,*A;
  iftDataSet *Z, *Zc;
  iftImage *timg,*tbin;
  Z   = iftObjectToDataSet(bin);
  iftSetStatus(Z,IFT_TRAIN);

  Zc  = iftCentralizeDataSet(Z);

  A = iftCovarianceMatrix(Zc);

  iftSingleValueDecomp(A,&U,&S,&Vt);

  timg = iftTransformImage(img,Vt,TRANSFORMATION_MATRIX);
  tbin = iftTransformImage(bin,Vt,TRANSFORMATION_MATRIX);

  int maxval = iftMaximumValue(timg);
  for(int p=0;p<timg->n;p++){
     timg->val[p] = (int)(((float)timg->val[p]/(float)maxval)*255.0);
  }
  iftUpdateMinMax(timg);
  //int maxRemFrame = iftMaxRemFrame(tbin);
  //printf("maxRemFrame: %d\n",maxRemFrame);
  //rimg = iftRemFrame(timg, maxRemFrame);
  //rimg = iftCenterImage(timg, tbin);
  
  //iftWriteImageP5(tbin,"alignedbin.pgm");
  //iftWriteImageP5(timg,"aligned.pgm");

  //iftDestroyImage(&timg);
  iftDestroyImage(&tbin);
  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&Zc);
  iftDestroyMatrix(&A);
  iftDestroyMatrix(&U);
  iftDestroyMatrix(&S);
  iftDestroyMatrix(&Vt);

  return timg;
}

int main(int argc, char *argv[]) 
{
  iftImage *bin,*timg,*img;
  int             number_of_images;
  iftImageNames  *image_names;
  char *input_dir, *bin_dir, *output_dir;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=4)
    iftError("Usage: iftHRotateImage <input_directory> <bin_directory> <output_directory>","main");

  input_dir = argv[1];
  bin_dir = argv[2];
  output_dir = argv[3];

  number_of_images  = iftCountImageNames(input_dir, "pgm");
  image_names       = iftCreateAndLoadImageNames(number_of_images, input_dir, "pgm");

  //#pragma omp parallel for shared(number_of_images,image_names,input_dir,bin_dir,output_dir)
  for (int s = 0; s < number_of_images ; s++){
	char filename[200];
	char filename_bin[200];
	char filename_out[200];
	sprintf(filename,"%s/%s",input_dir,image_names[s].image_name);
	sprintf(filename_bin,"%s/%s",bin_dir,image_names[s].image_name);
	fprintf(stdout,"Processing %s\n",filename);
	img  = iftReadImageP5(filename);
	bin  = iftReadImageP5(filename_bin);
	timg = iftHRotate2DImage(img, bin);
	sprintf(filename_out,"%s/%s",output_dir,image_names[s].image_name);
	iftWriteImageP5(timg,filename_out);
	iftDestroyImage(&img);
	iftDestroyImage(&bin);
	iftDestroyImage(&timg);
  }

  iftDestroyImageNames(image_names);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

