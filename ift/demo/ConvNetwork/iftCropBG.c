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
    iftUpdateMinMax(imgc);
    return(imgc);
}

iftImage  *iftCrop2DBinImage(iftImage *img, int x1, int y1, int x2, int y2)
{
    iftImage *imgc=iftCreateImage(x2-x1,y2-y1,1);
    int pcrop;

    pcrop = 0;
    for (int y = y1; y < y2; y++){
        for (int x = x1; x < x2; x++){
    	    int p = x + img->tby[y];
    	    imgc->val[pcrop]=img->val[p]*255;
    	    pcrop++;
        }
    }
    iftUpdateMinMax(imgc);
    return(imgc);
}

void  *iftCropBG(iftImage *img, iftImage *bin, char *output_dir, char *output_bin_dir, char *filename)
{
  iftImage *crop_img, *crop_bin;
  char filename_out[200], filename_out_bin[200];
  int xmin,ymin,xmax,ymax,p;
  xmin = 999999999;
  ymin = 999999999;
  xmax = -1;
  ymax = -1;
  for(p=0;p<img->n;p++){
    iftVoxel u   = iftGetVoxelCoord(img ,p);
    if(img->val[p] > 0){
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
  
  printf("(%d,%d) - (%d,%d) \n",xmin,ymin,xmax,ymax);
  
  crop_img = iftCrop2DImage(img,xmin,ymin,xmax,ymax);
  crop_bin = iftCrop2DBinImage(bin,xmin,ymin,xmax,ymax);
  
  sprintf(filename_out,"%s/%s",output_dir,filename);
  iftWriteImageP5(crop_img,filename_out);
  
  sprintf(filename_out_bin,"%s/%s",output_bin_dir,filename);
  iftWriteImageP5(crop_bin,filename_out_bin);
  
  iftDestroyImage(&crop_img);
  iftDestroyImage(&crop_bin);
}


int main(int argc, char *argv[]) 
{
  iftImage *bin,*img;
  int             number_of_images;
  iftImageNames  *image_names;
  char *input_dir, *bin_dir, *output_dir, *output_bin_dir;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=5)
    iftError("Usage: iftCropBG <image_directory> <bin_directory> <output_image_directory> <output_bin_directory>","main");

  input_dir = argv[1];
  bin_dir = argv[2];
  output_dir = argv[3];
  output_bin_dir = argv[4];

  number_of_images  = iftCountImageNames(input_dir, "pgm");
  image_names       = iftCreateAndLoadImageNames(number_of_images, input_dir, "pgm");

  //#pragma omp parallel for shared(number_of_images,image_names,input_dir,bin_dir,output_dir)
  for (int s = 0; s < number_of_images ; s++){
	char filename[200];
	char filename_bin[200];
	sprintf(filename,"%s/%s",input_dir,image_names[s].image_name);
	sprintf(filename_bin,"%s/%s",bin_dir,image_names[s].image_name);
	fprintf(stdout,"Processing %s\n",filename);
	img  = iftReadImageP5(filename);
	bin  = iftReadImageP5(filename_bin);
	iftCropBG(img, bin, output_dir, output_bin_dir, image_names[s].image_name);
	iftDestroyImage(&img);
	iftDestroyImage(&bin);
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

