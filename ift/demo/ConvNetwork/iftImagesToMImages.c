#include "ift.h"

void iftImagesToMImages(char *input_dir, char *output_dir, char *ext)
{
  int number_of_images;
  iftImageNames  *image_names;
  char command[200];
  DIR *dir=NULL;

  number_of_images  = iftCountImageNames(input_dir, ext);
  image_names       = iftCreateAndLoadImageNames(number_of_images, input_dir, ext);

  dir = opendir(output_dir);
  if (dir == NULL){
    sprintf(command,"mkdir %s",output_dir);
    if (system(command)!=0) 
      iftError("Could not open directory","iftImagesToMImages");
  }else
    closedir(dir);

#pragma omp parallel for shared(number_of_images,image_names,input_dir,output_dir)
    for (int s = 0; s < number_of_images ; s++){
      char imagename1[200], imagename2[200];
      sprintf(imagename1,"%s/%s",input_dir,image_names[s].image_name);
      fprintf(stdout,"Processing %s\n",imagename1);
      iftImage  *input;
      iftMImage *output;
      if (strcmp(ext,"pgm")==0){
	input  = iftReadImageP5(imagename1);
	output = iftImageToMImage(input,GRAY_CSPACE);
      } else {
	if (strcmp(ext,"ppm")==0){
	  input = iftReadImageP6(imagename1);
	  output = iftImageToMImage(input,WEIGHTED_YCbCr_CSPACE);
	} else {
	  if (strcmp(ext,"scn")==0){
	    input = iftReadImage(imagename1);
	    output = iftImageToMImage(input,GRAY_CSPACE);
	  } else {
	    iftError("Invalid extension","iftImagesToMImages");
	  }
	}
      }

      strcpy(imagename1,image_names[s].image_name);
      char *basename = strtok(imagename1,".");
      sprintf(imagename2,"%s/%s.mig",output_dir,basename);
      iftWriteMImage(output,imagename2);
      iftDestroyMImage(&output);
      iftDestroyImage(&input);
    }

  iftDestroyImageNames(image_names);
}

int main(int argc, char **argv) 
{
  timer          *t1=NULL,*t2=NULL;


  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/


  if (argc!=4)
    iftError("Usage: iftImagesToMImages <input_dir> <output_dir> <extension [pgm,ppm,scn]>","main");

  t1 = iftTic();
  iftImagesToMImages(argv[1],argv[2],argv[3]);
  t2 = iftToc();
  fprintf(stdout,"Images converted in %f ms\n",iftCompTime(t1,t2));


  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   






  return(0);
}
