#include "ift.h"

/* Author: Alexandre Xavier Falcão (March 11th, 2024)

   Description: Detects objects (e.g., parasite eggs) from decoded
   saliency maps.

*/

#define BORDER_DIST   1
#define MAX_OBJ_AREA  9000
#define MIN_OBJ_AREA  1000


/* Returns all components with average saliency value above Otsu */

iftImage *DrawBoxes(iftImage *orig, iftImage *comp, iftColor YCbCr, float scale)
{
  iftImage  *box     = iftCopyImage(orig);

  if (iftMaximumValue(comp)==0)
    return(box);

  iftAdjRel *A       = iftCircular(1.5);
  iftImage  *label   = iftFastLabelComp(comp, A);
  int        nlabels = iftMaximumValue(label);
  
  for (int l=1; l <= nlabels; l++) {
    iftImage *bin     = iftExtractObject(label,l);
    iftVoxel center;
    iftBoundingBox bb = iftMinBoundingBox(bin, &center);
    iftDestroyImage(&bin);
    int xsize = (bb.end.x - bb.begin.x)*scale;
    int ysize=(bb.end.y - bb.begin.y)*scale;
    if (xsize*ysize > 0){
      iftAdjRel *B = iftRectangular(xsize,ysize);
      iftAdjRel *C = iftAdjacencyBoundaries(B,A);    
      iftDestroyAdjRel(&B);
      for (int i=1; i < C->n; i++) {
	iftVoxel v = iftGetAdjacentVoxel(C,center,i);
	if (iftValidVoxel(box,v)){
	  int q       = iftGetVoxelIndex(box,v);
	  box->val[q] = YCbCr.val[0];
	  box->Cb[q]  = YCbCr.val[1];
	  box->Cr[q]  = YCbCr.val[2];
	}
      }
      iftDestroyAdjRel(&C);
    }
  }
  iftDestroyAdjRel(&A);
  iftDestroyImage(&label);
  
  return(box);
}

iftImage *BoxesFromComp(iftImage *comp, float scale)
{
  iftImage  *label   = iftFastLabelComp(comp,NULL);
  int        nlabels = iftMaximumValue(label);
  iftImage  *boxes   = iftCreateImageFromImage(comp);
  
  for (int l=1; l <= nlabels; l++) {
    iftImage *bin     = iftExtractObject(label,l);
    iftVoxel center;
    iftBoundingBox bb = iftMinBoundingBox(bin, &center);
    int xsize = (bb.end.x - bb.begin.x)*scale;
    int ysize = (bb.end.y - bb.begin.y)*scale;
    iftVoxel u; u.z=0;
    for (u.y=center.y-ysize/2; u.y < center.y+ysize/2; u.y++)
      for (u.x=center.x-xsize/2; u.x < center.x+xsize/2; u.x++){
	if (iftValidVoxel(boxes,u)){
	  int p = iftGetVoxelIndex(boxes,u);
	  boxes->val[p] = 1;
	}
      }
    iftDestroyImage(&bin);
  }
  
  iftDestroyImage(&label);
  
  return(boxes);
}

int main(int argc, char *argv[])
{
  
  if (argc!=4){ 
    iftError("Usage: iftSMansoniDetection <P1> <P2> <P3>\n"
	     "[1] folder with the salience maps\n"
	     "[2] layer (1,2,...) to create the results\n"
	     "[3] output folder with the resulting images\n",	 
	     "main");
  }
  
  timer *tstart = iftTic();

  char *filename     = iftAllocCharArray(512);
  int layer          = atoi(argv[2]);
  char suffix[12];
  sprintf(suffix,"_layer%d.png",layer);
  iftFileSet *fs     = iftLoadFileSetFromDirBySuffix(argv[1], suffix, true);
  char *output_dir1   = argv[3];
  char  fileout[30];
  sprintf(fileout,"result_layer%d.csv",layer);
  iftMakeDir(output_dir1);
  iftColorTable *ctb = iftCreateRandomColorTable(10,65535);
  iftAdjRel *A       = iftCircular(5.0); 
  iftAdjRel *B       = iftCircular(1.5);      
  iftAdjRel *C       = iftCircular(1.0);
  FILE      *fp      = fopen(fileout,"w");
  float      mscore  = 0.0;

  for(int i = 0; i < fs->n; i++) {
    printf("Processing image %d of %ld\r", i + 1, fs->n);

    char *basename1      = iftFilename(fs->files[i]->path,suffix);      
    char *basename2      = iftFilename(fs->files[i]->path,".png");      
    iftImage *salie      = iftReadImageByExt(fs->files[i]->path);
    sprintf(filename,"./images/%s.png",basename1);
    iftImage *orig       = iftReadImageByExt(filename); 
    sprintf(filename,"./truelabels/%s.png",basename1);
    iftImage *aux1       = iftReadImageByExt(filename); 
    iftImage *gt         = iftThreshold(aux1,128,IFT_INFINITY_INT,1);
    iftDestroyImage(&aux1);
    iftImage *img        = iftCopyImage(orig);

    /* Detect binary components */

    if (iftMaximumValue(salie)>0){      
      iftImage  *aux2 = iftThreshold(salie,iftOtsu(salie),IFT_INFINITY_INT,255);      aux1            = iftFastLabelComp(aux2,NULL);
      iftDestroyImage(&aux2);
      aux2  = iftSelectCompInAreaInterval(aux1,NULL,
					  MIN_OBJ_AREA,MAX_OBJ_AREA);
      iftDestroyImage(&aux1);
      if (iftMaximumValue(aux2)>0){
	iftImage *bin1 = BoxesFromComp(aux2,1.50);
	iftImage *bin2 = BoxesFromComp(gt,1.0);
	float score    = iftFScore(bin1, bin2);
	fprintf(fp,"%s;%.2f\n",basename1,score);
	mscore += score;
	if (iftMaximumValue(bin1)>0){
	  iftDestroyImage(&img);
	  img = DrawBoxes(orig, aux2, ctb->color[1], 1.1);
	}	
	iftDestroyImage(&aux2);
	iftDestroyImage(&bin1);
	iftDestroyImage(&bin2);
      }else{
	if (iftMaximumValue(gt)==0){
	  fprintf(fp,"%s;1.0\n",basename1);
	  mscore += 1.0;
	} else {
	  fprintf(fp,"%s;0.0\n",basename1);
	  mscore += 0.0;
	}
      }
    } else {
      if (iftMaximumValue(gt)==0){
	fprintf(fp,"%s;1.0\n",basename1);
	mscore += 1.0;
      } else {
	fprintf(fp,"%s;0.0\n",basename1);
	mscore += 0.0;
      }
    }
    iftDestroyImage(&salie);
    iftDestroyImage(&orig);
    iftDestroyImage(&gt);
      
    /* save resulting image */

    sprintf(filename,"%s/%s.png",output_dir1,basename2);
    iftWriteImageByExt(img,filename);    
    
    iftDestroyImage(&img);
    iftFree(basename1);
    iftFree(basename2);
  }

  iftDestroyColorTable(&ctb);
  iftFree(filename);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  iftDestroyAdjRel(&C);
  mscore /= fs->n;
  fprintf(fp,"Average Fscore; %.2f\n",mscore);
  
  fclose(fp);
  iftDestroyFileSet(&fs);
  
  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}
