#include "ift.h"


int main(int argc, char *argv[]) 
{
  size_t           mem_start, mem_end;
  timer           *t1, *t2;

  mem_start = iftMemoryUsed();

  if ((argc != 6)&&(argc != 3)){
    printf("Usage: iftBinarizeActivations <P1> <P2> <P3> <P4> <P5>\n");
    printf("P1: input folder with activation images\n");
    printf("P2: output folder with detected components\n");
    printf("P3: sx (optional scale for the original size)\n");
    printf("P4: sy (optional scale for the original size)\n");
    printf("P5: sz (optional scale for the original size)\n");
    exit(-1);
  }

  t1 = iftTic();

  // Read input parameters

  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(argv[1], ".mimg", 1);
  iftMakeDir(argv[2]);

  int   sx=1, sy=1, sz=1;
  if (argc == 6) {
    sx = atoi(argv[3]);
    sy = atoi(argv[4]);
    sz = atoi(argv[5]);
  } 
  
  char filename[200];

  for (int i=0; i < fs->n; i++) {    
    iftMImage *img = iftReadMImage(fs->files[i]->path);    
    char *basename = iftFilename(fs->files[i]->path, ".mimg");

    printf("Image %03d\n",i+1);
    
    // detect components 
    
    for (int b=0; b < img->m; b++) {
      iftImage *mask = iftCreateImage(img->xsize,img->ysize,img->zsize);
      iftImage *norm = iftMImageToImage(img,255,b);
      
      for (int p=0; p < img->n; p++) {
	if (norm->val[p] >= 127){
	  mask->val[p] = 255;
	}
      }

      iftDestroyImage(&norm);
    
    
      // perform interpolation if asked to

      if (sx*sy*sz > 1) {
	iftImage *interp_mask;
	if (iftIs3DImage(mask)){
	  interp_mask = iftCreateImage(sx*mask->xsize,sy*mask->ysize,
				       sz*mask->zsize);
	} else {
	  interp_mask = iftCreateImage(sx*mask->xsize,sy*mask->ysize,
				       1);
	}
	
	iftVoxel u;
	for (u.z=0; u.z < interp_mask->zsize; u.z++){
	  for (u.y=0; u.y < interp_mask->ysize; u.y++){
	    for (u.x=0; u.x < interp_mask->xsize; u.x++){
	      iftVoxel v;
	      v.x = u.x / sx;
	      v.y = u.y / sy;
	      v.z = u.z / sz;
	      int p = iftGetVoxelIndex(interp_mask,u);
	      if (iftValidVoxel(mask,v)){
		int q = iftGetVoxelIndex(mask,v);
		interp_mask->val[p] = mask->val[q];
	      }
	    }
	  }
	}
	iftDestroyImage(&mask);
	mask = interp_mask;
      }
      
      // estimate minuciae as the geodesic centers of the detected
      // components
    
      if (iftIs3DImage(mask)){
	sprintf(filename,"%s/%s-%03d.nii.gz",argv[2],basename,b+1);
      }else{
	sprintf(filename,"%s/%s-%03d.png",argv[2],basename,b+1);
      }
      iftWriteImageByExt(mask,filename);
      iftDestroyImage(&mask);
    }
    iftDestroyMImage(&img);
    iftFree(basename);
  }
  
  mem_end = iftMemoryUsed();
  iftVerifyMemory(mem_start,mem_end);
    
  t2 = iftToc();
  puts(iftFormattedTime(iftCompTime(t1,t2)));

  return(0);
}
