#include "ift.h"

void iftWriteSeedsPython(iftLabeledSet* seed,   iftImage* img, const char *_filename, ...)
{
    va_list args;
    char filename[IFT_STR_DEFAULT_SIZE];
    va_start(args, _filename);
    vsprintf(filename, _filename, args);
    va_end(args);

    FILE *file = fopen(filename,"w");
    if(file == NULL)
        iftError("Invalid destination file", "iftWriteSeeds");

    iftLabeledSet *s = seed;
    int nseeds = 0;

    while(s != NULL){
        nseeds++;
        s = s->next;
    }

    if (!iftIs3DImage(img)) {
        fprintf(file,"%d %d %d\n", nseeds, img->xsize, img->ysize);
        s = seed;
        while(s != NULL){
            iftVoxel voxel = iftGetVoxelCoord(img,s->elem);
            fprintf(file, "%d %d %d %d\n", voxel.x, voxel.y, s->marker, s->label);
            s = s->next;
        }
    } else {
      fprintf(file,"%d %d %d %d\n", nseeds, img->xsize, img->ysize, img->zsize);
        s = seed;
        while(s != NULL){
            iftVoxel voxel = iftGetVoxelCoord(img,s->elem);
            fprintf(file, "%d %d %d %d %d %d\n", voxel.x, voxel.y, voxel.z, s->marker, s->label, s->handicap);
            s = s->next;
        }
    }

    fclose(file);
}


int main(int argc, char *argv[])
{
  
  if (argc != 5){
    iftError("Usage: iftLabel2Seeds <...>\n"
	     "[1] input folder with annotated images to create seeds.\n"
	     "[2] output folder with markers (*.txt).\n"
	     "[3] 0: preserve object labels, 1: use image labels.\n"
	     "[4] 0: output *-seeds.txt for ift 1: output *.txt for python\n.",
	     "main");
  }

  iftFileSet *fs = iftLoadFileSetFromDirOrCSV(argv[1], 1, 1);
  iftMakeDir(argv[2]);
  int conversion_type  = atoi(argv[3]);
  int output_type      = atoi(argv[4]);  
  int nimages          = fs->n;
  char ext[20];
  char filename[200];
  
  for (int i=0; i < nimages; i++) {
    iftImage *img    = iftReadImageByExt(fs->files[i]->path);    
    sprintf(ext,"%s",iftFileExt(fs->files[i]->path));
    char *basename   = iftFilename(fs->files[i]->path,ext);
    iftSList *info   = iftSplitString(basename,"_");
    iftSNode *node   = info->head;
    int image_label  = atoi(node->elem);
    iftDestroySList(&info);

    if (output_type == 0)
      sprintf(filename,"%s/%s-seeds.txt",argv[2],basename);
    else
      sprintf(filename,"%s/%s.txt",argv[2],basename);
      
    iftLabeledSet *S = NULL;
    if (conversion_type==1){ /* use image labels */
      for (int p = 0; p < img->n; p++) {
	if (img->val[p]!=0){
	  iftInsertLabeledSet(&S,p,image_label);
	}
      }
    }else{ /* preserve object labels */
      for (int p = 0; p < img->n; p++) {
	if (img->val[p]!=0){
	  iftInsertLabeledSet(&S,p,img->val[p]);
	}
      }
    }
    if (output_type == 0)
      iftWriteSeeds(S,img,filename);
    else
      iftWriteSeedsPython(S,img,filename);
  
    iftFree(basename);
    iftDestroyImage(&img);
    iftDestroyLabeledSet(&S);
  }

  iftDestroyFileSet(&fs);
  
  return (0);
}
