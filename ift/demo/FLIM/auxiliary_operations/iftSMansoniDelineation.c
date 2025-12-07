#include "ift.h"

/* Author: Alexandre Xavier Falcão (September 10th, 2023)

   Description: Delineates objects (e.g., parasite eggs) from the
   decoded saliency maps.

*/

#define BORDER_DIST   1
#define SEED_DILATION 30
#define SEED_EROSION  1
#define MAX_OBJ_AREA  9000
#define MIN_OBJ_AREA  1000

/* iftImage *DynamicTrees(iftImage *orig, iftImage *seeds_in, iftImage *seeds_out) */
/* { */
/*   iftMImage  *mimg     = iftImageToMImage(orig,LAB_CSPACE); */
/*   iftImage   *pathval  = NULL, *label = NULL, *root = NULL; */
/*   float      *tree_L   = NULL; */
/*   float      *tree_A   = NULL; */
/*   float      *tree_B   = NULL; */
/*   int        *nnodes   = NULL; */
/*   int Imax             = iftRound(sqrtf(3.0)* */
/* 				  iftMax(iftMax(iftMMaximumValue(mimg,0), */
/* 						iftMMaximumValue(mimg,1)), */
/* 					 iftMMaximumValue(mimg,2))); */
/*   iftGQueue  *Q        = NULL; */
/*   iftAdjRel  *A        = iftCircular(1.0); */
/*   int         i, p, q, r, tmp; */
/*   iftVoxel    u, v; */

/*   // Initialization */
  
/*   pathval    = iftCreateImage(orig->xsize, orig->ysize, orig->zsize); */
/*   label      = iftCreateImage(orig->xsize, orig->ysize, orig->zsize); */
/*   root       = iftCreateImage(orig->xsize, orig->ysize, orig->zsize); */
/*   tree_L     = iftAllocFloatArray(orig->n); */
/*   tree_A     = iftAllocFloatArray(orig->n); */
/*   tree_B     = iftAllocFloatArray(orig->n); */
/*   nnodes     = iftAllocIntArray(orig->n); */
/*   Q          = iftCreateGQueue(Imax+1, orig->n, pathval->val); */

/*   /\* Initialize costs *\/ */
    
/*   for (p = 0; p < orig->n; p++) */
/*   { */
/*     pathval->val[p] = IFT_INFINITY_INT; */
/*     if (seeds_in->val[p] != 0){ */
/*       root->val[p]    = p; */
/*       label->val[p]   = 255; */
/*       pathval->val[p] = 0; */
/*     }else{ */
/*       if (seeds_out->val[p] != 0){ */
/* 	root->val[p]    = p; */
/* 	label->val[p]   = 0; */
/* 	pathval->val[p] = 0; */
/*       } */
/*     }     */
/*     iftInsertGQueue(&Q, p); */
/*   }       */

/*   // Image Foresting Transform */

/*   while (!iftEmptyGQueue(Q)) */
/*   { */
/*     p           = iftRemoveGQueue(Q); */
/*     r           = root->val[p]; */
/*     tree_L[r]  += mimg->val[p][0]; */
/*     tree_A[r]  += mimg->val[p][1]; */
/*     tree_B[r]  += mimg->val[p][2]; */
/*     nnodes[r]  += 1; */
/*     u           = iftGetVoxelCoord(orig, p); */

/*     for (i = 1; i < A->n; i++) */
/*     { */
/*       v = iftGetAdjacentVoxel(A, u, i); */

/*       if (iftValidVoxel(orig, v)) */
/*       { */
/*         q = iftGetVoxelIndex(orig, v); */

/* 	if (Q->L.elem[q].color != IFT_BLACK) { */

/* 	  int Wi = iftRound( */
/* 		   sqrtf(powf((mimg->val[q][0]-tree_L[r]/nnodes[r]),2.0) +  */
/* 			 powf((mimg->val[q][1]-tree_A[r]/nnodes[r]),2.0) + */
/* 			 powf((mimg->val[q][2]-tree_B[r]/nnodes[r]),2.0)) */
/* 		   ); */
	  
/*           tmp = iftMax(pathval->val[p], Wi); */

/*           if (tmp < pathval->val[q])  { */
/* 	    if (Q->L.elem[q].color == IFT_GRAY) */
/* 	      iftRemoveGQueueElem(Q,q); */
/*             label->val[q]    = label->val[p]; */
/*             root->val[q]     = root->val[p]; */
/*             pathval->val[q]  = tmp; */
/*             iftInsertGQueue(&Q, q); */
/*           } */
/*         } */
/*       } */
/*     } */
/*   } */


/*   iftDestroyAdjRel(&A); */
/*   iftDestroyGQueue(&Q); */
/*   iftDestroyImage(&pathval); */
/*   iftDestroyImage(&root); */
/*   iftFree(tree_L); */
/*   iftFree(tree_A); */
/*   iftFree(tree_B); */
/*   iftFree(nnodes); */

/*   return (label); */
/* } */

/* iftImage *ImageGradient(iftImage *img, iftAdjRel *A) */
/* { */
/*   iftImage *gradI = iftCreateImage(img->xsize,img->ysize,img->zsize); */
/*   float    *mag   = iftAllocFloatArray(A->n);  */
/*   float    *gx    = iftAllocFloatArray(3); */
/*   float    *gy    = iftAllocFloatArray(3); */
/*   float    *gz    = iftAllocFloatArray(3); */
  
/*   for (int i=0; i < A->n; i++) */
/*     mag[i]=sqrt(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]+A->dz[i]*A->dz[i]); */
  
/*   for (ulong p=0; p < img->n; p++){ */
/*     iftVoxel u  = iftGetVoxelCoord(img,p); */

/*     for (int b=0; b < 3; b++){ */
/*       gx[b] = 0; gy[b] = 0; gz[b] = 0; */
/*     } */

/*     for (int i=1; i < A->n; i++) { */
/*       iftVoxel v = iftGetAdjacentVoxel(A,u,i); */
/*       if (iftValidVoxel(img,v)){ */
/* 	int q = iftGetVoxelIndex(img,v);	     */
/* 	for (int b=0; b < 3; b++){ */
/* 	  gx[b] += ((float)img->val[q]-(float)img->val[p])*A->dx[i]/mag[i]; */
/* 	  gy[b] += ((float)img->Cb[q]-(float)img->Cb[p])*A->dy[i]/mag[i]; */
/* 	  gz[b] += ((float)img->Cr[q]-(float)img->Cr[p])*A->dz[i]/mag[i]; */
/* 	} */
/*       } */
/*     } */
/*     float Gx=0.0, Gy=0.0, Gz=0.0; */
/*     for (int b=0; b < 3; b++){ */
/*       gx[b] = gx[b] / (A->n-1); */
/*       gy[b] = gy[b] / (A->n-1); */
/*       gz[b] = gz[b] / (A->n-1); */
/*       Gx += gx[b]; Gy += gy[b]; Gz += gz[b];  */
/*     } */
/*     Gx /= 3; Gy /= 3; Gz /= 3; */
/*     gradI->val[p] = iftRound(sqrtf(Gx*Gx + Gy*Gy + Gz*Gz)); */
/*   } */

/*   iftFree(mag); */
/*   iftFree(gx); */
/*   iftFree(gy); */
/*   iftFree(gz); */
  
/*   return(gradI); */
/* } */

/* iftImage *Watershed(iftImage *gradI, iftImage *seeds_in, iftImage *seeds_out) */
/* { */
/*   iftImage   *pathval = NULL, *label = NULL; */
/*   iftGQueue  *Q = NULL; */
/*   int         i, p, q, tmp; */
/*   iftVoxel    u, v; */
/*   iftAdjRel     *A = iftCircular(1.0); */

/*   // Initialization */
  
/*   pathval    = iftCreateImage(gradI->xsize, gradI->ysize, gradI->zsize); */
/*   label      = iftCreateImage(gradI->xsize, gradI->ysize, gradI->zsize); */
/*   Q          = iftCreateGQueue(iftMaximumValue(gradI)+1, gradI->n, pathval->val); */

/*   /\* Initialize costs *\/ */
    
/*   for (p = 0; p < gradI->n; p++) */
/*   { */
/*     pathval->val[p] = IFT_INFINITY_INT; */
/*     if (seeds_in->val[p] != 0){ */
/*       label->val[p]   = 255; */
/*       pathval->val[p] = 0; */
/*     }else{ */
/*       if (seeds_out->val[p] != 0){ */
/* 	label->val[p] = 0; */
/* 	pathval->val[p] = 0; */
/*       } */
/*     }     */
/*     iftInsertGQueue(&Q, p); */
/*   }       */
  
/*   /\* Propagate Optimum Paths by the Image Foresting Transform *\/ */

/*   while (!iftEmptyGQueue(Q)) */
/*   { */
/*     p = iftRemoveGQueue(Q); */
/*     u = iftGetVoxelCoord(gradI, p); */

/*     for (i = 1; i < A->n; i++) */
/*     { */
/*       v = iftGetAdjacentVoxel(A, u, i); */

/*       if (iftValidVoxel(gradI, v)) */
/*       { */
/*         q = iftGetVoxelIndex(gradI, v); */

/* 	if (Q->L.elem[q].color != IFT_BLACK) { */
	  
/*           tmp = iftMax(pathval->val[p], gradI->val[q]); */

/*           if (tmp < pathval->val[q])  { */
/* 	    iftRemoveGQueueElem(Q,q); */
/*             label->val[q]    = label->val[p]; */

/*             pathval->val[q]  = tmp; */
/* 	    iftInsertGQueue(&Q,q); 	     */
/* 	  } */
/*         } */
/*       } */
/*     } */
/*   } */
  
/*   iftDestroyAdjRel(&A); */
/*   iftDestroyGQueue(&Q); */
/*   iftDestroyImage(&pathval); */

/*   return (label); */
/* } */

/* /\* Returns all components with average saliency value above a */
/*    threshold *\/ */

/* iftSet *iftSalientCompsAboveOtsu(iftImage *label, iftImage *salie) */
/* { */
/*   int T         = iftOtsu(salie); */
/*   int nlabels   = iftMaximumValue(label); */
/*   int *csalie   = iftAllocIntArray(nlabels+1); */
/*   int *nelems   = iftAllocIntArray(nlabels+1); */
/*   iftSet *SelectedComps=NULL; */

/*   for (int p = 0; p < label->n; p++) { */
/*     if (label->val[p]>0){ */
/*       if (csalie[label->val[p]]==0){ */
/* 	csalie[label->val[p]]=salie->val[p]; */
/*       }else{ */
/* 	if (csalie[label->val[p]]<salie->val[p]) */
/* 	  csalie[label->val[p]]=salie->val[p]; */
/*       } */
/*       nelems[label->val[p]]+=1; */
/*     } */
/*   } */

/*   for (int j = 1; j <= nlabels; j++){ */
/*     if (nelems[j]>0){ */
/*       if (csalie[j]>=T){ */
/*   	iftInsertSet(&SelectedComps,j); */
/*       } */
/*     } */
/*   } */
  
/*   iftFree(csalie); */
/*   iftFree(nelems); */
  
/*   return(SelectedComps); */
/* } */

/* Assuming the parasites will not touch each other, delineate
   them. If they do, then delineation must be done component by
   component. */

/* iftImage *DelineateComps(iftImage *comps, */
/* 			 iftSet *SelectedComps, */
/* 			 iftImage *orig) */
/* { */
/*   /\* Set inner markers *\/ */

/*   iftImage *seeds_in = iftCreateImage(comps->xsize,comps->ysize,comps->zsize); */
/*   for (int p=0; p < comps->n; p++){ */
/*     if (iftSetHasElement(SelectedComps, comps->val[p])){ */
/*       seeds_in->val[p] = 1;        */
/*     } */
/*   } */

/*   /\* Set outer markers *\/ */
  
/*   iftSet   *S   = NULL; */
/*   iftImage *bin = iftErodeBin(seeds_in,&S,SEED_EROSION); */
/*   iftDestroyImage(&seeds_in); */
/*   seeds_in      = bin; */
/*   bin           = iftDilateBin(seeds_in,&S,SEED_DILATION); */
/*   iftImage *seeds_out = iftComplement(bin); */
/*   iftDestroyImage(&bin); */
/*   iftDestroySet(&S); */

/*   /\* Delineate components simultaneously *\/ */

  
/*   iftImage *label = DynamicTrees(orig,seeds_in,seeds_out); */
/*   iftDestroyImage(&seeds_in); */
/*   iftDestroyImage(&seeds_out); */

/*   return(label); */
/* } */

int main(int argc, char *argv[])
{
  
  if (argc!=6){ 
    iftError("Usage: iftSMansoniDelineation <P1> <P2> <P3> <P4>\n"
	     "[1] folder to dataset (with img and label folders)\n"
	     "[2] folder with the salience maps\n"
	     "[3] layer (1,2,...) to create the results\n"
	     "[4] label of interest (1,2,...)\n"
	     "[5] output folder for resulting images and masks\n", 
	     "main");
  }
  
  timer *tstart = iftTic();

  char *filename     = iftAllocCharArray(512);
  int layer          = atoi(argv[3]);
  int label          = atoi(argv[4]);
  char suffix[12];
  sprintf(suffix,"_layer%d.png",layer);
  sprintf(filename,"%s/label%d",argv[2],label);
  iftFileSet *fs     = iftLoadFileSetFromDirBySuffix(filename, suffix, true);
  
  char output_dir1[100];
  char output_dir2[100];
  sprintf(output_dir1, "%s/%s", argv[5], "out_images");
  sprintf(output_dir2, "%s/%s", argv[5], "masks");
  char  fileout[130];
  sprintf(fileout,"%s/result_layer%d_label%d.csv", argv[5], layer, label);
  iftMakeDir(output_dir1);
  iftMakeDir(output_dir2);
  iftColorTable *ctb = iftCreateRandomColorTable(10,65535);
  iftAdjRel *B       = iftCircular(1.5);      
  iftAdjRel *C       = iftCircular(1.0);
  FILE      *fp      = fopen(fileout,"w");
  float      mscore  = 0.0;

  for(int i = 0; i < fs->n; i++) {
    printf("Processing image %d of %ld\r", i + 1, fs->n);

    char *basename1      = iftFilename(fs->files[i]->path,suffix);      
    char *basename2      = iftFilename(fs->files[i]->path,".png");      
    iftImage *salie      = iftReadImageByExt(fs->files[i]->path);
    sprintf(filename,"%s/images/%s.png", argv[1], basename1);
    iftImage  *orig      = iftReadImageByExt(filename);
    iftMImage *mimg      = iftImageToMImage(orig,LAB_CSPACE);
    sprintf(filename,"%s/truelabels/%s.png", argv[1], basename1);
    iftImage *aux1       = iftReadImageByExt(filename); 
    iftImage *gt         = iftThreshold(aux1,128,IFT_INFINITY_INT,1);
    iftDestroyImage(&aux1);
    iftImage *img        = iftCopyImage(orig);
    iftImage *label      = iftSMansoniDelineation(mimg, salie, BORDER_DIST, MIN_OBJ_AREA, MAX_OBJ_AREA, SEED_EROSION, SEED_DILATION); 

    if (iftMaximumValue(label)>0) {    
      if (iftMaximumValue(gt)==0){
	fprintf(fp,"%s;%.2f\n",basename1,0.0);
      }else{
	float score    = iftFScore(label, gt);
	fprintf(fp,"%s;%.2f\n",basename1,score);
	mscore += score;
      }
      iftDrawBorders(img, label, C, ctb->color[1], B);	
    } else {
      if (iftMaximumValue(gt)==0){
	mscore += 1.0;
	fprintf(fp,"%s;%.2f\n",basename1,1.0);
      } else {
	fprintf(fp,"%s;%.2f\n",basename1,0.0);
      }
    }
    iftDestroyImage(&salie);
    iftDestroyImage(&orig);
    iftDestroyImage(&gt);
    iftDestroyMImage(&mimg);
    
    sprintf(filename,"%s/%s.png",output_dir1,basename2);
    iftWriteImageByExt(img,filename);    
    sprintf(filename,"%s/%s.png",output_dir2,basename2);
    iftWriteImageByExt(label,filename);
    
    iftDestroyImage(&img);
    iftDestroyImage(&label);
    iftFree(basename1);
    iftFree(basename2);
  }

  iftDestroyColorTable(&ctb);
  iftFree(filename);
  iftDestroyAdjRel(&B);
  iftDestroyAdjRel(&C);
  mscore /= fs->n;
  fprintf(fp,"Average Fscore; %.2f\n",mscore);
  
  fclose(fp);
  iftDestroyFileSet(&fs);
  
  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));
  
  return (0);
}
