
#include "ift.h"
#include <float.h>
/* Author: Marcelo Rodrigues (May 20th, 2025)

   Title: Adaptive Decoder-Based Marker-Relaxed Dynamic Trees (ADMRDT)
   Description: Delineates objects (e.g., parasite eggs) from the
   decoded saliency maps with propagation of saliency maps and
   Malmberg (10.1117/12.840019) relaxation procedure.

   Last update: May 20th, 2025
*/

#define BORDER_DIST 1    // 1
#define SEED_DILATION 30 // 30 // 15 // 19
#define SEED_EROSION 1   // 1
#define MAX_OBJ_AREA 9000//12000 (entamoeba)
#define MIN_OBJ_AREA 1000//100 (entamoeba)



int main(int argc, char *argv[])
{
 
  if (argc < 5 || argc > 10)
  {
    {
      iftError("Usage: '/bin/iftSMansoniDelineationWithADMRDT '/SCHISTO/split1/test' 'saliences/' '2' '2' 'results_admrdt' '2' '30' '1' '0.001' \n"
               "[1] folder to dataset (with img and label folders)\n"
               "[2] folder with the salience maps\n"
               "[3] layer (1,2,...) to create the results\n"
               "[4] label of interest (1,2,...)\n"
               "[5] output folder for resulting images and masks\n"
               "[6] (Optional) relaxation iterations between 2 to N (integer). Default is 2.  \n"
               "[7] (Optional) dilation factor of seeds markers (integer). Default is 30.\n"
               "[8] (Optional) erosion factor of seeds markers (integer). Default is 1.\n"
               "[9] (Optional) alpha confidence used to combine the label map of Dynamic Trees/IFT with "
                               " the propagated saliency map of Adaptive Decoder (float in range [0,1]).  "
                               "The higher the alpha, the more weight is given to the propagated saliency values in fusion rule. "
                               "Otherwise, the retrict label value in Dynamic Trees becomes dominant." 
                               " Default is 0.001.\n",
               "main");
    }
  }
  timer *tstart = iftTic();

  char *filename = iftAllocCharArray(512);
  int layer = atoi(argv[3]);
  int label = atoi(argv[4]);
  char suffix[12];
  sprintf(suffix, "_layer%d.png", layer);
  sprintf(filename,"%s/label%d",argv[2],label);

  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(filename, suffix, true);

  char output_dir1[500];
  char output_dir2[500];
  sprintf(output_dir1, "%s/%s", argv[5], "out_images");
  sprintf(output_dir2, "%s/%s", argv[5], "masks");
  char fileout[500];
  //sprintf(fileout, "%s/result_layer%d_label%d.csv", argv[5], layer, label);
  iftMakeDir(output_dir1);
  iftMakeDir(output_dir2);
  iftColorTable *ctb = iftCreateRandomColorTable(10, 65535);
  iftAdjRel *B = iftCircular(1.5); 
  iftAdjRel *C = iftCircular(1.0); 
  //FILE *fp = fopen(fileout, "w");

  int relaxation_iterations = (argc > 6) ? atoi(argv[6]) : 0;

  if (relaxation_iterations != 0)
  {
    printf("[INFO] Relaxation iterations set to %d.\n", relaxation_iterations);
  } else {
    relaxation_iterations = 1;
  }

  int dilation_factor = (argc > 7) ? atoi(argv[7]) : SEED_DILATION;

  if (dilation_factor >= 0)
  {
    printf("[INFO] Dilation factor set to %d.\n", dilation_factor);
  }

  int erosion_factor = (argc > 8) ? atoi(argv[8]) : SEED_EROSION;

  if (erosion_factor >= 0)
  {
    printf("[INFO] Erosion factor set to %d.\n", erosion_factor);
  }

  float alpha_confidence = (argc > 9) ? atof(argv[9]) :  0.001;
  if (alpha_confidence >= 0)
  {
    printf("[INFO] Alpha confidence value of fusion rule set to %f.\n", alpha_confidence);
  }


  for (int i = 0; i < fs->n; i++)
  {
    printf("Processing image %d of %ld\r", i + 1, fs->n);
    
    char *basename1 = iftFilename(fs->files[i]->path, suffix);
    char *basename2 = iftFilename(fs->files[i]->path, ".png");
    iftImage *salie = iftReadImageByExt(fs->files[i]->path);
    
    sprintf(filename, "%s/images/%s.png", argv[1], basename1);
    iftImage *orig = iftReadImageByExt(filename);
    iftMImage *mimg = iftImageToMImage(orig, LAB_CSPACE);
    sprintf(filename, "%s/truelabels/%s.png", argv[1], basename1);
    iftImage *aux1 = iftReadImageByExt(filename);
   // iftImage *gt = iftBinarize(aux1);
    iftDestroyImage(&aux1);
    iftImage *img = iftCopyImage(orig);

    
    iftFImage *result_saliency = NULL;
    iftImage *label = iftCreateImage(salie->xsize, salie->ysize, salie->zsize); // assume that the decoder saliency has the same shape as the input image
    
    result_saliency = iftSMansoniDelineationADMRDT(
        mimg, 
        salie, 
        BORDER_DIST, 
        MIN_OBJ_AREA, 
        MAX_OBJ_AREA, 
        erosion_factor, 
        dilation_factor,
        relaxation_iterations, 
        alpha_confidence
    );
    
    // normalize saliency in range 0 and 1 before multiply by 255 for visualization purpose and 
    // allow consistency with modern Salient Object Detection metrics computation
    if (iftFMaximumValue(result_saliency) > 0)
    {
    
      Min_Max_Normalize(result_saliency, 255, false); // this ensure 8-bit for visualization
     
    }

   
    
    for (int p = 0; p < label->n; p++) 
    {
        
      
        label->val[p] = (int)result_saliency->val[p]; // 255
 
    }
 
    // ensure ground truth with unique values 0 and 255
   /* for (int p = 0; p < gt->n; p++) 
    {
      if (gt->val[p] > 0)
      {
        gt->val[p] = 255;
      }
      else
      {
        gt->val[p] = 0;
      }
    }*/
    
    iftDestroyFImage(&result_saliency);

    if (iftMaximumValue(label) > 0)
    {
      
      iftDrawBorders(img, label, C, ctb->color[1], B);
    }

    iftDestroyImage(&salie);
    iftDestroyImage(&orig);
   // iftDestroyImage(&gt);
    iftDestroyMImage(&mimg);

    sprintf(filename, "%s/%s.png", output_dir1, basename2);
    iftWriteImageByExt(img, filename);
    sprintf(filename, "%s/%s.png", output_dir2, basename2);
    iftWriteImageByExt(label, filename);

   
    iftDestroyImage(&img);
    iftDestroyImage(&label);

    iftFree(basename1);
    iftFree(basename2);
  }

  iftDestroyColorTable(&ctb);
  iftFree(filename);
  iftDestroyAdjRel(&B);
  iftDestroyAdjRel(&C);

  //fclose(fp);
  iftDestroyFileSet(&fs);


  printf("\nDone ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

  return (0);
}

