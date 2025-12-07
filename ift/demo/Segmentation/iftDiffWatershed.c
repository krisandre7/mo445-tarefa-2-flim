#include "ift.h"

void help()
{
  printf("\n");
  printf("iftDiffWatershed image[scn|ppm|pgm] adjacency_radius smooth_iterations[0|1..N] output[scn|pgm] markers1 [markers2]\n\n");

  printf("Examples:\n");
  printf("\tiftDiffWatershed img.scn 1.73 5 result.scn seeds1.txt\n");
  printf("\tiftDiffWatershed img.scn 1.73 5 result.scn seeds1.txt seeds2.txt\n");
  printf("\tiftDiffWatershed img.pgm 1.41 15 result.pgm seeds1.txt seeds2.txt\n");
  printf("\n");
  iftError("Please check your program call.", "iftDiffWatershed");
}
void writeAppropriateImage(iftImage * img, char *filename)
{
  printf("Segmentation saved in %s\n", filename);
  iftWriteImageByExt(img, filename);
}


int main(int argc, char *argv[])
{
  iftImage        *img             = NULL;
  iftImage        *basins          = NULL;
  iftImageForest  *fst             = NULL;
  iftLabeledSet   *seed1           = NULL;
  iftLabeledSet   *seed2           = NULL;
  iftAdjRel       *A               = NULL;
  iftSmoothBorder *smooth          = NULL;
  iftSet          *removal_markers = NULL;
  timer           *t1, *t2;
  char *output_filename            = NULL;
  int   smooth_iterations          = 0;

  if (argc != 6 && argc != 7)
    help();

  img = iftReadImageByExt(argv[1]);

  if (iftIs3DImage(img))
    A   = iftSpheric(atof(argv[2]));
  else
    A   = iftCircular(atof(argv[2]));

  smooth_iterations = atoi(argv[3]);
  output_filename   = argv[4];

  seed1  = iftReadSeeds(img, argv[5]);
  basins = iftImageBasins(img, A);
  fst    = iftCreateImageForest(basins, A);
  smooth = iftCreateSmoothBorder(fst->img, fst->A, smooth_iterations, 0.5);

  removal_markers = iftExtractRemovalMarkers(&seed1);
  fprintf(stdout, "Executing iftDiffWatershed, seed set 1\n");
  fprintf(stdout, "----------------------------------------\n");
  t1 = iftTic();
  iftDiffWatershed(fst, seed1, removal_markers);
  t2 = iftToc();
  fprintf(stdout, "Segmentation time: %.2f s\n", iftCompTime(t1, t2) / 1000);

  if (smooth_iterations > 0)
  {
    t1 = iftTic();
    iftRelaxObjects(fst, smooth);
    t2 = iftToc();
    fprintf(stdout, "Smoothing time: %.2f s\n", iftCompTime(t1, t2) / 1000);
    fprintf(stdout, "----------------------------------------\n");
  }

  /* if (!iftIsSegmentationConsistent(fst)) */
  /*   iftError("Relaxation caused inconsistency","main"); */
  

  /* iftColor RGB, YCbCr; */
  /* iftAdjRel *B          = iftCircular(0.0); */
  /* RGB.val[0] = 255; */
  /* RGB.val[1] = 0; */
  /* RGB.val[2] = 0; */
  /* YCbCr      = iftRGBtoYCbCr(RGB); */
  /* iftImage *aux  = iftCopyImage(img); */
  /* iftDrawBorders(aux,fst->label,A,YCbCr,B); */
  /* iftWriteImageP6(aux,"regions.ppm"); */

  writeAppropriateImage(fst->label, output_filename);

  if (argc == 7)
  {
    seed2 = iftReadSeeds(img, argv[6]);

    removal_markers = iftExtractRemovalMarkers(&seed2);
    t1 = iftTic();
    fprintf(stdout, "\nExecuting iftDiffWatershed, seed set 2\n");
    fprintf(stdout, "--------------------------------------\n");
    iftDiffWatershed(fst, seed2, removal_markers);
    t2 = iftToc();
    fprintf(stdout, "Segmentation time [2]: %.2f s\n", iftCompTime(t1, t2) / 1000);

    if (smooth_iterations > 0)
    {
      t1 = iftTic();
      iftRelaxObjects(fst, smooth);
      t2 = iftToc();
      fprintf(stdout, "Smoothing time [2]: %.2f s\n", iftCompTime(t1, t2) / 1000);
      fprintf(stdout, "----------------------------------------\n");
    }
    writeAppropriateImage(fst->label, output_filename);
  }


  iftDestroyImageForest(&fst);
  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);
  iftDestroyImage(&basins);
  iftDestroySmoothBorder(&smooth);

  return (0);
}
