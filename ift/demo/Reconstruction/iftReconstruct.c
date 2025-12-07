#include "ift.h"

#define ReductionFactor 0.5
#define CropSize 5
#define LogMaxValue 8000.0 
#define RotorSlewVelocity 30.0
#define CroppingRadius 10.0

iftImage *iftReduceXYResolution(iftImage *proj)
{
  iftImage *reduced_proj; 
  float     sx = ROUND(ReductionFactor*(float)proj->xsize)/(float)proj->xsize;
  float     sy = ROUND(ReductionFactor*(float)proj->ysize)/(float)proj->ysize;
  
  reduced_proj = iftInterp(proj,sx,sy,1.0);

  return(reduced_proj);
}

iftImage *iftCropColimator(iftImage *proj)
{ 
  iftVoxel     uo, uf; 
  iftImage    *cropped_proj;
  int left   = CropSize; 
  int right  = (proj->xsize - CropSize); 
  int top    = CropSize; 
  int bottom = (proj->ysize - CropSize); 

  uo.x=left;
  uo.y=top;
  uo.z=0;
  uf.x=right;
  uf.y=bottom;
  uf.z=(proj->zsize -1);

  cropped_proj = iftExtractROI(proj,uo,uf);

  return(cropped_proj);
}

iftFImage *iftApplyLogOnProjections(iftImage *proj)
{
  iftFImage *log_proj = iftCreateFImage(proj->xsize, proj->ysize, proj->zsize);
  
  for (int p = 0; p < log_proj->n; p++)	
    log_proj->val[p] = LogMaxValue*log(65535.0f/(float)(proj->val[p] + 0.1f));

  return(log_proj);
}

/* Read and compute angles of projections to radians */

double *iftAnglesOfProjections(char *file_with_angles, int num_of_proj)
{
  double *angle = iftAllocDoubleArray(num_of_proj);
  FILE   *fp    = fopen(file_with_angles,"r");

  for(int i=0; i < num_of_proj; i++){
    if (fscanf(fp,"%lf",&angle[i])!=1) iftError("Reading error","iftAnglesOfProjections");
    angle[i] = angle[i] * (IFT_PI / 180.0);
  }
  fclose(fp);

  return(angle);
}

/* Compute the reconstruction weight function */

double *iftRecWeightFunction(double *angle, int num_of_proj)
{
  double *weight = iftAllocDoubleArray(num_of_proj);
  double  rotorVelocity;

  for ( int projIndex = 0; projIndex < num_of_proj; projIndex++ )
    {
      if ( projIndex == 0 )
	rotorVelocity = RotorSlewVelocity * 180.0 / IFT_PI * 
	  fabs( angle[projIndex] - angle[projIndex + 1] );		
      else if ( projIndex < num_of_proj - 1 )
	rotorVelocity = (RotorSlewVelocity/2.0) * 180.0 / IFT_PI * 
	  fabs( angle[projIndex - 1] - angle[projIndex + 1] );
      else 
	rotorVelocity = RotorSlewVelocity * 180.0 / IFT_PI * 
	  fabs( angle[projIndex - 1] - angle[projIndex] );

      weight[projIndex] = ( rotorVelocity / RotorSlewVelocity ) * 
	1.0/ReductionFactor;
    }

  return(weight);
}

/* Apply weights to the log of the projections */

double *iftWeightLogProjections(iftFImage *log_proj, double *weight)
{
  double *weighted_proj = iftAllocDoubleArray(log_proj->n);

  for (int p = 0; p < log_proj->n; p++){
    weighted_proj[p] = log_proj->val[p] * weight[iftGetZCoord(log_proj,p)];
  }

  return(weighted_proj);
}

iftFImage *iftCircularCropping(double *voxel, int xsize, int ysize, int zsize)
{
  iftFImage *frec = iftCreateFImage(xsize, ysize, zsize);
  float xcenter = xsize/2.0;
  float ycenter = ysize/2.0;
  float cmargin = 0.0;
  float radius_square = (xcenter-cmargin)*(xcenter-cmargin);
  float dist_square;
  for (int p=0; p < frec->n; p++){
    iftVoxel u = iftFGetVoxelCoord(frec,p);
    dist_square = pow(((float)u.x-xcenter),2)+pow(((float)u.y-ycenter),2);
    if (dist_square > radius_square || voxel[p] < CroppingRadius)
      frec->val[p] = 0.0;	
    else
      frec->val[p] = voxel[p];
  }
  
  return(frec);
}

iftImage *iftReconstruct(iftFImage *log_proj, char *geometry_file)
{
  iftJson *json  = iftReadJson(geometry_file);
  double dXInc   = iftGetJDouble(json,"dXInc")*(1.0/ReductionFactor); 
  double dYInc   = iftGetJDouble(json,"dYInc")*(1.0/ReductionFactor);
  double dFocalLength = iftGetJDouble(json,"dFocalLength");
  double dSourceDetectorLength = iftGetJDouble(json,"dSourceDetectorLength");	
  double AOR       = iftGetJDouble(json,"AOR") * (log_proj->xsize-1);
  double midPlane  = iftGetJDouble(json,"midPlane") * (log_proj->ysize-1);      
  int    xsize    = iftGetJInt(json,"xsize");
  int    ysize     = iftGetJInt(json,"ysize");
  int    zsize    = iftGetJInt(json,"zsize");
  int    bWindow   = iftGetJInt(json,"bWindow");
  char  *anglefile = iftGetJString(json,"anglefile");
  double *angle    = iftAnglesOfProjections(anglefile,log_proj->zsize);
  double *weight   = iftRecWeightFunction(angle,log_proj->zsize);
  double *weighted_proj = iftWeightLogProjections(log_proj,weight);
  double *voxel    = iftAllocDoubleArray(xsize*ysize*zsize);

  iftDestroyJson(&json);

  /* Reconstruct voxel array from projection array */
  
  iftFilteredBackProjection(weighted_proj, log_proj->zsize, log_proj->xsize, log_proj->ysize, dXInc, dYInc, \
			    AOR, midPlane, dFocalLength, dSourceDetectorLength, angle, voxel, zsize, xsize, ysize, 1.0, 1.0, bWindow);  

  free(weighted_proj);
  free(angle);
  free(weight);


  /* Apply 2D circular cropping */

  iftFImage *frec = iftCircularCropping(voxel, xsize, ysize, zsize);

  /* Convert image to 12 bits */

  iftImage *rec  = iftFImageToImage(frec, 4095);
  iftDestroyFImage(&frec);

  free(voxel);

  return(rec);
}

int main(int argc, char *argv[])
{
  iftImage  *proj, *aux, *rec;
  iftFImage *log_proj, *faux;

  if (argc != 4) {
    printf("Usage: %s projections.scn reconstruction.scn geometry-reconstruction.json\n", argv[0]);
    exit(1);
  }

  /* Read and reduce original image of projections */

  /* proj = iftReadImage(argv[1]);  */
  /* aux = iftReduceXYResolution(proj); */
  /* iftDestroyImage(&proj); */
  /* proj = aux; */

  /* Crop colimator from projections */

  /* aux = iftCropColimator(proj); */
  /* iftDestroyImage(&proj); */
  /* proj = aux; */

  proj = iftReadImage(argv[1]);

  /* Apply log on projections */

  faux = iftApplyLogOnProjections(proj);
  iftDestroyImage(&proj);
  log_proj = faux;

  /* Reconstruct image from projections */

  rec = iftReconstruct(log_proj, argv[3]);
  iftDestroyFImage(&log_proj);

  iftWriteImage(rec,argv[2]);
  iftDestroyImage(&rec);

  return(0);
}
