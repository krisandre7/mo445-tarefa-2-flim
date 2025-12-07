#include "ift.h"

#define ReductionFactor 0.5

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

int iftSimpleForwardProject(
   unsigned short *proj,             //projection data
   int numProj,                      //number of projections
   int nSignalXPts,                  //pixels detector
   int nSignalYPts,                  //lines detector
   double dSignalXInc,               //pixel spacing
   double dSignalYInc,               //line spacing
   double AOR,			     //column on detector that corresponds to the axis of rotation 
   double midPlane,                  //row on detector that corresponds to the midplane
   double dFocalLength,              //source to actual AOR distance
   double dSourceDetectorLength,     //source to detector distance
   double theta[356],                //angle data
   unsigned char *vox,               //slice data
   int numSlices,                    //number of slices,
   int pixels,			     //cols per slice
   int lines,                        //rows per slice
   double subSample                  //sub sampling factor
			    )
{
	
  int iProj, iSlice, iX, iY, xx, yy, j, size;
	
  int subSampleX = (int)subSample;
  int subSampleY = (int)subSample;
  int subSampleZ = (int)subSample;
  
  // scale signalInc  to adjust for imaginary detector through origin of phantom 
  dSignalXInc *= dFocalLength / dSourceDetectorLength;
  dSignalYInc *= dFocalLength / dSourceDetectorLength;
	
  //center in pixel coordinates
  double centerX = (double)AOR;         
  double centerY = (double)midPlane;
  double centerZ = centerY;
  double P[3][4]; // projection matrix
  double Vx[3], Vy[3], Vz[3], PM0[3], M0[3];
  double p, eta, con;
  int lo_p, lo_eta;
  unsigned short *pProj, *pLength, *p1;
  unsigned char  *pVox;
  double ccos, ssin;
  double dFocalLength_pix = dFocalLength / dSignalXInc; //focal length in pixel units
  
  int nProj = numProj;
  
  
  printf("\n");

  for(iProj = 0; iProj < nProj; iProj++)
    {
      printf("\rprojection # %d",iProj);
      fflush(stdout);
      ccos = cos(theta[iProj]);
      ssin = sin(theta[iProj]);
      
      //calculate projection matrix for given view angle
      centerY = centerX;
      
      P[0][0] = AOR * ccos + dFocalLength_pix * ssin * subSampleX;
      P[0][1] = -AOR * ssin + dFocalLength_pix * ccos * subSampleY;
      P[0][2] = 0.0;
      P[0][3] = dFocalLength_pix * AOR;
      P[1][0] = centerZ * ccos;
      P[1][1] = -centerZ * ssin;
      P[1][2] = dFocalLength_pix * subSampleZ;
      P[1][3] = dFocalLength_pix * centerZ;
      P[2][0] = ccos;
      P[2][1] = -ssin;
      P[2][2] = 0.0;
      P[2][3] = dFocalLength_pix;
      
      pProj = proj + iProj * nSignalYPts * nSignalXPts;
      pVox = vox;
      
      M0[0] = -0.5 * (double)(pixels-1);
      M0[1] = -0.5 * (double)(lines-1);
      M0[2] = -0.5 * (double)(numSlices-1);
      
      PM0[0] = P[0][0] * M0[0] + P[0][1] * M0[1] + P[0][2] * M0[2] + P[0][3];
      PM0[1] = P[1][0] * M0[0] + P[1][1] * M0[1] + P[1][2] * M0[2] + P[1][3];
      PM0[2] = P[2][0] * M0[0] + P[2][1] * M0[1] + P[2][2] * M0[2] + P[2][3];
      
      Vz[0] = PM0[0];
      Vz[1] = PM0[1];
      Vz[2] = PM0[2];
      
      for(iSlice = 0; iSlice < numSlices; iSlice++) //loop over slices of reconstructed image
	{
	  Vy[0] = Vz[0];
	  Vy[1] = Vz[1];
	  Vy[2] = Vz[2];
	  for(iY = 0; iY < lines; iY++) //loop over lines of reconstructed image
	    {
	      Vx[0] = Vy[0];
	      Vx[1] = Vy[1];
	      Vx[2] = Vy[2];
	      for(iX = 0; iX < pixels; iX++)  //loop over pixes of reconstructed image
		{						
		  con = 1.0 / Vx[2];
		  //determine p (in pixels) location in distance on virtual detector for the ray going thru pt(x,y,z)
		  p = Vx[0] * con;
		  //determine eta location in distance on virtual detector for the ray going thru pt(x,y,z)
		  eta = Vx[1] * con;						
		  lo_p = (int)(p + 1.0) - 1;
		  lo_eta = (int)(eta + 1.0) - 1;
		  if(lo_p >= 0 && lo_p < nSignalXPts-1 && lo_eta >= 0 && lo_eta < nSignalYPts-1 && *pVox > 0)
		    pProj[lo_eta*nSignalXPts + lo_p] += 1;
		  pVox++;
		  
		  Vx[0] += P[0][0];
		  Vx[1] += P[1][0];
		  Vx[2] += P[2][0];
		}
	      Vy[0] += P[0][1];
	      Vy[1] += P[1][1];
	      Vy[2] += P[2][1];
	    }
	  Vz[0] += P[0][2];
	  Vz[1] += P[1][2];
	  Vz[2] += P[2][2];
	}
      
    }
  printf("\n");

  return(1);
}

iftImage *iftProjectSeeds(iftImage *seeds, char *geometry_file)
{
  iftJson *json    = iftReadJson(geometry_file);
  int    xsizeProj = iftGetJInt(json,"xsizeProj");
  int    ysizeProj = iftGetJInt(json,"ysizeProj");
  int    numProj   = iftGetJInt(json,"numProj");
  double dXInc   = iftGetJDouble(json,"dXInc")*(1.0/ReductionFactor); 
  double dYInc   = iftGetJDouble(json,"dYInc")*(1.0/ReductionFactor);
  double dFocalLength = iftGetJDouble(json,"dFocalLength");
  double dSourceDetectorLength = iftGetJDouble(json,"dSourceDetectorLength");	
  double AOR       = iftGetJDouble(json,"AOR") * (xsizeProj-1);
  double midPlane  = iftGetJDouble(json,"midPlane") * (ysizeProj-1);      
  char  *anglefile = iftGetJString(json,"anglefile");
  double *angle    = iftAnglesOfProjections(anglefile,numProj);
  unsigned short *uproj = iftAllocUShortArray(xsizeProj*ysizeProj*numProj);
  unsigned char  *cvox  = iftAllocUCharArray(seeds->n);
  iftImage *proj; 

  for (int p = 0; p < seeds->n; p++)
    cvox[p] = seeds->val[p];

  iftSimpleForwardProject(uproj, numProj, xsizeProj, ysizeProj, dXInc, dYInc, AOR, midPlane, dFocalLength, dSourceDetectorLength, angle, cvox, seeds->zsize, seeds->xsize, seeds->ysize, 1.0);

  free(cvox);
  free(angle);
  iftDestroyJson(&json);

  proj = iftCreateImage(xsizeProj,ysizeProj,numProj);

  for (int p = 0; p < seeds->n; p++)
    proj->val[p] = uproj[p];
 
  free(uproj);

  return(proj);
}

int main(int argc, char *argv[])
{
  iftImage *rec; 
  iftImage *mask, *marker, *aux[2], *seeds[2];

  if (argc != 3) 
    iftError("Usage: %s <reconstruction.scn> <geometry-reconstruction.json>", "main", argv[0]);

  /* Estimate internal seeds */

  rec         = iftReadImage(argv[1]);
  marker      = iftThreshold(rec, iftCumHistogramThres(rec,0.998),4095,1);
  mask        = iftThreshold(rec, iftCumHistogramThres(rec,0.995),4095,1);
  seeds[0]    = iftInfRec(mask,marker);
  iftDestroyImage(&mask);
  iftDestroyImage(&marker);
  aux[0]      = iftComplement(seeds[0]);
  aux[1]      = iftMask(rec,aux[0]);
  marker      = iftThreshold(aux[1], iftCumHistogramThres(rec,0.995),4095,1);
  mask        = iftThreshold(aux[1], iftCumHistogramThres(rec,0.993),4095,1);
  iftDestroyImage(&aux[1]);
  aux[1]      = iftOr(mask,seeds[0]);
  iftDestroyImage(&mask);
  seeds[1]    = iftInfRec(aux[1],marker);
  iftDestroyImage(&aux[0]);
  iftDestroyImage(&aux[1]);
  iftDestroyImage(&marker);
  iftDestroyImage(&seeds[0]);


  iftWriteImage(seeds[1],"inner_seeds.scn");

  iftDestroyImage(&rec);

  /* aux[0]   = iftProjectSeeds(seeds[1],argv[2]); */
  /* seeds[0] = iftThreshold(aux,iftOtsu(aux),IFT_INFINITY_INT,1); */
  /* iftDestroyImage(&aux[0]); */
  /* iftWriteImage(seeds[0],"inner_seeds_proj.scn"); */
  /* iftDestroyImage(&seeds[0]); */
  /* iftDestroyImage(&seeds[1]); */
  return(0);
}




















