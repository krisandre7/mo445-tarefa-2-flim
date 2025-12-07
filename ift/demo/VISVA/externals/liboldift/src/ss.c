#include "ss.h"

/**********************************************/
/* SS - Segment Saliences                     */
/**********************************************/


/***************SS EXTRACTION ALGORITHM*********************/

Curve *SS_ExtractionAlgorithm(Image *in, int maxdist, int nsamples, int side){
  Curve *inner = NULL;
  Curve *outer = NULL;
  Curve *diff = NULL;
  Curve *ninner = NULL;
  Curve *nouter = NULL;
  Curve *ndiff = NULL;
  Curve *output = NULL;

  Image *mbb = NULL;
  Image *bin = NULL;
  Image *contour = NULL;
  Image *segments = NULL;
  
  AdjRel *A=NULL;
  AnnImg *aimg= NULL;
  
  int p,i,Lmax, maxcost = maxdist*maxdist;
  double nin, nout, maxin, maxout;

  mbb  = MBB(in);
  bin = AddFrame(mbb,maxdist,0);

  DestroyImage(&mbb);
  
  segments = LabelContPixel(bin);

  /* Compute Euclidean IFT */
  contour    = LabelContPixel(bin);
  
  aimg    = Annotate(bin,NULL,contour); 
  A       = Circular(1.5);
  iftDilation(aimg,A);  
  
  Lmax    = MaximumValue(aimg->label);
  //printf("Lmax = %d\n", Lmax);
  inner   = CreateCurve(Lmax);
  outer   = CreateCurve(Lmax);
  diff    = CreateCurve(Lmax);
  
  for (i=0; i<Lmax; i++){
    diff->X[i] = inner->X[i] = outer->X[i]= (double)(i*nsamples)/Lmax;
  }  
  
  /* Compute influence areas */  
  nin = nout = 0.0;
  for (p=0; p < bin->ncols*bin->nrows; p++){
    if (segments->val[p] != 0){
      segments->val[p]=((((segments->val[p]*nsamples)/Lmax))/*%2*/)+1;
    }
    if ((aimg->label->val[p] > 0)&&(aimg->cost->val[p] <= maxcost)) {
      if (aimg->img->val[p] != 0){
	nin++;
	inner->Y[aimg->label->val[p]-1]++;
      } else {
	nout++;
	outer->Y[aimg->label->val[p]-1]++;
      }
    }
  }
  
  maxin = INT_MIN;
  maxout = INT_MIN;
  for (i=0; i<Lmax; i++){
    if (inner->Y[i] > maxin){
      maxin = inner->Y[i];
    }
    if (outer->Y[i] > maxout){
      maxout = outer->Y[i];
    }
  }
  
  for (i=0; i<Lmax; i++){
    inner->Y[i] /= nin;
    outer->Y[i] /= nout;
    diff->Y[i] = outer->Y[i] - inner->Y[i];
  }
  
  ninner   = CreateCurve(nsamples);
  nouter   = CreateCurve(nsamples);
  ndiff    = CreateCurve(nsamples);
  
  for (i=0; i<nsamples; i++){
    ninner->X[i] = nouter->X[i] = ndiff->X[i] = i;
  }
  for (i=0; i<Lmax; i++){
    ninner->Y[(int)inner->X[i]] += inner->Y[i];
    nouter->Y[(int)outer->X[i]] += outer->Y[i];
  }
  for (i=0; i<nsamples; i++){
    ndiff->Y[i] =  nouter->Y[i] - ninner->Y[i];
  }
  
  
  if (side == INTERIOR){
    output = CopyCurve(ninner);
  }
  else if (side==EXTERIOR){
    output = CopyCurve(nouter);
  }
  else if (side == BOTH){
    output = CopyCurve(ndiff);
  }
  else{
    printf("Invalid \"side\" option <%d>\n", side);
    exit(-1);
  }
  
  DestroyImage(&segments);
  DestroyCurve(&ninner);
  DestroyCurve(&nouter);
  DestroyCurve(&ndiff);

  DestroyImage(&contour);
  DestroyAdjRel(&A);
  DeAnnotate(&aimg);

  DestroyImage(&bin);
  DestroyImage(&mbb);
  DestroyCurve(&inner);
  DestroyCurve(&outer);
  DestroyCurve(&diff);

  return output;
}
/************************************************************/



/***************SS SIMILARITY ALGORITHM*********************/
double SS_getMin(double Dist1, double Dist2, double Dist3){
  if((Dist1<=Dist2) && (Dist1<=Dist3)) 
    return(Dist1);
  else if((Dist2<=Dist1) && (Dist2<=Dist3)) 
    return(Dist2);
  //else if((Dist3<=Dist1) && (Dist3<=Dist2)) 
  return(Dist3);
}

double SS_OCS(FeatureVector1D *fv1, FeatureVector1D *fv2){
  
  int i,j, dim1 = fv1->n, dim2 = fv2->n;
  double temp_dist;
  double penalty;
  double *DISTANCE = NULL;
  
  DISTANCE=(double *) calloc((dim1+1)*(dim2+1),sizeof(double));

  penalty=20.0;
  /* OPTIMAL CORRESPONDENCE OF STRINGS
   */
  DISTANCE[0*(dim2+1)+0]=0;
  for(j=1;j<=dim2;j++)
    DISTANCE[0*(dim2+1)+j]=j * penalty;
  
  for(i=1;i<=dim1;i++)
    DISTANCE[i*(dim2+1)+0]=i * penalty;
  
  for(i=1;i<=dim1;i++)
    for(j=1;j<=dim2;j++)
      if(abs(i-j) < (5)){
	temp_dist=abs(fv1->X[i-1]-fv2->X[j-1]);
		
	DISTANCE[i*(dim2+1)+j]= temp_dist +
	  SS_getMin(DISTANCE[(i-1)*(dim2+1)+(j-1)],
		 DISTANCE[(i-1)*(dim2+1)+(j)] + penalty,
		 DISTANCE[(i)*(dim2+1)+(j-1)] + penalty); 
      }
  
  temp_dist = DISTANCE[(dim1)*(dim2+1)+(dim2)]/dim2;
  free(DISTANCE);
  
  return temp_dist;
}

double SS_OCSMatching(FeatureVector1D *fv_1, FeatureVector1D *fv_2){
  double distance,temp_dist;
  int i,k;
  FeatureVector1D *temp1, *temp2, *fv1, *fv2;
  
  fv1 = CreateFeatureVector1D(fv_1->n);
  fv2 = CreateFeatureVector1D(fv_2->n);
  for (i = 0; i<fv1->n; i++){
    fv1->X[i] = 100*fv_1->X[i];
    fv2->X[i] = 100*fv_2->X[i];
  }
  
  temp1 = CreateFeatureVector1D(fv2->n);
  temp2 = CreateFeatureVector1D(fv2->n);
  
  temp_dist=INT_MAX; 
  for(k=0; k<fv2->n; k++){
    for(i=0;i<fv2->n;i++){
      temp2->X[i]=fv2->X[(i+k)%fv2->n];
    }   
    distance= SS_OCS(fv1,temp2);
    if(temp_dist>distance) 
      temp_dist=distance;
  }
  /***Taking the mirror of fv2 *****/
  for(i=0;i<fv2->n;i++){
    temp2->X[i]=fv2->X[(fv2->n-1)-i];
  }
  
  for(k=0; k<fv2->n; k++){
    for(i=0;i<fv2->n;i++){
      temp1->X[i]= temp2->X[(i+k)%fv2->n];
    }
    distance=SS_OCS(fv1,temp1);
    if(temp_dist>distance) 
      temp_dist=distance;
  }
  
  distance=temp_dist;
  DestroyFeatureVector1D(&temp1);
  DestroyFeatureVector1D(&temp2);
  DestroyFeatureVector1D(&fv1);
  DestroyFeatureVector1D(&fv2);
  return(distance);
}


double SS_SimilarityAlgorithm(FeatureVector1D *fv1d1, FeatureVector1D *fv1d2){
  
  double dist = 0;

  dist = SS_OCSMatching(fv1d1, fv1d2);
    
  return dist;
}
/************************************************************/
