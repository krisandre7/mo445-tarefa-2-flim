#include "ift.h"

typedef struct _em {
  double **mean;
  double ***covarianceMatrix;
  double *prior;
  double **posterior;
  double **w;
  int nclusters;
  int nnodes;   
  int nfeats;
} EM;


EM *CreateEM(Subgraph *sg){
   
	int i,j;
	EM *em=(EM *)calloc(1,sizeof(EM));	

	em->mean = (double **) calloc(sg->nlabels,sizeof(double *));  
	for (i=0; i < sg->nlabels; i++) 
		em->mean[i] = (double *)calloc(sg->nfeats,sizeof(double));		

	em->covarianceMatrix = (double ***) calloc(sg->nlabels,sizeof(double **)); 
	for (i=0; i < sg->nlabels; i++) 
	{
		em->covarianceMatrix[i] = (double **)calloc(sg->nfeats,sizeof(double*));		
		for (j=0; j<sg->nfeats; j++)
			em->covarianceMatrix[i][j] = (double *)calloc(sg->nfeats,sizeof(double));		
	}

	em->prior = (double *) calloc(sg->nlabels,sizeof(double ));

	em->posterior = (double **) calloc(sg->nnodes,sizeof(double *));  
	for (i=0; i < sg->nnodes; i++) 
		em->posterior[i] = (double *)calloc(sg->nlabels,sizeof(double));		

	
	em->w = (double **) calloc(sg->nnodes,sizeof(double *));  
	for (i=0; i < sg->nnodes; i++) 
		em->w[i] = AllocDoubleArray(sg->nlabels);		


	em->nclusters = sg->nlabels;   

	em->nnodes = sg->nnodes;   

	em->nfeats = sg->nfeats;   
	
	return em;
}

void DestroyEM(EM **em){

	int i,j;
	if((*em) != NULL){
	
		for (i=0; i < (*em)->nclusters; i++) 
			free((*em)->mean[i]);			
		free((*em)->mean);
	
	
		for (i=0; i < (*em)->nclusters; i++) 		
			for (j=0; j<(*em)->nfeats; j++)
					free((*em)->covarianceMatrix[i][j]);						
	
		for (i=0; i < (*em)->nclusters; i++) 		
			free((*em)->covarianceMatrix[i]);			
		free((*em)->covarianceMatrix);			


		free((*em)->prior);	

		for (i=0; i < (*em)->nnodes; i++) {
			free((*em)->posterior[i]);
			free((*em)->w[i]);
		}

		free((*em)->posterior);  	
		free((*em)->w);  	

		*em = NULL;
	}		
}

void InitialMean(EM *em, Subgraph *sg){

  int i, j, *Nj = AllocIntArray(em->nnodes);	
  
	for(i = 0; i<em->nnodes; i++)
    {
        for(j = 0; j<em->nfeats; j++)
		{
		  em->mean[ sg->node[i].label ][ j ] += sg->node[i].feat[j];
		}
		Nj[sg->node[i].label]++;	
	}

    for(i = 0; i<em->nclusters; i++)
    {
      for(j = 0; j<em->nfeats; j++)
		{
		  em->mean[ i ][ j ] = em->mean[ i ][ j ]/Nj[i];		 
		}
	}
	
  free(Nj);
}

void SumCovarianceMatrix(double **matrix, double *vector, int nfeats) {

  int m,n;

  for (m=0; m<nfeats; m++)
    {
      for (n=0; n<nfeats; n++)
		{
		  matrix[ m ][ n ] += vector[m] * vector[n];		  
		}
    }	
}


void ResetCovarianceMatrix(EM *em){
	int i,j, k;
	
	
	for(i=0;i<em->nclusters; i++)
	{
		for(j=0;j<em->nfeats; j++)
		{
			for(k=0;k<em->nfeats; k++)
			{
				em->covarianceMatrix[i][j][k]=0.0;			
			}
		}
	}		
}

void InitialCovarianceMatrix(EM *em, Subgraph *sg){

  int i, j, k, *Nj = AllocIntArray(em->nclusters);
  double *diff= AllocDoubleArray(em->nfeats);		
  
  ResetCovarianceMatrix(em);
  
  for(i=0; i<em->nnodes; i++)
    {
      
      for(j=0; j<em->nfeats; j++)
	{	
	  diff[j] = sg->node[i].feat[j] - em->mean[ sg->node[i].label ][j];		
	  }
      
      Nj[sg->node[i].label]++;			
      SumCovarianceMatrix(em->covarianceMatrix[ sg->node[i].label ], diff, em->nfeats);					
    }
  
  for(i=0; i<em->nclusters; i++)
  {
     for(j=0; j<em->nfeats; j++)
	{
		  for(k=0; k<em->nfeats; k++)
			{						 
			  em->covarianceMatrix[ i ][ j ][ k ] = em->covarianceMatrix[ i ][ j ][ k ]/Nj[i];
			}
		}
    } 
  
  free(diff);
  free(Nj);
}

void InitialPriorProbability(EM *em, Subgraph *sg){

  int i;
  
  for(i=0; i<em->nnodes; i++)
    {		
      em->prior[ sg->node[i].label ] +=1;	
    }
  
  for(i=0; i<em->nclusters; i++)
    {		
      em->prior[ i ]/=em->nnodes ;
    }	
}
 

double Determinant(double **a,int n)
{
  int i,j,j1,j2;
  double det=0;
  double **m = NULL;

  if (n == 1) 
    { 
      det = a[0][0];
    } 
  else if (n == 2) 
    {
      det = a[0][0]*a[1][1] - a[1][0]*a[0][1];
      
    } 
  else if (n > 2) 
    {	
      det = 0;
      for (j1=0;j1<n;j1++) 
	 {
	 	m = malloc((n-1)*sizeof(double *));
	  	for (i=0;i<n-1;i++)
            m[i] = malloc((n-1)*sizeof(double));
	  
	  for (i=1;i<n;i++) 
	    {
	      j2 = 0;
	      for (j=0;j<n;j++) 
		{
		  if (j == j1)
		    continue;
		  m[i-1][j2] = a[i][j];
		  j2++;
		}
	    }
	  
	  det += pow(-1.0,j1+2.0) * a[0][j1] * Determinant(m,n-1);
	  for (i=0;i<n-1;i++)
            free(m[i]);
	  free(m);
	}
    }
  
  return(det);
}


void CoFactor(double **a,int n,double **b)
{
   int i,j,ii,jj,i1,j1;
   double det;
   double **c;

   c = malloc((n-1)*sizeof(double *));
   for (i=0;i<n-1;i++)
     c[i] = malloc((n-1)*sizeof(double));


   for (j=0;j<n;j++) 
   {
      for (i=0;i<n;i++) 
      {
         i1 = 0;
         for (ii=0;ii<n;ii++) 
         {
            if (ii == i)
               continue;
            j1 = 0;
            for (jj=0;jj<n;jj++) 
            {
               if (jj == j)
                  continue;
               c[i1][j1] = a[ii][jj];
               j1++;
            }
            i1++;
         }
         det = Determinant(c,n-1);
         b[i][j] = pow(-1.0,i+j+2.0) * det;
      }
   }
   for (i=0;i<n-1;i++)
      free(c[i]);
   free(c);
}


void TransposeMatrix(double **a,int n)
{
   int i,j;
   double tmp;

   for (i=1;i<n;i++) 
   {
      for (j=0;j<i;j++) 
      {
         tmp = a[i][j];
         a[i][j] = a[j][i];
         a[j][i] = tmp;
      }
   }
}


void Inverse(EM *em, double **matrix, double **inverse){
 
 	int i,j;
	double det;

	det=Determinant(matrix, em->nfeats);	

	if(det>0.0)
	{
	  CoFactor(matrix,em->nfeats,inverse);
	  TransposeMatrix(inverse,em->nfeats);
	  
	  for(i=0; i<em->nfeats; i++)
	    {		
	      for(j=0; j<em->nfeats; j++)
		  {	
		   inverse[i][j] = inverse[i][j]/det;		  
		  }
	    } 		
	//}else{
	//	if(Determinant(inverse,em->nfeats) < 0)
//		printf("inverse %g  ",Determinant(inverse,em->nfeats));
		//printf("determinante menor igual que zero %g  ",det);
	}

}

double MahalanobisDistance(EM *em, int cluster, float *featureVector){
  
  int i,j,m,n;
  double dist=0.0, sum;
  double  *diff=NULL,*aux = NULL, **inverse=NULL;
  
  diff = AllocDoubleArray(em->nfeats);	
  aux  = AllocDoubleArray(em->nfeats); 
  
  inverse = (double **) calloc(em->nfeats,sizeof(double *)); 
  for (i=0; i < em->nfeats; i++) 
    {
      inverse[i] = AllocDoubleArray(em->nfeats);
    }
  
  Inverse(em,em->covarianceMatrix[cluster], inverse);
  
  for(j=0 ; j<em->nfeats; j++)
    {
      diff[j] = featureVector[j] -  em->mean[cluster][j];			
    }
  
  for(m=0 ; m<em->nfeats; m++)
    {
      sum = 0.0;						
      for(n=0 ; n<em->nfeats; n++)
		sum += inverse[m][n]*diff[n];			
      
      aux[m] = sum;
    }	
  
  
  for(n=0 ; n<em->nfeats; n++)
    dist+=diff[n]*aux[n];

  if (dist < 0) {printf("Por que dist eh menor que zero?\n"); dist = 0.0;}
    
	
  for(n=0 ; n<em->nfeats; n++)
    dist = dist*(-0.5);		
 

  for (i=0; i < em->nfeats; i++) {
    free(inverse[i]);
  }				
  free(inverse);
  
  free(diff);
  free(aux);
  
  return (dist);
} 
 
void PosteriorProbability(EM *em,Subgraph *sg){

	int i,j;
	double **expMahalanobisDist=NULL,  denominador;
	
	expMahalanobisDist = (double **) calloc(em->nnodes,sizeof(double *)); 
	
	for (i=0; i < em->nnodes; i++) 
	{
		expMahalanobisDist[i] = AllocDoubleArray(em->nclusters);
	}

	//printf("\n");
			
				
	for	(i=0; i < em->nclusters; i++) 
	{	  
	  for	(j=0; j < em->nnodes; j++) 
	    expMahalanobisDist[j][i] = exp(MahalanobisDistance(em, i, sg->node[j].feat) );			
	}
	
	
	//printf("\n");
	for	(i=0; i < em->nnodes; i++) 
	{
		
	  denominador = 0.0;
	  for (j=0; j < em->nclusters; j++)
	    {		
	      denominador += expMahalanobisDist[i][j]*em->prior[j];			
	    }
						
		for	(j=0; j < em->nclusters; j++) 
		  {
		    if (denominador != 0.0) 
		      em->posterior[i][j] = (expMahalanobisDist[i][j]*em->prior[j] )/denominador;
		    else
		      em->posterior[i][j]=1.0;
		    //			printf("Posterior( %d | %d ) = %g\n",j,i,em->posterior[i][j]);
			if(MahalanobisDistance(em, j,sg->node[i].feat )> 0)
			  printf("dist( %d | %d ) = (%g,%g)\n",j, i, MahalanobisDistance(em, j,sg->node[i].feat ), em->posterior[i][j]);
		}
	}	
	
	for (i=0; i < em->nnodes; i++) 
	{
		free(expMahalanobisDist[i]);
	}
	free(expMahalanobisDist); 	
	
}


typedef struct _arcweights {
  float **val;
  int     n;
} ArcWeights;

ArcWeights *CreateArcWeights(int nlabels)
{
  ArcWeights *oindex=(ArcWeights *)calloc(1,sizeof(ArcWeights));
  int i;

  oindex->val = (float **) calloc(nlabels,sizeof(float *));
  for (i=0; i < nlabels; i++) 
    oindex->val[i] = (float *)calloc(nlabels,sizeof(float));
  
  oindex->n = nlabels;
  return(oindex);
}

void DestroyArcWeights(ArcWeights **oindex)
{
  int i;

  if ((*oindex) != NULL){
    for (i=0; i < (*oindex)->n; i++) 
      free((*oindex)->val[i]);
    free((*oindex)->val);
    free((*oindex));
    *oindex = NULL;
  }
}

void PrintArcWeights(ArcWeights *oindex)
{
  int i,j;

  printf("\n");

  for(i=0; i < oindex->n; i++) {
    for(j=0; j < oindex->n; j++) 
      printf("%5.2f ",oindex->val[i][j]);
    printf("\n");
  }
}

ArcWeights *OverlappingIndex(Subgraph *sg){ 
  int   i, j, k;
  float weight,tot;
  ArcWeights *oindex;

  oindex = CreateArcWeights(sg->nlabels);

  for (i = 0; i < sg->nnodes; i++){
    for (j = 0; (j < sg->nnodes); j++){
      k = sg->ordered_list_of_nodes[j];
      weight = opf_ArcWeight(sg->node[k].feat,sg->node[i].feat,sg->nfeats);

      if (weight <= sg->node[k].radius){
	oindex->val[sg->node[i].label][sg->node[k].label]++;
      }
    }
  }
  
  // Normalize the overlapping index

  for(i=0; i < sg->nlabels; i++){
    tot=0;
    for(j=0; j < sg->nlabels; j++) 
      tot += oindex->val[i][j];
    for(j=0; j < sg->nlabels; j++) 
      oindex->val[i][j] /= tot;
  }

  return(oindex);
}


int FindSubgraphRoot(Subgraph *sg, int i)
{
  if (sg->node[i].root == i)
    return(i);
  else
    return(sg->node[i].root=FindSubgraphRoot(sg, sg->node[i].root));
}


void MergeOverlapClusters(Subgraph *sg, ArcWeights *oindex)
{
  int i,j;

  printf("initial number of clusters %d\n",sg->nlabels);

  for (i=0; i < sg->nnodes; i++) {
    for (j=0; j < sg->nnodes; j++) {
      if (sg->node[i].label != sg->node[j].label)
	if (oindex->val[sg->node[i].label][sg->node[j].label] >= 0.10){
	  sg->node[i].root = FindSubgraphRoot(sg,j);
	}
    }
  }

  for (i=0; i < sg->nnodes; i++) 
    sg->node[i].root = FindSubgraphRoot(sg,i);

  j=0;
  for (i=0; i < sg->nnodes; i++) 
    if (i==sg->node[i].root){
      sg->node[sg->node[i].root].label=j;
      j++;
    }
  sg->nlabels=j;
  
  for (i=0; i < sg->nnodes; i++) 
    sg->node[i].label = sg->node[sg->node[i].root].label;

  printf("final number of clusters %d\n",sg->nlabels);
  
}

CImage *CGaussianFilter(CImage *cimg1)
{
  AdjRel *A=Circular(3.0);
  Kernel *K=GaussianKernel(A,0.5);
  CImage *cimg2=(CImage *)calloc(1,sizeof(CImage));

  cimg2->C[0] = LinearFilter2(cimg1->C[0],K);
  cimg2->C[1] = LinearFilter2(cimg1->C[1],K);
  cimg2->C[2] = LinearFilter2(cimg1->C[2],K);

  DestroyAdjRel(&A);
  DestroyKernel(&K);

  return(cimg2);
}

Subgraph *GetBorderSamples(Image *img, int size, int nsamples)
{
  int max_nnodes = size*img->ncols*2+(img->nrows-(2*size))*2*size;
  Subgraph *sg=CreateSubgraph(nsamples);
  Pixel u;
  int i, j, *border=AllocIntArray(max_nnodes), *used=AllocIntArray(max_nnodes);

  if (max_nnodes < nsamples) {
    printf("Error: the number of samples exceeded the number of available border pixels.\n");
    exit(1);
  }

  i=0;
  for (u.y=0; u.y < size; u.y++) 
    for (u.x=0; u.x < img->ncols; u.x++){
      border[i] = u.x + img->tbrow[u.y]; i++;
    }
  for (u.y=img->nrows-size; u.y < img->nrows; u.y++) 
    for (u.x=0; u.x < img->ncols; u.x++) {
      border[i] = u.x + img->tbrow[u.y]; i++;
    }
  for (u.x=0; u.x < size; u.x++) 
    for (u.y=size; u.y < img->nrows-size; u.y++) {
      border[i] = u.x + img->tbrow[u.y]; i++;
    }
  for (u.x=img->ncols-size; u.x < img->ncols; u.x++) 
    for (u.y=size; u.y < img->nrows-size; u.y++) {
      border[i] = u.x + img->tbrow[u.y]; i++;
    }
 
  i    = 0;
  while( i < nsamples ) {
    j = RandomInteger( 0, max_nnodes - 1 );
    if ( used[ j ] == 0 ) {
      sg->node[ i ].pixel    = border[j];
      sg->node[ i ].position = border[j];
      used[ j ] = 1;
      i++;
    }
  }
  
  free( used );
  free( border );

  return( sg );
}

Image *PDFPropagation(Subgraph *sg, Features *f)
{
    int     i,p,k;
    Image  *pdf=CreateImage(f->ncols,f->nrows);
    float   dist;


    for (p=0; p < f->nelems; p++)
      pdf->val[p]=opf_MAXDENS;

    for (p=0; p < f->nelems; p++)
    {
      for (i = 0; i < sg->nnodes; i++){
		k = sg->ordered_list_of_nodes[i];
		dist = opf_ArcWeight(sg->node[k].feat,f->elem[p].feat,sg->nfeats);

		if (dist <= sg->node[k].radius){
		  pdf->val[p]=(int)(opf_MAXDENS-(sg->node[k].dens*exp(-dist/sg->K)));
		  break;
		}
      }
    }

    return(pdf);
}

void Weight(EM *em){

	int i,j,k;
	double sum;

	
	for(i=0; i<em->nnodes; i++)
	{
	
	  for(j=0; j<em->nclusters; j++)
	    {					
	      sum=0.0;
	      for(k=0; k<em->nclusters; k++)
		{					
		  sum += em->posterior[i][k];
		}
	      em->w[i][j] = em->posterior[i][j]/sum;
	    }
	  
	}
}

void ResetMean(EM *em){
	int i,j;
	
	
	for(i=0;i<em->nclusters; i++)
	{
	  for(j=0;j<em->nfeats; j++)
	    {
	      em->mean[i][j]=0.0;			
	    }
	}
}

void Mean(EM *em, Subgraph *sg){

  int i, j;	

  ResetMean(em);
  
  for(i = 0; i<em->nnodes; i++)
    {
      for(j = 0; j<em->nfeats; j++)
	{
	  em->mean[ sg->node[i].label ][ j ] += sg->node[i].feat[j]*em->w[ i ][ sg->node[i].label ];
	}
    }
  
}

void WeightedSumCovarianceMatrix(double **matrix, double *vector, double w, int nfeats) {

	int i, m,n;
	double **a = NULL; 
		
	a = (double **) calloc(nfeats,sizeof(double *));  
	for (i=0; i < nfeats; i++) 
		a[i] = (double *)calloc(nfeats,sizeof(double));		

	for (m=0; m<nfeats; m++)
	{
		for (n=0; n<nfeats; n++)
		{
			a[ m ][ n ] += vector[m] * vector[n];
		}
	}	
	
	for (m=0; m < nfeats; m++)
	{
		for (n=0; n < nfeats; n++)
		{
			matrix[ m ][ n ] += a[ m ][ n ] * w;		
		}
	}


	for (i=0; i < nfeats; i++) 
		free(a[i]);		
	free(a);
}


void CovarianceMatrix(EM *em, Subgraph *sg){

	int i,j,k;
	double *diff= (double *) calloc(em->nfeats,sizeof(double ));	
	
	ResetCovarianceMatrix(em);

	for(i=0;i<em->nclusters; i++)
	{
	  for(j=0;j<em->nnodes; j++)
	    {
	      for(k=0;k<em->nfeats; k++)
		{			
		  diff[k] = sg->node[j].feat[k] - em->mean[ i ][ k ];	       			
		  }			
	      WeightedSumCovarianceMatrix(em->covarianceMatrix[i], diff, em->w[j][i], em->nfeats);	
	    }
	}		
	
	free(diff);



}

void PriorProbability(EM *em){

	int i,j;
	double sum;
		//printf("\n");
	for(i=0;i<em->nclusters; i++)
	{
	  sum=0;
	  for(j=0;j<em->nnodes; j++)
	    sum+=em->w[j][i];
	  
	  em->prior[i]=sum/em->nnodes;
	   	 //printf("%lf ",em->prior[i]);
	}
}

Image *PDFPropagationEM(Features *f, EM *em){

  int i,j;
  double maxvalue=INT_MIN,minvalue=INT_MAX;	
  double **expMahalanobisDist=NULL;
  double  maxPDFV, pdfv;		
  
  DImage  *pdf=CreateDImage(f->ncols,f->nrows);	
  Image   *img=CreateImage(f->ncols,f->nrows);	
  

  expMahalanobisDist = (double **) calloc(f->nelems,sizeof(double *)); 

  
  for (i=0; i < f->nelems; i++) 
    {
      expMahalanobisDist[i] = AllocDoubleArray(em->nclusters);

    }
  
  for	(i=0; i < em->nclusters; i++) 
    {
      for (j=0; j < f->nelems; j++) {
	expMahalanobisDist[j][i] = exp(MahalanobisDistance(em, i,f->elem[j].feat));	
      }
    }
  
  for	(i=0; i < f->nelems; i++) 
    {
      maxPDFV=INT_MIN;
      for (j=0; j < em->nclusters; j++) 
	{	  
	  pdfv = expMahalanobisDist[i][j]*opf_MAXDENS*em->prior[j];
	  if( pdfv > maxPDFV){
	    maxPDFV = pdfv;
	  }			
	}		
      pdf->val[i] = maxPDFV;

      if (maxPDFV > maxvalue){
	maxvalue = maxPDFV;
      }
      if (maxPDFV < minvalue){
	minvalue = maxPDFV;
      }
    }
  
  //Normalizing

  if (minvalue < maxvalue)
    for (i=0; i < f->nelems; i++)
      img->val[i] = (int)(opf_MAXDENS*((maxvalue-pdf->val[i])/(maxvalue-minvalue)));

  for (i=0; i < f->nelems; i++) 
    free(expMahalanobisDist[i]);	
  free(expMahalanobisDist); 
  DestroyDImage(&pdf);	
	
  return img;
}

Curve *SubgraphHistogram(Subgraph *sg, int nvalues){
  int i,p,nbins;
  Curve *hist=NULL;
  
  nbins = nvalues+1;
  hist  = CreateCurve(nbins);

  for (p=0; p < sg->nnodes; p++)
    hist->Y[sg->node[p].truelabel]++;  

  for (i=0; i < nbins; i++) 
    hist->X[i] = i;
  
  return(hist);
}

Curve *SubgraphCumulativeHistogram(Subgraph *sg, int nvalues){
  int i,p,nbins;
  Curve *hist=NULL;
  float total;
  
  nbins = nvalues+1;
  hist  = CreateCurve(nbins);

  for (p=0; p < sg->nnodes; p++)
    hist->Y[sg->node[p].truelabel]++;

  for (i=0; i < nbins; i++) 
    hist->X[i] = i;
  

  total = hist->Y[ 0 ];
  int last = 0;
  
  for (i=1; i < nbins; i++) {	
    if(hist->Y[i]!=0){
		hist->Y[i] += hist->Y[ last ];
		total += hist->Y[ i ];
		last = i;
	}
  }
  
  for (i=0; i < nbins; i++) 
    hist->Y[i] = hist->Y[i]/total;
   
  return(hist);  
}


Subgraph *SamplingThreshold(Image *img, int nlearnsamples, float rate)
{
	Curve *hist=NULL, *histAccumulated=NULL;
	int p, size, i, thrF, thrB, nF=0, nB=0, *used=AllocIntArray(img->nrows*img->ncols);
	int nsamples, maxvalue=0, countlearnsamples, nsamplesF, nsamplesB;
	float rateB = 0, rateF =0;
	Subgraph *sgtrain = NULL;
	Subgraph *sglearn = CreateSubgraph(nlearnsamples);
	
	
	if(rate > 0.5)
		rate= 0.5;
			
	/*sanpling learning set*/
	countlearnsamples = sglearn->nnodes;
	size = img->nrows*img->ncols;

	srandom((int)time(NULL));	
	// selecting learning samples from image  
	while(countlearnsamples>0) {				
		p = RandomInteger( 0, size - 1 );	
		
		if(used[p]==0){
			sglearn->node[countlearnsamples-1].position = p;
			sglearn->node[countlearnsamples-1].pixel = p;
			sglearn->node[countlearnsamples-1].truelabel = img->val[p];
			used[p] = -1;
			if(img->val[p] > maxvalue)
				maxvalue = img->val[p];
			countlearnsamples--;
		}
	}
	
	hist = SubgraphHistogram(sglearn, maxvalue);		
	histAccumulated = SubgraphCumulativeHistogram(sglearn, maxvalue);
		
	// selecting training samples from image  	
	for (thrB=0; thrB<=maxvalue && rateB<rate; thrB++){
	    rateB += histAccumulated->Y[ thrB ];
		nB += (int)hist->Y[thrB];
	}

	for (thrF=maxvalue; thrF>=0 && rateF<rate; thrF--){
	    rateF += histAccumulated->Y[ thrF ];
		nF += (int)hist->Y[thrF];
	}
	
	sgtrain = CreateSubgraph(nF + nB);
  	sgtrain->nlabels = 2;
	nsamples =  sgtrain->nnodes;
	nsamplesF = nF;
	nsamplesB = nB;
	for(i=0; i < sglearn->nnodes; i++){				

		if(sglearn->node[ i ].truelabel > thrF){
			if(nsamplesF > 0 && used[i] != -1){
		  		sgtrain->node[ nsamples-1 ].position = sglearn->node[ i ].position;
		  		sgtrain->node[ nsamples-1 ].pixel = sglearn->node[ i ].pixel;
				sgtrain->node[ nsamples-1 ].truelabel = 1;
				
				used[sglearn->node[ i ].position] = -1;
		  		nsamplesF--; nsamples--;
		  	}
	  	}
		else if(sglearn->node[ i ].truelabel < thrB){
			if(nsamplesB > 0 && used[i] !=-1 ){
		  		sgtrain->node[ nsamples-1 ].position = sglearn->node[ i ].position;
		  		sgtrain->node[ nsamples-1 ].pixel = sglearn->node[ i ].position;
		  		sgtrain->node[ nsamples-1 ].truelabel = 0;
		  		
		  		used[sglearn->node[ i ].position] = -1;
		  		nsamplesB--; nsamples--;
		  	}
		}			
	}
	
	DestroyCurve(&hist);
	DestroyCurve(&histAccumulated);
	DestroySubgraph(&sglearn);
	free(used);
	return sgtrain;
}


Subgraph *Image2Subgraph(Image *img){
	
	int p,size = img->nrows*img->ncols;
	Subgraph *sgcomplement = CreateSubgraph(size);
	
	for (p = 0; p < size; p++){
  		sgcomplement->node[ p ].position = p;
  		sgcomplement->node[ p ].pixel = p;
	}
	return sgcomplement;
}

Image *Subgraph2Image(Image *img, Subgraph *sgtest,Subgraph *sgtraining){

	int  i, count;
	Image *label = CreateImage(img->ncols,img->nrows);
	
	count = sgtest->nnodes;
	for (i = 0; i < count; i += 1){
		label->val[sgtest->node[i].position] = sgtest->node[i].label;

	}
	
	count = sgtraining->nnodes;
	for (i = 0; i < count; i += 1){
		label->val[sgtraining->node[i].position] = sgtraining->node[i].label;
	}
	

	return label;
}

Subgraph *BinaryClassifying(Image *binary, Image *enhanced, int ntraining, float rate){
	int i, p, nsamplesB =0, nsamplesF=0, size = binary->nrows*binary->ncols, Imax=MaximumValue(enhanced);
	int countsamples = ntraining, nF=0, nB=0, countF = 0, countB = 0;
	int *used=AllocIntArray(binary->ncols*binary->nrows);
	float meanF = 0, meanB = 0, rateF, rateB;
	Subgraph *sgB=NULL, *sgF=NULL, *sg=NULL;	
	Curve *histB=NULL, *histF=NULL;

	for (i=0; i<size; i++){
		if (binary->val[ i ] == 1)
			nF++;
		else if (binary->val[ i ] == 0)
			nB++;
	}
	
	sgF = CreateSubgraph(nF);	
	sgB = CreateSubgraph(nB);	
	
	for(i=0; i < size; i++){	
		if(binary->val[ i ] == 1){
	  		sgF->node[ nsamplesF ].truelabel = enhanced->val[i]; 
	  		meanF += enhanced->val[i];
	  		nsamplesF++;
	  	}else if(binary->val[ i ] == 0){
	  		sgB->node[ nsamplesB ].truelabel = enhanced->val[i];
	  		meanB += enhanced->val[i];
	  		nsamplesB++;
	  	}
	}	
	
	histF = SubgraphHistogram(sgF, Imax);	
	histB = SubgraphHistogram(sgB, Imax);
	
	meanF = round(meanF/(float)nF);
	meanB = round(meanB/(float)nB);	
	
	rateF = histF->Y[(int)meanF]/(float)nF;
	rateB = histB->Y[(int)meanB]/(float)nB;		
	
	while(rateF <= rate){	
		countF++;	
		if(meanF + countF  <= Imax)
			rateF += (histF->Y[(int)meanF + countF]/(float)nF);			
		if (meanF - countF >= 0)
			rateF += (histF->Y[(int)meanF - countF]/(float)nF);	
	}
			
	while(rateB <= rate){
		countB++;	
		if(meanB + countB  <= Imax)
			rateB += histB->Y[(int)meanB + countB]/(float)nB;
		if (meanB - countB >= 0)
			rateB += histB->Y[(int)meanB - countB]/(float)nB;
	}

	sg = CreateSubgraph(ntraining);
	sg->nlabels = 2;
	
	if( meanB + countB <  meanF - countF){
		srandom((int)time(NULL));
		while(countsamples>0) {				
			p = RandomInteger( 0, size - 1 );
			if(used[p]!=-1){
				if( enhanced->val[p] >= (meanF - countF) && enhanced->val[p] <= (meanF + countF)){		
					sg->node[countsamples-1].position = p;
					sg->node[countsamples-1].pixel = p;
					sg->node[countsamples-1].truelabel = 1;
					used[p] = -1;			
					countsamples--;
				}
				else if( enhanced->val[p] >= (meanB - countB) && enhanced->val[p] < (meanB + countB)){		
					sg->node[countsamples-1].position = p;
					sg->node[countsamples-1].pixel = p;
					sg->node[countsamples-1].truelabel = 0;
					used[p] = -1;			
					countsamples--;
				}
			}	
		}
	}

		
	DestroyCurve(&histF);
	DestroyCurve(&histB);
	DestroySubgraph(&sgF);
	DestroySubgraph(&sgB);
	free(used);
	return sg;	
	
} 

Subgraph *GetTrainingSamples(Image *img)
{
  Curve *hist=NormHistogram(img);
  int i, j, p, nsamples_fg, nsamples_bg, nsamples;
  int *used, n=img->ncols*img->nrows;
  Subgraph *sg=NULL;
  float area,fg_thres,bg_thres,bg_arealim=0.30,fg_arealim=0.20;
  int max_num_fg_samples=200,max_num_bg_samples=300;

  // Compute threshold for foreground pixels

  if ((hist->Y[hist->n-1]*n)>2000){ // get only outliers
    fg_thres = hist->n-2;
    nsamples_fg = max_num_fg_samples;
  }else{
  
    area = 0.0;
    for (i=hist->n-1; (i > 0) && (area <= fg_arealim); i--) 
      area += hist->Y[i];
  
    fg_thres = i;
    nsamples_fg = (int)(area*n);
  }

  // Compute threshold for background pixels
  area = 0.0;
  for (i=0; (i < hist->n) && (area <= bg_arealim); i++) 
    area += hist->Y[i];

  bg_thres = i;
  nsamples_bg = (int)(area*n);

  // Compute number of samples in each part

  if (nsamples_fg > max_num_fg_samples) nsamples_fg = max_num_fg_samples;
  if (nsamples_bg > max_num_bg_samples) nsamples_bg = max_num_bg_samples;
  nsamples = nsamples_fg + nsamples_bg;
  sg = CreateSubgraph(nsamples);
  used = AllocIntArray(n);

  // Compute foreground samples

  p    = 0;
  while( p <= nsamples_fg ) {
    j = RandomInteger( 0, n - 1 );
    if (( used[ j ] == 0 ) && (img->val[j] > fg_thres)) {
      sg->node[ p ].pixel      = j;
      sg->node[ p ].truelabel  = 1;
      sg->node[ p ].position   = j;
      used[ j ] = 1;
      p++;
    }
  }

  // Compute background samples
 
  while( p <= nsamples ) {
    j = RandomInteger( 0, n - 1 );
    if (( used[ j ] == 0 ) && (img->val[j] <= bg_thres)) {
      sg->node[ p ].pixel      = j;
      sg->node[ p ].truelabel  = 0;
      sg->node[ p ].position   = j;
      used[ j ] = 1;
      p++;
    }
  }

  free(used);
  DestroyCurve(&hist);
  return(sg);
}

void SelectLargestComp(Image *bin)
{
  AdjRel *A=Circular(1.5);
  Image  *label=LabelBinComp(bin,A);
  int Lmax=MaximumValue(label);
  int *area=(int *)AllocIntArray(Lmax+1);
  int imax,i,p,n=bin->ncols*bin->nrows;

  for (p=0; p < n; p++)  
    if (label->val[p]>0)
      area[label->val[p]]++;
  imax = 0;
  for (i=1; i <= Lmax; i++) 
    if (area[i]>area[imax])
      imax = i;
  for (p=0; p < n; p++)  
    if (label->val[p]!=imax)
      bin->val[p]=0;
  DestroyImage(&label);
  DestroyAdjRel(&A);
  free(area);
}


int main(int argc, char **argv)
{

  Image    *img=NULL,*label=NULL, *flabel=NULL, *pdf=NULL;
  Subgraph *sg=NULL,*sgtrain=NULL, *sgtest=NULL;
  Features *f=NULL;
  CImage   *cimg2=NULL,*cimg1=NULL;
  EM       *em=NULL;
  char ext[10],*pos;
  int i, it;
  char filename[200],basename[100]=" ";
  timer *t1=NULL,*t2=NULL;
  ArcWeights *oindex=NULL;
  
  opf_ArcWeight=opf_EuclDist;

	/*--------------------------------------------------------*/
	

  if (argc!=5){
    printf("opfem <image> <bordersize> <filter> <iterations>\n");
    printf("image: pgm or ppm\n");
    printf("bordersize: e.g., 20.\n");
    printf("filter: 1 - close holes and 2 (default) is area close.\n");
    printf("iterations: e.g., 1, 5, 10.\n");
    exit(1);
  }

  pos = strrchr(argv[1],'.') + 1;  
  strncpy(basename,argv[1],strlen(argv[1])-strlen(pos)-1);
  sscanf(pos,"%s",ext);

  t1 = Tic();

  if (strcmp(ext,"pgm")==0){
    img = ReadImage(argv[1]);
    sg  = GetBorderSamples(img, atoi(argv[2]), 200);
    f   = LMSImageFeats(img,3);
  }else{
    if (strcmp(ext,"ppm")==0){
      cimg2 = ReadCImage(argv[1]);
      cimg1 = CGaussianFilter(cimg2);
      img   = ift_CopyImage(cimg2->C[1]);      
      DestroyCImage(&cimg2);
      sg    = GetBorderSamples(img,atoi(argv[2]), 200);
      f     = LabCImageFeats(cimg1);   
    }else{
      printf("Invalid image format: %s\n",ext);
      exit(-1);
    }
  }

  SetSubgraphFeatures(sg,f);
  opf_BestkMinCut(sg,(int)(0.10*sg->nnodes),(int)(0.40*sg->nnodes));
  opf_OPFClustering(sg);
  oindex = OverlappingIndex(sg);
  //PrintArcWeights(oindex);
  MergeOverlapClusters(sg, oindex);

  // Expectation Maximization

  em = CreateEM(sg);		

  //Initial Maximization:
	
  InitialMean(em,sg); 
  InitialCovarianceMatrix(em, sg);	
  InitialPriorProbability(em, sg);
  
		
  it = atoi(argv[4]);
  for (i=0; i<it; i++){
    //Expectation:
    PosteriorProbability(em, sg);	
    
    //Maximization:
    Weight(em);
    Mean(em, sg);
    CovarianceMatrix(em, sg);
    PriorProbability(em);		
  }

  pdf = PDFPropagationEM(f,em);
  sprintf(filename,"%s_pdf.ppm",basename);
  WriteImage(pdf,filename);

  // Classification

  sgtrain = GetTrainingSamples(pdf);
  SetSubgraphFeatures(sgtrain,f);
  opf_OPFTraining(sgtrain); 
  sgtest = Image2Subgraph(pdf); 
  SetSubgraphFeatures(sgtest,f);
  opf_OPFClassifying(sgtrain, sgtest); 	
  label = Subgraph2Image(pdf, sgtest,sgtrain);
  DestroySubgraph(&sgtrain);
  DestroySubgraph(&sgtest);	


  AdjRel *A=Circular(1.5);
  flabel  = OpenRec(label,A); 
  SelectLargestComp(flabel);
  DestroyImage(&label);
  DestroyAdjRel(&A);  
  if (atoi(argv[3])==1)
    label =  CloseHoles(flabel);
  else
    label =  FastAreaClose(flabel,300);


 t2 = Toc();


  fprintf(stdout,"opfem in %f ms\n",CTime(t1,t2));

  sprintf(filename,"%s_label.ppm",basename);
  WriteImage(label,filename);
  cimg2     = DrawLabeledRegions(img,label);
  sprintf(filename,"%s_result.ppm",basename);
  WriteCImage(cimg2,filename);
  DestroySubgraph(&sg);
  if (strcmp(ext,"pgm")==0)
    DestroyImage(&img);
  else{
    DestroyCImage(&cimg1);
    DestroyImage(&img);
  }
   DestroyImage(&label);
   DestroyImage(&flabel);
   DestroyImage(&pdf);
   DestroyCImage(&cimg2);
   DestroyFeatures(&f);
   DestroySubgraph(&sgtrain);
   DestroySubgraph(&sgtest);	  
   DestroyArcWeights(&oindex);
   DestroyEM(&em);	
   DestroySubgraph(&sg);
   DestroyImage(&img);
   DestroyCImage(&cimg1);
 		
   return (0);
}
