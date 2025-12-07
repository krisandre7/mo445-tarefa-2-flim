#include "ift.h"

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
  timer *t1=NULL,*t2=NULL;
  Image    *img=NULL,*label=NULL,*flabel=NULL,*pdf=NULL;
  Subgraph *sg=NULL, *sgtrain = NULL, *sgtest=NULL;
  Features *f=NULL;
  CImage *cimg2=NULL,*cimg1=NULL;
  char ext[10],*pos;
  ArcWeights *oindex=NULL;
  char filename[200],basename[100]=" ";
  AdjRel *A;

  opf_ArcWeight=opf_EuclDist;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=4){
    printf("backgroundremoval <image> <bordersize> <filter>\n");
    printf("image: pgm or ppm\n");
    printf("bordersize: e.g., 20.\n");
    printf("filter: 1 - close holes and 2 (default) is area close.\n");
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
      sg    = GetBorderSamples(img,atoi(argv[2]),200);
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

  pdf=PDFPropagation(sg,f);

  sprintf(filename,"%s_pdf.ppm",basename);
  WriteImage(pdf,filename);

  
  sgtrain = GetTrainingSamples(pdf);
  SetSubgraphFeatures(sgtrain,f);
  opf_OPFTraining(sgtrain); 
  sgtest = Image2Subgraph(pdf); 
  SetSubgraphFeatures(sgtest,f);
  opf_OPFClassifying(sgtrain, sgtest); 	
  label = Subgraph2Image(pdf, sgtest,sgtrain);
  DestroySubgraph(&sgtrain);
  DestroySubgraph(&sgtest);	
 

  /*
  sgtrain = SamplingThreshold(pdf, 200, 0.2); 
  SetSubgraphFeatures(sgtrain,f);
  opf_OPFTraining(sgtrain); 
  sgtest = Image2Subgraph(pdf); 
  SetSubgraphFeatures(sgtest,f);
  opf_OPFClassifying(sgtrain, sgtest); 	
  label = Subgraph2Image(pdf, sgtest,sgtrain);
  DestroySubgraph(&sgtrain);
  DestroySubgraph(&sgtest);	

  sgtrain = BinaryClassifying(label, pdf, 300, 0.3);
  opf_OPFTraining(sgtrain); 
  SetSubgraphFeatures(sgtrain,f);				
  sgtest = Image2Subgraph(pdf); 
  SetSubgraphFeatures(sgtest,f);	
  opf_OPFClassifying(sgtrain, sgtest); 
  DestroyImage(&label);
  label = Subgraph2Image(pdf, sgtest,sgtrain);

  //--------------------------------------

  */

  sprintf(filename,"%s_label.ppm",basename);
  WriteImage(label,filename);

  A=Circular(1.5);
  flabel  = OpenRec(label,A); 
  SelectLargestComp(flabel);
  DestroyImage(&label);
  DestroyAdjRel(&A);  
  if (atoi(argv[3])==1)
    label =  CloseHoles(flabel);
  else
    label =  FastAreaClose(flabel,300);

  t2 = Toc();

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

  fprintf(stdout,"backgroundremoval in %f ms\n",CTime(t1,t2));

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return(0);
}
