#include <sys/stat.h>
#include <sys/types.h>

#include "iftTrainPixels.h"
#include "common.h"

#define TRAIN_PIXELS_DEBUG

iftLabelPixelsData  *iftCreateLabelPixelsData(int npixels)
{
	iftLabelPixelsData *le = NULL;

	le = (iftLabelPixelsData *) calloc(1,sizeof(iftLabelPixelsData));
	if (le == NULL)
		iftError(MSG1,"iftCreateLabelPixelsData");
	le->n         = npixels;
	le->labelPixel = (iftLabelPixel *) calloc(le->n,sizeof(iftLabelPixel));

	return(le);
}

iftTrainPixels* iftCreateTrainPixels(int nimages, int nsamplesbi)
{
	iftTrainPixels *Z =(iftTrainPixels*) calloc(1,sizeof(iftTrainPixels));
	int s;
	
	Z->data = (iftLabelPixelsData*) calloc(nimages,sizeof(iftLabelPixelsData));

	if (Z->data == NULL) 
		iftError(MSG1,"iftCreateTrainPixels");

	for (s=0; s < nimages; s++) {
		Z->data[s].filename       = NULL;
		Z->data[s].image          = NULL;
		Z->data[s].labelPixel     = NULL;
	}

	Z->nimages    = nimages;
	Z->nsamplesbi = nsamplesbi;

	return(Z);
}

void iftDestroyLabelPixelsData(iftLabelPixelsData* pLPD )
{
  iftLabelPixelsData *aux = pLPD;

  if (aux != NULL) {
    if (aux->labelPixel) free(aux->labelPixel);
  }
}

void iftDestroyTrainPixels(iftTrainPixels **Z)
{
  iftTrainPixels *aux = *Z;
  int s;

  if (aux != NULL) {
    for (s=0; s < aux->nimages; s++) {
      if (aux->data[s].filename)  free(aux->data[s].filename );
      iftDestroyLabelPixelsData(&(aux->data[s]));
      //  if (aux->data[s].labelPixel) free(aux->data[s].labelPixel);
      if (aux->data[s].image != NULL)
	iftDestroyMImage(&(aux->data[s].image));			
    }
    free(aux->data);
    free(aux);
    *Z = NULL;
  }
}
/* end Data Structures for iftTrainPixels */


/* Extract Label Pixels */
iftLabelPixelsData* iftExtractLabelPixelsGTBorderWGNonBorder(iftMImage *mimg,iftImage *imgGT, iftAdjRel* A,int nsamplesbi,float percBorder, float radius)
{
  // create label epixels
  int nsamplesBorderbi = (int) (nsamplesbi*percBorder + 0.5);
  iftLabelPixelsData *LE = iftCreateLabelPixelsData(nsamplesbi);
  printf("Creating label Pixel: %d \n",LE->n);
	
  // Label image components from imgGT - borders must have 0 value
  iftAdjRel* adjcomp = iftCircular(1.5);
  iftImage *imgNGT;
  iftImage *imglabels;

  imgNGT=iftCreateImage(imgGT->xsize,imgGT->ysize,imgGT->zsize);
  int k;
  iftCopyVoxelSize(imgGT,imgNGT);
  for (k=0; k < imgNGT->n; k++) {
    imgNGT->val[k]=(255-imgGT->val[k]);
  }
  imglabels = iftRelabelRegions(imgNGT,adjcomp);
  iftDestroyAdjRel(&adjcomp);
  iftDestroyImage(&imgNGT);

  // Creating imgWG
  iftImage *img;
  img = iftMImageToImage(mimg,255,0);
  float volume_threshold = 2000.;
  iftImage* basins = iftImageBasins(img,A);
	
  int nregions = 257;
  iftImage* marker=NULL;
  iftImage* imgWG=NULL;
  volume_threshold /= 2;
  while(nregions > 256) {
    volume_threshold *= 2;
    marker = iftVolumeClose(basins,volume_threshold);
    imgWG = iftWaterGray(basins,marker,A);
    nregions = iftMaximumValue(imgWG);
    printf("volume_threshold: %f, nregions: %d\n",volume_threshold,nregions);
  }

  iftWriteImageP2(imgWG,"watergray.pgm");
  iftWriteImageP2(imglabels,"labels.pgm");
  iftWriteImageP2(imglabels,"GT.pgm");
  iftDestroyImage(&basins);
  iftDestroyImage(&marker);
  // imgWG created
	

  // generating pixels
  int t,p,q,i,v,high;
  int numB,countB;
  int numNB,countNB;
  numB = nsamplesBorderbi;
  numNB = (int)nsamplesbi-nsamplesBorderbi;
  countB = 0;
  countNB = 0;

  int numPixelsTotal = mimg->n;
  int* sample  = iftAllocIntArray(numPixelsTotal);
  int* count   = iftAllocIntArray(numPixelsTotal);

  // extract label edges
  printf("nsamples: %d, non edges: %d, edges:%d (percBorder: %f)\n",nsamplesbi,numB,numNB,percBorder);

  // randomly select K samples 
  sample = iftAllocIntArray(numPixelsTotal); 
  count  = iftAllocIntArray(numPixelsTotal); 
  for (i=0; i < numPixelsTotal; i++) {
    sample[i]=i; 
    count[i]=100;
  }

  t = 0; high = numPixelsTotal-1; 
  while(t < nsamplesbi){
    // random pixel
    i = iftRandomInteger(0,high);
    // verify random pixel
    if (count[i] == 0){
      p = sample[i];

      iftVoxel pixel   = iftMGetVoxelCoord(mimg ,p);
      int ctGT=0,ctLabel=0,ctWG=0;
      for(v=1;v<A->n;v++) {
	iftVoxel neighbor   = iftGetAdjacentVoxel(A,pixel,v);
	if (iftMValidVoxel(mimg,neighbor))
	  q = iftMGetVoxelIndex(mimg,neighbor);
	else
	  break;
				
	if (imglabels->val[p] == imglabels->val[q])
	  ctLabel++;
				  
	if (imgWG->val[p] == imgWG->val[q])
	  ctWG++;
      }
      if (v==A->n) {
	if (imgGT->val[p] != 0)
	  ctGT++;
	if ( (countB < numB) && (ctGT) ) { // (ctLabel != A->n-1 ) ){ 
	  // set pixels attributes as Border
	  LE->labelPixel[t].truelabel = 2;
	  LE->labelPixel[t].p = p;
	  countB++;
	  t++;
	}
	else if ( (countNB < numNB) && (!ctGT) && (ctWG != A->n-1) ) {
		  // && (ctLabel == A->n-1 ) && (ctWG != A->n-1) ) {
	  // set edge attributes Non Border
	  LE->labelPixel[t].truelabel = 1;
	  LE->labelPixel[t].p = p;
	  countNB++;
	  t++;
	}

      }

      iftSwitchValues(&sample[i],&sample[high]);
      iftSwitchValues(&count [i],&count [high]);
      high--;

    }else{
      count[i]--;
    }
  }

  free(count);
  free(sample);
	
  return LE;
}
/* end Extract LabelPixels*/


/* Extract Label Pixels */
iftLabelPixelsData* iftExtractLabelPixelsWG(iftMImage *mimg,iftImage *imgGT, iftAdjRel* A,int nsamplesbi,float percBorder, float radius)
{
  // create label epixels
  int nsamplesPixelsbi = (int) (nsamplesbi*percBorder + 0.5);
  iftLabelPixelsData *LE = iftCreateLabelPixelsData(nsamplesbi);
  printf("Creating label Pixel: %d \n",LE->n);
	
  // Label image components from imgGT - borders must have 0 value
  iftAdjRel* adjcomp = iftCircular(1.0);
  iftImage *imgNGT;
  iftImage *imglabels;

  imgNGT=iftCreateImage(imgGT->xsize,imgGT->ysize,imgGT->zsize);
  int k;
  iftCopyVoxelSize(imgGT,imgNGT);
  for (k=0; k < imgNGT->n; k++) {
    imgNGT->val[k]=(255-imgGT->val[k]);
  }
  imglabels = iftRelabelRegions(imgNGT,adjcomp);
  iftDestroyAdjRel(&adjcomp);
  iftDestroyImage(&imgNGT);

  // Creating imgWG
  iftImage *img;
  img = iftMImageToImage(mimg,255,0);
  float volume_threshold = 2000.;
  iftImage* basins = iftImageBasins(img,A);
	
  int nregions = 257;
  iftImage* marker=NULL;
  iftImage* imgWG=NULL;
  while(nregions > 256) {
    marker = iftVolumeClose(basins,volume_threshold);
    imgWG = iftWaterGray(basins,marker,A);
    nregions = iftMaximumValue(imgWG);
    printf("volume_threshold: %f, nregions: %d\n",volume_threshold,nregions);
    volume_threshold *= 2;
  }
  volume_threshold /= 2;
  //	iftWriteImageP2(imgWG,"watergray.pgm");
  iftDestroyImage(&basins);
  iftDestroyImage(&marker);
  // imgWG created
	

  // generating pixels
  int s,p,q,i;
  int numB,countB;
  int numNB,countNB;
  numB = nsamplesPixelsbi;
  numNB = (int)nsamplesbi-nsamplesPixelsbi;
  countB = 0;
  countNB = 0;

  int numPixelsTotal = mimg->n;
  int* selectedPIX = iftAllocIntArray(numPixelsTotal);

  // extract label edges
  s = 0;
  printf("nsamples: %d, non edges: %d, edges:%d (percBorder: %f)\n",nsamplesbi,numB,numNB,percBorder);
  while(s < nsamplesbi){
    // random pixel
    p = iftRandomInteger(0,numPixelsTotal-1);
    // verify random pixel
    if (selectedPIX[p] == 0){
      selectedPIX[p] = 1;
      iftVoxel pixel   = iftMGetVoxelCoord(mimg ,p);
      int ctLabel=0,ctWG=0;
      for(i=1;i<A->n;i++)
	{
	  iftVoxel neighbor   = iftGetAdjacentVoxel(A,pixel,i);
	  if (iftMValidVoxel(mimg,neighbor))
	    q = iftMGetVoxelIndex(mimg,neighbor);
	  else
	    continue;
				
	  if (imglabels->val[p] == imglabels->val[q])
	    ctLabel++;
				  
	  if (imgWG->val[p] == imgWG->val[q])
	    ctWG++;
	}
				
      if ( (countB < numB) && (ctLabel != A->n-1 ) ){ 
	// set edge attributes Border
	LE->labelPixel[s].truelabel = 2;
	LE->labelPixel[s].p = p;
	s++;
	countB++;
				
      }
      else if ( (countNB < numNB) && (ctLabel == A->n-1 ) && (ctWG == A->n-1) )
	{ 
	  // set edge attributes Non Border
	  LE->labelPixel[s].truelabel = 1;
	  LE->labelPixel[s].p = p;
	  s++;
	  countNB++;
	  /*
	  // test pixel markers
	  if(s == 2){
	  exemplo=iftCreateImage(imglabels->xsize,imglabels->ysize,imglabels->zsize);
	  for (k=0; k < imglabels->n; k++) {
	  exemplo->val[k]=imglabels->val[k];
	  }
	  exemplo->val[p] = 255;
	  exemplo->val[q] = 255;
	  printf("************p: %d, q: %d **********\n",p,q);
	  iftWriteImageP2(exemplo,"testimg/labels-pq.pgm");
					
	  exemplo=iftCreateImage(imgWG->xsize,imgWG->ysize,imgWG->zsize);
	  for (k=0; k < imgWG->n; k++) {
	  exemplo->val[k]=imgWG->val[k];
	  }
	  exemplo->val[p] = 255;
	  exemplo->val[q] = 255;
	  iftWriteImageP2(exemplo,"testimg/wg-pq.pgm");
	  }
	  */
	}
    }
  }
	
  free(selectedPIX);
	
  return LE;
}
/* end Extract LabelPixels*/

/* Extract Label Pixels */
iftLabelPixelsData* iftExtractLabelPixels(iftMImage *mimg,iftImage **pimgGT, iftAdjRel* A,int nsamplesbi ,float percBorder, float radius)
{
  // create label edges
  int nsamplesPixelsbi = (int) (nsamplesbi*percBorder + 0.5);
  iftLabelPixelsData *LE;
  LE = iftCreateLabelPixelsData(nsamplesbi);
  printf("Creating label Pixel: %d \n",LE->n);

  iftImage *img,*imgWG,*imglabels,*basins,*marker;
  img       = iftMImageToImage(mimg,255,0);
  basins    = iftImageBasins(img,A);
  // Label image components from BorderImageToLabelImage (Rauber & Falcão)	
  imglabels = iftBorderImageToLabelImage(basins,*pimgGT,radius);
	

  // Creating imgWG
  float volume_threshold = 2000.;
  marker=NULL;imgWG=NULL;
  marker = iftVolumeClose(basins,volume_threshold);
  imgWG = iftWaterGray(basins,marker,A);

  iftDestroyImage(&basins);
  iftDestroyImage(&marker);
  iftWriteImageP2(imgWG,"watergray.pgm");
  iftWriteImageP2(imglabels,"labels.pgm");

  //
  iftDestroyImage(&imglabels);
  imglabels = iftCopyImage(*pimgGT); // ignoring the imgGT correction
  //
	
  int p,s,q,i,high;
  iftVoxel u,v;
  int numB,countB;
  int numNB,countNB;
  numB    = nsamplesPixelsbi;
  numNB   = nsamplesbi-nsamplesPixelsbi;
  countB  = 0;
  countNB = 0;

  iftAdjRel* adjA  = iftCircular(7.0);  // Menotti's call 7.0 (2013/09/14)
  iftAdjRel* adjB  = iftCircular(1.5);  // Falcao's call 1.5 (2013/07/24-menotti)
  iftAdjRel* adjNB = iftCircular(5.0);  // Falcao's call 5.0 (2013/07/26-menotti)

  int dx,dy,dz;
  iftMaxAdjShifts(adjA, &dx, &dy, &dz);

  int* count  = iftAllocIntArray(mimg->n);
  int* sample = iftAllocIntArray(mimg->n);
  for (p=0; p < mimg->n;p++) {
    count [p] = 100;
    sample[p] = p;
  }

  // extract label pixels
  s = 0; high = mimg->n-1;
  printf("nsamples: %d, border: %d, nonborder: %d (percBorder: %f), adjB->n: %d\n",nsamplesbi,numB,numNB,percBorder,adjB->n);
  int ntrials = 0;
  while( (s < nsamplesbi) && (ntrials < mimg->n) ) {
    // random pixel
    int index = iftRandomInteger(0,high);
    if (count[index]==0) {
      ntrials++;

      p = sample[index];
      u = iftMGetVoxelCoord(mimg,p);

      char select_p   = 1;
      for (i=high+1; i < mimg->n; i++) { 
	q = sample[i];
	v = iftMGetVoxelCoord(mimg,q);

	/* region overlapping constraint */
	if ( (abs(v.x-u.x)<=dx/4) && // before /2 - allowing the selection of more
	     (abs(v.y-u.y)<=dy/4) &&
	     (abs(v.z-u.z)<=dz/4)    ){
	  select_p = 0;
	  break;
	}
      }
      /* image border constraint */
      if ( ( (u.x+3*dx >= mimg->xsize) || (u.x-3*dx < 0) ) ||
	   ( (u.y+3*dy >= mimg->ysize) || (u.y-3*dy < 0) ) )
	select_p=0;

      if (!select_p)
	continue;

      int ctLabel=0,ctWG=0;
      for(i=1;i<adjB->n;i++)
	{
	  v = iftGetAdjacentVoxel(adjB,u,i);
	  if (iftMValidVoxel(mimg,v))
	    q = iftMGetVoxelIndex(mimg,v);
	  else {
	    ctLabel=0;
	    break; // avoiding selecting border image pixels
	  }

	  if (imglabels->val[p] == imglabels->val[q])
	    ctLabel++;
	}

      if ( (countB < numB) && (ctLabel != adjB->n-1) && (ctLabel > 0) ) {
	// set pixel attributes Border
	LE->labelPixel[s].truelabel = 2;
	LE->labelPixel[s].p = p;
	//printf("s: %6d, b: %4d, p: %7d, q:%7d",s,b,p,q);
	//				fprintf(stderr,"%2d/%2d(B)\t",ctLabel,adjB->n-1);
	iftSwitchValues(&sample[index],&sample[high]);
	iftSwitchValues(&count[index],&count[high]);      
	high--;s++;countB++;
      }
      else {
	ctLabel=0;ctWG=0;
	for(i=1;i<adjNB->n;i++)
	  {
	    v = iftGetAdjacentVoxel(adjNB,u,i);
	    if (iftMValidVoxel(mimg,v))
	      q = iftMGetVoxelIndex(mimg,v);
	    else {
	      ctWG=0;
	      break; // avoiding selecting border image pixels
	    }

	    if (imglabels->val[p] == imglabels->val[q])
	      ctLabel++;
					  
	    if (imgWG->val[p] == imgWG->val[q])
	      ctWG++;
	  }
	if ( (countNB < numNB) && (ctLabel == adjNB->n-1 ) && (ctWG == adjNB->n-1) ) {
	  // set pixel attributes Non Border
	  LE->labelPixel[s].truelabel = 1;
	  LE->labelPixel[s].p = p;
	  //printf("s: %6d, b: %4d, p: %7d, q:%7d",s,b,p,q);
	  //				fprintf(stderr,"%2d/%2d(N)\t",ctLabel,adjNB->n-1);
	  iftSwitchValues(&sample[index],&sample[high]);
	  iftSwitchValues(&count[index],&count[high]);      
	  high--; s++; countNB++;
	}
      }
    }
    else
      count[index]--;
  }
  fprintf(stderr,"B: %d, NB: %d, high: %d\n",countB,countNB,high);

  free(count);
  free(sample);
  iftDestroyImage(&imgWG);
  iftDestroyImage(pimgGT);
  *pimgGT = imglabels;
  iftDestroyAdjRel(&adjA);
  iftDestroyAdjRel(&adjB);
  iftDestroyAdjRel(&adjNB);

  return LE;
}
/* end Extract LabelPixels*/

/* Extract train edges */
iftTrainPixels* iftExtractTrainPixels(char *file_imageNames,char *file_imageNamesGT,int nsamplesbi ,float percBorder, float radius)
{
	int idxImg;
	int nimages,nimagesGT = 0;
	char *folder_name = iftAllocCharArray(256);
	char *folder_nameGT = iftAllocCharArray(256);
	char fullPath[128];
	FileNames * files = NULL;
	FileNames * filesGT = NULL;
	iftTrainPixels *TE;
	iftLabelPixelsData *auxLE;

	int nsamplesPixelsbi = (int)(nsamplesbi*percBorder + 0.5);

	folder_name = getFolderName(file_imageNames);
	nimages = countImages(file_imageNames);
	folder_nameGT = getFolderName(file_imageNamesGT);
	nimagesGT = countImages(file_imageNamesGT);

	files = createFileNames(nimages);
	loadFileNames(files, file_imageNames);

	filesGT = createFileNames(nimagesGT);
	loadFileNames(filesGT, file_imageNamesGT);

	printf("Folder: %s\n", folder_name);
	printf("Number of Images: %d\n", nimages);
	printf("File with image names: %s\n"   , file_imageNames);
	printf("File with GT image names: %s\n", file_imageNamesGT);
	printf("Number of samples per image: %d (%d-Border/%d-NonBorder %6.2f%%)\n",nsamplesbi,nsamplesPixelsbi,nsamplesbi-nsamplesPixelsbi,100.*percBorder);	
	printf("Spatial radius: %.4f\n",radius);	
	puts("");


	// initialize TE
	TE = iftCreateTrainPixels(nimages, nsamplesbi);
	TE -> radius = radius;
	// define adjacency
	iftAdjRel* adj_relation = iftCircular(radius);

#ifdef TRAIN_PIXELS_DEBUG
	mkdir("output/",S_IRWXU);
#endif
	
	for( idxImg = 0; idxImg < TE->nimages; idxImg++){
		// read image
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		//printf("(%4d/%4d) %s - %s\n",idxImg,TE->nimages,folder_name,files[idxImg].filename);
		strcat(fullPath, files[idxImg].filename);
		iftImage *img;
		if (strstr(fullPath,".ppm") != NULL)
			img = iftReadImageP6(fullPath);
		else
			img = iftReadImageP5(fullPath);

		// read image GT
		fullPath[0] = 0;
		strcpy(fullPath, folder_nameGT);
		strcat(fullPath, filesGT[idxImg].filename);
		iftImage *imgGT = iftReadImageP5(fullPath);

		iftVerifyImageDomains(img,imgGT,"iftExtractTrainPixels");

#ifdef TRAIN_PIXELS_DEBUG
		iftColor Yellow,Blue,Red,Green,YCbCrYellow,YCbCrBlue,YCbCrRed,YCbCrGreen;
		Yellow.val[0] = 255; Yellow.val[1] = 255; Yellow.val[2] =   0;YCbCrYellow = iftRGBtoYCbCr(Yellow);
		Blue.val  [0] =   0; Blue.val  [1] =   0; Blue.val  [2] = 255;YCbCrBlue   = iftRGBtoYCbCr(Blue  );
		Red.val   [0] = 255; Red.val   [1] =   0; Red.val   [2] =   0;YCbCrRed    = iftRGBtoYCbCr(Red   );
		Green.val [0] =   0; Green.val [1] = 255; Green.val [2] =   0;YCbCrGreen  = iftRGBtoYCbCr(Green );
		iftAdjRel *A          = iftCircular(sqrtf(2.0));
		iftAdjRel *B          = iftCircular(0.0);
		iftAdjRel *C          = iftCircular(3.0);
		iftImage *imgout = iftCopyImage(img);
		iftDrawBorders(imgout,imgGT,A,YCbCrYellow,B);
#endif


		// Generating Deep Features and Pixels
		iftMImage*  mimg  = iftImageToMImage(img,YCbCr_CSPACE); // RGB_CSPACE);
		// extract ed and its labels
		//auxLE = iftExtractLabelPixelsWG(mimg, imgGT,adj_relation,nsamplesbi,percBorder,radius);
		auxLE = iftExtractLabelPixels(mimg,&imgGT,adj_relation,nsamplesbi,percBorder,radius);

#ifdef TRAIN_PIXELS_DEBUG
		iftDrawBorders(imgout,imgGT,A,YCbCrGreen,B);
		iftVoxel u;
		for(int s=0;s<auxLE->n;s++)
		{
		        int p = auxLE->labelPixel[s].p;
			u.x = iftGetXCoord(imgout,p);
			u.y = iftGetYCoord(imgout,p);
			u.z = iftGetZCoord(imgout,p);
			if (auxLE->labelPixel[s].truelabel == 1) // Non-Border
			  iftDrawPoint(imgout,u,YCbCrBlue,C);
			else if (auxLE->labelPixel[s].truelabel == 2) // Border
			  iftDrawPoint(imgout,u,YCbCrRed ,C);
		}

		char imgpath[256];
		sprintf(imgpath,"output/%s",files[idxImg].filename);
		iftWriteImageP6(imgout,imgpath);
		iftDestroyImage(&imgout);
		iftDestroyAdjRel(&A);iftDestroyAdjRel(&B);iftDestroyAdjRel(&C);
#endif
		// set image and label edges into train edges
                TE->data[idxImg].labelPixel = auxLE->labelPixel;
		TE->data[idxImg].filename = iftAllocCharArray(strlen(files[idxImg].filename));
		strcpy(TE->data[idxImg].filename,files[idxImg].filename);
		TE->data[idxImg].nfilename = strlen(files[idxImg].filename);
		TE->data[idxImg].n = nsamplesbi;
		TE->data[idxImg].image = mimg;
		
		iftDestroyImage(&img);
		iftDestroyImage(&imgGT);
	}
	
	iftDestroyAdjRel(&adj_relation);
	
	return TE;
}
/* end Extract train edges */

/* Write train edges*/
void iftWriteTrainPixels(iftTrainPixels *trainPixels,char *filenameTrainPixels) {
	int i,j;
	FILE *fileTrainPixels = fopen(filenameTrainPixels, "wb");
	if ( fileTrainPixels != NULL ){
		// write TrainPixels
		fwrite(&trainPixels->nimages, sizeof(int), 1, fileTrainPixels);
		fwrite(&trainPixels->nsamplesbi, sizeof(int), 1, fileTrainPixels);
		fwrite(&trainPixels->radius, sizeof(float), 1, fileTrainPixels);
		// write LabelPixelsData
		for( i = 0; i < trainPixels->nimages; i++) {
			fwrite(&trainPixels->data[i].nfilename, sizeof(int), 1, fileTrainPixels);
			fwrite(&trainPixels->data[i].n, sizeof(int), 1, fileTrainPixels);
			// write filename
			printf("file: %s\n",trainPixels->data[i].filename);
			for( j = 0; j < trainPixels->data[i].nfilename; j++) {
				fwrite(&trainPixels->data[i].filename[j], sizeof(char), 1, fileTrainPixels);
			}
			// write labelPixel
			for( j = 0; j < trainPixels->data[i].n; j++) {
				fwrite(&trainPixels->data[i].labelPixel[j].truelabel, sizeof(int), 1, fileTrainPixels);
				fwrite(&trainPixels->data[i].labelPixel[j].p, sizeof(int), 1, fileTrainPixels);
			}
		}
	}
	fclose(fileTrainPixels);
}
/* end Write train edges*/

/* Read train edges*/
iftTrainPixels* iftReadTrainPixels(char *filenameTrainPixels,int bReadImages) {
	FILE *fileTrainPixels;
	iftTrainPixels *trainPixels;
	int i,j,rd;
	int nimages,nsamplesbi;
	float radius; 
	printf("filenameTrainPixels: %s \n",filenameTrainPixels);
	//read from the file in binary mode 
	fileTrainPixels = fopen(filenameTrainPixels, "rb"); 
	rd=fread(&nimages   , sizeof(int)  , 1, fileTrainPixels);   if (rd != 1) iftError("non read data from binary file","iftReadTrainPixels");
	rd=fread(&nsamplesbi, sizeof(int)  , 1, fileTrainPixels);   if (rd != 1) iftError("non read data from binary file","iftReadTrainPixels");
	rd=fread(&radius    , sizeof(float), 1, fileTrainPixels);   if (rd != 1) iftError("non read data from binary file","iftReadTrainPixels");
	// Initilize train edges
	trainPixels = iftCreateTrainPixels(nimages, nsamplesbi);
	trainPixels->radius = radius;

	// read LabelPixelsData
	for( i = 0; i < trainPixels->nimages; i++) {
		rd=fread(&trainPixels->data[i].nfilename, sizeof(int), 1, fileTrainPixels);   if (rd != 1) iftError("non read data from binary file","iftReadTrainPixels");
		rd=fread(&trainPixels->data[i].n, sizeof(int), 1, fileTrainPixels);   if (rd != 1) iftError("non read data from binary file","iftReadTrainPixels");
		//printf("i:%d, nfilename %d, n: %d\n", i, trainPixels->data[i].nfilename,trainPixels->data[i].n);
		// read filename
		trainPixels->data[i].filename = iftAllocCharArray(trainPixels->data[i].nfilename);
		for( j = 0; j < trainPixels->data[i].nfilename; j++) {
			rd=fread(&trainPixels->data[i].filename[j], sizeof(char), 1, fileTrainPixels);   if (rd != 1) iftError("non read data from binary file","iftReadTrainPixels");
			//printf("j: %d c: %c\n",j,trainPixels->data[i].filename[j]);
		}
		// read labelPixel
		trainPixels->data[i].labelPixel = (iftLabelPixel *) calloc(trainPixels->data[i].n,sizeof(iftLabelPixel));
		for( j = 0; j < trainPixels->data[i].n; j++) {
			rd=fread(&trainPixels->data[i].labelPixel[j].truelabel, sizeof(int), 1, fileTrainPixels);  if (rd != 1) iftError("non read data from binary file","iftReadTrainPixels");
			rd=fread(&trainPixels->data[i].labelPixel[j].p    , sizeof(int), 1, fileTrainPixels);      if (rd != 1) iftError("non read data from binary file","iftReadTrainPixels");
			//printf("j: %d class: %d, p: %d\n",j,trainPixels->data[i].labelPixel[j].truelabel,trainPixels->data[i].labelPixel[j].p);
		}

		if (bReadImages) {

		  // read mimage
		  // OBS the TrainPixels file is inside the images folder
		  char *folder_name;    
		  folder_name = getFolderName(filenameTrainPixels);
		  //printf("%s\n",folder_name);
		  char *fullPath;
		  fullPath = strcat(folder_name,trainPixels->data[i].filename);
		  //printf("%s\n",fullPath);
		  iftImage *img;
		  if (strstr(fullPath,".ppm") != NULL)
		    img = iftReadImageP6(fullPath);
		  else
		    img = iftReadImageP5(fullPath);

		  //trainPixels->data[i].image = iftImageToMImage(img,RGB_CSPACE);
		  trainPixels->data[i].image = iftImageToMImage(img,YCbCr_CSPACE);
		  //trainPixels->data[i].image = iftImageToMImage(img,WEIGHTED_YCbCr_CSPACE);
		}
		else
		  trainPixels->data[i].image = NULL;

	}
	fclose(fileTrainPixels);
	
	return trainPixels;
}

/* Train Pixels to DataSet */
iftDataSet* iftTrainPixelsToDataSet(iftTrainPixels *trainPixels, iftMSConvNetwork* msconvnet){
	int i,b,indexIData,indexPixel;
	iftDataSet *DS;
	// msconvnet first image
	iftMImage*  mimg  = iftApplyMSConvNetwork(trainPixels->data[0].image, msconvnet);
	// Create DataSet
	DS = iftCreateDataSet(trainPixels->data[0].n, mimg->m);
	DS->nclasses = 2;
	// Add the first sample to the DataSet
	DS->sample[0].truelabel = trainPixels->data[0].labelPixel[0].truelabel;
	// get features
	for( b = 0; b < mimg->m; b++) {
		DS->sample[0].feat[b] = mimg->band[b].val[trainPixels->data[0].labelPixel[0].p];
	}
	indexIData = 0;
	indexPixel = 1;
	// Add the other samples to the DataSet
	for( i = 1; i < DS->nsamples; i++) {
		// msconvnet to image
		if((i%(trainPixels->nsamplesbi)) == 0){
			mimg  = iftApplyMSConvNetwork(trainPixels->data[i].image, msconvnet);
			indexIData ++;
			indexPixel = 0;
		}
		// Set Dataset values
		DS->sample[i].truelabel = trainPixels->data[indexIData].labelPixel[indexPixel].truelabel;
		// get features
		for( b = 0; b < mimg->m; b++) {
			DS->sample[i].feat[b] = mimg->band[b].val[trainPixels->data[indexIData].labelPixel[indexPixel].p];
		}
		indexPixel++;
	}
	return DS;
}
/* end Train Pixels to DataSet*/


int iftSelectSupTrainPixels(iftDataSet *Z, iftTrainPixels* trainpixels,float train_perc,float border_train_perc)
{ 
  int s, *imageS=NULL,*imageN=NULL,*count=NULL, i;
  int t, high, nsamples, c, *class_size=NULL,*class_train_size=NULL,*class_train_size_sample=NULL;

  if (Z->nclasses == 0)
    iftError("There are no classes","iftSelectSupTrainPixels");

  if ((train_perc <= 0.0)||(train_perc > 1.0))
    iftError("Invalid percentage of training samples","iftSelectSupTrainPixels");

  if ((border_train_perc <= 0.0)||(border_train_perc > 1.0))
    iftError("Invalid percentage of border samples","iftSelectSupTrainPixels");

  // Reset status
  iftSetStatus(Z, IFT_TEST);

  // Verify if it is the trivial case of selecting all samples.
  if (train_perc == 1.0) {
    for (s=0; s < Z->nsamples; s++) 
      Z->sample[s].status = IFT_TRAIN;
    Z->ntrainsamples = Z->nsamples;  
    return(Z->nsamples);
  }

  // Compute the number of training samples
  Z->ntrainsamples = (int) (train_perc*Z->nsamples);
  if (Z->ntrainsamples == 0)
    iftError("Percentage is too low for this dataset","iftSelectSupTrainPixels");
  
  int ntrainimages = (int)(train_perc * trainpixels->nimages + 0.5);
  if (ntrainimages == 0)
    iftError("Percentage is too low for this trainpixels set","iftSelectSupTrainPixels");

  // Count number of samples per class
  class_size              = iftAllocIntArray(Z->nclasses+1);
  class_train_size        = iftAllocIntArray(Z->nclasses+1);
  class_train_size_sample = iftAllocIntArray(Z->nclasses+1);
  for (s=0; s < Z->nsamples; s++){
    class_size[Z->sample[s].truelabel]++;
  }

  // Verify percentage and number of training samples per class - in this case the iftTranPixels structure should guarantee that property
  Z->ntrainsamples = 0;
  for (c=1; c <= Z->nclasses; c++) {
    nsamples = (int)(train_perc*class_size[c]);
    if (nsamples > class_size[c] || nsamples == 0)
      nsamples = class_size[c];
    Z->ntrainsamples += nsamples;
    if (nsamples <= 0){
      fprintf(stderr,"For class %d\n",c);
      iftError("No available samples","iftSelectSupTrainPixels");
    }
  }

  int desired_classNB = (int) ( (1.-border_train_perc)/border_train_perc * class_size[2] +0.5);
  if (class_size[1] > desired_classNB) {
    class_train_size[1] = desired_classNB;
    class_train_size[2] = class_size[2];
  }
  else if (class_size[1] < desired_classNB) {
    class_train_size[1] = class_size[1];
    class_train_size[2] = (border_train_perc)/(1.-border_train_perc) * class_size[1];
  }
  else {
    class_train_size[1] = class_size[1];
    class_train_size[2] = class_size[2];
  }
  


  // Randomly select images 
  ntrainimages = (int)(train_perc * trainpixels->nimages + 0.5);
  imageS = iftAllocIntArray(trainpixels->nimages);
  imageN = iftAllocIntArray(trainpixels->nimages);
  count = iftAllocIntArray(trainpixels->nimages); 
  for (i=0; i < trainpixels->nimages; i++) {
      imageS[i]= (i == 0) ? 0 : imageS[i-1]+trainpixels->data[i-1].n;
      imageN[i]= trainpixels->data[i].n; 
      count[i] =100;
  }

  for (c=1; c <= Z->nclasses; c++) {
    class_train_size_sample[c] = class_train_size[c] / trainpixels->nimages;
  }

  t = 0; high = trainpixels->nimages-1;
  while (t < ntrainimages) {
    i = iftRandomInteger(0,high);
    if (count[i]==0){
      // then select all samples from that image
      for(s=imageS[i]; s < imageS[i]+imageN[i]; s++) {
	if (class_train_size_sample[Z->sample[s].truelabel]-- > 0) {
	  Z->sample[s].status=IFT_TRAIN;
	}
	else
	  Z->sample[s].status=IFT_OUTLIER;
      }
      class_train_size_sample[1] = class_train_size[1] / trainpixels->nimages;
      class_train_size_sample[2] = class_train_size[2] / trainpixels->nimages;
      
      iftSwitchValues(&imageS[i],&imageS[high]);
      iftSwitchValues(&imageN[i],&imageN[high]);
      iftSwitchValues(&count [i],&count [high]);
      t++; high--;
    }else{
      count[i]--;
    }
  }
  free(imageS);free(imageN);free(count);
  free(class_train_size_sample);
  free(class_train_size);
  free(class_size);

  return(Z->ntrainsamples);
}
