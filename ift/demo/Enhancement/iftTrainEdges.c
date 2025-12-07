#include "iftTrainEdges.h"
#include "common.h"

iftLabelEdgesData  *iftCreateLabelEdgesData(int nedges)
{
	iftLabelEdgesData *le = NULL;

	le = (iftLabelEdgesData *) calloc(1,sizeof(iftLabelEdgesData));
	if (le == NULL)
		iftError(MSG1,"iftCreateLabelEdgesData");
	le->n         = nedges;
	le->labelEdge = (iftLabelEdge *) calloc(le->n,sizeof(iftLabelEdge));

	return(le);
}

iftTrainEdges* iftCreateTrainEdges(int nimages, int nsamplesbi)
{
	iftTrainEdges *Z =(iftTrainEdges*) calloc(1,sizeof(iftTrainEdges));
	int s;
	
	Z->data = (iftLabelEdgesData*) calloc(nimages,sizeof(iftLabelEdgesData));

	if (Z->data == NULL) 
		iftError(MSG1,"iftCreateTrainEdges");

	for (s=0; s < nimages; s++) {
		Z->data[s].filename       = NULL;
		Z->data[s].image          = NULL;
		Z->data[s].labelEdge     = NULL;
	}

	Z->nimages    = nimages;
	Z->nsamplesbi = nsamplesbi;

	return(Z);
}

void iftDestroyTrainEdges(iftTrainEdges **Z)
{
	iftTrainEdges *aux = *Z;
	int s;

	if (aux != NULL) {
		for (s=0; s < aux->nimages; s++) {
			if (aux->data[s].filename)  free(aux->data[s].filename );
			if (aux->data[s].labelEdge) free(aux->data[s].labelEdge);
			iftDestroyMImage(&(aux->data[s].image));			
		}
		free(aux->data);
		free(aux);
		*Z = NULL;
	}
}
/* end Data Structures for iftTrainEdges */

/* Extract Label Edges */
iftLabelEdgesData* iftExtractLabelEdgesWG(iftMImage *mimg,iftImage *imgGT, iftAdjRel* A,int nsamplesbi,float percBorder, float radius)
{
	// create label edges
 	int nsamplesEdgesbi = (int) (nsamplesbi*percBorder + 0.5);
	iftLabelEdgesData *LE;
	LE = iftCreateLabelEdgesData(nsamplesbi);
        printf("Creating label Edge: %d \n",LE->n);
	
	// Label image components from imgGT - borders must have 0 value
	iftAdjRel* adjcomp = iftCircular(1.0);
	iftImage *imgNGT;
	iftImage *imglabels;
//	iftImage *exemplo;

	imgNGT=iftCreateImage(imgGT->xsize,imgGT->ysize,imgGT->zsize);
    	int k;
    	iftCopyVoxelSize(imgGT,imgNGT);
    	for (k=0; k < imgNGT->n; k++) {
       	  imgNGT->val[k]=(255-imgGT->val[k]);
	}
	imglabels = iftRelabelRegions(imgNGT,adjcomp);
	iftWriteImageP5(imglabels,"labels.pgm");
	iftDestroyAdjRel(&adjcomp);

	// Creating imgWG
//	iftImage *imgGW;
	iftImage *img;
	img = iftMImageToImage(mimg,255,0);
	float spatial_radius = 1.5;
	float volume_threshold = 2000.;
	iftAdjRel* adj_relation = iftCircular(spatial_radius);
	iftImage* basins = iftImageBasins(img,adj_relation);
	
	int nregions = -1;
	iftImage* marker=NULL; iftImage* imgWG=NULL;
	while(nregions < 0)
	{
		marker = iftVolumeClose(basins,volume_threshold);
		imgWG = iftWaterGray(basins,marker,adj_relation);
		nregions = iftMaximumValue(imgWG);
		printf("volume_threshold: %f, nregions: %d\n",volume_threshold,nregions);
		volume_threshold *= 2;
	}
	volume_threshold /= 2;
	iftWriteImageP2(imgWG,"watergray.pgm");
	iftDestroyImage(&basins);
	iftDestroyImage(&marker);
	iftDestroyAdjRel(&adj_relation);
	

	// generating edges
	int s,p,q,r,i;
	int numB,countB;
	int numNB,countNB;
	numB = nsamplesEdgesbi;
	numNB = (int)nsamplesbi-nsamplesEdgesbi;
	countB = 0;
	countNB = 0;

	int numEdgesTotal = mimg->n*(A->n-1);
	//int numEdgesTotal = mimg->n;
	int* selectedPIX = iftAllocIntArray(numEdgesTotal);

	// extract label edges
	s = 0;
	printf("nsamples: %d, non edges: %d, edges:%d (percBorder: %f)\n",nsamplesbi,numB,numNB,percBorder);
	while(s < nsamplesbi){
		// random pixel
		r = iftRandomInteger(0,numEdgesTotal-1);
		// verify random pixel
		if (selectedPIX[r] == 0){
			selectedPIX[r] = 1;
			//p = r;
			p = r / (A->n-1);
			i = r % (A->n-1) + 1;
			iftVoxel pixel   = iftMGetVoxelCoord(mimg ,p);
		//for (i=1; i < A->n; i++) {

			iftVoxel neighbor   = iftGetAdjacentVoxel(A,pixel,i);
			if (iftMValidVoxel(mimg,neighbor))
				q = iftMGetVoxelIndex(mimg ,neighbor);
			else
				continue;
			if ((countB < numB) && (imglabels->val[p] != imglabels->val[q]) && (imgWG->val[p] != imgWG->val[q]) ){ 
				// set edge attributes Border
				LE->labelEdge[s].truelabel = 2;
				LE->labelEdge[s].p = p;
				LE->labelEdge[s].q = q;
				s++;
				countB++;
				
			}
			else if ((countNB < numNB) && (imglabels->val[p] == imglabels->val[q]) && (imgWG->val[p] == imgWG->val[q]))
			{ 
				// set edge attributes Non Border
				LE->labelEdge[s].truelabel = 1;
				LE->labelEdge[s].p = p;
				LE->labelEdge[s].q = q;
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
		//}
		}
	}
	
	free(selectedPIX);
	
	return LE;
}
/* end Extract LabelEdges*/

/* Extract Label Edges */
iftLabelEdgesData* iftExtractLabelEdges(iftMImage *mimg,iftImage *imgGT, iftAdjRel* A,int nsamplesbi ,float percBorder, float radius)
{
	// create label edges
 	int nsamplesEdgesbi = (int) (nsamplesbi*percBorder + 0.5);
	iftLabelEdgesData *LE;
	LE = iftCreateLabelEdgesData(nsamplesbi);
        printf("Creating label Edge: %d \n",LE->n);
	
	// Label image components from BorderImageToLabelImage
	iftAdjRel* adjcomp = iftCircular(radius);
	iftImage *img,*basins,*imglabels;
	img = iftMImageToImage(mimg,255,0);
	basins = iftImageBasins(img,adjcomp);
	imglabels = iftBorderImageToLabelImage(basins,imgGT,radius);
	iftWriteImageP5(imglabels,"labels.pgm");
	iftDestroyAdjRel(&adjcomp);

	// generating edges
	int s,p,q,r,i;
	int numB,countB;
	int numNB,countNB;
	numB = nsamplesEdgesbi;
	numNB = (int)nsamplesbi-nsamplesEdgesbi;
	countB = 0;
	countNB = 0;

	//int numEdgesTotal = mimg->n;
	int numEdgesTotal = mimg->n*(A->n-1);
	int* selectedPIX = iftAllocIntArray(numEdgesTotal);

	// extract label edges
	s = 0;
	printf("nsamples: %d, non edges: %d, edges:%d (percBorder: %f)\n",nsamplesbi,numB,numNB,percBorder);
	while(s < nsamplesbi){
		// random pixel
		r = iftRandomInteger(0,numEdgesTotal-1);
		// verify random pixel
		if (selectedPIX[r] == 0){
			selectedPIX[r] = 1;
			//p = r;
			p = r / (A->n-1);
			i = r % (A->n-1) + 1;
			iftVoxel pixel   = iftMGetVoxelCoord(mimg ,p);
		//for (i=1; i < A->n; i++) {
			iftVoxel neighbor   = iftGetAdjacentVoxel(A,pixel,i);
			if (iftMValidVoxel(mimg,neighbor))
				q = iftMGetVoxelIndex(mimg ,neighbor);
			else
				continue; // q  = p;

			if ((countB < numB) && (imglabels->val[p] != imglabels->val[q])){ 
				// set edge attributes Border
				LE->labelEdge[s].truelabel = 2;
				LE->labelEdge[s].p = p;
				LE->labelEdge[s].q = q;
				//printf("s: %6d, b: %4d, p: %7d, q:%7d",s,b,p,q);
				s++;
				countB++;
			}
			else if ((countNB < numNB) && (imglabels->val[p] == imglabels->val[q]))
			{ 
				// set edge attributes Non Border
				LE->labelEdge[s].truelabel = 1;
				LE->labelEdge[s].p = p;
				LE->labelEdge[s].q = q;
				//printf("s: %6d, b: %4d, p: %7d, q:%7d",s,b,p,q);
				s++;
				countNB++;
			}
		//}
		}
	}
	
	free(selectedPIX);
	
	return LE;
}
/* end Extract LabelEdges*/

/* Extract train edges */
iftTrainEdges* iftExtractTrainEdges(char *file_imageNames,char *file_imageNamesGT,int nsamplesbi ,float percBorder, float radius)
{
	int idxImg;
	int nimages,nimagesGT = 0;
	char *folder_name = iftAllocCharArray(256);
	char *folder_nameGT = iftAllocCharArray(256);
	char fullPath[128];
	FileNames * files = NULL;
	FileNames * filesGT = NULL;
	iftTrainEdges *TE;
	iftLabelEdgesData *auxLE;

	int nsamplesEdgesbi = (int)(nsamplesbi*percBorder + 0.5);

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
	printf("Number of samples per image: %d (%d-Border/%d-NonBorder %6.2f%%)\n",nsamplesbi,nsamplesEdgesbi,nsamplesbi-nsamplesEdgesbi,100.*percBorder);	
	printf("Spatial radius: %.4f\n",radius);	
	puts("");


	// initialize TE
	TE = iftCreateTrainEdges(nimages, nsamplesbi);
	TE -> radius = radius;
	// define adjacency
	iftAdjRel* adj_relation = iftCircularEdges(radius);

	for( idxImg = 0; idxImg < TE->nimages; idxImg++){
		// read image
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		printf("%s - %s\n",folder_name,files[idxImg].filename);
		strcat(fullPath, files[idxImg].filename);
		iftImage *img = iftReadImageP5(fullPath);
		// read image GT
		fullPath[0] = 0;
		strcpy(fullPath, folder_nameGT);
		strcat(fullPath, filesGT[idxImg].filename);
		iftImage *imgGT = iftReadImageP5(fullPath);
		// Generating Deep Features and Edges
		//iftMImage*  mimg  = iftExtractDeepFeaturesMultiScale(img, msconvnet);
		iftMImage*  mimg  = iftImageToMImage(img,'R');
		// extract edges and its labels
		auxLE = iftExtractLabelEdges(mimg,imgGT,adj_relation,nsamplesEdgesbi,percBorder,radius);
                TE->data[idxImg].labelEdge = auxLE->labelEdge;
		// set image and label edges into train edges
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
void iftWriteTrainEdges(iftTrainEdges *trainEdges,char *filenameTrainEdges) {
  int i,j;
  FILE *fileTrainEdges = fopen(filenameTrainEdges, "wb");
  if ( fileTrainEdges != NULL ){
    // write TrainEdges
    fwrite(&trainEdges->nimages, sizeof(int), 1, fileTrainEdges);
    fwrite(&trainEdges->nsamplesbi, sizeof(int), 1, fileTrainEdges);
    fwrite(&trainEdges->radius, sizeof(float), 1, fileTrainEdges);
    // write LabelEdgesData
    for( i = 0; i < trainEdges->nimages; i++) {
      fwrite(&trainEdges->data[i].nfilename, sizeof(int), 1, fileTrainEdges);
      fwrite(&trainEdges->data[i].n, sizeof(int), 1, fileTrainEdges);
      // write filename
      printf("file: %s\n",trainEdges->data[i].filename);
      for( j = 0; j < trainEdges->data[i].nfilename; j++) {
        fwrite(&trainEdges->data[i].filename[j], sizeof(char), 1, fileTrainEdges);
      }
      // write labelEdge
      for( j = 0; j < trainEdges->data[i].n; j++) {
        fwrite(&trainEdges->data[i].labelEdge[j].truelabel, sizeof(int), 1, fileTrainEdges);
        fwrite(&trainEdges->data[i].labelEdge[j].p, sizeof(int), 1, fileTrainEdges);
        fwrite(&trainEdges->data[i].labelEdge[j].q, sizeof(int), 1, fileTrainEdges);
      }
    }
  }
  fclose(fileTrainEdges);
  
}
/* end Write train edges*/

/* Read train edges*/
iftTrainEdges* iftReadTrainEdges(char *filenameTrainEdges) {
  FILE *fileTrainEdges;
  iftTrainEdges *trainEdges;
  int i,j,rd;
  int nimages,nsamplesbi;
  float radius; 
  printf("filenameTrainEdges: %s \n",filenameTrainEdges);
  //read from the file in binary mode 
  fileTrainEdges = fopen(filenameTrainEdges, "rb"); 
  rd=fread(&nimages   , sizeof(int)  , 1, fileTrainEdges);   if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
  rd=fread(&nsamplesbi, sizeof(int)  , 1, fileTrainEdges);   if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
  rd=fread(&radius    , sizeof(float), 1, fileTrainEdges);   if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
  // Initilize train edges
  trainEdges = iftCreateTrainEdges(nimages, nsamplesbi);
  trainEdges->radius = radius;
  // read LabelEdgesData
  for( i = 0; i < trainEdges->nimages; i++) {
    rd=fread(&trainEdges->data[i].nfilename, sizeof(int), 1, fileTrainEdges);   if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
    rd=fread(&trainEdges->data[i].n, sizeof(int), 1, fileTrainEdges);   if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
    //printf("i:%d, nfilename %d, n: %d\n", i, trainEdges->data[i].nfilename,trainEdges->data[i].n);
    // read filename
    trainEdges->data[i].filename = iftAllocCharArray(trainEdges->data[i].nfilename);
    for( j = 0; j < trainEdges->data[i].nfilename; j++) {
      rd=fread(&trainEdges->data[i].filename[j], sizeof(char), 1, fileTrainEdges);   if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
      //printf("j: %d c: %c\n",j,trainEdges->data[i].filename[j]);
    }
    // read labelEdge
    trainEdges->data[i].labelEdge = (iftLabelEdge *) calloc(trainEdges->data[i].n,sizeof(iftLabelEdge));
    for( j = 0; j < trainEdges->data[i].n; j++) {
      rd=fread(&trainEdges->data[i].labelEdge[j].truelabel, sizeof(int), 1, fileTrainEdges);  if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
      rd=fread(&trainEdges->data[i].labelEdge[j].p    , sizeof(int), 1, fileTrainEdges);      if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
      rd=fread(&trainEdges->data[i].labelEdge[j].q    , sizeof(int), 1, fileTrainEdges);      if (rd != 1) iftError("non read data from binary file","iftReadTrainEdges");
      //printf("j: %d class: %d, p: %d ,q: %d \n",j,trainEdges->data[i].labelEdge[j].truelabel,trainEdges->data[i].labelEdge[j].p,trainEdges->data[i].labelEdge[j].q);
    }
    // read mimage
    // OBS the TrainEdges file is inside the images folder
    char *folder_name;    
    folder_name = getFolderName(filenameTrainEdges);
    printf("%s\n",folder_name);
    char *fullPath;
    fullPath = strcat(folder_name,trainEdges->data[i].filename);
    printf("%s\n",fullPath);
    iftImage *img = iftReadImageP5(fullPath);
    trainEdges->data[i].image = iftImageToMImage(img,'R');
  }
  fclose(fileTrainEdges);
  
  return trainEdges;
}

/* Train Edges to DataSet */
iftDataSet* iftTrainEdgesToDataSet(iftTrainEdges *trainEdges, iftMSConvNetwork* msconvnet){
  int i,b,indexIData,indexEdge;
  iftDataSet *DS;
  // msconvnet first image
  iftMImage*  mimg  = iftApplyMSConvNetwork(trainEdges->data[0].image, msconvnet);
  // Create DataSet
  DS = iftCreateDataSet(trainEdges->data[0].n, mimg->m);
  DS->nclasses = 2;
  // Add the first sample to the DataSet
  DS->sample[0].truelabel = trainEdges->data[0].labelEdge[0].truelabel;
  // get features
  for( b = 0; b < mimg->m; b++) {
    DS->sample[0].feat[b] = mimg->band[b].val[trainEdges->data[0].labelEdge[0].p] - mimg->band[b].val[trainEdges->data[0].labelEdge[0].q];
  }
  indexIData = 0;
  indexEdge = 1;
  // Add the other samples to the DataSet
  for( i = 1; i < DS->nsamples; i++) {
    // msconvnet to image
    if((i%(trainEdges->nsamplesbi)) == 0){
       mimg  = iftApplyMSConvNetwork(trainEdges->data[i].image, msconvnet);
       indexIData ++;
       indexEdge = 0;
    }
    // Set Dataset values
    DS->sample[i].truelabel = trainEdges->data[indexIData].labelEdge[indexEdge].truelabel;
    // get features
    for( b = 0; b < mimg->m; b++) {
      DS->sample[i].feat[b] = mimg->band[b].val[trainEdges->data[indexIData].labelEdge[indexEdge].p] - mimg->band[b].val[trainEdges->data[indexIData].labelEdge[indexEdge].q];
    }
    indexEdge++;
  }
  return DS;
}
/* end Train Edges to DataSet*/