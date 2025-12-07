#include "common.h"
#include "ift.h"
#include "iftTrainPixels.h"

iftMImage *iftImageToMImage02Y(iftImage *img1, char color_space)
{
  iftMImage *img2;
  int p;

  img1->maxval = iftMaximumValue(img1);

  if (iftIsColorImage(img1)) {
    img2=iftCreateMImage(img1->xsize,img1->ysize,img1->zsize,3);

    if (color_space == YCbCr_CSPACE){
      for (p=0; p < img2->n; p++) {
	img2->val[p][0]=0.2*(((float)img1->val[p])/img1->maxval);
	img2->val[p][1]=((float)img1->Cb[p])/img1->maxval;
	img2->val[p][2]=((float)img1->Cr[p])/img1->maxval;
      }
    }else{ /* RGB_CSPACE */
      iftColor  YCbCr, RGB;
      for (p=0; p < img2->n; p++) {
	YCbCr.val[0] = img1->val[p];
	YCbCr.val[1] = img1->Cb[p];
	YCbCr.val[2] = img1->Cr[p];
	RGB          = iftYCbCrtoRGB(YCbCr);
	img2->val[p][0]=((float)RGB.val[0])/img1->maxval;
	img2->val[p][1]=((float)RGB.val[1])/img1->maxval;
	img2->val[p][2]=((float)RGB.val[2])/img1->maxval;
      }
    }
  }else{
    img2=iftCreateMImage(img1->xsize,img1->ysize,img1->zsize,1);
    for (p=0; p < img2->n; p++) {
      img2->val[p][0]=((float)img1->val[p])/img1->maxval;
    }
  }

  img2->dx = img1->dx;
  img2->dy = img1->dy;
  img2->dz = img1->dz;

  return(img2);
}

iftMImage *iftImageToMImage533(iftImage *img1, char color_space)
{
  iftMImage *img2;
  int p;

  img1->maxval = iftMaximumValue(img1);

  if (iftIsColorImage(img1)) {
    img2=iftCreateMImage(img1->xsize,img1->ysize,img1->zsize,11);

    if (color_space == YCbCr_CSPACE){
      for (p=0; p < img2->n; p++) {
	img2->val[p][0]=((float)img1->val[p])/img1->maxval;
	img2->val[p][1]=((float)img1->val[p])/img1->maxval;
	img2->val[p][2]=((float)img1->val[p])/img1->maxval;
	img2->band[3].val[p]=((float)img1->val[p])/img1->maxval;
	img2->band[4].val[p]=((float)img1->val[p])/img1->maxval;
	img2->band[5].val[p]=((float)img1->Cb[p])/img1->maxval;
	img2->band[6].val[p]=((float)img1->Cb[p])/img1->maxval;
	img2->band[7].val[p]=((float)img1->Cb[p])/img1->maxval;
	img2->band[8].val[p]=((float)img1->Cr[p])/img1->maxval;
	img2->band[9].val[p]=((float)img1->Cr[p])/img1->maxval;
	img2->band[10].val[p]=((float)img1->Cr[p])/img1->maxval;
      }
    }else{ /* RGB_CSPACE */
      iftColor  YCbCr, RGB;
      for (p=0; p < img2->n; p++) {
	YCbCr.val[0] = img1->val[p];
	YCbCr.val[1] = img1->Cb[p];
	YCbCr.val[2] = img1->Cr[p];
	RGB          = iftYCbCrtoRGB(YCbCr);
	img2->val[p][0]=((float)RGB.val[0])/img1->maxval;
	img2->val[p][1]=((float)RGB.val[1])/img1->maxval;
	img2->val[p][2]=((float)RGB.val[2])/img1->maxval;
      }
    }
  }else{
    img2=iftCreateMImage(img1->xsize,img1->ysize,img1->zsize,1);
    for (p=0; p < img2->n; p++) {
    }
  }
      img2->val[p][0]=((float)img1->val[p])/img1->maxval;

  img2->dx = img1->dx;
  img2->dy = img1->dy;
  img2->dz = img1->dz;

  return(img2);
}

float iftCheckBorders(iftImage *img, iftImage *label, iftAdjRel *A, iftAdjRel *ACheck)
{
  iftVoxel u,v;
  int i,p,q,br;
  int count,numBorders;
  count = 0;
  numBorders = 0;
  iftImage* BL   = iftCreateImage(img->xsize,img->ysize,img->zsize);
  if ((img->xsize != label->xsize)||
      (img->ysize != label->ysize)||
      (img->zsize != label->zsize))
    iftError("Images must have the same domain","iftCheckBorders");

  // write border of image label
  for (p=0; p < img->n; p++) {
    u.x = iftGetXCoord(label,p);
    u.y = iftGetYCoord(label,p);
    u.z = iftGetZCoord(label,p);
    br = 0;
    for (i=0; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (iftValidVoxel(label,v)){
	q = iftGetVoxelIndex(label,v);
	if (label->val[p] < label->val[q]){
	  BL->val[p] = 255;
	  break;
	}
      }
    }
  }

  // check in the radio
  for (p=0; p < img->n; p++) {
      u.x = iftGetXCoord(label,p);
      u.y = iftGetYCoord(label,p);
      u.z = iftGetZCoord(label,p);
      br = 0;
      for (i=0; i < ACheck->n; i++) {
        v.x = u.x + ACheck->dx[i];
        v.y = u.y + ACheck->dy[i];
        v.z = u.z + ACheck->dz[i];
        if (iftValidVoxel(label,v)){
  	q = iftGetVoxelIndex(label,v);
  	if (img->val[p]==255 && BL->val[q]==255){
  	  br++;
  	  break;
  	}
        }
      }
      if(br>0){
    	  count++;
      }
      if (img->val[p]==255){
    	  numBorders++;
      }
  }


  //printf("match: %d, Total: %d \n",count,numBorders);
  return ((float)count/(float)numBorders);
}

int iftLabelsWG(iftImage *basins, iftImage *pathval, iftAdjRel  *A)
{
  iftImage   *label=NULL;
  iftGQueue  *Q=NULL;
  int         i,p,q,l=1,tmp;
  iftVoxel    u,v;

  // Initialization

  label   = iftCreateImage(basins->xsize,basins->ysize,basins->zsize);
  Q       = iftCreateGQueue(iftMaximumValue(pathval)+2,pathval->n,pathval->val);

  for (p=0; p < basins->n; p++) {
    pathval->val[p] += 1;
    label->val[p]=NIL;
    iftInsertGQueue(&Q,p);
  }

  // Image Foresting Transform

  while(!iftEmptyGQueue(Q)) {
    p=iftRemoveGQueue(Q);

    if (label->val[p]==NIL) { // root voxel
      pathval->val[p]  -= 1;
      label->val[p]=l; l++;
    }

    basins->val[p] = pathval->val[p]; // set the reconstruction value

    u = iftGetVoxelCoord(basins,p);

    for (i=1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(basins,v)){
	q = iftGetVoxelIndex(basins,v);
	if (Q->L.elem[q].color != BLACK){
	  tmp = MAX(pathval->val[p],basins->val[q]);
	  if (tmp < pathval->val[q]){
	    iftRemoveGQueueElem(Q,q);
	    label->val[q]      = label->val[p];
	    pathval->val[q]    = tmp;
	    iftInsertGQueue(&Q, q);
	  }
	}
      }
    }
  }

  iftDestroyGQueue(&Q);
  iftCopyVoxelSize(basins,label);

  return(l-1);
}

int main(int argc, char **argv) 
{
	if (argc != 5)	iftError("Please provide the following parameters:\n <FILE_INPUT_NAMES> <FILE_INPUT_GT_NAMES> <parameters.convnet> <VOLUME>\n\n", "main");

	char *file_inputImages;
	char *file_inputImagesGT;
	char *file_convnet;
	iftConvNetwork *convnet;
	iftMImage *input, *output;
	iftImage       *img,   *orig, *imgGT,   *origGT;
	timer          *t1=NULL,*t2=NULL;
	int idxImg,numSupNew,numSupOld,countPos;
	float vol,perBNew,perBOld,perSup,perB_Total,accNumSupNew,accPerNew,accNumSupOld,accPerOld;


	file_inputImages    = argv[1];
	file_inputImagesGT  = argv[2];
	file_convnet        = argv[3];
	vol 				= atof(argv[4]);


	// Read Convnet
	convnet = iftReadConvNetwork(file_convnet);

	// Read Input images
	int nimages = 0;
	char *folder_name = iftAllocCharArray(256);
	char fullPath[128];
	FileNames * files = NULL;
	
	folder_name = getFolderName(file_inputImages);
	nimages = countImages(file_inputImages);
	files = createFileNames(nimages);
	loadFileNames(files, file_inputImages);

	printf("Input Folder: %s\n", folder_name);
	printf("Number of Images: %d\n", nimages);
	printf("File with image names: %s\n"   , file_inputImages);
	puts("");


	// Read Input GT images
	int nimagesGT = 0;
	char *folder_nameGT = iftAllocCharArray(256);
	FileNames * filesGT = NULL;
	
	folder_nameGT = getFolderName(file_inputImagesGT);
	nimagesGT = countImages(file_inputImagesGT);
	filesGT = createFileNames(nimagesGT);
	loadFileNames(filesGT, file_inputImagesGT);

	// variables for Draw borders
	iftAdjRel *A = iftCircular(sqrtf(2));
	iftColor RGB, YCbCr;
	iftAdjRel *B = iftCircular(0.0);
	iftAdjRel *ACheck = iftCircular(2);

	countPos = 0;
	perSup,perB_Total = 0;

	// For all input images
	for( idxImg = 0; idxImg < nimages; idxImg++)
	{
		// Read input images
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		orig = iftReadImageP6(fullPath);

		// Apply ConvNetwork
		input  = iftImageToMImage533(orig,YCbCr_CSPACE);
		convnet->input_xsize = input->xsize;
		convnet->input_ysize = input->ysize;
		convnet->input_zsize = input->zsize;

		iftCreateAdjRelAlongNetwork(convnet);
		iftMImageIndexMatrices(convnet);
		printf("%d - %s Size: %d x %d x %d - %d \n", idxImg, fullPath,input->xsize,input->ysize, input->zsize, input->m);

		iftAdjRel   *R = iftRectangular(convnet->k_bank_adj_param[0],
								  convnet->k_bank_adj_param[0]);
		printf("before prob\n");
		iftImage    *prob = iftBorderProbImage(input);
		printf("before K\n");
		iftMMKernel *K    = iftUnsupLearnKernels(input,R,4000,0.01,prob);
		printf("before Destroy R\n");
		iftDestroyAdjRel(&R);
		printf("before Destroy prob\n");
		iftDestroyImage(&prob);
		printf("before Destroy MMKernel\n");
		iftDestroyMMKernel(&convnet->k_bank[0]);
		printf("after Destroy MMKernel\n");
		convnet->k_bank[0]   = K;
		convnet->nkernels[0] = K->nkernels;



		t1     = iftTic();
		printf("before apply\n");
		output = iftApplyConvNetwork(input,convnet);
		printf("after apply\n");
		t2     = iftToc();


		// Read input GT images
		fullPath[0] = 0;
		strcpy(fullPath, folder_nameGT);
		strcat(fullPath, filesGT[idxImg].filename);
		origGT = iftReadImageP5(fullPath);

		// for new Watergray
		img = iftMImageBasins(output, A);
		fullPath[0] = 0;
		strcpy(fullPath, "gtwgoutput/");
		strcat(fullPath, filesGT[idxImg].filename);
	  	//strcat(fullPath, ".basins.pgm");
		iftWriteImageP2(img,fullPath);

		iftImage *marker = iftVolumeClose(img,vol);
		iftImage *label  = iftWaterGray(img,marker,A);
		numSupNew = iftLabelsWG(img,marker,A);
		RGB.val[0] = 255;
		RGB.val[1] = 0;
		RGB.val[2] = 0;
		YCbCr      = iftRGBtoYCbCr(RGB);

		int sz = (orig->xsize - img->xsize)/2;
		iftDestroyImage(&img);    
		imgGT        = iftRemFrame(origGT,sz);
		// Count number of matches
		perBNew = iftCheckBorders(imgGT,label,A,ACheck);
		// Draw borders
		iftDrawBorders(imgGT,label,A,YCbCr,B);
		fullPath[0] = 0;
		strcpy(fullPath, "gtwgoutput/");
		strcat(fullPath, filesGT[idxImg].filename);
	  	strcat(fullPath, ".new.pgm");
		//iftWriteImageP6(imgGT,fullPath);

		iftDestroyImage(&img);

		// for Traditional Watergray
		img = iftImageBasins(orig, A);
		iftDestroyImage(&marker);    
		iftDestroyImage(&label);

		marker = iftVolumeClose(img,vol*2);
		label  = iftWaterGray(img,marker,A);
		numSupOld = iftLabelsWG(img,marker,A);
		RGB.val[0] = 0;
		RGB.val[1] = 0;
		RGB.val[2] = 255;
		YCbCr      = iftRGBtoYCbCr(RGB);
		// Count number of matches
		perBOld = iftCheckBorders(origGT,label,A,ACheck);
		// Draw borders
		iftDrawBorders(origGT,label,A,YCbCr,B);
		fullPath[0] = 0;
		strcpy(fullPath, "gtwgoutput/");
		strcat(fullPath, filesGT[idxImg].filename);
	  	strcat(fullPath, ".old.pgm");
		//iftWriteImageP6(origGT,fullPath);


		perSup += (float)numSupNew/numSupOld;
		perB_Total+= (float) perBNew/perBOld;
		if( numSupNew < numSupOld && (perBNew > (perBOld-0.08))){
			countPos++;
		}

		accNumSupNew += numSupNew;
		accNumSupOld += numSupOld;
		accPerNew += perBNew;
		accPerOld += perBOld;

		printf("%s New/Old -> NumSuperPx %d/%d = %f, perBordaPreservada: %f/%f \n", filesGT[idxImg].filename, numSupNew, numSupOld,(float)numSupNew/numSupOld, perBNew, perBOld);
		// Free
		iftDestroyImage(&img);    
		iftDestroyImage(&orig);    
		iftDestroyImage(&marker);    
		iftDestroyImage(&label);
		iftDestroyMImage(&input);
		iftDestroyMImage(&output);
	}
	perSup = perSup / nimages;
	perB_Total = perB_Total / nimages;
	// Print results
	printf("====  Positive Cases : %d/%d  ======\n",countPos,nimages);

	accNumSupNew = accNumSupNew / nimages;
	accNumSupOld = accNumSupOld / nimages;
	accPerNew = accPerNew / nimages;
	accPerOld = accPerOld / nimages;

	printf(" accNumSupNew : %f ,accNumSupOld: %f, accPerNew: %f, accPerOld :%f \n",accNumSupNew,accNumSupOld, accPerNew, accPerOld);

	iftDestroyAdjRel(&A);
	iftDestroyAdjRel(&B);
	iftDestroyConvNetwork(&convnet);


	return 0;
}
