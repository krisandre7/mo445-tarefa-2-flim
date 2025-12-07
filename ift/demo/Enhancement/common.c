/* Extract file names functions */
#include "ift.h"
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FileNames * createFileNames (int nimages) {
	FileNames * ITS = (FileNames *) malloc (nimages * sizeof(FileNames));
	return (ITS);
}

void destroyFileNames (FileNames * ITS) {
	if (ITS != NULL)
	{
		free(ITS);
		ITS = NULL;
	}
}

//Count the number of images in the file with the image names
int countImages (char * file_imageNames) {
	char line[512],*rd=NULL;
	FILE *fp = fopen(file_imageNames, "r");
	if (!fp) iftError("File not found", "msconvnet - countImages");
	int nlines = 0;
	while(!feof(fp))
	{
		rd=fgets(line, sizeof(char)*512, fp);
		if ( (rd == NULL) && (line[0] == 0) & feof(fp) )
			break; // eof of file with a blank line
		if (rd == NULL) iftError("error while reading from the config file", "msconvnet - countImages");
		
		if (strlen(line) > 1)
			nlines++;
		line[0] = 0;
	}
	fclose(fp);
	return nlines;
}

//Get image database path based on the file with the image names
char * getFolderName (char * file_imageNames) {
	int i=0;
	int position = -1;
	char * output = iftAllocCharArray(256);
	for (i =0; i<strlen(file_imageNames); i++)
		if (file_imageNames[i] == '/')
			position = i;
	if (position != -1)
	{
		for (i= 0; i < position; i++)
			output[i] = file_imageNames[i];
		output[i] = 0;
		strcat(output, "/");
		return output;
	}
	else
	{
		iftError("The file with the name and class must be inside the folder with the images", "msconvnet - getFolderName");
		return output;
	}
}

//Load all the image names and classes in the FileNames structure
void loadFileNames (FileNames * FN, char file_imageNames[]) {
	char line[512];
	int i=0;
	char* rd=NULL;
	
	FILE *fp = fopen(file_imageNames, "r");
	
	if (!fp) iftError("File not found", "msconvnet - loadFileNames");
	int nimages = countImages(file_imageNames);
	while(!feof(fp) && i < nimages) {
		rd=fgets(line, sizeof(char)*512, fp);	
		if (rd == NULL) iftError("error while reading from the config file", ",msconvnet - loadFileNames");
		
		if (strlen(line) <= 1)
			continue;
		strcpy(FN[i].filename, line);
		if(FN[i].filename[strlen(FN[i].filename)-1] == '\n')
			FN[i].filename[strlen(FN[i].filename)-1] = '\0';
		
		i++;
	}
	fclose(fp);
//	printf("finish to read\n");
}
/* end Extract file names functions */


//Parse the config file to create an convolutional network
iftMSConvNetwork* parseConfigFile (char * file_config)
{
	FILE *fp = fopen(file_config, "r");
	if (!fp) iftError("File not found", "msconvnet - parseConfigFile");
	char line[512];
	char *pch;

	int bConv=1;
	int bPool=1;
	int bNorm=1;
	
	int nscales=-1;
	int nlayers=-1;

	float *scale_ratio       = 0;
	int   input_bands        = 0;
	int   *input_norm_ws     = 0;
	int   output_norm_ws     = 0;
	int   **nkernels         = 0;
	int   **kernel_ws        = 0;
	int   **pooling_ws       = 0;
	int   **stride           = 0;
	float **alpha            = 0;
	int   **normalization_ws = 0;

	int i=0,j,k;
	char* rd=NULL;
	while (!feof(fp))
	{
		j=0;k=0;
		rd=fgets(line, sizeof(char)*512, fp);
		if (rd == NULL) iftError("error while reading from the config file", "msconvnet - parseConfigFile");
		
		if (i >0 && nscales == -1) iftError("N_SCALES must be the first parameter on the config file", "parseConfigFile");
		
		pch = strtok (line,", ");
		if (strcmp(pch, "N_SCALES")==0)
		{
			pch = strtok (NULL,", ");
			nscales = atoi(pch);

			scale_ratio      = (float*)  calloc(nscales,sizeof(float) );
			input_norm_ws    = (int*)    calloc(nscales,sizeof(int)   );
			nkernels         = (int**)   calloc(nscales,sizeof(int*)  );
			kernel_ws        = (int**)   calloc(nscales,sizeof(int*)  );
			pooling_ws       = (int**)   calloc(nscales,sizeof(int*)  );
			stride           = (int**)   calloc(nscales,sizeof(int*)  );
			alpha            = (float**) calloc(nscales,sizeof(float*));
			normalization_ws = (int**)   calloc(nscales,sizeof(int*)  );
			
		}
		else if (strcmp(pch, "N_LAYERS")==0)
		{
			if (nscales == -1) iftError("N_SCALES should be defined first", " parseConfigFile");

			pch = strtok (NULL,", ");
			nlayers    = atoi(pch);

			while (j < nscales)
			{
				nkernels        [j] = iftAllocIntArray(nlayers);
				kernel_ws       [j] = iftAllocIntArray(nlayers);
				pooling_ws      [j] = iftAllocIntArray(nlayers);
				stride          [j] = iftAllocIntArray(nlayers);
				alpha           [j] = iftAllocFloatArray(nlayers);
				normalization_ws[j] = iftAllocIntArray(nlayers);
				j++;
			}
		}
		else if (strcmp(pch, "NO_CONVOLUTION")==0)
		{
			bConv = 0;   
		}
		else if (strcmp(pch, "NO_POOLING")==0)
		{
			bPool = 0;   
		}
		else if (strcmp(pch, "NO_NORMALIZATION")==0)
		{
			bNorm = 0;   
		}
		else if (strcmp(pch, "N_BANDS_INPUT")==0)
		{
			pch = strtok (NULL,", ");
			input_bands = atoi(pch);
		}
		else if (strcmp(pch, "SCALE_RATIO")==0)
		{
			while (j < nscales)
			{
				pch = strtok (NULL,", ");
				scale_ratio[j] = (float)atof(pch);
				j++;
			}
		}
		else if (strcmp(pch, "INPUT_NORM_SIZE")==0)
		{
			while (j < nscales)
			{
				pch = strtok (NULL,", ");
				input_norm_ws[j] = atoi(pch);
				j++;
			}
		}
		else if (strcmp(pch, "OUTPUT_NORM_SIZE")==0)
		{
			pch = strtok (NULL,", ");
			output_norm_ws = atoi(pch);
		}
		else if (strcmp(pch, "N_KERNELS")==0)
		{
			while (j < nscales)
			{
				while (k < nlayers)
				{
					pch = strtok (NULL,", ");
					nkernels[j][k] = atoi(pch);
					k++;
				}
				j++;k=0;
			}
		}
		else if (strcmp(pch, "SIZE_KERNELS")==0)
		{
			while (j < nscales)
			{
				while (k < nlayers)
				{
					pch = strtok (NULL,", ");
					kernel_ws[j][k] = atoi(pch);
					k++;
				}
				j++;k=0;
			}
		}
		else if (strcmp(pch, "SIZE_POOLING")==0)
		{
			while (j < nscales)
			{
				while (k < nlayers)
				{
					pch = strtok (NULL,", ");
					pooling_ws[j][k] = atoi(pch);
					k++;
				}
				j++;k=0;
			}
		}
		else if (strcmp(pch, "STRIDE")==0)
		{
			while (j < nscales)
			{
				while (k < nlayers)
				{
					pch = strtok (NULL,", ");
					stride[j][k] = atoi(pch);
					k++;
				}
				j++;k=0;
			}
		}
		else if (strcmp(pch, "ALPHA")==0)
		{
			while (j < nscales)
			{
				while (k < nlayers)
				{
					pch = strtok (NULL,", ");
					alpha[j][k] = atoi(pch);
					k++;
				}
				j++;k=0;
			}
		}
		else if (strcmp(pch, "SIZE_NORM")==0)
		{
			while (j < nscales)
			{
				while (k < nlayers)
				{
					pch = strtok (NULL,", ");
					normalization_ws[j][k] = atoi(pch);
					k++;
				}
				j++;k=0;
			}
		}

		
		line[0] = 0;
		i++;
	}
	fclose(fp);
	
	if (!bConv)
	{
		if (nkernels        )	for(j=0;j<nscales;j++) free(nkernels[j]        );	free(nkernels        );	nkernels         = 0;
		if (kernel_ws       )	for(j=0;j<nscales;j++) free(kernel_ws[j]       );	free(kernel_ws       );	kernel_ws        = 0;
	}
	if (!bPool)
	{
		if (pooling_ws      )	for(j=0;j<nscales;j++) free(pooling_ws[j]      );	free(pooling_ws      );	pooling_ws       = 0;
		if (stride          )	for(j=0;j<nscales;j++) free(stride[j]          );	free(stride          );	stride           = 0;
		if (alpha           )	for(j=0;j<nscales;j++) free(alpha[j]           );	free(alpha           );	alpha            = 0;
	}
	if (!bNorm)
	{
		if (normalization_ws)	for(j=0;j<nscales;j++) free(normalization_ws[j]);	free(normalization_ws);	normalization_ws = 0;
	}

	printf("\nNumber of Layers: %d", nlayers);
	printf("\nNumber of Scales: %d", nscales);
	printf("\nNumber of Bands: %d\n", input_bands);
	printf("\nScale ratio: ");
	for (j=0; j < nscales; j++)
		printf("%.2f ", scale_ratio[j]);
	printf("\nInput Norm. W.Size: ");
	for (j=0; j < nscales; j++)
		printf("%d ", input_norm_ws[j]);
	if (bConv)
	{
		printf("\nNumber of Kernels: ");
		for (j=0; j < nscales; j++)
		{
			printf("%d| ",j);
			for(k=0; k < nlayers; k++)
				printf("%d ",nkernels[j][k]);
			if (j != nscales-1) 
				printf(" ");
		}
		
		printf("\nSize of Kernels: ");
		for (j=0; j < nscales; j++)
		{
			printf("%d| ",j);
			for(k=0; k < nlayers; k++)
				printf("%d ",kernel_ws[j][k]);
			if (j != nscales-1) 
				printf(" ");
		}
	}
	if (bPool)
	{
		printf("\nSize of Pooling: ");
		for (j=0; j < nscales; j++)
		{
			printf("%d| ",j);
			for(k=0; k < nlayers; k++)
				printf("%d ",pooling_ws[j][k]);
			if (j != nscales-1) 
				printf(" ");
		}
		printf("\nStride: ");
		for (j=0; j < nscales; j++)
		{
			printf("%d| ",j);
			for(k=0; k < nlayers; k++)
				printf("%d ",stride[j][k]);
			if (j != nscales-1) 
				printf(" ");
		}
		printf("\nAlpha: ");
		for (j=0; j < nscales; j++)
		{
			printf("%d| ",j);
			for(k=0; k < nlayers; k++)
				printf("%.2f ",alpha[j][k]);
			if (j != nscales-1) 
				printf(" ");
		}
	}
	if (bNorm)
	{
		printf("\nNorm. W.Size: ");
		for (j=0; j < nscales; j++)
		{
			printf("%d| ",j);
			for(k=0; k < nlayers; k++)
				printf("%d ", normalization_ws[j][k]);
			if (j != nscales-1) 
				printf(" ");
		}
	}
	printf("\nOutput Norm. W.Size: %d", output_norm_ws);
	printf("\n--------------\n");

	//iftCreateMSConvNetwork* msconvnet = iftCreateMSConvNetwork(nlayers, nscales, scale_ratio, input_bands,input_norm_ws, nkernels,kernel_ws,pooling_ws,stride,alpha,normalization_ws,output_norm_ws);
	iftMSConvNetwork* msconvnet = iftCreateMSConvNetwork(nscales);

	if (scale_ratio     )	                                                 	free(scale_ratio     );	scale_ratio      = 0;
	if (input_norm_ws   )	                                                 	free(input_norm_ws   );	input_norm_ws    = 0;
	if (nkernels        )	for(j=0;j<nscales;j++) free(nkernels[j]        );	free(nkernels        );	nkernels         = 0;
	if (kernel_ws       )	for(j=0;j<nscales;j++) free(kernel_ws[j]       );	free(kernel_ws       );	kernel_ws        = 0;
	if (pooling_ws      )	for(j=0;j<nscales;j++) free(pooling_ws[j]      );	free(pooling_ws      );	pooling_ws       = 0;
	if (stride          )	for(j=0;j<nscales;j++) free(stride[j]          );	free(stride          );	stride           = 0;
	if (alpha           )	for(j=0;j<nscales;j++) free(alpha[j]           );	free(alpha           );	alpha            = 0;
	if (normalization_ws)	for(j=0;j<nscales;j++) free(normalization_ws[j]);	free(normalization_ws);	normalization_ws = 0;
       
	return msconvnet;
}

void iftCreateRandomWhitenedKernelBanks(iftConvNetwork *convnet) 
{
  iftAdjRel *A;
  int nbands[convnet->nlayers];

  nbands[0] = convnet->input_nbands; 
  for (int l=1; l < convnet->nlayers; l++) 
    nbands[l] = convnet->nkernels[l-1]; 

  for (int l=0; l < convnet->nlayers; l++) {
    if (convnet->input_zsize != 1) 
      A = iftCuboid(convnet->k_bank_adj_param[l],convnet->k_bank_adj_param[l],convnet->k_bank_adj_param[l]);
    else
      A = iftRectangular(convnet->k_bank_adj_param[l],convnet->k_bank_adj_param[l]);

    convnet->k_bank[l] = iftRandomZMMMKernel(A, nbands[l], convnet->nkernels[l]);
    /*
    for(int i=0;i<convnet->nkernels[l];i++)
    {
      for(int m=0;m<nbands[l];m++) {
        float mean=0.;
	for(int p=0;p<A->n;p++) {
	  mean+=convnet->k_bank[l]->weight[i][m].val[p];
	  fprintf(stderr,"%.4f ",convnet->k_bank[l]->weight[i][m].val[p]);
	}
        fprintf(stderr,"mu: %.4f",mean);
      }
      fprintf(stderr,"\n");
    }
    */
    //    fprintf(stderr,"whitening..start\n");
    iftDataSet* Z1 = iftMMKernelToDataSet( convnet->k_bank[l] );
    iftDataSet* Z2 = iftWhiteningTransform(Z1);
    iftMMKernel* mmkerneltmp = iftDataSetToMMKernel(Z2);
    iftDestroyMMKernel( &(convnet->k_bank[l]) );
    convnet->k_bank[l] = mmkerneltmp;

    /* norm 1 */
   
    for(int k=0;k < convnet->nkernels[l];k++) {
      float norm=0.;
      for (int b=0; b < nbands[l]; b++) {
	for (int i=0; i < A->n; i++) {
	  norm += (convnet->k_bank[l]->weight[k][b].val[i]*convnet->k_bank[l]->weight[k][b].val[i]);
	}
      }
      norm = sqrtf(norm);
      if (norm > Epsilon) {
	for (int b=0; b < nbands[l]; b++) {
	  for (int i=0; i < A->n; i++) {
	    convnet->k_bank[l]->weight[k][b].val[i] /= norm;
	  }
	}
      }
    }

    /*
    for(int i=0;i<convnet->nkernels[l];i++)
    {
      for(int m=0;m<nbands[l];m++) {
        float mean=0.;
	for(int p=0;p<A->n;p++) {
	  mean+=convnet->k_bank[l]->weight[i][m].val[p];
	  fprintf(stderr,"%.4f ",convnet->k_bank[l]->weight[i][m].val[p]);
	}
        fprintf(stderr,"mu: %.4f",mean);
      }
      fprintf(stderr,"\n");
    }
    */
    iftDestroyDataSet(&Z1);
    iftDestroyDataSet(&Z2);
    iftDestroyAdjRel(&A);
  }

}

void iftCreateOnesKernelBanks(iftConvNetwork *convnet) 
{
  iftAdjRel *A;
  int nbands[convnet->nlayers],nkernels[convnet->nlayers];

  nbands[0] = convnet->input_nbands; 
  for (int l=1; l < convnet->nlayers; l++) 
    nbands[l] = convnet->nkernels[l-1]; 

  for (int l=0; l < convnet->nlayers; l++) {
    if (convnet->input_zsize != 1) 
      nkernels[l] = convnet->k_bank_adj_param[l]*convnet->k_bank_adj_param[l]*convnet->k_bank_adj_param[l]*nbands[l];
    else
      nkernels[l] = convnet->k_bank_adj_param[l]*convnet->k_bank_adj_param[l]*nbands[l];
  }

  for (int l=0; l < convnet->nlayers; l++) {
    if (convnet->input_zsize != 1) 
      A = iftCuboid(convnet->k_bank_adj_param[l],convnet->k_bank_adj_param[l],convnet->k_bank_adj_param[l]);
    else
      A = iftRectangular(convnet->k_bank_adj_param[l],convnet->k_bank_adj_param[l]);

    convnet->k_bank[l] = iftOnesMMKernel(A, nbands[l], nkernels[l]);
    convnet->nkernels[l] = nkernels[l];
    iftDestroyAdjRel(&A);
  }
}


iftImage* iftWaterGrayBorder(iftImage *label_image,iftAdjRel* A){
  int p,j,q;
  iftVoxel u,v;
  iftImage *grad = iftCreateImage(label_image->xsize,label_image->ysize,label_image->zsize);
  for (p=0; p < label_image->n; p++) {
    u.x = iftGetXCoord(label_image,p);
    u.y = iftGetYCoord(label_image,p);
    u.z = iftGetZCoord(label_image,p);
    for (j=0; j < A->n; j++) {
      v.x = u.x + A->dx[j];
      v.y = u.y + A->dy[j];
      v.z = u.z + A->dz[j];
      if (iftValidVoxel(label_image,v)){
	q = iftGetVoxelIndex(label_image,v);
	if (label_image->val[p] < label_image->val[q]){
	  grad->val[p] = 255;
	  break;
	}
      }
    }
  }
  return grad;
}

iftImage* iftWaterGrayIBorder(iftImage* gradient_image,iftImage *label_image,iftAdjRel* A){
  int p,j,q;
  iftVoxel u,v;

  if ( (gradient_image->xsize != label_image->xsize) ||
       (gradient_image->ysize != label_image->ysize) ||
       (gradient_image->zsize != label_image->zsize) )
    iftError("Image dimensions are incompatible","iftWaterGrayBorder");

  for (p=0; p < label_image->n; p++) {
    u.x = iftGetXCoord(label_image,p);
    u.y = iftGetYCoord(label_image,p);
    u.z = iftGetZCoord(label_image,p);
    for (j=0; j < A->n; j++) {
      v.x = u.x + A->dx[j];
      v.y = u.y + A->dy[j];
      v.z = u.z + A->dz[j];
      if (iftValidVoxel(label_image,v)){
	q = iftGetVoxelIndex(label_image,v);
	if (label_image->val[p] < label_image->val[q]){
	  break;
	}
      }
    }
    if (j == A->n)
      gradient_image->val[p] = 0.;
  }

  return gradient_image;
}

iftFImage* iftWaterGrayFBorder(iftFImage* gradient_image,iftImage *label_image,iftAdjRel* A){
  int p,j,q;
  iftVoxel u,v;

  if ( (gradient_image->xsize != label_image->xsize) ||
       (gradient_image->ysize != label_image->ysize) ||
       (gradient_image->zsize != label_image->zsize) )
    iftError("Image dimensions are incompatible","iftWaterGrayBorder");

  for (p=0; p < label_image->n; p++) {
    u.x = iftGetXCoord(label_image,p);
    u.y = iftGetYCoord(label_image,p);
    u.z = iftGetZCoord(label_image,p);
    for (j=0; j < A->n; j++) {
      v.x = u.x + A->dx[j];
      v.y = u.y + A->dy[j];
      v.z = u.z + A->dz[j];
      if (iftValidVoxel(label_image,v)){
	q = iftGetVoxelIndex(label_image,v);
	if (label_image->val[p] < label_image->val[q]){
	  break;
	}
      }
    }
    if (j == A->n)
      gradient_image->val[p] = 0.;
  }

  return gradient_image;
}
