#include "common.h"
#include "ift.h"
#include "iftTrainPixels.h"

iftImage* iftWGImage(iftImage* G, float volume_threshold, iftAdjRel* A)
{
	int nregions = 400;
	float vol = volume_threshold;
	// Create WaterGray image taking as input the Gradient image
	iftImage *imgWG,*marker;
	while(nregions > 399)
	{
		marker = iftVolumeClose(G,vol);
		imgWG = iftWaterGray(G,marker,A);
		nregions = iftMaximumValue(imgWG);
		vol *= 2;
	}
	return imgWG;
}

iftMImage* iftGImage(iftMImage* mimg,float radius)
{ 

	iftFImage* output = iftCreateFImage(mimg->xsize,mimg->ysize,mimg->zsize);
	iftAdjRel *adj = iftCircular(radius);
	
	float val;
	float maxVal;
	int p,q,b,i;
	// local variables
	float total,K;
	// Fill output image G(p)
	for(p = 0; p < mimg->n; p++)
	{
		total = 0;
		// initialize sum vector
		iftVoxel pixel   = iftMGetVoxelCoord(mimg,p);
		// get maxVal de G para cada ponto p
		maxVal = -999999999;
		for(i=1;i<adj->n;i++)
		{
			val = 0;
			iftVoxel neighbor   = iftGetAdjacentVoxel(adj,pixel,i);
			if (iftMValidVoxel(mimg,neighbor))
			{
				q = iftMGetVoxelIndex(mimg,neighbor);
				for(b = 0; b < mimg->m ; b++){
					val += powf((mimg->val[p][b] - mimg->val[q][b]),2);
				}
				//| \vec{v}(p) - \vec{v}(q) |_2
				val = sqrtf(val);
			}
			if(val>maxVal){
				maxVal = val;
			}
		}
		// K value
		K = maxVal/3;
		// get G(p) para cada ponto p
		for(i=1;i<adj->n;i++)
		{
			iftVector vec;
			val = 0;
			iftVoxel neighbor   = iftGetAdjacentVoxel(adj,pixel,i);
			if (iftMValidVoxel(mimg,neighbor))
			{
				q = iftMGetVoxelIndex(mimg,neighbor);
				for(b = 0; b < mimg->m ; b++){
					val += powf((mimg->val[p][b] - mimg->val[q][b]),2);
				}
				// d(p,q)
				val = sqrtf(val);
			}
			total += (1 - (float)exp(-powf(val,2)/K));
		}
		// G(p) =( 1/|A(p)| ) * Sumatoria
		if(K == 0){
			output->val[p] = 0;
		}else{
			output->val[p] = total/(adj->n-1);
		}
	}
	return output;
}

int main(int argc, char **argv) 
{
	if (argc != 6)	iftError("Please provide the following parameters:\n <MSCONVNET_FILE> <FILE_IMAGE_NAMES> <FILE_IMAGE_NAMES_GT> <RADIUS> <N_THREADS>\n\n", "main");

	
	int idxImg,totalPixels;
	int it;
	
	char *file_inputImages;
	char *file_inputImagesGT;
	char *file_msconvnet  ;
	char *file_datasetpixels;
	float radius;

	file_msconvnet = argv[1];
	file_inputImages   = argv[2];
	file_inputImagesGT   = argv[3];
	radius            = atof(argv[4]);
	omp_set_num_threads(atoi(argv[5]));

	// Read MSCONVNET
	iftMSConvNetwork* msconvnet = iftReadMSConvNetwork(file_msconvnet);

	// Read Input images (Images to label) 
	int nimages = 0;
	int nimagesGT = 0;
	char *folder_name = iftAllocCharArray(256);
	char *folder_nameGT = iftAllocCharArray(256);
	char fullPath[128];
	FileNames * files = NULL;
	FileNames * filesGT = NULL;
	
	folder_name = getFolderName(file_inputImages);
	nimages = countImages(file_inputImages);
	files = createFileNames(nimages);
	loadFileNames(files, file_inputImages);

	folder_nameGT = getFolderName(file_inputImagesGT);
	nimagesGT = countImages(file_inputImagesGT);
	filesGT = createFileNames(nimagesGT);
	loadFileNames(filesGT, file_inputImagesGT);

	printf("Input Folder: %s\n", folder_name);
	printf("Number of Images: %d\n", nimages);
	printf("File with image names: %s\n"   , file_inputImages);
	puts("");


	// For all input images
	for( idxImg = 0; idxImg < nimages; idxImg++)
	{
		// Read input images
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		iftImage *img = iftReadImageP5(fullPath);

		// Read input images
		fullPath[0] = 0;
		strcpy(fullPath, folder_nameGT);
		strcat(fullPath, filesGT[idxImg].filename);
		iftImage *imgGT = iftReadImageP5(fullPath);

		// Extract Deep Features
		iftMImage *auximg = iftImageToMImage(img,RGB_CSPACE);
		iftMImage *mimg = iftApplyMSConvNetwork(auximg, msconvnet);
		
		iftFImage* fG = iftGImage(mimg,radius);

		// Transform Float image
		iftImage* G = iftFImageToImage(fG, 255);
		
		// WaterGray variables
		iftImage *imgWG1,*imgWG2,*basins;
		iftAdjRel* adj = iftCircular(radius);
		basins    = iftImageBasins(img,adj);
		float volume_threshold1 = 10.;
		float volume_threshold2 = 100.;

		// Draw Border variables
		iftColor         RGB,YCbCr;
		iftAdjRel *A,*B;
		RGB.val[0] = 255;
	    RGB.val[1] = 0;
	    RGB.val[2] = 0;
	    YCbCr      = iftRGBtoYCbCr(RGB);
	    A          = iftCircular(sqrtf(2.0));
	    B          = iftCircular(0.0);

		// Create WaterGray with MSCONVNET
		imgWG1 = iftWGImage(G,volume_threshold1,adj);
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		strcat(fullPath, ".wg1.pgm");
		iftFImage *imgf1 = iftImageToFImage(imgWG1);
		iftImage *imgout1 = iftFImageToImage(imgf1,255);
		iftDrawBorders(imgout1,imgGT,A,YCbCr,B);
		iftWriteImageP6(imgout1,fullPath);

		// Create WaterGray Normal
		imgWG2 = iftWGImage(basins,volume_threshold2,adj);
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		strcat(fullPath, ".wg2.pgm");
		iftFImage *imgf2 = iftImageToFImage(imgWG2);
		iftImage *imgout2 = iftFImageToImage(imgf2,255);
		iftDrawBorders(imgout2,imgGT,A,YCbCr,B);
		iftWriteImageP6(imgout2,fullPath);

		// Write G Image
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		strcat(fullPath, ".g.pgm");
		iftDrawBorders(G,imgGT,A,YCbCr,B);
		iftWriteImageP6(G,fullPath);

	  	iftDestroyImage(&basins);

	}

	return 0;
}
