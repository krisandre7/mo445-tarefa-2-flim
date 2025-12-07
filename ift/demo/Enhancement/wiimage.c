#include "common.h"
#include "ift.h"
#include "iftTrainPixels.h"

iftDataSet* iftMImageToPixelDataSet(iftMImage* mimg) 
{ 

	iftDataSet* Zpx = iftCreateDataSet(mimg->n, mimg->m);
	Zpx->nclasses = 2;
	
	int p,b;
	for(p = 0; p < mimg->n; p++)
	{
		for(b = 0; b < mimg->m ; b++)
			Zpx->sample[p].feat[b] = mimg->val[p][b];
		Zpx->sample[p].id = p;
		Zpx->sample[p].status = IFT_TEST;
	}
	return Zpx;
}

void  iftBinMembership(iftCplGraph *graph, iftDataSet *Z, int class)
{
	int u, s, t, i, indClass, indOtherClass;
	float tmp, weight, minCost[2];
	iftDataSet *Z1=graph->Z;

	if (Z1->nfeats != Z->nfeats)
		iftError("Incompatible datasets","iftBinMembership");

	if (Z1->nclasses != 2)
		iftError("Number of classes is not 2","iftBinMembership");

	for (t = 0; t < Z->nsamples; t++)
	{

		if ((Z==Z1)&&(Z->sample[t].status==IFT_TRAIN))
			continue;

		minCost[0] = INFINITY_FLT;
		minCost[1] = INFINITY_FLT;

		for (i=0; i < graph->nnodes; i++){

			u  = graph->ordered_nodes[i];
			s  = graph->node[u].sample;
			if (iftDist == NULL)
				weight = Z1->iftArcWeight(Z1->sample[s].feat,Z->sample[t].feat,Z1->alpha,Z1->nfeats);
			else
				weight  = iftDist->distance_table[Z1->sample[s].id][Z->sample[t].id];

			if (Z1->sample[s].label==1) {
				tmp = MAX(graph->pathval[u], weight);
				if(tmp < minCost[0]){
					minCost[0] = tmp;
				}
			}else{
				tmp = MAX(graph->pathval[u], weight);
				if(tmp < minCost[1]){
					minCost[1] = tmp;
				}
			}

		}

		// index Class
		if ( class == 1 ) {
			indClass = 0;
			indOtherClass = 1;
		}else{
			indClass = 1;
			indOtherClass = 0;
		}
		// Membership of the sample to the class
		Z->sample[t].weight = minCost[indOtherClass]/(minCost[indClass]+minCost[indOtherClass]);
		if (minCost[0] < minCost[1]) {
			Z->sample[t].label  = 1;
		}else{
			Z->sample[t].label  = 2;
		}

	}

	Z->nlabels = 2;
}

iftMImage* iftWiImage(iftMImage* mimg, iftDataSet *Zpixels, int typeWi) 
{ 

	iftFImage* output = iftCreateFImage(mimg->xsize,mimg->ysize,mimg->zsize);
	iftAdjRel *adj = iftCircular(1.5);
	
	float val;
	float maxVal;
	int p,q,b,i;
	
	if(typeWi == 1)
	{
		// Fill output image  val = wi(p,q)
		for(p = 0; p < mimg->n; p++)
		{
			iftVoxel pixel   = iftMGetVoxelCoord(mimg,p);
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
					// wi(p,q)  = | \vec{v}(p) - \vec{v}(q) |_2
					val = sqrtf(val);
				}
				if(val>maxVal){
					maxVal = val;
				}
			}
			// W(s) = max_{forall t in A(s)} (wi(s,t))
			output->val[p] = maxVal;
		}
	}else if(typeWi == 2){
		// Fill output image  val = wi(p,q)
		for(p = 0; p < mimg->n; p++)
		{
			// initialize sum vector
			iftVector vecT;
			vecT.x = 0;
			vecT.y = 0;
			vecT.z = 0;
			iftVoxel pixel   = iftMGetVoxelCoord(mimg,p);
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
					// |vec(p) - vec(q)|_{2}
					val = sqrtf(val);
					// get unit vector vec(pq)
					vec.x = (float)adj->dx[i];
					vec.y = (float)adj->dy[i];
					vec.z = (float)adj->dz[i];
					iftVector norvec = iftNormalizeVector(vec);
					// Total vector
					vecT.x = vecT.x + (val*norvec.x);
					vecT.y = vecT.y + (val*norvec.y);
					vecT.z = vecT.z + (val*norvec.z);
				}
			}
			// |vecT(p)|_{2}
			output->val[p] = iftVectorMagnitude(vecT);
		}
	}else if(typeWi == 3){
		// Third case wi(s,t) = [B(t) + B(s)]/2
		timer *tic, *toc;
		iftDataSet *Ztrain[2], *Ztest[2];
		iftCplGraph *graph = NULL;

		iftDataSet* Zimg = iftMImageToPixelDataSet(mimg);

		// Select Train Samples
		tic = iftTic();
		iftSelectSupTrainSamples(Zpixels,(float)0.9999); // Selecting training samples
		iftSetStatus(Zpixels, IFT_TRAIN);
		
		Ztrain[0] = iftExtractSamples(Zpixels,IFT_TRAIN);
		Ztest [0] = iftExtractSamples(Zimg,IFT_TEST);

		Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
		iftDestroyDataSet(&Ztrain[0]);
		Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		iftDestroyDataSet(&Ztest [0]);
		toc = iftToc();
		fprintf(stderr,"selecting training(%d)/testing(%d) data: %f\n",Ztrain[1]->nsamples,Ztest[1]->nsamples,iftCompTime(tic,toc));
		
		tic = iftTic();	
		
		graph = iftCreateCplGraph(Ztrain[1]);          // Training
		iftSupTrain(graph);
		
		toc = iftToc();
		fprintf(stderr,"training svm classifier: %f\n", iftCompTime(tic,toc));

	
		tic = iftTic();
		iftBinMembership(graph,Ztest[1],2);                  // Get Membership test samples
		toc = iftToc();
		fprintf(stderr,"classifying the test set: %f\n", iftCompTime(tic,toc));

		// Fill output image  val = wi(p,q)
		for(p = 0; p < mimg->n; p++)
		{
			iftVoxel pixel   = iftMGetVoxelCoord(mimg,p);
			maxVal = -999999999;
			for(i=1;i<adj->n;i++)
			{
				val = 0;
				iftVoxel neighbor   = iftGetAdjacentVoxel(adj,pixel,i);
				if (iftMValidVoxel(mimg,neighbor))
				{
					q = iftMGetVoxelIndex(mimg,neighbor);
					val = (Ztest[1]->sample[p].weight + Ztest[1]->sample[q].weight)/2;
				}
				if(val>maxVal){
					maxVal = val;
				}
			}
			output->val[p] = maxVal;
		}

	}else if(typeWi == 4){
		// local variables
		float total,K;
		// Fill output image G(p)
		for(p = 0; p < mimg->n; p++)
		{
			total = 0;
			// initialize sum vector
			iftVector vecT;
			iftVoxel pixel   = iftMGetVoxelCoord(mimg,p);
			// get maxVal para cada ponto p
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
	}
	return output;
}

int main(int argc, char **argv) 
{
	if (argc != 5)	iftError("Please provide the following parameters:\n <DATASET_PIXELS_FILE> <MSCONVNET_FILE> <FILE_INPUT_NAMES> <N_THREADS>\n\n", "main");

	
	int idxImg,totalPixels;
	int it;
	
	char *file_inputImages;
	char *file_msconvnet  ;
	char *file_datasetpixels;

	file_datasetpixels = argv[1];
	file_msconvnet = argv[2];
	file_inputImages   = argv[3];
	omp_set_num_threads(atoi(argv[4]));

	
	// Read DataSet
	iftDataSet *Zpixels = iftReadOPFDataSet(file_datasetpixels);

	// Read MSCONVNET
	iftMSConvNetwork* msconvnet = iftReadMSConvNetwork(file_msconvnet);

	// Read Input images (Images to label) 
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


	// For all input images
	for( idxImg = 0; idxImg < nimages; idxImg++)
	{
		// Read input images
		fullPath[0] = 0;
		strcpy(fullPath, folder_name);
		strcat(fullPath, files[idxImg].filename);
		iftImage *img = iftReadImageP5(fullPath);

		// Extract Deep Features
		iftMImage *auximg = iftImageToMImage(img,RGB_CSPACE);
		iftMImage *mimg = iftApplyMSConvNetwork(auximg, msconvnet);
		
		iftFImage* imgwi = iftWiImage(mimg,Zpixels,4);


		// write labeled image
		iftImage* imgout = iftFImageToImage(imgwi, 4095);

	  	fullPath[0] = 0;
	  	strcpy(fullPath, folder_name);
	  	strcat(fullPath, files[idxImg].filename);
	  	strcat(fullPath, ".lbl.pgm");

		iftWriteImageP2(imgout,fullPath);

	}


	return 0;
}
