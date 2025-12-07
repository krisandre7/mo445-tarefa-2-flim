#include "ift.h"

typedef enum _dist_type {
   EUCLIDEAN, 
   COSINE
} Dist_Type;

#define DIST_TYPE COSINE /* Euclidean: 0 Cosine: 1 */

int main(int argc, char *argv[]){

	timer *tstart = NULL;

    if (argc != 4) {
        iftError("Usage: iftPrototypeEvaluation <P1> <P2> <P3>\n"
                 "P1: DataSet with training samples (.zip).\n"
                 "P2: Prototypes (.csv file with FLIM's training images).\n"
                 "P3: Output file with the misclassified images (.csv)\n",
                 "main");
	}

    tstart = iftTic();

    iftDataSet *Z    = iftReadDataSet(argv[1]);
    iftFileSet *fs   = NULL;

    /* Verify the dataset has a file set */
    
    if (Z->ref_data_type == IFT_REF_DATA_FILESET){
    	fs = (iftFileSet *)Z->ref_data;

    	if (Z->nsamples != fs->n){
			iftError("Dataset and file set do not have the same number of samples","main");
		}

		/* for (int i=0; i < fs->n; i++) */
		/* 	printf("%s\n",fs->files[i]->path); */

    	iftCSV *prototypes = iftReadCSV(argv[2],'/');

		/* for (int row=0; row < prototypes->nrows; row++) */
		/* 	printf("%s\n",prototypes->data[row][1]); */
		
		/* Find the prototypes (training images) in the dataset and set
		their status accordingly. Sample twice the number of impurity
		prototypes. */
      
    	for (int row=0; row < prototypes->nrows; row++){

			char *basename1 = iftBasename(prototypes->data[row][1]);
			
			for (int i=0; i < fs->n; i++){

	  			char *basename2 = iftFilename(fs->files[i]->path, iftFileExt(fs->files[i]->path));
				//printf("Basename1: %s, Basename2: %s\n", basename1, basename2);

				if (strcmp(basename1,basename2)==0){
					iftAddSampleStatus(&Z->sample[i], IFT_PROTOTYPE);
					printf("Sample %d is prototype %s from class %d\n",
						i,basename2,Z->sample[i].truelabel);
				}
				iftFree(basename2);
			}
		
			iftFree(basename1);
		}

    	int nimpurities=0;

    	while(nimpurities != 2*prototypes->nrows){

			int s = iftRandomInteger(0, Z->nsamples-1);

			if (Z->sample[s].truelabel == 9){
				iftAddSampleStatus(&Z->sample[s], IFT_PROTOTYPE);
				nimpurities++;  
			}
    	}

		int  nprototypes      = prototypes->nrows + nimpurities;
		int *prototype_sample = iftAllocIntArray(nprototypes);
		iftMatrix *P          = iftCreateMatrix(nprototypes,Z->nfeats);
		iftDestroyCSV(&prototypes);

		float *nerrors=iftAllocFloatArray(Z->nclasses+1);
		int   *nsamples_per_class = iftCountSamplesPerClassDataSet(Z);

      	if (DIST_TYPE==COSINE) {
      
			/* Normalize the feature vector of each sample, including
			prototypes that are copied to matrix P. */
	
			int p=0; /* prototype index */
			for (int s=0; s < Z->nsamples; s++){

				float norm=0.0;
				for (int f=0; f < Z->nfeats; f++)
					norm += Z->sample[s].feat[f]*Z->sample[s].feat[f];
				norm = sqrtf(norm);
				if (!iftAlmostZero(norm))
					for (int f=0; f < Z->nfeats; f++)
						Z->sample[s].feat[f] = Z->sample[s].feat[f]/norm;
				
				if (iftHasSampleStatus(Z->sample[s],IFT_PROTOTYPE)){
					prototype_sample[p]=s;
					for (int f=0; f < Z->nfeats; f++) 
						iftMatrixElem(P, p, f) = Z->sample[s].feat[f];
					p++;
				}

			}
	      
			/* Compute cosine similarity between all samples and the prototypes */
	
			iftMatrix *S = iftMultMatrices(Z->data,P);

			/* Verify if the most similar samples is from the same class
			and count the errors. */

			FILE *fp = fopen(argv[3],"w");
	
			fprintf(fp,"Misclassified Image; Certainty;\n");

			for (int s = 0; s < S->nrows; s++){ /* samples */
				float max_simil1=  IFT_INFINITY_FLT_NEG;
				float max_simil2=  IFT_INFINITY_FLT_NEG;
				int   p1        =  IFT_NIL;
				int   p2        =  IFT_NIL;
				Z->sample[s].weight = 0;

				/* Encontrar o protótipo mais similar para a amostra */
				for (int col = 0; col < S->ncols; col++){
					if (iftMatrixElem(S, col, s) > max_simil1){
						max_simil1 = iftMatrixElem(S, col, s);
						p1 = col;
					}
				} 

				/*Encontrar o segundo protótipo mais similar de classe diferente*/
				int t1 = prototype_sample[p1];

				for (int col = 0; col < S->ncols; col++){
					/*compara com max_simil1*/
					if (iftMatrixElem(S, col, s) > max_simil2){
						int t2 = prototype_sample[col];
						if (Z->sample[t1].truelabel != Z->sample[t2].truelabel){
							max_simil2 = iftMatrixElem(S, col, s);
							p2 = col;
						}
					}
				}

				Z->sample[s].weight = max_simil1 / (max_simil1 + max_simil2);


				// /*O que foi implementado na terça [24/06]*/
				// for (int col = 0; col < S->ncols; col++){ /* prototypes */
				// 	if (iftMatrixElem(S, col, s)>max_simil1){
				// 		if (p1 == IFT_NIL){
				// 			max_simil1 = iftMatrixElem(S, col, s);
				// 			p1         = col;
				// 		}else if (p2 == IFT_NIL){
				// 			int t1 = prototype_sample[p1];
				// 			if (Z->sample[t1].truelabel != Z->sample[s].truelabel){
				// 				max_simil2 = max_simil1;
				// 				p2 = p1;
				// 				Z->sample[s].weight =  iftMatrixElem(S, col, s) / (iftMatrixElem(S, col, s) + max_simil2);
				// 			}
				// 			max_simil1 = iftMatrixElem(S, col, s);
				// 			p1         = col;
				// 		}
						
				// 		else {
				// 			int t1 = prototype_sample[p1];
				// 			if (Z->sample[t1].truelabel != Z->sample[s].truelabel){
				// 				max_simil2 = max_simil1;
				// 				p2 = p1;
				// 			}
				// 			max_simil1 = iftMatrixElem(S, col, s);
				// 			p1         = col;
				// 			Z->sample[s].weight =  iftMatrixElem(S, col, s) / (iftMatrixElem(S, col, s) + max_simil2);
							
				// 		}
				// 	}
				// }

				/*normalizar entre 0 e 1*/

				// int t1 = prototype_sample[p1];
				if (Z->sample[t1].truelabel != Z->sample[s].truelabel){
					nerrors[Z->sample[s].truelabel]++;
					fprintf(fp,"%s;%f;\n",fs->files[s]->path,Z->sample[s].weight);
				}

			}


		fclose(fp);
		iftDestroyMatrix(&S);

      	} else { /* EUCLIDEAN */

			int perplexity = 50;
			int max_iter   = 2000;
			// Z2d  = iftDimReductionByTSNE(Z, 2, perplexity, max_iter);

			// /* Normalize the feature vector of each sample, including
			// prototypes that are copied to matrix P. */
			// for (int s=0; s < Z->nsamples; s++){
			// 	float norm=0.0;
			// 	for (int f=0; f < Z->nfeats; f++)
			// 		norm += Z->sample[s].feat[f]*Z->sample[s].feat[f];
			// 	norm = sqrtf(norm);
			// 	if (!iftAlmostZero(norm))
			// 		for (int f=0; f < Z->nfeats; f++)
			// 			Z->sample[s].feat[f] = Z->sample[s].feat[f]/norm;
			// }

			// /* Compute euclidean distance between all samples and the prototypes */
			// iftMatrix *S = iftCreateMatrix(P->nrows, Z->nsamples);
			// for (int s = 0; s < Z->nsamples; s++)
			// 	for (int p = 0; p < P->nrows; p++){
			// 		float dist = 0.0;
			// 		for (int f = 0; f < Z->nfeats; f++)
			// 			dist += (Z->sample[s].feat[f] - iftMatrixElem(P, p, f)) * (Z->sample[s].feat[f] - iftMatrixElem(P, p, f));
			// 		iftMatrixElem(S, p, s) = sqrtf(dist);
			// 	}
			

		}

      	iftDestroyMatrix(&P);
      	iftFree(prototype_sample);

      	printf("Accuracy per class: \n");
      	for (int c=1; c <= Z->nclasses; c++)	
			printf("Class %d: %f\n",c,1.0 - nerrors[c]/nsamples_per_class[c]);

      	iftFree(nsamples_per_class);
	}

    iftDestroyDataSet(&Z);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return 0;
}
