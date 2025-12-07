#include "ift.h"
#include "iftFLIM-BoVW.h"

/* To execute the similar network with convolution, uncomment Line 165
   here and replace Line 104 of iftFLIM-BoVW.h by the comment lines
   below it. The result should be equivalent to the BoVW network. */

iftMatrix *MyComputeKernelBank(iftDataSet *Z, int *ngroups) {
    iftMatrix *kernels = NULL;
    int i = 0;

    iftRandomSeed(42);

    if (*ngroups >= Z->nsamples) { /* use all markers instead */
      *ngroups = Z->nsamples;
      kernels  = iftCreateMatrix(*ngroups, Z->nfeats);
      for (int s = 0; s < Z->nsamples; s++) {
      	for (int j = 0; j < Z->nfeats; j++){
      	  iftMatrixElem(kernels, i, j) = Z->sample[s].feat[j];
      	}
      	i++;
      }
      return(kernels);
    }
    

    kernels = iftCreateMatrix(*ngroups, Z->nfeats);
    iftKMeans(Z, *ngroups, 100, 0.001, NULL, NULL, iftEuclideanDistance);
    for (int s = 0; s < Z->nsamples; s++) {
      if (iftHasSampleStatus(Z->sample[s], IFT_PROTOTYPE)) {
	for (int j = 0; j < Z->nfeats; j++){
	  iftMatrixElem(kernels, i, j) = Z->sample[s].feat[j];
	}
	i++;
      }
    }
        
    return (kernels);
}

void MyStatisticsFromAllSeeds(iftFileSet *fs_seeds, char *inputdata_dir, float *mean, float *stdev, float stdev_factor) {
    int nseeds = 0, ninput_channels = 0;
    char *basename = NULL;
    char filename[200];
    iftMImage *input = NULL;

    for (int i = 0; i < fs_seeds->n; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input = iftReadMImage(filename);
        iftFree(basename);

        ninput_channels = input->m;
        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        while (S != NULL) {
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            nseeds += 1;
            for (int b = 0; b < ninput_channels; b++) {
                mean[b] += input->val[p][b];
            }
        }
        iftDestroyMImage(&input);
    }

    for (int b = 0; b < ninput_channels; b++) {
        mean[b] = mean[b] / nseeds;
    }

    for (int i = 0; i < fs_seeds->n; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input = iftReadMImage(filename);
        iftFree(basename);

        ninput_channels = input->m;
        iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
        while (S != NULL) {
            int l;
            int p = iftRemoveLabeledSet(&S, &l);
            for (int b = 0; b < ninput_channels; b++) {
                stdev[b] += (input->val[p][b] - mean[b]) * (input->val[p][b] - mean[b]);
            }
        }
        iftDestroyMImage(&input);
    }

    for (int b = 0; b < ninput_channels; b++) {
        stdev[b] = sqrtf(stdev[b] / nseeds) + stdev_factor;
    }

}


void iftFLIMLearnNewModel(char *orig_dir, char *markers_dir, char *param_dir, iftFLIMArch *arch) {

    /* Set input parameters */

    iftMakeDir("tmp");

    iftFileSet *fs_orig  = iftLoadFileSetFromDirOrCSV(orig_dir, 1, 1);
    iftFileSet *fs_seeds = iftLoadFileSetFromDirBySuffix(markers_dir, "-seeds.txt", 1);
    iftAdjRel  *A        = NULL;
    iftMImage **output   = NULL;
    iftMImage *input     = NULL;
    int ninput_channels  = 0;
    int nimages          = fs_seeds->n;
    int atrous_factor    = 1;
    char *basename       = NULL;
    char filename[200], ext[10];
    
    sprintf(ext, "%s", iftFileExt(fs_orig->files[0]->path));
    printf("%s\n",ext);
    
    /* Generate input layer */
    for (int i = 0; i < nimages; i++) {
        basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
        sprintf(filename, "%s/%s%s", orig_dir, basename, ext);
        if (strcmp(ext,".mimg")==0){
            input = iftReadMImage(filename);
        } else {
            input = MyReadInputMImage(filename);
        }	
        sprintf(filename, "tmp/%s.mimg", basename);
        iftWriteMImage(input, filename);
        iftFree(basename);
        iftDestroyMImage(&input);
    }

    /* For each layer do */

    for (int l = 0; l < arch->nlayers; l++) {
        basename        = iftFilename(fs_seeds->files[0]->path, "-seeds.txt");
        sprintf(filename, "tmp/%s.mimg", basename);
        input           = iftReadMImage(filename);
        if (l == 0) {
            A = iftFLIMAdjRelFromKernel(arch->layer[l], iftIs3DMImage(input));
        } else {
            if (arch->apply_intrinsic_atrous) {
                atrous_factor *= arch->layer[l - 1].pool_stride;
                printf("Updating atrous factor\n");
                fflush(stdout);
            }
	    A = iftFLIMAdaptiveAdjRelFromKernel(arch->layer[l], atrous_factor, iftIs3DMImage(input));
        }
        ninput_channels = input->m;
        iftDestroyMImage(&input);

        float *mean  = iftAllocFloatArray(ninput_channels);
        float *stdev = iftAllocFloatArray(ninput_channels);
        MyStatisticsFromAllSeeds(fs_seeds, "tmp", mean, stdev, arch->stdev_factor);
        /* Learn kernels */
	
	printf("\nLayer %d\n", l + 1);
        fflush(stdout);

	iftDataSet *Z = NULL;
	int nsamples  = 0, nfeats = A->n * ninput_channels;

	/* count the number of samples */
	

        for (int i = 0; i < nimages; i++) {
	  basename         = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
	  sprintf(filename, "tmp/%s.mimg", basename);
	  input            = iftReadMImage(filename);
	  /* MyNormalizeImageByZScore(input, mean, stdev); */
	  iftWriteMImage(input,filename);
	  iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
	  nsamples += iftLabeledSetSize(S);
	  iftFree(basename);
	  iftDestroyMImage(&input);
	  iftDestroyLabeledSet(&S);
	}

	/* create a dataset with patches from seeds selected in all
	   training images */
	
	Z = iftCreateDataSet(nsamples,nfeats);
	int s = 0;
	
        for (int i = 0; i < nimages; i++) {
	  basename         = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
	  sprintf(filename, "tmp/%s.mimg", basename);
	  input            = iftReadMImage(filename);	  	  
	  iftLabeledSet *S = iftMReadSeeds(input, fs_seeds->files[i]->path);
	  while (S != NULL) {
	    int label;
	    int p = iftRemoveLabeledSet(&S, &label);
	    Z->sample[s].id = s;
	    Z->sample[s].truelabel = label;
	    if (Z->sample[s].truelabel > Z->nclasses)
	      Z->nclasses = Z->sample[s].truelabel;
	    iftVoxel u = iftMGetVoxelCoord(input, p);
	    int j = 0;
	    for (int k = 0; k < A->n; k++) {
	      iftVoxel v = iftGetAdjacentVoxel(A, u, k);
	      if (iftMValidVoxel(input, v)) {
		int q = iftMGetVoxelIndex(input, v);
		for (int b = 0; b < ninput_channels; b++) {
		  Z->sample[s].feat[j] = input->val[q][b];
		  j++;
		}
	      } else {
		for (int b = 0; b < input->m; b++) {
		  Z->sample[s].feat[j] = 0;
		  j++;
		}
	      }
	    }
	    s++;
	  }
	  iftFree(basename);
	  iftDestroyMImage(&input);
	  iftDestroyLabeledSet(&S);
	}
	iftSetStatus(Z, IFT_TRAIN);
	iftAddStatus(Z, IFT_SUPERVISED);

	/* compute arch->layer[l].nkernels_per_marker (class in this
	   case) */

	printf("nclasses %d\n",Z->nclasses);
	
	iftMatrix **kernels    = (iftMatrix **) calloc(Z->nclasses + 1, sizeof(iftMatrix *));
	int nkernels_per_class = arch->layer[l].noutput_channels/Z->nclasses;
	int total_nkernels  = 0;
	for (int c = 1; c <= Z->nclasses; c++) {
	  iftDataSet *Z1  = iftExtractSamplesFromClass(Z, c);
	  int ngroups     = nkernels_per_class;
	  kernels[c]      = MyComputeKernelBank(Z1, &ngroups);
	  total_nkernels += ngroups;
	  iftDestroyDataSet(&Z1);
	}

	printf("total_nkernels %d\n",total_nkernels);
	
	iftMatrix *kernelbank = iftCreateMatrix(total_nkernels, Z->nfeats);

	int k = 0;
	for (int c = 1; c <= Z->nclasses; c++) {
	  for (int col = 0; col < kernels[c]->ncols; col++, k++) {
            for (int row = 0; row < kernels[c]->nrows; row++) {
	      iftMatrixElem(kernelbank, k, row) = iftMatrixElem(kernels[c], col, row);
            }
	  }
	  iftDestroyMatrix(&kernels[c]);
	}

	iftFree(kernels);
	iftDestroyDataSet(&Z);

	/* save kernels */
	
	sprintf(filename, "%s/conv%d-kernels.npy", param_dir, l+1);
	iftWriteMatrix(kernelbank, filename);
	iftDestroyMatrix(&kernelbank);
	iftDestroyAdjRel(&A);
	
        sprintf(filename, "%s/conv%d", param_dir, l+1);
        MyWriteMeanStdev(filename, mean, stdev, ninput_channels);
        iftFree(mean);
        iftFree(stdev);
	
	/* Apply convolutional layer */
	int pool_stride            = arch->layer[l].pool_stride;
	arch->layer[l].pool_stride = 1;
	for (int i = 0; i < nimages; i++) {
	  basename = iftFilename(fs_seeds->files[i]->path, "-seeds.txt");
	  sprintf(filename, "tmp/%s.mimg", basename);
	  input = iftReadMImage(filename);
	  iftFree(basename);
	  output = MyConvolutionalLayer(&input, 1, arch, l, l+1, atrous_factor, param_dir);
	  iftDestroyMImage(&input);	    
	  iftWriteMImage(output[0], filename);
	  iftDestroyMImage(&output[0]);
	  iftFree(output);
	}
	arch->layer[l].pool_stride = pool_stride;
    }
    
    iftRemoveDir("tmp");
    iftDestroyFileSet(&fs_seeds);
    iftDestroyFileSet(&fs_orig);
}

int main(int argc, char *argv[])
{
    timer *tstart;

    if (argc != 5)
      iftError("Usage: iftFLIM-LearnNewModel P1 P2 P3 P4\n"
	       "P1: input  FLIM network architecture (.json)\n"
	       "P2: input  folder with the original images (.png, .nii.gz)\n"
	       "P3: input  folder with the training markers (-seeds.txt)\n"
	       "P4: output folder with the FLIM network parameters per layer\n",
	       "main");
    
    tstart = iftTic();

    iftFLIMArch *arch      = iftReadFLIMArch(argv[1]);
    char *orig_dir         = argv[2];
    char *markers_dir      = argv[3];
    char *param_dir        = argv[4];
    iftMakeDir(param_dir);

    /* Learn new model */
    
    iftFLIMLearnNewModel(orig_dir,markers_dir, param_dir, arch);
    iftDestroyFLIMArch(&arch);
    
    printf("Done ... %s\n", iftFormattedTime(iftCompTime(tstart, iftToc())));

    return (0);
}
