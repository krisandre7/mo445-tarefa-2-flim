#include "ift.h"

iftImage *iftQuantizeImage(iftImage *image, int bins_per_band, char colorspace){
	if(colorspace != LAB_CSPACE && colorspace != RGB_CSPACE && colorspace != YCbCr_CSPACE)
		iftError("Invalid colorspace.", "iftQuantizeImage");
	if(bins_per_band < 1)
		iftError("Invalid number of bins", "iftQuantizeImage");
        
        iftImage *quant = iftCreateImage(image->xsize,image->ysize,image->zsize);
        int p;
        if(iftIsColorImage(image)){
		float bin_size = 256.0/bins_per_band;
		for(p = 0; p < image->n; p++){
			iftColor color;
			color.val[0] = image->val[p];
			color.val[1] = image->Cb[p];
			color.val[2] = image->Cr[p];

			int binb1 = 0, binb2 = 0, binb3 = 0;

			if(colorspace == LAB_CSPACE){
				color = iftYCbCrtoRGB(color);
				iftFColor fcolor = iftRGBtoLab(color);
				color = iftLabtoQLab(fcolor);
			}else if (colorspace == RGB_CSPACE)
				color = iftYCbCrtoRGB(color);

			binb1 = (int)(color.val[0]/bin_size);
			binb2 = (int)(color.val[1]/bin_size);
			binb3 = (int)(color.val[2]/bin_size);

			if(binb1 < 0 || binb2 < 0 || binb3 < 0 || binb1 >= bins_per_band || binb2 >= bins_per_band || binb3 >= bins_per_band)
				iftError("Invalid color space", "iftQuantizeImage");

			quant->val[p] = binb1 + binb2*bins_per_band + binb3*bins_per_band*bins_per_band;
		}
	}else{
		for(p = 0; p < image->n; p++){
			quant->val[p] = (image->val[p]*bins_per_band)/256;
		}
	}

	return quant;
}

void iftComputeMeanColor(iftDataSet** dataset, int init_sample, int init_feat, iftImage *image, iftImage* label_image, char colorspace){
	if(colorspace != LAB_CSPACE && colorspace != RGB_CSPACE && colorspace != YCbCr_CSPACE)
		iftError("Invalid colorspace.", "iftQuantizeImage");

	int p;
        int nregions = iftMaximumValue(label_image);

	int *num_pix = iftAllocIntArray(nregions);
	for(p = 0; p < label_image->n; p++)
		num_pix[label_image->val[p] - 1]++;

	for(p = 0; p < image->n; p++){
		int region = label_image->val[p] - 1;
                int num_pix_region = num_pix[region];
		iftColor color;
		color.val[0] = image->val[p];
		color.val[1] = image->Cb[p];
		color.val[2] = image->Cr[p];

		if(colorspace == LAB_CSPACE){
			color = iftYCbCrtoRGB(color);
			iftFColor fcolor = iftRGBtoLab(color);

			(*dataset)->sample[region + init_sample].feat[init_feat] += (((float)fcolor.val[0]/100.0) / (float)num_pix_region);
			(*dataset)->sample[region + init_sample].feat[init_feat + 1] += ((((float)fcolor.val[1] + 86.185) / 184.439) / (float)num_pix_region);
			(*dataset)->sample[region + init_sample].feat[init_feat + 2] += ((((float)fcolor.val[2] + 107.863) / 202.345) / (float)num_pix_region);
		}else {
			if (colorspace == RGB_CSPACE)
				color = iftYCbCrtoRGB(color);

			(*dataset)->sample[region + init_sample].feat[init_feat] += ((float)color.val[0] / ((float)num_pix_region*255.0));
			(*dataset)->sample[region + init_sample].feat[init_feat + 1] += ((float)color.val[1] / ((float)num_pix_region*255.0));
			(*dataset)->sample[region + init_sample].feat[init_feat + 2] += ((float)color.val[2] / ((float)num_pix_region*255.0));
		}
	}

	(*dataset)->ref_data = (void*)label_image;
}

void iftComputeCH(iftDataSet** dataset, int init_sample, int init_feat, iftImage *quant, iftImage* label_image, int bins_per_band){
	if(bins_per_band < 1)
		iftError("Invalid number of bins", "iftComputeCH");

	int p;
        int nregions = iftMaximumValue(label_image);
        
	int **colorh = (int**)calloc(nregions, sizeof(int*));

	int *num_pix = iftAllocIntArray(nregions);
	for(p = 0; p < label_image->n; p++)
		num_pix[label_image->val[p] - 1]++;

	int i;
	for(i = 0; i < nregions; i++)
		colorh[i] = iftAllocIntArray(bins_per_band * bins_per_band * bins_per_band);
	
	for(p = 0; p < quant->n; p++){
		int region = label_image->val[p] - 1;
		colorh[region][quant->val[p]]++;
	}

	int j;
	for(i = 0; i < nregions; i++){
		
		for(j = 0; j < bins_per_band*bins_per_band*bins_per_band; j++){
			(*dataset)->sample[i + init_sample].feat[j + init_feat] = ((float)colorh[i][j] / (float)num_pix[i]);
		}
		free(colorh[i]);
	}

	(*dataset)->ref_data = (void*)label_image;

	free(colorh);
}


//Features: lbp bins (256)
void iftComputeLBP(iftDataSet** dataset, int init_sample, int init_feat, iftImage *image, iftImage* label_image){
	if(!iftIsColorImage(image))
		iftError("Color image expected", "iftComputeLBP");

        int nregions = iftMaximumValue(label_image);
        
	int **lbph = (int**)calloc(nregions,sizeof(int*));

	int i;
        int p;
	for(i = 0; i < nregions; i++){
		lbph[i] = iftAllocIntArray(256);
	}

        int *num_pix = iftAllocIntArray(nregions);
	for(p = 0; p < label_image->n; p++)
		num_pix[label_image->val[p] - 1]++;

	iftAdjRel *adj = iftCircular(1.5);

	for(p = 0; p < image->n; p++){
		int region = label_image->val[p] - 1;

		int lbp_code = 0;
		iftVoxel u = iftGetVoxelCoord(image, p);

		for(i = 1; i < adj->n; i++){
			lbp_code = lbp_code << 1;

			iftVoxel v = iftGetAdjacentVoxel(adj, u, i);

			if(iftValidVoxel(image, v)){
				int q = iftGetVoxelIndex(image, v);

				if(image->val[p] > image->val[q])
					lbp_code = lbp_code + 1;
			}
		}

		if(lbp_code < 0 || lbp_code > 255)
			iftError("Invalid lbp code", "iftComputeLBP");

		lbph[region][lbp_code]++;
	}

	int j;
	for(i = 0; i < nregions; i++){
		for(j = 0; j < 256; j++)
			(*dataset)->sample[i + init_sample].feat[j + init_feat] = ((float)lbph[i][j] / (float)num_pix[i]);

		free(lbph[i]);
	}

	(*dataset)->ref_data = (void*)label_image;

	free(lbph);
        free(num_pix);
	iftDestroyAdjRel(&adj);
}

void iftComputeBIC(iftDataSet** dataset, int init_sample, int init_feat, iftImage *quant, iftImage* label_image, int nbins){
	if(nbins < 1)
		iftError("Invalid number of bins", "iftComputeBIC");

	int p;
        int nregions = iftMaximumValue(label_image);
        
        iftAdjRel *adj = iftCircular(1.0);
	
        int **interiorh = (int**)calloc(nregions, sizeof(int*));
	int **borderh = (int**)calloc(nregions, sizeof(int*));

	int i;
	for(i = 0; i < nregions; i++){
		interiorh[i] = iftAllocIntArray(nbins);
		borderh[i] = iftAllocIntArray(nbins);
	}
 	
	int *num_pix = iftAllocIntArray(nregions);
	for(p = 0; p < label_image->n; p++)
		num_pix[label_image->val[p] - 1]++;
		
	for(p = 0; p < quant->n; p++){
		int region = label_image->val[p] - 1;
		iftVoxel u = iftGetVoxelCoord(quant,p);

		int border = 0;
		int i;
		for(i = 1; i < adj->n; i++){
			iftVoxel v = iftGetAdjacentVoxel(adj,u,i);
			if(iftValidVoxel(quant,v)){
				int q = iftGetVoxelIndex(quant,v);

				if(quant->val[p] != quant->val[q]){
					border = 1;
					break;
				}
			}
		}

		if(border){
			borderh[region][quant->val[p]]++;
		}else{
			interiorh[region][quant->val[p]]++;
		}
	}

	int j;
	for(i = 0; i < nregions; i++){
		for(j = 0; j < nbins; j++){
			//(*dataset)->sample[i + init_sample].feat[j + init_feat] = log2((borderh[i][j]/num_pix[i])*255 + 1);
			//(*dataset)->sample[i + init_sample].feat[j + nbins + init_feat] = log2((interiorh[i][j]/num_pix[i])*255 + 1);
			(*dataset)->sample[i + init_sample].feat[j + init_feat] = ((float)borderh[i][j]/(float)num_pix[i]);
			(*dataset)->sample[i + init_sample].feat[j + nbins + init_feat] = ((float)interiorh[i][j]/(float)num_pix[i]);
		}
		free(borderh[i]);
		free(interiorh[i]);
	}
	(*dataset)->ref_data = (void*)label_image;
	free(borderh);
	free(interiorh);
	iftDestroyAdjRel(&adj);
}


void iftComputeCCV(iftDataSet** dataset, int init_sample, int init_feat, iftImage *quant, iftImage* label_image, int nbins){
	if(nbins < 1)
		iftError("Invalid number of bins", "iftComputeCCV");
	
	int p;
        int nregions = iftMaximumValue(label_image);
	float thr = 0.10;
        
	iftAdjRel *adj = iftCircular(1.5);
	
        int **coh = (int**)calloc(nregions, sizeof(int*));
	int **incoh = (int**)calloc(nregions, sizeof(int*));
	int **totalh = (int**)calloc(nregions, sizeof(int*));

	int i;
	for(i = 0; i < nregions; i++){
		coh[i] = iftAllocIntArray(nbins);
		incoh[i] = iftAllocIntArray(nbins);
		totalh[i] = iftAllocIntArray(nbins);
	}
	
	int *num_pix = iftAllocIntArray(nregions);
	for(p = 0; p < label_image->n; p++)
		num_pix[label_image->val[p] - 1]++;
	
	iftImage *basins = iftImageBasins(quant,adj);
	iftImage *marker = iftVolumeClose(basins, 1);
	iftImage *label_quant = iftWaterGray(basins,marker,adj);
	for(p = 0; p < label_quant->n; p++){
		int region = label_image->val[p] - 1;
		totalh[region][label_quant->val[p] - 1]++;
	}
	
	for(p = 0; p < quant->n; p++){
		int region = label_image->val[p] - 1;
		if(totalh[region][label_quant->val[p]-1] > (int)(num_pix[region] * thr)){
			coh[region][quant->val[p]]++;
		}else{
			incoh[region][quant->val[p]]++;
		}
	}
	
	int j;
        printf("before dataset\n");
	for(i = 0; i < nregions; i++){
		for(j = 0; j < nbins; j++){
			(*dataset)->sample[i + init_sample].feat[j + init_feat] = ((float)coh[i][j]/(float)num_pix[i]);
			(*dataset)->sample[i + init_sample].feat[j + nbins + init_feat] = ((float)incoh[i][j]/(float)num_pix[i]);
			//printf("feat: %d \n",j + nbins + init_feat);
		}
		//free(coh[i]);
		//free(incoh[i]);
                //free(totalh[i]);
	}
	(*dataset)->ref_data = (void*)label_image;

	free(coh);
	free(incoh);
        free(totalh);
	
	iftDestroyImage(&basins);
	iftDestroyImage(&marker);
	iftDestroyImage(&label_quant);
}

void iftAddNumToDataset(iftDataSet** dataset, int index_feat, float val){
	int i;
	for(i = 0; i < (*dataset)->nsamples; i++)
		(*dataset)->sample[i].feat[index_feat] = val;
}

iftDataSet *iftCopyDataSetClass(iftDataSet *Z, int nfeats){
  int s;
  iftDataSet *Z1 = iftCreateDataSet(Z->nsamples, nfeats);
  int max_class = 1;
  for (s=0; s < Z->nsamples; s++) 
    if (Z->sample[s].truelabel > max_class)
      max_class = Z->sample[s].truelabel;
  Z1->nclasses = max_class;
  for (s=0; s < Z->nsamples; s++) 
    Z1->sample[s].truelabel  = Z->sample[s].truelabel;
  
  return Z1;
}

iftDataSet *iftCopyClsDataSet(iftDataSet *Z){
  int s,nsamples_cls,i,count;
  nsamples_cls = 0;
  for (s=0; s < Z->nsamples; s++) 
    if (Z->sample[s].truelabel > 0)
      nsamples_cls++;
  
  iftDataSet *Z1 = iftCreateDataSet(nsamples_cls, Z->nfeats);
  count = 0;
  for (s=0; s < Z->nsamples; s++) {
    if (Z->sample[s].truelabel > 0) {
      Z1->sample[count].truelabel  = Z->sample[s].truelabel;
      for (i=0; i < Z->nfeats; i++) {
        Z1->sample[count].feat[i] = Z->sample[s].feat[i];
      }
      count++;
    }
  }
  Z1->nclasses = Z->nclasses;
  return Z1;
}

int main(int argc, char *argv[]) 
{
  iftImage *image,*label_image;
  int bins_per_band, nfeats, nbins, number_of_images, number_of_labels, index_sample;
  char *input_dataset, *output_dataset, *image_dir, *label_dir;
  char filename_image[200], filename_label[200];
  iftImageNames  *image_names, *label_names;

  if (argc!=6)
    iftError("Usage: iftExtractDescriptorsFromDir <image_directory> <label_directory> <input_dataset> <output_dataset> <bins_per_band>","main");

  image_dir = argv[1];
  label_dir = argv[2];
  input_dataset = argv[3];
  output_dataset = argv[4];
  bins_per_band = atoi(argv[5]);

  number_of_images  = iftCountImageNames(image_dir, "ppm");
  image_names       = iftCreateAndLoadImageNames(number_of_images, image_dir, "ppm");
  
  number_of_labels  = iftCountImageNames(label_dir, "pgm");
  label_names       = iftCreateAndLoadImageNames(number_of_labels, label_dir, "pgm");
  
  nbins = bins_per_band*bins_per_band*bins_per_band;
  //nfeats = 3*nbins + 256 + 3 + 1;
  nfeats = 3*nbins + 256 + 3;


  iftDataSet *Z = iftReadOPFDataSet(input_dataset);
  iftDataSet *outZ = iftCopyDataSetClass(Z,nfeats);
  iftDestroyDataSet(&Z);
  
  index_sample = 0;
  for (int s = 0; s < number_of_images ; s++){
    sprintf(filename_image,"%s/%s",image_dir,image_names[s].image_name);
    sprintf(filename_label,"%s/%s",label_dir,label_names[s].image_name);
    fprintf(stdout,"Processing %s\n",filename_image);
    
    image  = iftReadImageP6(filename_image);
    label_image  = iftReadImageP2(filename_label);
    
    iftImage *quant = iftQuantizeImage(image, bins_per_band, LAB_CSPACE);
    printf("quant\n");
    iftComputeMeanColor((&outZ), index_sample, 0, image, label_image, LAB_CSPACE);
    printf("mean color\n");
    iftComputeCH((&outZ), index_sample, 3, quant, label_image, bins_per_band);
    printf("CH\n");
    iftComputeLBP((&outZ), index_sample, nbins + 3, image, label_image);
    printf("LBP\n");
    iftComputeBIC((&outZ), index_sample, nbins + 256 +  3, quant, label_image, nbins);
    printf("BIC\n");
    //iftAddNumToDataset((&outZ), nfeats - 1, (float)bins_per_band);
    index_sample += label_image->maxval;
    iftDestroyImage(&image);
    iftDestroyImage(&label_image);
  }
  iftDataSet *clsZ = iftCopyClsDataSet(outZ);
  printf("Num. samples: %d \n",clsZ->nsamples);
  iftDestroyDataSet(&outZ);
  iftWriteOPFDataSet(clsZ,output_dataset);
  iftDestroyDataSet(&clsZ);
  

  return(0);
}

