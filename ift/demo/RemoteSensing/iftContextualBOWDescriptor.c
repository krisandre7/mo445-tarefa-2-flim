#include "ift.h"

typedef iftFloatArray * (*iftExtractPatchDescriptor)(iftMImage* img, iftBoundingBox bb, int int_param1, iftImage *img_param2);

/**** BEGIN Auxiliary functions *****/
iftVoxel iftGetMVoxelCoord(iftMImage *img, int p)
{
  iftVoxel u;

  u.x = iftGetXCoord(img,p);
  u.y = iftGetYCoord(img,p);
  u.z = iftGetZCoord(img,p);

  return(u);
}

int max_int(int a, int b) {
	return (a > b) ? a : b;
}

int min_int(int a, int b) {
	return (a < b) ? a : b;
}
/**** END Auxiliary functions *****/


// Auxiliary function for Opencv python implementation
double iftCIEFunction(double t) {
	if (t > 0.008856) 
		return pow(t, 1.0/3.0);
	else
		return 7.787 * t + 16.0/116.0;
}
// Opencv python implementation
// https://github.com/opencv/opencv/blob/2f4e38c8313ff313de7c41141d56d945d91f47cf/modules/ts/misc/color.py
iftColor iftRGBtoQCIELabPythonOpenCV(iftColor rgb, int normalization_value)
{
	//RGB to XYZ
	double R = rgb.val[0]/(double)normalization_value;
	double G = rgb.val[1]/(double)normalization_value;
	double B = rgb.val[2]/(double)normalization_value;
	
	double X = (0.412453*R + 0.357580*G + 0.180423*B);
	double Y = (0.212671*R + 0.715160*G + 0.072169*B);
	double Z = (0.019334*R + 0.119193*G + 0.950227*B);
	
	double X_n = 0.950456;
	double Z_n = 1.088754;
	X /= X_n;
	Z /= Z_n;
	
	//XYZ to lab
	double L;
	if (Y > 0.008856)
		L = 116.0 * pow(Y, 1.0/3.0) - 16.0;
	else
		L = 903.3 * Y;
	
	double a = 500.0*(iftCIEFunction(X) - iftCIEFunction(Y)) + 128.0;
	double b = 200.0*(iftCIEFunction(Y) - iftCIEFunction(Z)) + 128.0;
	
	L *= 255.0/100.0;
	
	iftColor cielab;
	cielab.val[0] = (int)L;
	cielab.val[1] = (int)a;
	cielab.val[2] = (int)b;
	
	return cielab;
}

// Auxiliary function for colormine.org conversion
double PivotRgb(double n) {
    return (n > 0.04045 ? pow((n + 0.055) / 1.055, 2.4) : n / 12.92) * 100.0;
}

// Auxiliary function for colormine.org conversion
double PivotXyz(double n) {
	return n > 0.008856 ? pow(n, 1.0/3.0) : (903.3*n + 16.0) / 116.0;
}

// Convert RGB to quantized Lab  (http://colormine.org/convert/rgb-to-lab)
// output values are in the range [0-255]
iftColor iftRGBtoQCIELab(iftColor rgb, int normalization_value)
{
	//RGB to XYZ
	double R = PivotRgb(rgb.val[0]/(double)normalization_value);
	double G = PivotRgb(rgb.val[1]/(double)normalization_value);
	double B = PivotRgb(rgb.val[2]/(double)normalization_value);
	
	double X = (0.412453*R + 0.357580*G + 0.180423*B);
	double Y = (0.212671*R + 0.715160*G + 0.072169*B);
	double Z = (0.019334*R + 0.119193*G + 0.950227*B);
	
	double WhiteRef_X = 95.047;
	double WhiteRef_Y = 100.000;
	double WhiteRef_Z = 108.883;
	X /= WhiteRef_X;
	Y /= WhiteRef_Y;
	Z /= WhiteRef_Z;

	X = PivotXyz(X);
	Y = PivotXyz(Y);
	Z = PivotXyz(Z);
	
	//XYZ to lab
	double L = 116.0 * Y - 16.0;
	if (L < 0)
		L = 0;
	
	double a = 500.0*(X - Y) + 128.0;
	double b = 200.0*(Y - Z) + 128.0;
	
	L *= 255.0/100.0;
	
	iftColor cielab;
	cielab.val[0] = (int)L;
	cielab.val[1] = (int)a;
	cielab.val[2] = (int)b;
	
	return cielab;
}

iftMImage *iftImageToCIEQLabMImage(iftImage *img1) {
	iftMImage *img2 = NULL;
	iftColor QLab;
	iftColor  RGB,YCbCr;
	int p;
	int normalization_value = 255;
	img2 = iftCreateMImage(img1->xsize,img1->ysize,img1->zsize,3);      
	#pragma omp parallel for private(p, YCbCr, RGB, QLab)
	for (p=0; p < img2->n; p++) {
		YCbCr.val[0] = img1->val[p];
		YCbCr.val[1] = img1->Cb[p];
		YCbCr.val[2] = img1->Cr[p];
		RGB = iftYCbCrtoRGB(YCbCr,normalization_value);
		QLab = iftRGBtoQCIELab(RGB,normalization_value);
		img2->val[p][0]=QLab.val[0];
		img2->val[p][1]=QLab.val[1];
		img2->val[p][2]=QLab.val[2];
	}
	return img2;
}

iftImage* iftLabQuantize(iftImage *img, int bins_per_band, int normalization_value){

	iftImage *quant = iftCreateImage(img->xsize,img->ysize,img->zsize);
	int p;
	if(iftIsColorImage(img)){

		for(p = 0; p < img->n; p++){
			iftColor color_ycbcr;
			color_ycbcr.val[0] = img->val[p];
			color_ycbcr.val[1] = img->Cb[p];
			color_ycbcr.val[2] = img->Cr[p];
			
			iftColor color_rgb = iftYCbCrtoRGB(color_ycbcr,normalization_value);
			//iftFColor fcolor_lab = iftRGBtoLab(color_rgb, normalization_value);
			//iftColor color_qlab = iftLabtoQLab(fcolor_lab, normalization_value);
			iftColor color_qlab = iftRGBtoQCIELab(color_rgb, normalization_value);
			int l = (color_qlab.val[0]*bins_per_band)/(normalization_value+1);
			int a = (color_qlab.val[1]*bins_per_band)/(normalization_value+1);
			int b = (color_qlab.val[2]*bins_per_band)/(normalization_value+1);
			
			quant->val[p] = l + a*bins_per_band + b*bins_per_band*bins_per_band;
		}

	}else{
		for(p = 0; p < img->n; p++){
		  quant->val[p] = (img->val[p]*bins_per_band)/(normalization_value+1);
		}
	}
	return quant;
}

iftImage *iftGetBICMap(iftImage *quant) {
	iftAdjRel *A;
	iftVoxel u, v;
	iftImage *bic_map = iftCreateImage(quant->xsize, quant->ysize, quant->zsize); // default Interior pixel
	A = iftCircular(1.0);
	for (int p = 0; p < bic_map->n; ++p) {
		u = iftGetVoxelCoord(bic_map, p);
		for (int i = 1; i < A->n; ++i) {
			v = iftGetAdjacentVoxel(A, u, i);
			if (iftValidVoxel(quant, v)) {
				int q = iftGetVoxelIndex(quant, v);
				if (quant->val[p] != quant->val[q]) {
					bic_map->val[p] = 1; // Border pixel
				}
			}
		}
	}
	iftDestroyAdjRel(&A);
	return bic_map;
}

iftIntArray *iftSelectRandomCodeBook(iftImage *image, int nwords) {  // Inefficient
	int p, nselected;
	iftIntArray *code_book = iftCreateIntArray(nwords);
	iftImage *marked = iftCreateImage(image->xsize, image->ysize, image->zsize);
	nselected = 0;
	while(nselected < nwords) {
		p = iftRandomInteger(0, image->n-1);
		if (marked->val[p] == 0) {
			code_book->val[nselected] = p;
			marked->val[p] = 1;
			nselected++;
		}
	}
	iftDestroyImage(&marked);
	return code_book;
}

iftFloatArray *iftComputeBICInPatch(iftMImage *quant, iftBoundingBox bb, int nbins_per_band, iftImage *bic_map) {
	int nbins, nfeats, ninterior, nborder, q;
	iftVoxel v;
	nbins = nbins_per_band * nbins_per_band * nbins_per_band;
	nfeats = 2 * nbins;
	iftFloatArray *feature_vec = iftCreateFloatArray(nfeats);
	ninterior = 0;
	nborder = 0;
	// create histogram
	v.z = 0;
	for (v.y = bb.begin.y; v.y <= bb.end.y; v.y++) {
		for (v.x = bb.begin.x; v.x <= bb.end.x; v.x++) {
			q = iftGetVoxelIndex(quant, v);
			if (bic_map->val[q] == 0) {
				ninterior++;
				feature_vec->val[nbins + (int)quant->val[q][0]] += 1;
			} else {
				nborder++;
				feature_vec->val[(int)quant->val[q][0]] += 1;
			}
		}
	}
	// Normalize
	if (nborder > 0)
		for (int j = 0; j < nbins; ++j)
			feature_vec->val[j] /= (float)nborder;
	if (ninterior > 0)
		for (int j = 0; j < nbins; ++j)
			feature_vec->val[nbins + j] /= (float)ninterior;
	return feature_vec;
}

// It assumes that values in input image is in the range[0-255]
iftFloatArray *iftComputeMCInPatch(iftMImage *img, iftBoundingBox bb, int param_notused, iftImage *img_param_notused) {
	int nfeats, q, norm_term;
	iftVoxel v;
	nfeats = img->m;
	iftFloatArray *feature_vec = iftCreateFloatArray(nfeats);
	norm_term = 255 * (bb.end.y - bb.begin.y + 1) * (bb.end.x - bb.begin.x + 1);
	// create histogram
	v.z = 0;
	for (v.y = bb.begin.y; v.y <= bb.end.y; v.y++) {
		for (v.x = bb.begin.x; v.x <= bb.end.x; v.x++) {
			q = iftGetVoxelIndex(img, v);
			for (int j = 0; j < nfeats; ++j) {
				feature_vec->val[j] += (float)img->val[q][j];
			}
		}
	}
	for (int j = 0; j < nfeats; ++j)
		feature_vec->val[j] /= norm_term;
	return feature_vec;
}

iftDataSet *iftGetDatasetFromSpecifiedPatches(iftMImage *quant, iftIntArray *code_book, int patch_rad, iftExtractPatchDescriptor extractPatchDescriptor, int nbins_per_band, iftImage *bic_map) {
	int nwords, p;
	iftVoxel u;
	iftFloatArray *feature_vec;
	iftBoundingBox bb;
	iftDataSet *dataset_bic = NULL;
	nwords = code_book->n;
	bb.begin.z = 0;
	bb.end.z = 0;
	for (int i = 0; i < nwords; ++i) {
		p = code_book->val[i];
		u = iftGetMVoxelCoord(quant, p);
		bb.begin.x = max_int(0, u.x - patch_rad);
		bb.begin.y = max_int(0, u.y - patch_rad);
		bb.end.x = min_int(quant->xsize - 1, u.x + patch_rad);
		bb.end.y = min_int(quant->ysize - 1, u.y + patch_rad);
		feature_vec = extractPatchDescriptor(quant, bb, nbins_per_band, bic_map);
		if (dataset_bic == NULL)
			dataset_bic = iftCreateDataSet(nwords, feature_vec->n);

		for (int j = 0; j < dataset_bic->nfeats; ++j) {
			dataset_bic->sample[i].feat[j] = feature_vec->val[j];
		}
		iftDestroyFloatArray(&feature_vec);
	}
	return dataset_bic;
}

iftDataSet *iftGetDenseBOWDescriptor(iftMImage *quant, iftImage *label, iftIntArray *code_book, int patch_rad, iftExtractPatchDescriptor extractPatchDescriptor, int nbins_per_band, iftImage *bic_map) {
	int nsuperpixels, nwords, nfeats, p, index_sup, best_codebook;
	float best_codebook_dist, dist_to_codebook;
	iftIntArray *num_pixels_sup;
	iftFloatArray *bic_desc;
	iftVoxel u;
	iftBoundingBox bb;
	iftDataSet *data_codebook, *data_bow;
	nwords = code_book->n;
	nsuperpixels = iftMaximumValue(label);
	num_pixels_sup = iftCreateIntArray(nsuperpixels);
	data_bow = iftCreateDataSet(nsuperpixels, nwords);
	data_codebook = iftGetDatasetFromSpecifiedPatches(quant, code_book, patch_rad, extractPatchDescriptor, nbins_per_band, bic_map);
	nfeats = data_codebook->nfeats;
	printf("nwords %d nsuperpixels %d bpb %d nfeats %d\n", nwords, nsuperpixels, nbins_per_band, nfeats);

	bb.begin.z = 0;
	bb.end.z = 0;
	for (p = 0; p < quant->n; ++p) {
		index_sup = label->val[p] - 1;
		u = iftGetMVoxelCoord(quant, p);
		bb.begin.x = max_int(0, u.x - patch_rad);
		bb.begin.y = max_int(0, u.y - patch_rad);
		bb.end.x = min_int(quant->xsize - 1, u.x + patch_rad);
		bb.end.y = min_int(quant->ysize - 1, u.y + patch_rad);
		bic_desc = extractPatchDescriptor(quant, bb, nbins_per_band, bic_map);

		// coding and pooling 
		best_codebook = 0;
		best_codebook_dist = iftDistance1(bic_desc->val, data_codebook->sample[0].feat, data_codebook->alpha, nfeats);
		for (int i = 1; i < nwords; ++i) {
			dist_to_codebook = iftDistance1(bic_desc->val, data_codebook->sample[i].feat, data_codebook->alpha, nfeats);
			if (dist_to_codebook < best_codebook_dist) {
				best_codebook = i;
				best_codebook_dist = dist_to_codebook;
			}
		}
		data_bow->sample[index_sup].feat[best_codebook] += 1;
		num_pixels_sup->val[index_sup] += 1;
		iftDestroyFloatArray(&bic_desc);
	}
	// Normalize BOW descriptors
	for (int i = 0; i < nsuperpixels; ++i) 
		for (int j = 0; j < nwords; ++j) 
			if (num_pixels_sup->val[i] > 0)
				data_bow->sample[i].feat[j] /= (float)num_pixels_sup->val[i];
	
	iftDestroyIntArray(&num_pixels_sup);
	iftDestroyDataSet(&data_codebook);
	return data_bow;
}

iftIntArray *iftGetNumberOfPixelsInsideSuperpixels(iftImage *label) {
	iftIntArray *npixels_by_sup;
	int nsuperpixels;
	nsuperpixels = iftMaximumValue(label);
	npixels_by_sup = iftCreateIntArray(nsuperpixels);

	for (int p = 0; p < label->n; ++p) {
		int index_sup = label->val[p] - 1;
		npixels_by_sup->val[index_sup] += 1;
	}
	return npixels_by_sup;
}

iftDataSet *iftContextualDescriptorWith3levels(iftDataSet *input_dataset, iftImage *label) {
  iftDataSet *dataset;
  iftSet *adj_set;
  iftRegionGraph *region_graph;
  iftAdjRel *A;
  iftIntArray *queue, *dist, *npixels_by_sup, *total_npixels_by_level;
  int nsamples, back_queue, front_queue, nfeats_input, max_level, nblocks;
  nsamples = input_dataset->nsamples;
  nfeats_input = input_dataset->nfeats;
  max_level = 3;
  nblocks = 3;

  A = iftCircular(1.0);
  region_graph = iftRegionGraphFromLabelImage(label, input_dataset, A);
  dataset = iftCreateDataSet(nsamples, nblocks*nfeats_input);
  queue = iftCreateIntArray(nsamples);
  dist = iftCreateIntArray(nsamples);
  npixels_by_sup = iftGetNumberOfPixelsInsideSuperpixels(label);

  // clean queue
  for (int i = 0; i < nsamples; ++i) {
    queue->val[i] = -1;
    dist->val[i] = IFT_INFINITY_INT;
  }
  
  for (int s = 0; s < nsamples; ++s) {
    total_npixels_by_level = iftCreateIntArray(nblocks);

    dataset->sample[s].id = input_dataset->sample[s].id;
    // copy data from sample s (contextual level = 0)
    for (int j = 0; j < nfeats_input; ++j) 
      dataset->sample[s].feat[j] = input_dataset->sample[s].feat[j];

    front_queue = 0;
    back_queue = 0;
    queue->val[back_queue] = s;
    dist->val[s] = 0;
    back_queue++;
    while (front_queue < back_queue) {
      int u = queue->val[front_queue];
      front_queue++;
      // iterate over neighbours
      adj_set = region_graph->node[u].adjacent;
      while(adj_set){
        int v = adj_set->elem; // neighbour index
        if (dist->val[v] != IFT_INFINITY_INT) {
          adj_set = adj_set->next;
          continue;
        }
        dist->val[v] = dist->val[u] + 1;

        int index_block = dist->val[v];
        if (dist->val[v] == max_level) // special case
          index_block = max_level - 1;
        // accumulate weighted values in feature vector
        for (int j = 0; j < nfeats_input; ++j)
          dataset->sample[s].feat[index_block*nfeats_input  + j] += (input_dataset->sample[v].feat[j] * (float)npixels_by_sup->val[v]);

        total_npixels_by_level->val[index_block] += npixels_by_sup->val[v];

        if (dist->val[v] < max_level) {
          queue->val[back_queue] = v;
          back_queue++;
        }
        adj_set = adj_set->next;
      }
      iftDestroySet(&adj_set);
    }
    // Normalize
    for (int b = 1; b < nblocks; ++b)
      for (int j = 0; j < nfeats_input; ++j) 
        dataset->sample[s].feat[b*nfeats_input + j] /= (float)total_npixels_by_level->val[b];
    // clear queue
    for (int i = 0; i < back_queue; ++i) 
      queue->val[i] = -1;
    for (int i = 0; i < nsamples; ++i)
      dist->val[i] = IFT_INFINITY_INT;

    iftDestroyIntArray(&total_npixels_by_level);
  }
  
  // Free
  iftDestroyIntArray(&queue);
  iftDestroyIntArray(&dist);
  iftDestroyIntArray(&npixels_by_sup);
  iftDestroyRegionGraph(&region_graph);
  iftDestroyAdjRel(&A);
  
  iftSetDistanceFunction(dataset, 1);
  dataset->ref_data      = input_dataset->ref_data;
  dataset->ref_data_type = input_dataset->ref_data_type;

  return dataset;
}

int main(int argc, char *argv[])
{
	iftImage *img, *label, *quant, *bic_map, *gt_img;
	iftMImage *mquant, *qlabmimg;
	iftDataSet *contextual_bow_bic, *contextual_bow_mc, *concatenated_dataset;
	iftDataSet *data_bow_bic, *data_bow_mc, *output_dataset;
	iftIntArray *code_book;
	int *superpixel_clslabel;
	int nbins_per_band, patch_rad, nwords, is_contextual, desc_type;
	data_bow_bic = NULL;
	data_bow_mc = NULL;
	contextual_bow_bic = NULL;
	contextual_bow_mc = NULL;
	concatenated_dataset = NULL;
	quant = NULL;
	mquant = NULL;
	bic_map = NULL;
	qlabmimg = NULL;
	
	if (argc != 10 && argc!=11)
		iftError("Usage: iftContextualBOWDescriptor <image.ppm> <superpixels.pgm> <num_bins_per_band> <num_visual_words> <patch_radius> <classification_map.pgm> <contextual 1: Yes, 2: No> <type 1: BIC, 2: MC, 3: all> <output_dataset>  <(optional) codebook_file>", "main");
	
	img = iftReadImageByExt(argv[1]);
	label = iftReadImageByExt(argv[2]);
	nbins_per_band = atoi(argv[3]);
	nwords = atoi(argv[4]);
	patch_rad = atoi(argv[5]);
	gt_img = iftReadImageByExt(argv[6]);
	is_contextual = atoi(argv[7]);
	desc_type = atoi(argv[8]);

	//iftRandomSeed(time(NULL));

	// obtain codebook
	if (argc == 10)
		code_book = iftSelectRandomCodeBook(img, nwords);
	else {
		FILE *fcodebook = fopen(argv[10], "r");
		code_book = iftCreateIntArray(nwords);
		for(int i=0; i< nwords; i++){
			int cb_pos;
			fscanf(fcodebook,"%d",&cb_pos);
			code_book->val[i] = cb_pos;
		}
		fclose(fcodebook);
	}
	printf("Preprocessing .....\n");
	// Get base descriptors
	if (desc_type == 1 || desc_type ==3) { // BOW-BIC
		quant = iftLabQuantize(img, nbins_per_band, 255);
		mquant = iftImageToMImage(quant, GRAY_CSPACE);
		bic_map = iftGetBICMap(quant);
		data_bow_bic = iftGetDenseBOWDescriptor(mquant, label, code_book, patch_rad, iftComputeBICInPatch, nbins_per_band, bic_map);
	}
	if (desc_type == 2 || desc_type ==3) { // BOW-MC
		qlabmimg = iftImageToCIEQLabMImage(img);
		data_bow_mc = iftGetDenseBOWDescriptor(qlabmimg, label, code_book, patch_rad, iftComputeMCInPatch, 3, NULL);
	} 
	printf("Preprocessing Ok\n");
	
	// Get contextual descritor if required
	if (desc_type == 1) { // BOW-BIC
		// Contextual descriptor
		if (is_contextual == 1) { 
			contextual_bow_bic = iftContextualDescriptorWith3levels(data_bow_bic, label);
			iftDestroyDataSet(&data_bow_bic);
			output_dataset = contextual_bow_bic;
			printf("Contextual BOW-BIC Ok\n");
		} else {
			output_dataset = data_bow_bic;
			printf("BOW-BIC Ok\n");
		}
	} else if (desc_type == 2) { // BOW-MC
		// Contextual descriptor
		if (is_contextual == 1) { // Contextual descriptor
			contextual_bow_mc = iftContextualDescriptorWith3levels(data_bow_mc, label);
			iftDestroyDataSet(&data_bow_mc);
			output_dataset = contextual_bow_mc;
			printf("Contextual BOW-MC Ok\n");
		} else {
			output_dataset = data_bow_bic;
			printf("BOW-MC Ok\n");
		}
	} else { // Concatenate both descriptors
		if (is_contextual == 1) {
			iftDataSet *datasets[2];
			contextual_bow_bic = iftContextualDescriptorWith3levels(data_bow_bic, label);
			iftDestroyDataSet(&data_bow_bic);
			contextual_bow_mc = iftContextualDescriptorWith3levels(data_bow_mc, label);
			iftDestroyDataSet(&data_bow_mc);
			datasets[0] = contextual_bow_bic;
			datasets[1] = contextual_bow_mc;
			concatenated_dataset = iftConcatDatasetFeatures(datasets, 2);
			iftDestroyDataSet(&contextual_bow_bic);
			iftDestroyDataSet(&contextual_bow_mc);
			output_dataset = concatenated_dataset;
			printf("Contextual BOW-MC + Contextual BOW-BIC Ok\n");
		} else {
			iftDataSet *datasets[2];
			datasets[0] = data_bow_bic;
			datasets[1] = data_bow_mc;
			concatenated_dataset = iftConcatDatasetFeatures(datasets, 2);
			output_dataset = concatenated_dataset;
			printf("BOW-BIC + BOW-MC Ok\n");
		}
	}

	// get superpixel labels
	int nclasses = iftMaximumValue(gt_img);
	int nsuperpixels = iftMaximumValue(label);
	superpixel_clslabel = iftAllocIntArray(nsuperpixels);
	for (int p = 0; p < gt_img->n; ++p) {
		int index_sup = label->val[p] - 1;
		superpixel_clslabel[index_sup] = gt_img->val[p];
	}

	printf("Copying labels ...\n");
	// copy labels
	output_dataset->nclasses = nclasses;
	for (int i = 0; i < output_dataset->nsamples; ++i) {
		output_dataset->sample[i].truelabel = superpixel_clslabel[i];
	}
	printf("nfeats %d\n", output_dataset->nfeats);
	printf("nsamples %d\n", output_dataset->nsamples);
	
	// write dataset
	iftSetDistanceFunction(output_dataset, 1);
    iftWriteOPFDataSet(output_dataset, argv[9]);
    output_dataset = NULL;

    // Free
	iftDestroyIntArray(&code_book);
	iftDestroyImage(&gt_img);
	iftDestroyImage(&img);
	iftDestroyImage(&label);
	free(superpixel_clslabel);

	if (quant != NULL) iftDestroyImage(&quant);
	if (mquant != NULL) iftDestroyMImage(&mquant);
	if (qlabmimg != NULL) iftDestroyMImage(&qlabmimg);
	if (bic_map != NULL) iftDestroyImage(&bic_map);

	if (data_bow_bic != NULL) iftDestroyDataSet(&data_bow_bic);
	if (data_bow_mc != NULL) iftDestroyDataSet(&data_bow_mc);
	if (contextual_bow_bic != NULL) iftDestroyDataSet(&contextual_bow_bic);
	if (contextual_bow_mc != NULL) iftDestroyDataSet(&contextual_bow_mc);
	if (concatenated_dataset != NULL) iftDestroyDataSet(&concatenated_dataset);
}