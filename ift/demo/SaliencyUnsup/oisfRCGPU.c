#include "ift.h"

#include <cuda.h>
#include <cublas.h>

// Retorna os pontos mais proximos do centro geometrico de cada label

//iftVoxelArray *iftGeometricCentersFromLabelImage(objMask); - Return

/*
 *
      centers = iftGeometricCentersFromLabelImage(objMask);

      sampled->val[i-1] = iftGetVoxelIndex(objMask, centers->val[1]);


//Retorna quantidade de elementos no superpixel (obj_label)
int iftCountObjectSpels(  iftImage *label, int obj_label);


 * iftCompEuclDistanceTable(  iftDataSet *Z1,   iftDataSet* Z2);
 *
 * iftDistanceTable *iftCreateDistanceTable(int nsamples1, int nsamples2);

 */


int iftUnsupClassifyWithoutCheckingOutliers(iftKnnGraph *graph, iftDataSet *Z);

iftImage *imageToGlobalSaliencyImage(iftImage *img, iftDataSet *Z, int cluster_number);

iftImage *imageToRegionSaliencyImage(iftImage *img, iftDataSet *Z, iftImage *label_img, int cluster_number);

iftDataSet *computeQuantizedDataset(iftImage *img, int nsplits, float train_perc, int max_number_of_clusters, int *cluster_number, int kmax);

iftMImage *iftExtendMImageByGrayObjSalMap(iftMImage *mimg, iftImage* objsm);

iftIGraph *iftInitOISFIGraph(iftImage *img, iftImage *mask, iftImage *objsm);

iftDataSet *computeQuantizedDatasetKMeans(iftImage *img, int max_number_of_clusters, int *cluster_number,  int maxIterations, float minImprovement);

void saliencyCalculationGPU(  iftMatrix *D,   iftMatrix *P,   iftFloatArray *W, iftFloatArray **saliency, iftDistanceTable *centroid_distances, float sigma_squared, double *border_region_ratio);

iftSet *iftObjSalMapSamplByHighestValue(iftImage *objsm, int num_seeds );

iftImage *iftSamplingByOSMOX(iftImage* objsm, int num_seeds, float obj_perc);

void iftWriteOverlay(iftImage* orig, iftImage *label, const char *filename);

iftImage *convertToLab(iftImage *image);

iftImage* quantizedDatasetToImage(iftDataSet* Z,iftImage* img);

iftDataSet *replicateMedoidToCluster(iftDataSet *Z);

iftFloatArray *saliencySmoothing(iftDataSet *Z, iftImage *label_img, iftFloatArray *saliency, iftColorTable *colors, int cluster_number, iftMatrix *region_color_prob, iftMatrix *region_has_color, double *border_region_ratio, iftFloatArray *global_color_prob);

/**********************************************************************
																MAIN
**********************************************************************/
int main(int argc, const char *argv[])
{
    iftImage  *img, *original_image, *saliency_img, *label_img = NULL, *mask, *seeds, *combined_maps_img;
    iftDataSet *Z = NULL, *Z_quantized;
    iftIGraph *igraph, *out_igraph;
    int  maxIter = 30, max_number_of_clusters, number_superpixels, cluster_number, iterations; //ARG
    float thr = 0.1, alpha=0.5, beta=12, gamma=4, iter=1, minImprovement=0.01, obj_perc = 0.005;
    char      filename[250];
    iftFloatArray **combined_maps_pixels;

    /*
     * image is the input image
     * label_img is a superpixel map (optional based on whether you want to use region or global based)
     * input_train_perc  is the percentage of pixels used for training of OPF`s cluster algorithm
     * output-basename is the file to output the saliency map
     * maximum number of colors after the quantization by OPF clustering
     */
//
    if (argc != 6){
        iftError("Usage: Region Saliency <image.jpg> <number of iterations> <output-basename> <maximum number of colors> <number of superpixels> ", "main");
    }

    sprintf(filename,"../datasets/%s/images/%s",argv[3] ,argv[1]);

    original_image = iftReadImageByExt(filename);
    original_image = iftLinearFilter(original_image, iftGaussianKernel2D(8, 0.5));
    img = convertToLab(original_image);
    iftDestroyImage(&original_image);
    combined_maps_img = iftCreateImage(img->xsize, img->ysize, img->zsize);
    iterations = atoi(argv[2]);
    max_number_of_clusters = atoi(argv[4]);
    number_superpixels = atoi(argv[5]);
    mask = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);

    combined_maps_pixels = (iftFloatArray**) calloc(img->n, sizeof(iftFloatArray*));

#pragma omp parallel for
    for(int p = 0; p < img->n; p++)
        combined_maps_pixels[p] = iftCreateFloatArray(iterations);


    //Create Dataset for quantization
    Z = computeQuantizedDatasetKMeans(img, max_number_of_clusters, &cluster_number, maxIter, minImprovement);

    //Z_quantized = replicateMedoidToCluster(Z);

    Z_quantized = iftCopyDataSet(Z, false);

    img = quantizedDatasetToImage(Z_quantized, img);
//    iftWriteImageByExt(saliency_img, "global-result.png");


    saliency_img = imageToGlobalSaliencyImage(img, Z_quantized, cluster_number);


    printf("Computing saliency for image: %s\n", filename);
    for(int i = 0; i<iterations; i++) {
        printf("Iteration %d\n", i);
        obj_perc*= 0.95;
        number_superpixels*= 0.96;
        gamma+=0.1;
        seeds = iftSamplingByOSMOX(saliency_img, number_superpixels, obj_perc);
        //seeds = iftGrayObjMapGridSamplOnMaskByArea(saliency_img, mask, number_superpixels, thr, obj_perc);

        /* Uncomment below line to change OISF Sampling to Grid */
        //seeds = iftGrayObjMapGridSamplOnMaskByArea(saliency_img, mask, number_superpixels, thr, 0.9);
        igraph = iftInitOISFIGraph(img, mask, saliency_img);


        out_igraph = iftIGraphOISF(igraph, seeds, alpha, beta, gamma, iter);

        label_img = iftIGraphLabel(out_igraph);
        iftDestroyImage(&saliency_img);

        iftDestroyIGraph(&out_igraph);
        iftDestroyIGraph(&igraph);
        iftDestroyImage(&seeds);

        /* */
        iftSList *list = iftSplitString(argv[1],".");
        iftSNode *L    = list->head;
        saliency_img = imageToRegionSaliencyImage(img, Z, label_img, cluster_number);

        for(int p = 0; p < img->n; p++)
            combined_maps_pixels[p]->val[i] = saliency_img->val[p];

	if((i+1)%5 == 0){
		sprintf(filename,"../datasets/%s/oisfRC/%s-%d.png",argv[3] ,L->elem, i);
		iftWriteImageByExt(saliency_img, filename);
        sprintf(filename,"../datasets/%s/oisfRC/%s-%d-label.png",argv[3] ,L->elem, i);
        iftImage *img_copy = iftCopyImage(img);
        iftWriteOverlay(quantizedDatasetToImage(Z_quantized, img_copy), label_img, filename);
        iftDestroyImage(&img_copy);
	}

	iftDestroySList(&list);

        iftDestroyImage(&label_img);
    }
    iftDestroyDataSet(&Z);



    iftSList *list = iftSplitString(argv[1],".");
    iftSNode *L    = list->head;
    /* Combined maps */
    for(int p = 0; p < combined_maps_img->n; p++){
        combined_maps_img->val[p] = iftRound(iftMedianFloatArray(combined_maps_pixels[p]->val, iterations));

    }
    sprintf(filename,"../datasets/%s/oisfRC/%s-comb.png",argv[3] ,L->elem);
    iftWriteImageByExt(combined_maps_img, filename);
    iftDestroyImage(&combined_maps_img);


    for(int p = 0; p < img->n; p++)
        iftDestroyFloatArray(&combined_maps_pixels[p]);

    free(combined_maps_pixels);


    // Free
    iftDestroyImage(&img);
    iftDestroyImage(&saliency_img);
    iftDestroyDataSet(&Z_quantized);
    iftDestroyImage(&mask);


    return 0;
}

/*
 * Variant to iftUnsupClassify, but never assign a pixel as an outlier
 */
int iftUnsupClassifyWithoutCheckingOutliers(iftKnnGraph *graph, iftDataSet *Z)
{

    iftDataSet *Z1=graph->Z;
    float K=2.0*graph->maxarcw[graph->k]*graph->maxarcw[graph->k]/9.0;

    if (Z1->nfeats != Z->nfeats)
        iftError("Incompatible datasets", "iftUnsupClassify");

    // Propagate group labels
    int noutliers=1;

#pragma omp parallel for shared(Z,Z1,graph,K,noutliers)
    for (int t=0; t < Z->nsamples; t++)
    {
        if ((Z==Z1)&&(iftHasSampleStatus(Z->sample[t], IFT_TRAIN) || iftHasSampleStatus(Z->sample[t], IFT_PROTOTYPE)))
            continue;

        /* Adan Echemendia modified this code, each label will be has default label -1, the previous code put this label
         * to 0 but this carry problems with methods of propagating the ids of the cluster representatives, so in this
         * methods a sample can be classified with label 0 because it was conquered by the representative with id 0 of a
         * cluster. This doesn't produce any consequence to the calls to this method */
        Z->sample[t].group   = -1;
        Z->sample[t].weight  = 0.0;
        float dist,min_dist = IFT_INFINITY_FLT;
        int closest_node = IFT_NIL;

        for (int i = 0; i < graph->nnodes; i++){
            int u    = graph->ordered_nodes[i];
            int s    = graph->node[u].sample;
            if (iftDist==NULL)
                dist = Z1->iftArcWeight(Z1->sample[s].feat,Z->sample[t].feat,Z1->alpha,Z1->nfeats);
            else
                dist = iftDist->distance_table[s][t];
            if (dist <= graph->node[u].maxarcw){
                Z->sample[t].group    = Z1->sample[s].group;
                Z->sample[t].weight   = graph->pathval[u]*expf(-dist/K);
                break;
            }else{ // compute closest node for the case of outliers
                if (dist < min_dist) {
                    min_dist       = dist;
                    closest_node   = u;
                }
            }
        }

        if (Z->sample[t].group==-1){ /* t is an outlier */
#pragma omp atomic
            noutliers++;
            iftAddSampleStatus(&Z->sample[t], IFT_OUTLIER);
            /* it is important for background removal */
            Z->sample[t].weight = graph->pathval[closest_node]*exp(-(min_dist)/K);
            int s = graph->node[closest_node].sample;
            Z->sample[t].group  = Z1->sample[s].group;
        }
    }

    for (int t=0; t < Z->nsamples; t++)
        if (Z->sample[t].group > Z->ngroups )
            Z->ngroups = Z->sample[t].group;

    return(noutliers);
}

/*
 * Return a quantized dataset and change the value of cluster_number to the number of clusters created
 */
iftDataSet *computeQuantizedDataset(iftImage *img, int nsplits, float train_perc, int max_number_of_clusters, int *cluster_number, int kmax){
    iftDataSet *Z;
    iftSampler *sampler=NULL;
    iftKnnGraph *knn_graph;

    Z = iftImageToDataSet(img);

    //Sample for faster training
    sampler = iftRandomSubsampling(Z->nsamples, nsplits, train_perc*Z->nsamples);

    iftDataSetSampling(Z, sampler, 0);

    knn_graph = iftCreateKnnGraph(Z, kmax);

    *cluster_number = iftUnsupTrainWithCClusters(knn_graph, max_number_of_clusters);


    printf("Cluster number = %d\n", *cluster_number);

    iftUnsupClassifyWithoutCheckingOutliers(knn_graph, Z);

    iftDestroyKnnGraph(&knn_graph);
    iftDestroySampler(&sampler);

    return Z;
}


/*
 * Return a quantized dataset and change the value of cluster_number to the number of clusters created
 */
iftDataSet *computeQuantizedDatasetKMeans(iftImage *img, int max_number_of_clusters, int *cluster_number,  int maxIterations, float minImprovement){
    iftDataSet *Z;

    Z = iftImageToDataSet(img);

    iftClusterDataSetByKMeans(Z, max_number_of_clusters, maxIterations, minImprovement, 0, '0', '0');

    *cluster_number = Z->ngroups;

    printf("Cluster number = %d\n", *cluster_number);

    //iftUnsupClassifyWithoutCheckingOutliers(knn_graph, Z);

    return Z;
}


double calculateColorDistance(iftColorTable *colors, int color_index_1, int color_index_2){
    double distance=0;

    for (int i =0; i<3; i++)
        distance+=pow(colors->color[color_index_1].val[i] - colors->color[color_index_2].val[i],2);
    distance = sqrt(distance);
    return distance;
}



iftFloatArray *saliencySmoothing(iftDataSet *Z, iftImage *label_img, iftFloatArray *saliency, iftColorTable *colors, int cluster_number, iftMatrix *region_color_prob, iftMatrix *region_has_color, double *border_region_ratio, iftFloatArray *global_color_prob){
    int m;
    iftDataSet *Z_color_saliency, *Z_color_smoothing, *Z_color_saliency_new;
    iftKnnGraph *smoothing_knn;
    iftFloatArray *saliency_new;
    iftIntArray *color_sizes;


    /* ------------------ Saliency Smoothing -----------------*/
    m = cluster_number / 10;
    color_sizes = iftCreateIntArray(cluster_number);

    Z_color_saliency = iftCreateDataSet(cluster_number, 1);
    Z_color_saliency_new = iftCreateDataSet(cluster_number, 1);
    Z_color_smoothing = iftCreateDataSet(cluster_number,3);
    saliency_new = iftCreateFloatArray((int)saliency->n);


    //Find the mean saliency of each color
    for(int p =0; p < (int)Z->nsamples; p++){
        int c = Z->sample[p].group-1;
        Z_color_saliency->sample[c].feat[0]+=saliency->val[label_img->val[p]-1];
        color_sizes->val[c]++;
    }

    for(int c = 0; c<cluster_number; c++){
        Z_color_saliency->sample[c].feat[0] /= color_sizes->val[c];
        Z_color_smoothing->sample[c].feat[0] = colors->color[c].val[0];
        Z_color_smoothing->sample[c].feat[1] = colors->color[c].val[1];
        Z_color_smoothing->sample[c].feat[2] = colors->color[c].val[2];
    }

    iftSetStatus(Z_color_smoothing, IFT_TRAIN);


    smoothing_knn = iftCreateKnnGraph(Z_color_smoothing, m);

    for(int c =0; c < cluster_number; c++) {
        double sum = 0;
        double t = 0;
        double distance = 0;
        for (int nn = 0; nn < smoothing_knn->k; nn++){
            t+= smoothing_knn->node[c].adj[smoothing_knn->ordered_nodes[nn]].arcw;
        }
        for (int nn = 0; nn < m; nn++){
            distance = smoothing_knn->node[c].adj[smoothing_knn->ordered_nodes[nn]].arcw;
            sum+= (t - distance)* Z_color_saliency->sample[c].feat[0];
        }
        Z_color_saliency_new->sample[c].feat[0] = (float)sum/((m-1)*t); //saliency average of color c
    }
    iftDestroyDataSet(&Z_color_saliency);
    iftDestroyDataSet(&Z_color_smoothing);
    iftDestroyKnnGraph(&smoothing_knn);

    for(int ri = 0; ri < (int)saliency->n-1; ri++){
        double new_saliency = 0;
        float color_counter = 0;
        for(int ci = 0; ci < cluster_number; ci++) {
            if (!iftMatrixElem(region_has_color, ci, ri))
                continue;
            new_saliency+= exp(-global_color_prob->val[ci]) * iftMatrixElem(region_color_prob, ci, ri) * Z_color_saliency_new->sample[ci].feat[0]; //applying weight
            color_counter+= 1 * iftMatrixElem(region_color_prob, ci, ri);
        }
        saliency_new->val[ri] = (new_saliency*border_region_ratio[ri])/color_counter; //compute new saliency based on neighbors

    }
    iftDestroyDataSet(&Z_color_saliency_new);
    iftDestroyIntArray(&color_sizes);
    return saliency_new;
}


/*
 * Return an saliency map given an image based on global contrast
 */
iftImage *imageToRegionSaliencyImage(iftImage *img, iftDataSet *Z, iftImage *label_img, int cluster_number){
    iftDataSet *Z_centroid, *Z_region_map = NULL, *Z_region_map_normalized = NULL, *Z_center_voxel;
    int  c =0, border_radius = 20; //m is used for smoothing
    double  max_saliency = 0, max_region_size = 0, min_saliency = IFT_INFINITY_DBL, *border_region_ratio, max_color = 0, min_color = IFT_INFINITY_DBL;
    float sigma_squared = -1, border_penalty = 0.95;
    iftColorTable *colors;
    iftImage *saliency_image;
    iftMatrix *region_color_prob, *region_has_color, *color_distance_matrix;
    iftIntArray *region_sizes;
    iftFloatArray *saliency, *smoothed_saliency, *region_sizes_prob, *global_color_prob;
    iftDistanceTable *centroid_distances, *region_to_center;
    iftVoxelArray *centroids;


    saliency_image = iftCreateImage(label_img->xsize,label_img->ysize,label_img->zsize);

    colors = iftCreateColorTable(cluster_number);

    region_sizes = iftCountLabelSpels(label_img);
    region_sizes_prob = iftCreateFloatArray(region_sizes->n-1);

    region_color_prob = iftCreateMatrix(cluster_number, ((int)region_sizes_prob->n));
    region_has_color = iftCreateMatrix(cluster_number, ((int)region_sizes_prob->n));

    global_color_prob = iftCreateFloatArray(cluster_number);

    /* -------------------- Calculate the probabilities of each color to appear on each region ---------------------- */
    for(int p = 0; p < label_img->n; p++){
        c = Z->sample[p].group-1;
        iftMatrixElem(region_color_prob, c, label_img->val[p]-1)++;
        iftMatrixElem(region_has_color, c, label_img->val[p]-1) = 1;
        //global_color_prob->val[c]++;

        colors->color[c].val[0] = Z->sample[p].feat[0];
        colors->color[c].val[1] = Z->sample[p].feat[1];
        colors->color[c].val[2] = Z->sample[p].feat[2];
    }

    for (int c = 0; c < global_color_prob->n; c++)
        for(int r =0; r < region_sizes_prob->n; r++)
            global_color_prob->val[c]+=iftMatrixElem(region_has_color, c, r);

    // Find min and max color for normalization
    for (int c = 0; c < global_color_prob->n; c++){
        if(global_color_prob->val[c] > max_color)
            max_color = global_color_prob->val[c];
        if(global_color_prob->val[c] < min_color)
            min_color = global_color_prob->val[c];
    }

    for (int c = 0; c < global_color_prob->n; c++) {
        global_color_prob->val[c] =  ((global_color_prob->val[c]) / (float) (max_color - min_color));
        //printf("prob %f \n", global_color_prob->val[c]);
    }

    //Normalize the probabilities from 0-1 based on region size
    for(int reg = 0; reg < (int)region_sizes_prob->n; reg++){
        for(int color = 0; color < cluster_number; color++) {
            iftMatrixElem(region_color_prob, color, reg) = ((float)iftMatrixElem(region_color_prob, color, reg)) / region_sizes->val[reg+1];
        }
    }


    for(int r = 1; r < region_sizes->n; r++){
        if(region_sizes->val[r] > max_region_size)
            max_region_size = region_sizes->val[r];
    }

    for(int r = 0; r < region_sizes_prob->n; r++){
        region_sizes_prob->val[r] = ((float)region_sizes->val[r+1]/max_region_size);
    }

    /*------------------------- Calculate the distances of the centroids and to center of image ----------------------*/
    Z_centroid = iftCreateDataSet((int)region_sizes_prob->n, 2);
    Z_region_map = iftCreateDataSet(img->n, 2);
    Z_region_map->ngroups = ((int)region_sizes_prob->n);
    Z_region_map_normalized = iftCreateDataSet(img->n, 2);
    Z_region_map_normalized->ngroups = ((int)region_sizes_prob->n);
    centroids = iftGeometricCentersFromLabelImage(label_img);

#pragma omp parallel for
    for(int reg = 1; reg < (int)region_sizes->n; reg++){
        Z_centroid->sample[reg-1].feat[0] = (float)centroids->val[reg].x / (img->xsize-1);
        Z_centroid->sample[reg-1].feat[1] = (float)centroids->val[reg].y / (img->ysize-1);
        //printf("%f\n", Z_centroid->sample[reg-1].feat[0]);
        for(int p = 0; p < img->n; p++){
            iftVoxel u;
            u = iftGetVoxelCoord(img, p);
            //Z_region_map is used in Find border regions
            Z_region_map->sample[p].feat[0] = (float)u.x;
            Z_region_map->sample[p].feat[1] = (float)u.y;
            Z_region_map->sample[p].group = label_img->val[p];
            // Z_region_map_normalized is used in Calculate the distances of the centroids and to center of image
            Z_region_map_normalized->sample[p].feat[0] = (float)u.x / (img->xsize-1);
            Z_region_map_normalized->sample[p].feat[1] = (float)u.y / (img->ysize-1);
            Z_region_map_normalized->sample[p].group = label_img->val[p];
        }
    }

    Z_center_voxel = iftCreateDataSet(1, 2);

    Z_center_voxel->sample[0].feat[0] = ((float)(img->xsize/2)-1) / (img->xsize-1);
    Z_center_voxel->sample[0].feat[1] = ((float)(img->ysize/2)-1) / (img->ysize-1);

    region_to_center = iftCompEuclDistanceTable(Z_region_map_normalized, Z_center_voxel);
    iftDestroyDataSet(&Z_region_map_normalized);
    iftDestroyDataSet(&Z_center_voxel);

    /* -------------------------------- Find border regions ----------------------------------*/
    border_region_ratio = iftAllocDoubleArray((int)region_sizes_prob->n);

#pragma omp parallel for
    for(int reg = 1; reg < (int)region_sizes->n; reg++){
        int border_counter = 0;
        iftDataSet *Z_region = iftExtractGroup(Z_region_map, reg);
        for(int d = 0; d < Z_region->nsamples; d++){
            //Count how many pixels are within the border (x pixels adjacency)
            if(Z_region->sample[d].feat[0] >= (img->xsize - border_radius) ||  Z_region->sample[d].feat[1] >= (img->ysize - border_radius) || Z_region->sample[d].feat[0] <= border_radius || Z_region->sample[d].feat[1] <= border_radius)
                border_counter++;
        }
        if(border_counter > (Z_region->nsamples / 70)) {
            border_region_ratio[reg - 1] = border_penalty;
        }
        else
            border_region_ratio[reg-1] = 1;

        //border_region_ratio[reg-1] = 1;
        iftDestroyDataSet(&Z_region);
    }

    iftDestroyIntArray(&region_sizes);
    iftDestroyDataSet(&Z_region_map);

    centroid_distances = iftCompEuclDistanceTable(Z_centroid, Z_centroid);

    iftDestroyDataSet(&Z_centroid);

    saliency = iftCreateFloatArray(region_sizes_prob->n);
    // The variables are named according to the paper

    /* --------------------------- Calculate Color Distance Table---------------------------------------------------- */
    color_distance_matrix = iftCreateMatrix(cluster_number, cluster_number);
    for(int ci = 0; ci < cluster_number; ci++) {
        for (int cj = 0; cj < cluster_number; cj++) {
            iftMatrixElem(color_distance_matrix, ci, cj) = calculateColorDistance(colors, ci, cj);// * exp(-global_color_prob->val[ci]);
        }
    }


    /*----------------------------- Calculate Saliency ---------------------------------------------------------------*/
    saliencyCalculationGPU(color_distance_matrix, region_color_prob, region_sizes_prob, &saliency, centroid_distances, sigma_squared, border_region_ratio);

    iftDestroyMatrix(&color_distance_matrix);

    /* Apply Final Penalties */
    for(int ri = 0; ri < region_sizes_prob->n; ri++){
        saliency->val[ri] = saliency->val[ri] * border_region_ratio[ri] * region_sizes_prob->val[ri];
    }

    /* Smoothing by color*/
    smoothed_saliency = saliencySmoothing(Z, label_img, saliency, colors, cluster_number, region_color_prob, region_has_color, border_region_ratio, global_color_prob);

    iftDestroyFloatArray(&global_color_prob);
    iftDestroyFloatArray(&saliency);

    iftDestroyVoxelArray(&centroids);
    iftDestroyMatrix(&region_has_color);
    iftDestroyMatrix(&region_color_prob);
    iftDestroyDistanceTable(&centroid_distances);
    iftDestroyColorTable(&colors);
    iftDestroyDistanceTable(&region_to_center);
    free(border_region_ratio);
    /* Normalize Saliency to 0-255 */
    for(int rk = 0; rk < (int)region_sizes_prob->n; rk++) { //Get Max
        if(smoothed_saliency->val[rk] > max_saliency)
            max_saliency = smoothed_saliency->val[rk];
        if(smoothed_saliency->val[rk] < min_saliency)
            min_saliency = smoothed_saliency->val[rk];
    }

    for(int rk = 0; rk < (int)region_sizes_prob->n; rk++) //Normalize
        smoothed_saliency->val[rk] = 255*(smoothed_saliency->val[rk]-min_saliency)/(max_saliency-min_saliency);


    for(int p = 0; p < saliency_image->n; p++)
        saliency_image->val[p] = smoothed_saliency->val[label_img->val[p]-1];

    iftDestroyFloatArray(&smoothed_saliency);
    iftDestroyFloatArray(&region_sizes_prob);

    return saliency_image;

}

/*
 * Return an saliency map given an image based on region (superpixel) contrast
 */
iftImage *imageToGlobalSaliencyImage(iftImage *img, iftDataSet *Z, int cluster_number){
    int c =0;
    double *probabilities, *saliency, distance = 0, max_saliency = 0, sum = 0, min_saliency = IFT_INFINITY_DBL;
    iftColorTable *colors;
    iftImage *saliency_image;
    iftDataSet *Z_center_pixel, *Z_pixel_map_normalized;
    iftDistanceTable *pixel_to_center;

    saliency_image = iftCreateImage(img->xsize,img->ysize,img->zsize);

    probabilities = (double*) calloc(cluster_number, sizeof(double));
    saliency = (double*) calloc(cluster_number, sizeof(double));
    colors = iftCreateColorTable(cluster_number);

    //Calculate the probabilities of each color to appear and create the color vector according to the features
    for(int s = 0; s < Z->nsamples; s++) {
        c = Z->sample[s].group-1;
        probabilities[c]++;
        colors->color[c].val[0] = Z->sample[s].feat[0];
        colors->color[c].val[1] = Z->sample[s].feat[1];
        colors->color[c].val[2] = Z->sample[s].feat[2];
    }

    //Normalize the probabilities from 0-1
    for(int color = 0; color < cluster_number; color++) {
        probabilities[color] = probabilities[color] / Z->nsamples;
    }

    /* Distance from pixel to center */

    Z_center_pixel = iftCreateDataSet(1, 2);

    Z_pixel_map_normalized = iftCreateDataSet(img->n, 2);

    Z_center_pixel->sample[0].feat[0] = ((float)(img->xsize/2)-1) / (img->xsize-1);
    Z_center_pixel->sample[0].feat[1] = ((float)(img->ysize/2)-1) / (img->ysize-1);

#pragma omp parallel for
    for(int p = 0; p < img->n; p++){
        iftVoxel u;
        u = iftGetVoxelCoord(img, p);
        Z_pixel_map_normalized->sample[p].feat[0] = (float)u.x / (img->xsize-1);
        Z_pixel_map_normalized->sample[p].feat[1] = (float)u.y / (img->ysize-1);
    }

    pixel_to_center = iftCompEuclDistanceTable(Z_pixel_map_normalized, Z_center_pixel);
    iftDestroyDataSet(&Z_pixel_map_normalized);
    iftDestroyDataSet(&Z_center_pixel);


    /* ------ Calculate Global Saliency ------- */
    for(int cl = 0; cl < cluster_number; cl++){
        for(int cj = 0; cj < cluster_number; cj++){
            for(int i =0; i < Z->nfeats; i++){
                sum+=pow(colors->color[cl].val[i] - colors->color[cj].val[i],2);
            }
            distance = distance + (sqrt(sum) * probabilities[cj]);
            sum = 0;
        }
        saliency[cl] = distance;
        distance = 0;
    }

    /* Normalize Saliency to 0-255 */
    for(int cl = 0; cl < cluster_number; cl++){ //Get Max
        if(saliency[cl] > max_saliency)
            max_saliency = saliency[cl];
        if(saliency[cl] < min_saliency)
            min_saliency = saliency[cl];
    }

    for(int cl = 0; cl < cluster_number; cl++) //Normalize
        saliency[cl] = 255*(saliency[cl]-min_saliency)/(max_saliency-min_saliency);


    for(int p = 0; p < saliency_image->n; p++)
        saliency_image->val[p] = saliency[Z->sample[p].group-1] * pixel_to_center->distance_table[p][0];

    max_saliency = 0;
    min_saliency = IFT_INFINITY_DBL;

    free(saliency);

    for(int p = 0; p < img->n; p++){ //Get Max
        if(saliency_image->val[p] > max_saliency)
            max_saliency = saliency_image->val[p];
        if(saliency_image->val[p] < min_saliency)
            min_saliency = saliency_image->val[p];
    }

    for(int p = 0; p < saliency_image->n; p++)
        saliency_image->val[p] = 255*(saliency_image->val[p]-min_saliency)/(max_saliency-min_saliency);

    // Free
    iftDestroyColorTable(&colors);
    iftDestroyDistanceTable(&pixel_to_center);
    free(probabilities);
    //return iftDataSetClustersToQuantizedImage(Z, true);
    return saliency_image;

}


/* OISF */

iftMImage *iftExtendMImageByGrayObjSalMap
        (iftMImage *mimg, iftImage* objsm)
{
    // Variables
    int max, norm;
    iftMImage *emimg;

    // Init
    max = iftMaximumValue(objsm);
    norm = iftNormalizationValue(max);
    emimg = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m+1);

    // For all pixels
    for (int p = 0; p < mimg->n; p++)  {
        // In all bands
        for(int b = 0; b < mimg->m; b++ )  emimg->val[p][b] = mimg->val[p][b];
        // Put the value of the object saliency map in the last band
        emimg->band[mimg->m].val[p] = objsm->val[p]/(float)norm;
    }

    return emimg;
}

iftIGraph *iftInitOISFIGraph
        (iftImage *img, iftImage *mask, iftImage *objsm)
{
    // Variables
    iftMImage *mimg, *obj_mimg;
    iftAdjRel *A;
    iftIGraph *igraph;

    // Assign
    if (iftIs3DImage(img)) A = iftSpheric(1.0);
    else A = iftCircular(1.0);

    if (iftIsColorImage(img)) mimg   = iftImageToMImage(img,LABNorm_CSPACE);
    else mimg   = iftImageToMImage(img,GRAY_CSPACE);

    // Extend MImage
    obj_mimg = iftExtendMImageByGrayObjSalMap(mimg, objsm);

    // Convert to IGraph
    igraph = iftImplicitIGraph( obj_mimg, mask, A);

    //Free
    iftDestroyMImage(&mimg);
    iftDestroyMImage(&obj_mimg);
    iftDestroyAdjRel(&A);

    return igraph;
}


void saliencyCalculationGPU(  iftMatrix *D,   iftMatrix *P,   iftFloatArray *W, iftFloatArray **saliency, iftDistanceTable *centroid_distances, float sigma_squared, double *border_region_ratio) {

    double alpha = 1.0, beta = 0.0;

    int colsD, rowsD, rowsP, colsP, rowsW, colsW, rowsDP, colsDP, rowsPDP, colsPDP, rowsS, colsS;

    iftFloatArray *s = *saliency;

    iftMatrix *PDP;


    colsD = D->ncols;
    rowsD = D->nrows;

    rowsP = P->nrows;
    colsP = P->ncols;

    rowsDP = rowsD;
    colsDP = rowsP;

    float *dD;

    float *dP;
    float *dDP;
    float *dPDP;
    float *dS;
    float *dW;

    /* Allocation */
    dD = iftAllocFloatArrayGPU(D->n);
    dP = iftAllocFloatArrayGPU(P->n);
    dDP = iftAllocFloatArrayGPU(rowsDP*colsDP);


    iftCopyFloatArrayToGPU(dD, D->val, D->n);
    iftCopyFloatArrayToGPU(dP, P->val, P->n);
    /* DP Calculation */
    cublasSgemm('n', 't', rowsD, rowsP, colsD, alpha, dD, rowsD, dP, rowsP, beta, dDP, rowsDP);


    cublasStatus_t status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
        iftError("Check Cuda documentation for error %d.\n", "cublasSgemmDP", status);
    }

    iftFreeGPU(dD);

    /* PyDPR */

    rowsP = P->nrows;
    colsP = P->ncols;


    rowsPDP = rowsP;
    colsPDP = colsDP;

    PDP = iftCreateMatrix(colsPDP, rowsPDP);

    dPDP = iftAllocFloatArrayGPU(rowsPDP*colsPDP);

    cublasSgemm('n', 'n', rowsP, colsDP, colsP, alpha, dP, rowsP, dDP, rowsDP, beta, dPDP, rowsPDP);


    iftFreeGPU(dP);
    iftFreeGPU(dDP);

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
        iftError("Check Cuda documentation for error %d.\n", "cublasSgemmPDP", status);
    }

    /* Apply centroid distances on CPU */

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
        iftError("Check Cuda documentation for error %d.\n", "iftCopyToGPU", status);
    }


    iftCopyFloatArrayFromGPU(PDP->val, dPDP, (int)PDP->n);

    for (int i = 0; i < PDP->nrows; i++) {
        for (int j = 0; j < PDP->ncols; j++) {
            if(border_region_ratio[j] < 1)
                iftMatrixElem(PDP, j, i) = iftMatrixElem(PDP, j, i) * exp(centroid_distances->distance_table[i][j]/sigma_squared) * 1;
            else
                iftMatrixElem(PDP, j, i) = iftMatrixElem(PDP, j, i) * 100;
        }
    }

    iftCopyFloatArrayToGPU(dPDP, PDP->val, (int)PDP->n);


    /* Back to GPU */

    rowsW = W->n;
    colsW = 1;

    rowsS = rowsPDP;
    colsS = colsW;//

    dS = iftAllocFloatArrayGPU(rowsS*colsS);

    dW = iftAllocFloatArrayGPU(W->n);

    iftFloatArray *W_new;

    W_new = iftCreateFloatArray(W->n);

    for(int r = 0; r < W->n; r++){
        W_new->val[r] = exp(-2.5*W->val[r]);
        //region_sizes_prob->val[r] = 1.0;
    }

    iftCopyFloatArrayToGPU(dW, W_new->val, (int)W_new->n);

    cublasSgemm('n', 't', rowsPDP, colsW, colsPDP, alpha, \
                                 dPDP, rowsPDP, dW, colsW, beta, dS, rowsS);

    iftDestroyFloatArray(&W_new);
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
        iftError("Check Cuda documentation for error %d.\n", "cublasSgemmS", status);
    }

    if ((*saliency) == NULL) {
        (*saliency) = iftCreateFloatArray(rowsS);
    }

    iftCopyFloatArrayFromGPU(s->val, dS, s->n);

    iftFreeGPU(dPDP);
    iftFreeGPU(dS);
    iftFreeGPU(dW);
}

iftImage *iftSamplingByOSMOX
        (iftImage* objsm, int num_seeds, float obj_perc)
{
    int obj_seeds, bkg_seeds;
    iftSet *seed_set, *obj_set, *bkg_set, *s;
    iftImage *seed_img;

    seed_set = NULL;
    obj_set = NULL;
    bkg_set = NULL;
    s = NULL;

    seed_img = iftCreateImage(objsm->xsize, objsm->ysize, objsm->zsize);

    // Establish the number of seeds
    obj_seeds = iftRound(num_seeds * obj_perc);
    bkg_seeds = num_seeds - obj_seeds;

#pragma omp sections
    {
#pragma omp section
        {
            obj_set = iftObjSalMapSamplByHighestValue(objsm, obj_seeds);
        }
#pragma omp section
        {
            iftImage *invsm;

            // Invert the object saliency map
            invsm = iftComplement(objsm);

            bkg_set = iftObjSalMapSamplByHighestValue(invsm, bkg_seeds);

            // Free
            iftDestroyImage(&invsm);
        }
    }

    // Merge sets
    seed_set = iftSetConcat(obj_set, bkg_set);

    // Convert into an image
    s = seed_set;
    while( s != NULL ) {
        seed_img->val[s->elem] = 1;
        s = s->next;
    }

    // Free
    iftDestroySet(&seed_set);
    iftDestroySet(&obj_set);
    iftDestroySet(&bkg_set);
    iftDestroySet(&s);

    return (seed_img);
}


iftSet *iftObjSalMapSamplByHighestValue
        (iftImage *objsm, int num_seeds )
{
    int patch_width, seed_count, max_val;
    float stdev;
    int *pixel_val;
    iftAdjRel *A;
    iftKernel *gaussian;
    iftGQueue *queue;
    iftSet *seed;

    max_val = iftMaximumValue(objsm) + 1;

    // Estimate the gaussian influence zone
    patch_width = iftRound(sqrtf(objsm->n/(float)num_seeds));
    stdev = patch_width/3.0;

    if(iftIs3DImage(objsm)) A = iftSpheric(patch_width);
    else A = iftCircular(patch_width);

    // Copy of the object saliency map values
    pixel_val = (int *)calloc(objsm->n, sizeof(int));

    // Gaussian penalty
    gaussian = iftCreateKernel(A);

    for(int i = 0; i < A->n; i++) {
        float dist;

        dist = A->dx[i]*A->dx[i] + A->dy[i]*A->dy[i] + A->dz[i]*A->dz[i];
        gaussian->weight[i] = exp(-dist/(2*stdev*stdev));
    }

    seed = NULL;
    queue = iftCreateGQueue(max_val, objsm->n, pixel_val );

    // Removing the highest values
    iftSetRemovalPolicy(queue, MAXVALUE);
    for( int p = 0; p < objsm->n; p++ ) {
        pixel_val[p] = objsm->val[p];
        iftInsertGQueue(&queue, p);
    }

    seed_count = 0;
    while( seed_count < num_seeds ) {
        int p;
        iftVoxel voxel_p;

        p = iftRemoveGQueue(queue);
        voxel_p = iftGetVoxelCoord(objsm, p);

        iftInsertSet(&seed, p);

        // Mark as removed
        pixel_val[p] = IFT_NIL;

        // For every adjacent voxel in the influence zone
        for(int i = 1; i < gaussian->A->n; i++) {
            iftVoxel voxel_q;

            voxel_q = iftGetAdjacentVoxel(gaussian->A, voxel_p, i);

            if(iftValidVoxel(objsm, voxel_q)) {
                int q;

                q = iftGetVoxelIndex(objsm, voxel_q);

                // If it was not removed (yet)
                if( pixel_val[q] != IFT_NIL ) {
                    iftRemoveGQueueElem(queue, q);

                    // Penalize
                    pixel_val[q] = iftRound((1 - gaussian->weight[i]) * pixel_val[q]);

                    iftInsertGQueue(&queue, q);
                }
            }
        }

        seed_count++;
    }

    // Free
    iftDestroyKernel(&gaussian);
    iftDestroyGQueue(&queue);
    iftDestroyAdjRel(&A);

    return (seed);
}

void iftWriteOverlay
        (iftImage* orig, iftImage *label, const char *filename)
{
    int normvalue;
    iftImage *overlay;
    iftAdjRel *A;
    iftColor RGB, YCbCr;

    normvalue = iftNormalizationValue(iftMaximumValue(orig));

    A = iftCircular(0.0);

    overlay = iftBorderImage(label,0);

    RGB = iftRGBColor(normvalue, 0, 0);
    YCbCr = iftRGBtoYCbCr(RGB, normvalue);

    iftDrawBorders(orig,overlay,A,YCbCr,A);

    iftWriteImageByExt(orig, filename);

    iftDestroyImage(&overlay);
    iftDestroyAdjRel(&A);
}

iftImage *convertToLab(iftImage *image){
    iftImage *lab_image;
    lab_image = iftCreateColorImage(image->xsize, image->ysize, image->zsize, iftImageDepth(image));
    int normalization_value = iftNormalizationValue(iftMaximumValue(image));

    iftColor origin_color, lab_color;
    iftFColor lab_color_f;
    int p;
    for(p = 0; p < image->n; p++){
        origin_color.val[0] = image->val[p];
        origin_color.val[1] = image->Cb[p];
        origin_color.val[2] = image->Cr[p];

        lab_color_f = iftYCbCrtoLab(origin_color,normalization_value);
        lab_color = iftLabtoQLab(lab_color_f, normalization_value);

        lab_image->val[p] = lab_color.val[0];
        lab_image->Cb[p] = lab_color.val[1];
        lab_image->Cr[p] = lab_color.val[2];
    }
    return lab_image;
}

iftImage* quantizedDatasetToImage(iftDataSet* Z,iftImage* img) {
    //  fprintf(stderr,"(%d x %d):\n",img->n,dataset->nsamples);
    if ( img->n != Z->nsamples) {
        iftError("incompatible dimensions ", "iftEdgesDataSetToImage");
    }

    iftImage* output = iftCreateColorImage(img->xsize,img->ysize,img->zsize, iftImageDepth(img));

    for(int sample = 0;sample < Z->nsamples; sample++) {
        output->val[sample] = (int) Z->sample[sample].feat[0];
        output->Cb[sample] = (ushort) Z->sample[sample].feat[1];
        output->Cr[sample] = (ushort) Z->sample[sample].feat[2];
    }


    return output;
}

iftDataSet *replicateMedoidToCluster(iftDataSet *Z){
    iftDataSet *Z_quantized;
    float **sum_feat, distance;
    iftIntArray *count_samples, *group_medoid;
    iftFloatArray *min_distance;

    /* Compute the Centroid */

    sum_feat = (float**) calloc(Z->ngroups,sizeof(float*));
    for(int i = 0;i<Z->ngroups;i++) {
        sum_feat[i] = (float*) calloc(Z->nfeats, sizeof(float));
    }

    count_samples = iftCreateIntArray(Z->ngroups);

    for(int sample = 0; sample < Z->nsamples; sample++) {
        for (int feat = 0; feat < Z->nfeats; feat++) {
            sum_feat[Z->sample[sample].group-1][feat] += Z->sample[sample].feat[feat];
        }
        count_samples->val[Z->sample[sample].group-1]++;
    }

    for(int sample = 0; sample < Z->nsamples; sample++) {
        for (int feat = 0; feat < Z->nfeats; feat++)
            sum_feat[Z->sample[sample].group-1][feat]/= (float) count_samples->val[Z->sample[sample].group-1];

    }


    /* Compute the medoid */

    group_medoid = iftCreateIntArray(Z->ngroups);
    min_distance = iftCreateFloatArray(Z->ngroups);

    for(int group = 0; group < Z->ngroups; group++)
        min_distance->val[group] = IFT_INFINITY_FLT;

    for(int sample = 0; sample < Z->nsamples; sample++) {
        distance = iftFeatDistance(Z->sample[sample].feat, sum_feat[Z->sample[sample].group-1], Z->nfeats);

        if(distance < min_distance->val[Z->sample[sample].group-1]){
            min_distance->val[Z->sample[sample].group-1] = distance;
            group_medoid->val[Z->sample[sample].group-1] = sample;
        }
    }


    Z_quantized = iftCopyDataSet(Z, true);

    for(int sample = 0; sample < Z->nsamples; sample++) {
        for (int feat = 0; feat < Z->nfeats; feat++)
            Z_quantized->sample[sample].feat[feat] = Z->sample[group_medoid->val[Z->sample[sample].group-1]].feat[feat];
    }

    for(int i = 0;i<Z->ngroups;i++) {
        free(sum_feat[i]);
    }

    free(sum_feat);

    iftDestroyIntArray(&group_medoid);
    iftDestroyIntArray(&count_samples);
    iftDestroyFloatArray(&min_distance);

    return Z_quantized;

}
