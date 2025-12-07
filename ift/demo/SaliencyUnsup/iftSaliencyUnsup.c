#include "ift.h"

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

iftImage *imageToGlobalSaliencyImage(iftMImage *img, iftDataSet *Z, int cluster_number);

iftImage *imageToRegionSaliencyImage(iftMImage *img, iftDataSet *Z, iftImage *label_img, int cluster_number);

iftDataSet *computeQuantizedDataset(iftMImage *img, int nsplits, float train_perc, int max_number_of_clusters, int *cluster_number, int kmax);

iftImage *iftOGRIDSampling(iftImage *objsm, iftImage *mask, int k, float thresh );

iftMImage *iftExtendMImageByGrayObjSalMap(iftMImage *mimg, iftImage* objsm);

iftIGraph *iftInitOISFIGraph(iftImage *img, iftImage *mask, iftImage *objsm);

iftSet *iftMultiLabelCenterSamplingOnMaskByArea(  iftImage *label, iftImage *mask, float thresh);

iftImage *iftCENTERSampling(iftImage *objsm, iftImage *mask, int k, float thresh );

iftDataSet *computeQuantizedDatasetKMeans(iftMImage *img, int max_number_of_clusters, int *cluster_number,  int maxIterations, float minImprovement);

/**********************************************************************
																MAIN
**********************************************************************/
int main(int argc, const char *argv[])
{
    iftImage  *img, *saliency_img, *label_img = NULL, *mask, *seeds;
    iftDataSet *Z = NULL;
    iftIGraph *igraph, *out_igraph;
    iftMImage *mimg, *saliency_mimg, *gb_mimg;
    int  maxIter = 30, max_number_of_clusters, number_superpixels, cluster_number, iterations; //ARG
    float thr = 0.5, alpha=0.5, beta=12, gamma=5, minImprovement=0.01;

    int iter=10;

    /*
     * image is the input image
     * label_img is a superpixel map (optional based on whether you want to use region or global based)
     * input_train_perc  is the percentage of pixels used for training of OPF`s cluster algorithm
     * output-basename is the file to output the saliency map
     * maximum number of colors after the quantization by OPF clustering
     */

    if (argc != 6){
        iftError("Usage: Region Saliency <image.jpg> <number of iterations> <output-basename> <maximum number of colors> <number of superpixels> ", "main");
    }

    img = iftReadImageByExt(argv[1]);
    iterations = atoi(argv[2]);
    mimg = iftImageToMImage(img, YCbCr_CSPACE);
    gb_mimg = iftImageToMImage(img, RGBNorm_CSPACE);
    max_number_of_clusters = atoi(argv[4]);
    number_superpixels = atoi(argv[5]);
    mask = iftSelectImageDomain(img->xsize, img->ysize, img->zsize);

    printf("Calculating Global Saliency\n");


    printf("Quantizing\n");

    //Create Dataset for quantization
    Z = computeQuantizedDatasetKMeans(mimg, max_number_of_clusters, &cluster_number, maxIter, minImprovement);

//    saliency_img = imageToGlobalSaliencyImage(mimg, Z, cluster_number);
//
//    iftWriteImageByExt(saliency_img, "global-result.png");

    saliency_img = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);

    iftWriteImageByExt(saliency_img, "saliency.png");
    for(int i = 0; i<iterations; i++) {
        printf("Iteration: %d\n", i);
        printf("Creating Superpixels");
        seeds = iftGrayObjMapCentrSamplOnMaskByArea(saliency_img, mask, number_superpixels, thr, 0.5);
        /* Uncomment below line to change OISF Sampling to Grid */
        //seeds = iftGrayObjMapGridSamplOnMaskByArea(saliency_img, mask, number_superpixels, thr, 0.9);
        printf("Found %d seeds!\n", iftNumberOfElements(seeds));

        igraph = iftInitOISFIGraph(img, mask, saliency_img);

        printf("Segmenting superpixels... \n");
        out_igraph = iftIGraphOISF(igraph, seeds, alpha, beta, gamma, iter);

        int max = iftMaximumValue(saliency_img);

        label_img = iftIGraphLabel(out_igraph);
        iftWriteImageByExt(iftThreshold(saliency_img, max*thr, max, 255), "threshold.png");
        iftWriteImageByExt(label_img, "saida.png");
        iftDestroyImage(&saliency_img);

        iftDestroyIGraph(&out_igraph);
        iftDestroyIGraph(&igraph);
        iftDestroyImage(&seeds);

        /* */
        iftSList *list = iftSplitString(argv[3],".");
        iftSNode *L    = list->head;
        char      filename[50];
        sprintf(filename,"output/%s-%d.png",L->elem, i);
        saliency_img = imageToRegionSaliencyImage(mimg, Z, label_img, cluster_number);

        double beta = iftGraphCutBeta(gb_mimg);

        printf("Segmenting with Grabcut\n");
        max = iftMaximumValue(saliency_img);

        iftImage *norm = iftNormalize(iftThreshold(saliency_img, 70, max, 255), 0, 1);
        iftImage *reg = iftMaybeForeground(norm);
        iftImage *segm = iftGrabCut(gb_mimg, reg, beta, iterations);

        iftImage *out = iftMask(img, segm);
        iftWriteImageByExt(out, filename);
        iftDestroyImage(&out);
        iftDestroyImage(&norm);
        iftDestroyImage(&reg);
        printf("Done!\n");

        //iftWriteImageByExt(saliency_img, filename);
        iftDestroyImage(&label_img);
    }

    // Free
    iftDestroyImage(&img);
    iftDestroyMImage(&gb_mimg);
    iftDestroyImage(&saliency_img);
    //printf("OI\n");


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
iftDataSet *computeQuantizedDataset(iftMImage *img, int nsplits, float train_perc, int max_number_of_clusters, int *cluster_number, int kmax){
    iftDataSet *Z;
    iftSampler *sampler=NULL;
    iftKnnGraph *knn_graph;

    Z = iftMImageToDataSet(img, NULL);

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
iftDataSet *computeQuantizedDatasetKMeans(iftMImage *img, int max_number_of_clusters, int *cluster_number,  int maxIterations, float minImprovement){
    iftDataSet *Z;

    Z = iftMImageToDataSet(img, NULL);

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



iftFloatArray *saliencySmoothing(iftDataSet *Z, iftImage *label_img, iftFloatArray *saliency, iftColorTable *colors, int cluster_number, iftMatrix *region_color_prob, iftMatrix *region_has_color, double *border_region_ratio){
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
        Z_color_saliency_new->sample[c].feat[0] = (float)sum/((m-1)*t);
    }
    iftDestroyDataSet(&Z_color_saliency);

    for(int ri = 0; ri < (int)saliency->n-1; ri++){
        double new_saliency = 0;
        float color_counter = 0;
        for(int ci = 0; ci < cluster_number; ci++) {
            if (!iftMatrixElem(region_has_color, ci, ri))
                continue;
            new_saliency+= iftMatrixElem(region_color_prob, ci, ri) * Z_color_saliency_new->sample[ci].feat[0];
            color_counter+=1*iftMatrixElem(region_color_prob, ci, ri);
        }
        saliency_new->val[ri] = (new_saliency*border_region_ratio[ri])/color_counter;

    }
    iftDestroyFloatArray(&saliency);
    iftDestroyDataSet(&Z_color_saliency_new);
    iftDestroyDataSet(&Z_color_smoothing);
    return saliency_new;
}


/*
 * Return an saliency map given an image based on global contrast
 */
iftImage *imageToRegionSaliencyImage(iftMImage *img, iftDataSet *Z, iftImage *label_img, int cluster_number){
    iftDataSet *Z_centroid, *Z_region_map = NULL, *Z_region = NULL, *Z_center_voxel;
    int  c =0, border_radius = 20; //m is used for smoothing
    double  max_saliency = 0, max_region_size = 0, min_saliency = IFT_INFINITY_DBL, *border_region_ratio;
    float sigma_squared = -0.4; //spatial weight control
    iftColorTable *colors;
    iftImage *saliency_image;
    iftMatrix *region_color_prob, *region_has_color;
    iftIntArray *region_sizes;
    iftFloatArray *region_sizes_prob, *saliency;
    iftDistanceTable *centroid_distances, *region_to_center;
    iftVoxelArray *centroids;


    saliency_image = iftCreateImage(label_img->xsize,label_img->ysize,label_img->zsize);

    colors = iftCreateColorTable(cluster_number);

    region_sizes = iftCountLabelSpels(label_img);
    region_sizes_prob = iftCreateFloatArray(region_sizes->n);

    region_color_prob = iftCreateMatrix(cluster_number, ((int)region_sizes->n-1));
    region_has_color = iftCreateMatrix(cluster_number, ((int)region_sizes->n-1));



    printf("Compute saliency\n");

    //Calculate the probabilities of each color to appear on each region
    for(int p = 0; p < label_img->n; p++){
        c = Z->sample[p].group-1;
        iftMatrixElem(region_color_prob, c, label_img->val[p]-1)++;
        iftMatrixElem(region_has_color, c, label_img->val[p]-1) = 1;

        colors->color[c].val[0] = Z->sample[p].feat[0];
        colors->color[c].val[1] = Z->sample[p].feat[1];
        colors->color[c].val[2] = Z->sample[p].feat[2];

    }

    //Normalize the probabilities from 0-1 based on region size
    for(int reg = 0; reg < (int)region_sizes->n-1; reg++){
        for(int color = 0; color < cluster_number; color++) {
            iftMatrixElem(region_color_prob, color, reg) = (float)iftMatrixElem(region_color_prob, color, reg) / region_sizes->val[reg+1];
        }
    }

    for(int r = 0; r <= region_sizes->n; r++){
        if(region_sizes->val[r] > max_region_size)
            max_region_size = region_sizes->val[r];
    }

    for(int r = 0; r < region_sizes->n; r++){
        region_sizes_prob->val[r] = (float)region_sizes->val[r]/max_region_size;
    }

    printf("Computing Distance of Centroids\n");
    /* Calculate the distances of the centroids and to center of image*/
    Z_centroid = iftCreateDataSet((int)region_sizes->n-1, 2);
    Z_region_map = iftCreateDataSet(img->n, 2);
    Z_region_map->ngroups = ((int)region_sizes->n-1);
    centroids = iftGeometricCentersFromLabelImage(label_img);

#pragma omp parallel for
    for(int reg = 1; reg < (int)region_sizes->n; reg++){
        Z_centroid->sample[reg-1].feat[0] = (float)centroids->val[reg].x / img->xsize-1;
        Z_centroid->sample[reg-1].feat[1] = (float)centroids->val[reg].y / img->ysize-1;
#pragma omp parallel for
        for(int p = 0; p < img->n; p++){
            iftVoxel u;
            u = iftMGetVoxelCoord(img, p);
            Z_region_map->sample[p].feat[0] = (float)u.x / (img->xsize-1);
            Z_region_map->sample[p].feat[1] = (float)u.y / (img->ysize-1);
            Z_region_map->sample[p].group = label_img->val[p];
        }
    }

    Z_center_voxel = iftCreateDataSet(1, 2);

    Z_center_voxel->sample[0].feat[0] = (float)(img->xsize/2)-1;
    Z_center_voxel->sample[0].feat[1] = (float)(img->ysize/2)-1;

    region_to_center = iftCompEuclDistanceTable(Z_region_map, Z_center_voxel);

    /* -------------------------------- Find border regions ----------------------------------*/
    Z_region_map = iftCreateDataSet(img->n, 2);
    Z_region_map->ngroups = ((int)region_sizes->n-1);
    border_region_ratio = iftAllocDoubleArray((int)region_sizes->n-1);

#pragma omp parallel for
    for(int reg = 1; reg < (int)region_sizes->n; reg++){
#pragma omp parallel for
        for(int p = 0; p < img->n; p++){
            iftVoxel u;
            u = iftMGetVoxelCoord(img, p);
            Z_region_map->sample[p].feat[0] = (float)u.x;
            Z_region_map->sample[p].feat[1] = (float)u.y;
            Z_region_map->sample[p].group = label_img->val[p];
        }
    }

    // Average of distance to the center of image
    for(int reg = 1; reg < (int)region_sizes->n; reg++){
        int border_counter = 0;
        Z_region = iftExtractGroup(Z_region_map, reg);
#pragma omp parallel for
        for(int d = 0; d < Z_region->nsamples; d++){
            //Count how many pixels are within the border (16 pixels adjacency)
            if(Z_region->sample[d].feat[0] >= (img->xsize - border_radius) ||  Z_region->sample[d].feat[1] >= (img->ysize - border_radius) || Z_region->sample[d].feat[0] <= border_radius || Z_region->sample[d].feat[1] <= border_radius)
                border_counter++;
        }
        if(border_counter > (Z_region->nsamples / 80))
            border_region_ratio[reg-1] = 0.2;
        else
            border_region_ratio[reg-1] = 1;

        //border_region_ratio[reg-1] = 1;
        iftDestroyDataSet(&Z_region);
    }

    iftDestroyDataSet(&Z_region_map);

    centroid_distances = iftCompEuclDistanceTable(Z_centroid, Z_centroid);

    iftDestroyDataSet(&Z_centroid);

    saliency = iftCreateFloatArray(region_sizes->n-1);
    // The variables are named according to the paper
#pragma omp parallel for
    for(int ri = 0; ri < (int)region_sizes->n-1; ri++){
        double sum = 0;
        double sum_rk = 0;
        double dr = 0;
#pragma omp parallel for
        for(int rj = 0; rj < (int)region_sizes->n-1; rj++){
#pragma omp parallel for
            for(int ci = 0; ci < cluster_number; ci++) {
                if (!iftMatrixElem(region_has_color, ci, ri))
                    continue;
#pragma omp parallel for
                for (int cj = 0; cj < cluster_number; cj++) {
                    if (!iftMatrixElem(region_has_color, cj, rj))
                        continue;
                    sum+= calculateColorDistance(colors, ci, cj) * iftMatrixElem(region_color_prob, cj, rj) * iftMatrixElem(region_color_prob, ci, ri);
                }
                dr = dr + sum;
                sum = 0;
            }
            sum_rk = dr;// * exp(centroid_distances->distance_table[ri][rj]/sigma_squared); //* (float)  ((region_sizes_prob->val[ri+1])); //w(r_i) on the paper
            dr = 0;
        }
        saliency->val[ri] = sum_rk * region_to_center->distance_table[ri][0] * border_region_ratio[ri]; // region_to_center_distances is the paper's w_s(r_k)
        sum_rk = 0;
    }

    /* Smoothing by color*/
    //saliency = saliencySmoothing(Z, label_img, saliency, colors, cluster_number, region_color_prob, region_has_color, border_region_ratio);


    iftDestroyVoxelArray(&centroids);
    iftDestroyMatrix(&region_has_color);
    iftDestroyMatrix(&region_color_prob);
    iftDestroyDistanceTable(&centroid_distances);
    iftDestroyFloatArray(&region_sizes_prob);
    iftDestroyColorTable(&colors);

    /* Normalize Saliency to 0-255 */
    for(int rk = 0; rk < (int)region_sizes->n-1; rk++){ //Get Max
        if(saliency->val[rk] > max_saliency)
            max_saliency = saliency->val[rk];
        if(saliency->val[rk] < min_saliency)
            min_saliency = saliency->val[rk];
    }

    for(int rk = 0; rk < (int)region_sizes->n-1; rk++) //Normalize
        saliency->val[rk] = 255*(saliency->val[rk]-min_saliency)/(max_saliency-min_saliency);


    for(int p = 0; p < saliency_image->n; p++)
        saliency_image->val[p] = saliency->val[label_img->val[p]-1];

    return saliency_image;

}

/*
 * Return an saliency map given an image based on region (superpixel) contrast
 */
iftImage *imageToGlobalSaliencyImage(iftMImage *img, iftDataSet *Z, int cluster_number){
    int c =0;
    double *probabilities, *saliency, distance = 0, max_saliency = 0, sum = 0, min_saliency = IFT_INFINITY_DBL;
    iftColorTable *colors;
    iftImage *saliency_image;

    saliency_image = iftCreateImage(img->xsize,img->ysize,img->zsize);

    probabilities = (double*) calloc(cluster_number, sizeof(double));
    saliency = (double*) calloc(cluster_number, sizeof(double));
    colors = iftCreateColorTable(cluster_number);

    printf("Computing Saliency\n");



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

    /* ------ Calculate Global Saliency ------- */
    for(int cl = 0; cl < cluster_number; cl++){
        for(int cj = 0; cj < cluster_number; cj++){
            for(int i =0; i < Z->nfeats; i++){
                sum+=pow(colors->color[cl].val[i] - colors->color[cj].val[i],2);
            }
            distance = distance + sqrt(sum) * probabilities[cj];
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
        saliency_image->val[p] = saliency[Z->sample[p].group-1];

    // Free
    printf("Max %d\n", iftMaximumValue(saliency_image));
    iftDestroyColorTable(&colors);
    //return iftDataSetClustersToQuantizedImage(Z, true);
    return saliency_image;

}


/* OISF */

iftImage *iftOGRIDSampling
        (iftImage *objsm, iftImage *mask, int k, float thresh )
{
    int seedsForBkg, seedsForObj, maxVal;
    iftImage *bin, *invBin, *seeds;
    iftSet *bkgSeeds, *objSeeds, *S;

    objSeeds = NULL;
    bkgSeeds = NULL;
    S = NULL;

    seedsForObj = k;
    maxVal = iftMaximumValue(objsm);

    seeds = iftCreateImage(objsm->xsize, objsm->ysize, objsm->zsize);
    bin = iftThreshold(objsm, thresh*maxVal, maxVal, 1);
    invBin = iftComplement(bin);

    objSeeds = iftMultiLabelGridSamplingOnMaskByArea(bin, mask, seedsForObj);

    seedsForBkg = iftMax(k - iftSetSize(objSeeds), 1);
    bkgSeeds = iftMultiLabelGridSamplingOnMaskByArea(invBin, mask, seedsForBkg);

    S = objSeeds;
    while (S != NULL) { seeds->val[S->elem] = 1; S = S->next;}
    iftDestroySet(&S);

    S = bkgSeeds;
    while (S != NULL) { seeds->val[S->elem] = 1; S = S->next; }

    // Free
    iftDestroyImage(&bin);
    iftDestroyImage(&invBin);
    iftDestroySet(&objSeeds);
    iftDestroySet(&bkgSeeds);
    iftDestroySet(&S);

    return seeds;
}

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


iftImage *iftCENTERSampling
        (iftImage *objsm, iftImage *mask, int k, float thresh )
{
    int seedsForBkg, maxVal;
    iftImage *bin, *invBin, *seeds, *ero, *ero2;
    iftSet *bkgSeeds, *objSeeds, *S;

    objSeeds = NULL;
    bkgSeeds = NULL;
    S = NULL;

    seedsForBkg = k;
    maxVal = iftMaximumValue(objsm);

    seeds = iftCreateImage(objsm->xsize, objsm->ysize, objsm->zsize);
    bin = iftThreshold(objsm, thresh*maxVal, maxVal, 1);
    invBin = iftComplement(bin);

    ero = iftErodeLabelImage(invBin, 20.0);
    ero2 = iftErodeLabelImage(bin, 5.0);

    bkgSeeds = iftMultiLabelGridSamplingOnMaskByArea(ero, mask, seedsForBkg);
    objSeeds = iftMultiLabelCenterSamplingOnMaskByArea(ero2, mask, 0.5);

    S = objSeeds;
    while (S != NULL) { seeds->val[S->elem] = 1; S = S->next;}
    iftDestroySet(&S);

    S = bkgSeeds;
    while (S != NULL) { seeds->val[S->elem] = 1; S = S->next; }

    // Free
    iftDestroyImage(&bin);
    iftDestroyImage(&ero);
    iftDestroyImage(&ero2);
    iftDestroyImage(&invBin);
    iftDestroySet(&objSeeds);
    iftDestroySet(&bkgSeeds);
    iftDestroySet(&S);

    return seeds;
}

iftSet *iftMultiLabelCenterSamplingOnMaskByArea
        (  iftImage *label, iftImage *mask, float thresh)
{
    int totalArea, numObj;
    iftImage *newLabels;
    iftAdjRel *A;
    iftSet *seeds;
    iftIntArray *sampled;

    seeds = NULL;

    if( iftIs3DImage(label) ) A = iftSpheric(1.74);
    else A = iftCircular(1.45);

    newLabels = iftFastLabelComp(label, A);
    numObj = iftMaximumValue(newLabels);
    sampled = iftCreateIntArray(numObj);

    totalArea = 0;
    for( int p = 0; p < newLabels->n; p++ )
    {
        if(newLabels->val[p] > 0) totalArea++;
    }

    for( int i = 1; i <= numObj; i++ )
    {
        iftImage* objMask;
        int objArea;

        objArea = 0;

        objMask = iftThreshold(newLabels, i, i, 1);

        for( int p = 0; p < objMask->n; p++ )
        {
            if(objMask->val[p] > 0) objArea++;
        }

        float objPerc = objArea / (float)totalArea;

        if( objPerc >= thresh ) {
            iftVoxelArray *centers;

            centers = iftGeometricCentersFromLabelImage(objMask);

            sampled->val[i-1] = iftGetVoxelIndex(objMask, centers->val[1]);

            // Free
            iftDestroyVoxelArray(&centers);
        }
        else sampled->val[i-1] = -1;

        // Free
        iftDestroyImage(&objMask);
    }

    for( int i = 0; i < numObj; i++ ) {
        if( sampled->val[i] != -1 ) {
            if( mask->val[sampled->val[i]] != 0 ) iftInsertSet(&seeds, sampled->val[i]);
        }
    }

    // Free
    iftDestroyImage(&newLabels);
    iftDestroyAdjRel(&A);
    iftDestroyIntArray(&sampled);

    return seeds;

}