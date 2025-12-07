%module pygc
%{
  #include "graph.h"
  
  #define class class_
    extern "C" {
		#include "ift.h"
	}
  #undef class
%}

typedef struct{
  int    *val;
  ushort *Cb,*Cr;
  int xsize,ysize,zsize;
  float dx,dy,dz;
  int *tby, *tbz;        // speed-up voxel access tables
  int maxval, minval, n; // minimum and maximum values, and number of voxels
} iftImage;

%extend iftImage{
	~iftImage(){
		iftImage *ptr = ($self);
		iftDestroyImage(&ptr);
	}
}

%newobject graph_cut_superpixel_segmentation;
iftImage* graph_cut_superpixel_segmentation(iftRegionGraph* region_graph, iftImage* label_image, iftLabeledSet *seed, int beta);

%newobject graph_cut_pixel_grad_segmentation;
iftImage* graph_cut_pixel_grad_segmentation(iftImage* basins, iftLabeledSet *seed, int beta);

%newobject graph_cut_pixel_segmentation;
iftImage* graph_cut_pixel_segmentation(iftDataSet* dataset, iftAdjRel* adj, iftLabeledSet *seed, int beta);

%{
	iftImage* graph_cut_superpixel_segmentation(iftRegionGraph* region_graph, iftImage* label_image, iftLabeledSet *seed, int beta){
		typedef Graph<double,double,double> GraphType;
		GraphType *graph = new GraphType(region_graph->nnodes, region_graph->nnodes);
		iftDataSet *dataset = region_graph->dataset;

		float max_dist = -INFINITY_FLT;
		float min_dist = INFINITY_FLT;
		int r;
		for(r = 0; r < region_graph->nnodes; r++){
			graph->add_node();

			iftSet *adj = region_graph->node[r].adjacent;
			while(adj){
				int v = adj->elem;
				float dist = dataset->iftArcWeight(dataset->sample[r].feat,dataset->sample[v].feat, dataset->alpha,dataset->nfeats);
				if(dist > max_dist)
					max_dist = dist;
				if(dist < min_dist)
					min_dist = dist;
				adj = adj->next;
			}
		}
		
		for(r = 0; r < region_graph->nnodes; r++){
			iftSet *adj = region_graph->node[r].adjacent;
			while(adj){
				int v = adj->elem;			
				if(r < v){
					float dist = dataset->iftArcWeight(dataset->sample[r].feat,dataset->sample[v].feat, dataset->alpha,dataset->nfeats);
					dist = (dist - min_dist)/(max_dist - min_dist);
					float similarity = exp(-dist/0.5);

					double edge_weight = pow(similarity, beta);
					graph->add_edge(r,v, edge_weight, edge_weight);
				}
			
				adj = adj->next;
			}
		}
		
		timer *t1 = iftTic();

		iftBMap *labeled = iftCreateBMap(region_graph->nnodes);
		
		iftLabeledSet *s = seed;
		while(s){
			int p = s->elem;
			
			r = label_image->val[p] - 1;
			
			if(!iftBMapValue(labeled, r)){
				int label = s->label;
				
				if(label == 1)
					graph->add_tweights(r, INFINITY_FLT, 0);
				else
					graph->add_tweights(r, 0, INFINITY_FLT);
					
				iftBMapSet1(labeled,r);
			}
		
			s = s->next;
		}
		
		graph->maxflow();
		
		iftImage *segmentation = iftCreateImage(label_image->xsize, label_image->ysize, label_image->zsize);
		
		int p;
		for(p = 0; p < segmentation->n; p++){
			r = label_image->val[p] - 1;
			
			if (graph->what_segment(r) == GraphType::SOURCE)
				segmentation->val[p] = 1;
		}
		
		delete graph;
		iftDestroyBMap(&labeled);
		
		timer *t2 = iftToc();

		printf("gc super segmentation time: %fs\n",iftCompTime(t1,t2)/1000);

		return segmentation;
	}
	
	iftImage* graph_cut_pixel_grad_segmentation(iftImage* basins, iftLabeledSet *seed, int beta){
		typedef Graph<double,double,double> GraphType;
		GraphType *graph = new GraphType(basins->n, basins->n);

		iftImage *normalized_basins = iftNormalize(basins, 0, 255);

		iftAdjRel *adj = iftCircular(1.0);
		
		int p;
		for(p = 0; p < basins->n; p++)
			graph->add_node();
			
		for(p = 0; p < basins->n; p++){
			iftVoxel u = iftGetVoxelCoord(basins,p);
		
			int i;
			for(i = 1; i < adj->n; i++){
				iftVoxel v = iftGetAdjacentVoxel(adj,u,i);
				
				if(iftValidVoxel(basins, v)){
					int q = iftGetVoxelIndex(basins,v);
					
					if(p < q){
						double edge_weight = pow(1.0 - (normalized_basins->val[p] + normalized_basins->val[q])/(2*255.), beta);
						graph->add_edge(p,q, edge_weight, edge_weight);
					} 
				}
			}
		}
		
		timer *t1 = iftTic();

		iftLabeledSet *s = seed;
		while(s){
			p = s->elem;
			int label = s->label;
			
			if(label == 1)
				graph->add_tweights(p, INFINITY_FLT, 0);
			else
				graph->add_tweights(p, 0, INFINITY_FLT);
		
			s = s->next;
		}
		
		graph->maxflow();
		
		iftImage *segmentation = iftCreateImage(basins->xsize, basins->ysize, basins->zsize);
		
		for(p = 0; p < segmentation->n; p++){
			if (graph->what_segment(p) == GraphType::SOURCE)
				segmentation->val[p] = 1;
		}
		
		iftDestroyImage(&normalized_basins);
		delete graph;
		iftDestroyAdjRel(&adj);
		
		timer *t2 = iftToc();

		printf("gc pixel grad segmentation time: %fs\n",iftCompTime(t1,t2)/1000);

		return segmentation;
	}
	
	iftImage* graph_cut_pixel_segmentation(iftDataSet* dataset, iftAdjRel* adj, iftLabeledSet *seed, int beta){
		iftImage *image = (iftImage*)dataset->ref_data;
		iftSetDistanceFunction(dataset, 5);
	
		typedef Graph<double,double,double> GraphType;
		GraphType *graph = new GraphType(image->n, 2*image->n);

		float max_dist = -INFINITY_FLT;
		float min_dist = INFINITY_FLT;
		int p;
		for(p = 0; p < image->n; p++){
			graph->add_node();	
			iftVoxel u = iftGetVoxelCoord(image, p);

			int i;
			for(i = 1; i < adj->n; i++){
				iftVoxel v = iftGetAdjacentVoxel(adj, u, i);
				
				if(iftValidVoxel(image,v)){
					int q = iftGetVoxelIndex(image,v);
					
					float dist = dataset->iftArcWeight(dataset->sample[p].feat,dataset->sample[q].feat, dataset->alpha,dataset->nfeats);
					if(dist > max_dist)
						max_dist = dist;					
					if(dist < min_dist)
						min_dist = dist;
				}
			
			}
		}

		for(p = 0; p < image->n; p++){
			iftVoxel u = iftGetVoxelCoord(image,p);
		
			int i;
			for(i = 1; i < adj->n; i++){
				iftVoxel v = iftGetAdjacentVoxel(adj,u,i);
				
				if(iftValidVoxel(image, v)){
					int q = iftGetVoxelIndex(image,v);
					
					if(p < q){
						float dist = dataset->iftArcWeight(dataset->sample[p].feat,dataset->sample[q].feat, dataset->alpha,dataset->nfeats);
						dist = (dist - min_dist)/(max_dist - min_dist);
						float similarity = exp(-dist/0.5);

						double edge_weight = pow(similarity, beta);
						graph->add_edge(p,q, edge_weight, edge_weight);
					} 
				}
			}
		}
		
		timer *t1 = iftTic();

		iftLabeledSet *s = seed;
		while(s){
			p = s->elem;
			int label = s->label;
			
			if(label == 1)
				graph->add_tweights(p, INFINITY_FLT, 0);
			else
				graph->add_tweights(p, 0, INFINITY_FLT);
		
			s = s->next;
		}
		
		graph->maxflow();
		
		iftImage *segmentation = iftCreateImage(image->xsize, image->ysize, image->zsize);
		
		for(p = 0; p < segmentation->n; p++){
			if (graph->what_segment(p) == GraphType::SOURCE)
				segmentation->val[p] = 1;
		}
		
		delete graph;
		
		timer *t2 = iftToc();

		printf("gc super segmentation time: %fs\n",iftCompTime(t1,t2)/1000);

		return segmentation;
	}
	
%}
