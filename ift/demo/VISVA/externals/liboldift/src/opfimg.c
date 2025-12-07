#include "opfimg.h"
#include "sort.h"
#include "common.h"
#include "spectrum.h"
#include "radiometric3.h"
#include "mathematics3.h"
#include "segmentation.h"

///int MAXDENS=1000;   // Maximum value for pdf computation

//variables used for precomputed distances
///bool   PrecomputedDistance;
///float  **DistanceValue;

//distfun DistFun = opf_EuclDist;

/*----------- Constructor and destructor ------------------------*/

// Allocate nodes without feature space and adjacency list
//
///Subgraph *CreateSubgraph(int nnodes)
//{
//    Subgraph *sg=(Subgraph *)calloc(1,sizeof(Subgraph));
//    int i;
//
//    sg->nnodes = nnodes;
//    sg->node   = (SNode *)calloc(nnodes,sizeof(SNode));
//    sg->ordered_list_of_nodes = (int *) calloc(nnodes,sizeof(int));
//
//    if (sg->node == NULL)
//    {
//        Error("Cannot allocate nodes","CreateSubgraph");
//    }
//
//    for (i=0; i < sg->nnodes; i++)
//    {
//        sg->node[i].feat   = NULL;
//        sg->node[i].adj    = NULL;
//    }
//
//    return(sg);
//}
//
//// Deallocate memory for subgraph
//
///void DestroySubgraph(Subgraph **sg)
//{
//    int i;
//
//    if ((*sg)!=NULL)
//    {
//        for (i=0; i < (*sg)->nnodes; i++)
//        {
//            if ((*sg)->node[i].feat != NULL)
//                free((*sg)->node[i].feat);
//            if ((*sg)->node[i].adj != NULL)
//                DestroySet(&(*sg)->node[i].adj);
//        }
//        free((*sg)->node);
//        free((*sg)->ordered_list_of_nodes);
//        free((*sg));
//        *sg = NULL;
//    }
//}

// Allocate nodes of labelgraph

Labelgraph *CreateLabelgraph(int nnodes)
{
    Labelgraph *lg = ( Labelgraph * )calloc( 1, sizeof( Labelgraph ) );
    int i;

    lg->nnodes = nnodes;
    lg->node   = ( LNode * ) calloc( nnodes, sizeof( LNode ) );
    if ( lg->node == NULL )
    {
        Error("Cannot allocate nodes","CreateLabelgraph");
    }

    for ( i = 0; i < lg->nnodes; i++ )
    {
        lg->node[i].adj = NULL;
        lg->node[i].voxels = NULL;
        lg->node[i].histogram = NULL;
    }

    return( lg );
}

// Deallocate memory for labelgraph

void DestroyLabelgraph( Labelgraph **lg )
{
    int i;

    if ( ( *lg ) != NULL )
    {
        for ( i = 0; i < ( *lg )->nnodes; i++ )
        {
            if ( ( *lg )->node[ i ].adj != NULL ) DestroySet( &( *lg )->node[ i ].adj );
            if ( ( *lg )->node[ i ].voxels != NULL ) DestroySet( &( *lg )->node[ i ].voxels );
            if ( ( *lg )->node[ i ].histogram != NULL ) DestroyCurve( &( *lg )->node[ i ].histogram );
        }
        free( ( *lg )->node );
        free( *lg );
        *lg = NULL;
    }
}

// Copy a labelgraph

Labelgraph *CopyLabelgraph( Labelgraph *in )
{
    Labelgraph *lg = ( Labelgraph * )calloc( 1, sizeof( Labelgraph ) );
    int i;

    lg->nnodes = in->nnodes;
    lg->max_mean = in->max_mean;
    lg->node   = ( LNode * ) calloc( in->nnodes, sizeof( LNode ) );
    if ( lg->node == NULL )
    {
        Error("Cannot allocate nodes","CreateLabelgraph");
    }

    for ( i = 0; i < lg->nnodes; i++ )
    {
        lg->node[ i ].adj = NULL;
        lg->node[ i ].voxels = NULL;
        lg->node[ i ].histogram = NULL;
        lg->node[ i ].size = in->node[ i ].size;
        lg->node[ i ].label = in->node[ i ].label;
        lg->node[ i ].mean = in->node[ i ].mean;
        lg->node[ i ].rep = in->node[ i ].rep;
        if ( in->node[ i ].histogram != NULL ) lg->node[ i ].histogram = CopyCurve( in->node[ i ].histogram );
        if ( in->node[ i ].adj != NULL ) lg->node[ i ].adj = CloneSet( in->node[ i ].adj );
        if ( in->node[ i ].voxels != NULL ) lg->node[ i ].voxels = CloneSet( in->node[ i ].voxels );
    }

    return( lg );
}

// Set adjacent regions in labelgraph
void SetLabelSubGraphData( Labelgraph *lg, Subgraph *sg, Scene *scn, Scene *mask )
{
    int p,q,i;
    uint *size = NULL;
    float *mean = NULL;
    Voxel   u,v;
    AdjRel3 *A;

    //printf("It found %d clusters\n",sg->nlabels);
    if ( sg->nlabels < 2 )
    {
        printf( "ERROR!!! Graph with only one cluster !\n" );
        return;
    }

    lg->max_mean = -1.0;

    // compute mean brightness and size of every label region
    mean = AllocFloatArray( sg->nlabels );
    size = AllocUIntArray( sg->nlabels );
    for ( p = 0; p < sg->nnodes; p++ )
    {
        mean[ sg->node[ p ].label ] += scn->data[ sg->node[ p ].pixel ];
        size[ sg->node[ p ].label ] += 1;
    }
    for ( p = 0; p < sg->nlabels; p++ )
    {
        mean[ p ] /= size[ p ];
        lg->node[ p ].mean  = mean[ p ];
        lg->node[ p ].size  = size[ p ];
        lg->node[ p ].rep   = p;
        lg->node[ p ].label = p;
        if ( lg->max_mean < mean[ p ] ) lg->max_mean = mean[ p ];
        //printf("mean[%d]:%5.1f ", p, mean[p]);
    }
    // printf("\n");
    // sets
    A = Spheric( 1.0 );
    //printf("a->n=%d\n",A->n);
    for ( p = 0; p < sg->nnodes; p++ )
    {
        if ( mask->data[ p ] )
        {
            //printf("In mask\n");
            u.x  = VoxelX( scn, p );
            u.y  = VoxelY( scn, p );
            u.z  = VoxelZ( scn, p );
            InsertSet( &( lg->node[ sg->node[ p ].label ].voxels ), p );
            for ( i = 1; i < A->n; i++ )
            {
                v.x = u.x + A->dx[ i ];
                v.y = u.y + A->dy[ i ];
                v.z = u.z + A->dz[ i ];
                if ( ValidVoxel( scn, v.x, v.y, v.z ) )
                {
                    //printf("Valid voxel\n");
                    q = VoxelAddress( scn, v.x, v.y, v.z );
                    if ( ( mask->data[ q ] ) && ( !IsInSet( lg->node[ sg->node[ p ].label ].adj, sg->node[ q ].label ) ) && ( sg->node[ p ].label != sg->node[ q ].label ) )
                    {
                        //printf("New label\n");
                        InsertSet( &( lg->node[ sg->node[ p ].label ].adj ), sg->node[ q ].label );
                    }
                }
            }
        }
    }

    free( mean );
    free( size );
    DestroyAdjRel3( &A );
}

// Set adjacent regions in labelgraph by label scene
void SetLabelSubGraphDatabyScene( Labelgraph *lg, Scene *label, Scene *scn, Scene *mask )
{
    int p,q,i,n;
    uint *size = NULL;
    float *mean = NULL;
    Voxel   u,v;
    AdjRel3 *A;

    printf("It found %d clusters\n",lg->nnodes);
    if ( lg->nnodes < 2 )
    {
        printf( "ERROR!!! Graph with only one cluster !\n" );
        return;
    }
    lg->max_mean = -1.0;
    n = scn->xsize * scn->ysize * scn->zsize;

    // compute mean brightness and size of every label region
    mean = AllocFloatArray( lg->nnodes );
    size = AllocUIntArray( lg->nnodes );
    for ( p = 0; p < n; p++ )
    {
        mean[ label->data[ p ] ] += scn->data[ p ];
        size[ label->data[ p ] ] += 1;
    }
    for ( p = 0; p < lg->nnodes; p++ )
    {
        mean[ p ] /= size[ p ];
        lg->node[ p ].mean  = mean[ p ];
        lg->node[ p ].size  = size[ p ];
        lg->node[ p ].rep   = p;
        lg->node[ p ].label = p;
        if ( lg->max_mean < mean[ p ] ) lg->max_mean = mean[ p ];
        //printf("mean[%d]:%5.1f ", p, mean[p]);
    }
    //printf("\n");
    // sets
    A = Spheric( 1.0 );
    //printf("a->n=%d\n",A->n);
    for ( p = 0; p < n; p++ )
    {
        if ( mask->data[ p ] )
        {
            //printf("In mask\n");
            u.x  = VoxelX( scn, p );
            u.y  = VoxelY( scn, p );
            u.z  = VoxelZ( scn, p );
            InsertSet( &( lg->node[ label->data[ p ] ].voxels ), p );
            for ( i = 1; i < A->n; i++ )
            {
                v.x = u.x + A->dx[ i ];
                v.y = u.y + A->dy[ i ];
                v.z = u.z + A->dz[ i ];
                if ( ValidVoxel( scn, v.x, v.y, v.z ) )
                {
                    //printf("Valid voxel\n");
                    q = VoxelAddress( scn, v.x, v.y, v.z );
                    if ( ( mask->data[ q ] ) && ( !IsInSet( lg->node[ label->data[ p ] ].adj, label->data[ q ] ) ) &&
                            ( label->data[ p ] != label->data[ q ] ) )
                    {
                        //printf("New label\n");
                        InsertSet( &( lg->node[ label->data[ p ] ].adj ), label->data[ q ] );
                    }
                }
            }
        }
    }

    free( mean );
    free( size );
    DestroyAdjRel3( &A );
}

// Random sampling

Subgraph *RandomSampl( Image *img, int nnodes ) {
  Subgraph *sg=NULL;
  Image *used=CreateImage( img->ncols, img->nrows);
  int i, n=img->ncols*img->nrows, p, max_nnodes;
  AdjRel *A=Circular(1.5);
  Image *edges,*dilated_edges;  
  // Detect edges to avoid samples on edges

  edges  = EdgeDetection(img);
  dilated_edges  = Dilate(edges,A);
  DestroyImage(&edges);
  DestroyAdjRel(&A);

  max_nnodes=0;
  for (i=0; i < n; i++) 
    if (dilated_edges->val[i]==0)
      max_nnodes++;

  if (nnodes > max_nnodes) {
    Warning("The number of samples exceeded the number of available pixels","RandomSampl");
    nnodes = max_nnodes;
  }
 
  // Obtain nnodes random samples with similar histogram

  sg   = CreateSubgraph(nnodes);
  i    = 0;
  while( i < nnodes ) {
    p = RandomInteger( 0, n - 1 );
    if (( used->val[ p ] == 0 )&&( dilated_edges->val[p]==0)) {
      sg->node[ i ].pixel = p;
      sg->node[ i ].position = p;
      used->val[ p ] = 1;
      i++;
    }
  }
  
  DestroyImage( &used );
  DestroyImage(&dilated_edges);

  return( sg );
}

// Create subgraph by uniform sampling

Subgraph *UnifSampl(Image *img, Image *mask, int dx, int dy, int xo, int yo)
{
    Subgraph *sg=NULL;
    int x,y,nnodes,i,p;

    if (mask != NULL)
    {
        nnodes=0;
        for (y=yo; y < img->nrows; y=y+dy)
            for (x=xo; x < img->ncols; x=x+dx)
            {
                p = x + img->tbrow[y];
                if (mask->val[p])
                {
                    nnodes++;
                }
            }
        sg = CreateSubgraph(nnodes);
        i=0;
        for (y=yo; y < img->nrows; y=y+dy)
            for (x=xo; x < img->ncols; x=x+dx)
            {
                p = x + img->tbrow[y];
                if (mask->val[p])
                {
                    sg->node[i].pixel = p;
                    sg->node[i].position = p;
                    i++;
                }
            }
    }
    else
    {
        nnodes=0;
        for (y=yo; y < img->nrows; y=y+dy)
            for (x=xo; x < img->ncols; x=x+dx)
            {
                p = x + img->tbrow[y];
                nnodes++;
            }
        sg = CreateSubgraph(nnodes);
        i=0;
        for (y=yo; y < img->nrows; y=y+dy)
            for (x=xo; x < img->ncols; x=x+dx)
            {
                p = x + img->tbrow[y];
                sg->node[i].pixel = p;
                sg->node[i].position = p;
                i++;
            }
    }

    return(sg);
}

Subgraph *UnifSampl3(Scene *scn, Scene *mask, int dx, int dy, int dz)
{
    Subgraph *sg=NULL;
    int x,y,z,nnodes,i,p;

    if (mask != NULL)
    {
        nnodes=0;
        for (z=0; z < scn->zsize; z=z+dz)
            for (y=0; y < scn->ysize; y=y+dy)
            {
                for (x=0; x < scn->xsize; x=x+dx)
                {
                    p = x + scn->tby[y] + scn->tbz[z];
                    if (mask->data[p])
                    {
                        nnodes++;
                    }
                }
            }
        sg = CreateSubgraph(nnodes);
        i=0;
        for (z=0; z < scn->zsize; z=z+dz)
            for (y=0; y < scn->ysize; y=y+dy)
            {
                for (x=0; x < scn->xsize; x=x+dx)
                {
                    p = x + scn->tby[y] + scn->tbz[z];
                    if (mask->data[p])
                    {
                        sg->node[i].pixel = p;
                        sg->node[i].position = p;
                        i++;
                    }
                }
            }
    }
    else
    {
        nnodes=0;
        for (z=0; z < scn->zsize; z=z+dz)
            for (y=0; y < scn->ysize; y=y+dy)
            {
                for (x=0; x < scn->xsize; x=x+dx)
                {
                    p = x + scn->tby[y] + scn->tbz[z];
                    nnodes++;
                }
            }
        sg = CreateSubgraph(nnodes);
        i=0;
        for (z=0; z < scn->zsize; z=z+dz)
            for (y=0; y < scn->ysize; y=y+dy)
            {
                for (x=0; x < scn->xsize; x=x+dx)
                {
                    p = x + scn->tby[y] + scn->tbz[z];
                    sg->node[i].pixel = p;
                    sg->node[i].position = p;
                    i++;
                }
            }
    }

    return(sg);
}

// Compute samples for image compression
Subgraph *ChessSampl(Image *img, int xinit)
{
    Subgraph *sg=NULL;
    int x,y,nnodes,i,p,xo;

    nnodes=0;
    xo = xinit;
    for (y=0; y < img->nrows; y=y+1)
    {
        for (x=xo; x < img->ncols; x=x+2)
        {
            nnodes++;
        }
        xo = 1 - xo;
    }

    sg = CreateSubgraph(nnodes);
    sg->nlabels=MaximumValue(img)+1;

    i=0;
    xo = xinit;
    for (y=0; y < img->nrows; y=y+1)
    {
        for (x=xo; x < img->ncols; x=x+2)
        {
            p = x + img->tbrow[y];
            sg->node[i].pixel  = p;
            sg->node[i].position  = p;
            sg->node[i].truelabel = img->val[p];
            i++;
        }
        xo = 1 - xo;
    }

    return(sg);
}

// Compute samples for image compression
Subgraph *VertSampl(Image *img)
{
    Subgraph *sg=NULL;
    int x,y,nnodes,i,p,yo;

    nnodes=0;
    yo = 1;
    for (y=yo; y < img->nrows; y=y+2)
    {
        for (x=0; x < img->ncols; x=x+1)
        {
            nnodes++;
        }
    }

    sg = CreateSubgraph(nnodes);
    sg->nlabels=MaximumValue(img)+1;

    i=0;
    for (y=yo; y < img->nrows; y=y+2)
    {
        for (x=0; x < img->ncols; x=x+1)
        {
            p = x + img->tbrow[y];
            sg->node[i].pixel  = p;
            sg->node[i].position  = p;
            sg->node[i].truelabel = img->val[p];
            i++;
        }
    }

    return(sg);
}

// Histogram Sampling

Subgraph *HistogramSampl3( Scene *scn, Scene *mask, float thres , int nnodes ) {
  Subgraph *sg = NULL;
  Scene *used = NULL, *ROI = NULL;
  Curve *c = NULL;
  int xthres, ythres, p, side, i, size;

  sg = CreateSubgraph( nnodes );
  used = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  if( mask == NULL ) ROI = CopyScene( scn );
  else ROI = Mult3( scn, mask );
  c = Histogram3( ROI );
  size = 0;
  side = 0;
  xthres = 2;

  for( i = 1; i < c->n; i++ ) size += ROUND( c->Y[ i ] );
  // Computing threshold point:
  for( ythres = ROUND( c->Y[ 1 ] ); ( ( float ) ythres ) / ( size - ythres ) <= thres; xthres++ ) {
    ythres += ROUND( c->Y[ xthres ] );
  }
  if( ( ythres < nnodes ) || ( size - ythres < nnodes ) ) {
    printf("Warning: Number of samples is bigger than the interval.\n");
    for( ; ( ( float ) ythres ) / ( size - ythres ) <= nnodes / 2; xthres++ ) {
      ythres += ROUND( c->Y[ xthres ] );
    }
  }
  // Computing samplings
  i = 0;
  while( i < nnodes ) {
    p = RandomInteger( 0, mask->n - 1 );
    if( used->data[ p ] == 0 ) {
      if( ( side == 0 ) && ( ROI->data[ p ] < xthres ) ) {
	side = 1;
	sg->node[ i ].pixel = p;
	sg->node[ i ].position = p;
	used->data[ p ] = 1;
	i++;
      }
      else {
	if ( ( side == 1 ) && ( ROI->data[ p ] >= xthres) ) {
	  side = 0;
	  sg->node[ i ].pixel = p;
	  sg->node[ i ].position = p;
	  used->data[ p ] = 1;
	  i++;
	}
      }
    }
  }
  DestroyScene( &used );
  DestroyScene( &ROI );
  DestroyCurve( &c );

  return( sg );
}

// Random sampling

Subgraph *RandomSampl3( Scene *scn, Scene *mask, int nnodes ) {
  Subgraph *sg=NULL;
  Scene *used=CreateScene( mask->xsize, mask->ysize, mask->zsize);
  int i, n, p;

  // Obtain nnodes random samples with similar histogram
  n = mask->n;
  sg   = CreateSubgraph(nnodes);
  i    = 0;
  while( i < nnodes ) {
    p = RandomInteger( 0, n - 1 );
    if( ( mask == NULL ) || ( mask->data[ p ] ) ) {
      if (used->data[ p ] == 0 ) {
	sg->node[ i ].pixel = p;
	sg->node[ i ].position = p;
	used->data[ p ] = 1;
	i++;
      }
    }
  }
  
  DestroyScene( &used );
  return( sg );
}

// Create subgraph by threshold sampling

Subgraph *ThresSampl3(Scene *scn, Scene *mask, int thres, int nnodes)
{
    Subgraph *sg=NULL;
    Scene *used=CreateScene(scn->xsize,scn->ysize,scn->zsize);
    int i,p,n;
    char side;

    sg   = CreateSubgraph(nnodes);
    i    = 0;
    n    = scn->xsize*scn->ysize*scn->zsize;
    side = 0;

    if (mask != NULL)
    {
        while (i < nnodes)
        {
            p = RandomInteger(0,n-1);
            if (mask->data[p])
            {
                if (used->data[p]==0)
                {
                    if ((side==0)&&(scn->data[p]<=thres))
                    {
                        side = 1; // set side to above
                        sg->node[i].pixel = p;
                        sg->node[i].position = p;
                        used->data[p]=1;
                        i++;
                    }
                    else
                    {
                        if ((side==1)&&(scn->data[p]>thres))
                        {
                            side = 0; // set side to below
                            sg->node[i].pixel = p;
                            sg->node[i].position = p;
                            used->data[p]=1;
                            i++;
                        }
                    }
                }
            }
        }
    }
    else
    {
        while (i < nnodes)
        {
            p = RandomInteger(0,n-1);
            if (used->data[p]==0)
            {
                if ((side==0)&&(scn->data[p]<=thres))
                {
                    side = 1; // set side to above
                    sg->node[i].pixel = p;
                    sg->node[i].position = p;
                    used->data[p]=1;
                    i++;
                }
                else
                {
                    if ((side==1)&&(scn->data[p]>thres))
                    {
                        side = 0; // set side to below
                        sg->node[i].pixel = p;
                        sg->node[i].position = p;
                        used->data[p]=1;
                        i++;
                    }
                }
            }
        }
    }
    DestroyScene(&used);

    return(sg);
}

Subgraph *ThresNSampl3(Scene *scn, Scene *mask, int *thres, int N, int nnodes)
{
    Subgraph *sg=NULL;
    Scene *used=CreateScene(scn->xsize,scn->ysize,scn->zsize);
    int i,p,n;
    int side;

    sg   = CreateSubgraph(nnodes);
    i    = 0;
    n    = scn->xsize*scn->ysize*scn->zsize;
    side = 0;

    if (mask != NULL)
    {
        while (i < nnodes)
        {
            p = RandomInteger(0,n-1);
            if (mask->data[p])
            {
                if (used->data[p]==0)
                {
                    if ((side==0)&&(scn->data[p]<=thres[0]))
                    {
                        side = 1; // set side to above
                        sg->node[i].pixel = p;
                        sg->node[i].position = p;
                        used->data[p]=1;
                        i++;
                    }
                    else if ((side<N-1)&&(scn->data[p]>thres[side-1])&&(scn->data[p]<=thres[side]))
                    {
                        side++; // set side to above
                        sg->node[i].pixel = p;
                        sg->node[i].position = p;
                        used->data[p]=1;
                        i++;
                    }
                    else
                    {
                        if ((side==N-1)&&(scn->data[p]>thres[N-1]))
                        {
                            side = 0; // set side to below
                            sg->node[i].pixel = p;
                            sg->node[i].position = p;
                            used->data[p]=1;
                            i++;
                        }
                    }
                }
            }
        }
    }
    else
    {
        while (i < nnodes)
        {
            p = RandomInteger(0,n-1);
            if (used->data[p]==0)
            {
                if ((side==0)&&(scn->data[p]<=thres[0]))
                {
                    side = 1; // set side to above
                    sg->node[i].pixel = p;
                    sg->node[i].position = p;
                    used->data[p]=1;
                    i++;
                }
                else if ((side<N-1)&&(scn->data[p]>thres[side-1])&&(scn->data[p]<=thres[side]))
                {
                    side++; // set side to above
                    sg->node[i].pixel = p;
                    sg->node[i].position = p;
                    used->data[p]=1;
                    i++;
                }
                else
                {
                    if ((side==N-1)&&(scn->data[p]>thres[N-1]))
                    {
                        side = 0; // set side to below
                        sg->node[i].pixel = p;
                        sg->node[i].position = p;
                        used->data[p]=1;
                        i++;
                    }
                }
            }
        }
    }
    DestroyScene(&used);

    for (i=0; i < sg->nnodes; i++)
        sg->node[i].status = 0; // training node


    return(sg);
}

/*----------- Feature computation based on Markov properties -----------*/

// The intensity/color of a node and of its neighboors within a dm
// radius in the image/scene are used to form its feature vector. It
// assumes that the sampling has been done.

void MarkovFeat3(Subgraph *sg, Scene *scn, Scene *mask, float dm)
{
    AdjRel3 *A;
    int     p,q,i,j,xysize=scn->xsize*scn->ysize,aux;
    Voxel   u,v;

    sg->Imax=MaximumValue3(scn);

    if (dm >= 1.0)
    {
        A=Spheric(dm);
        sg->dm=dm;
        sg->nfeats = A->n;
        for (i=0; i < sg->nnodes; i++)
            sg->node[i].feat = AllocFloatArray(sg->nfeats);

        for ( i = 0; i < sg->nnodes; i++ )
        {
            p   = sg->node[i].pixel;
            aux = p%xysize;
            u.x = aux%scn->xsize;
            u.y = aux/scn->xsize;
            u.z = p/xysize;
            for (j=0; j < A->n; j++)
            {
                v.x = u.x + A->dx[j];
                v.y = u.y + A->dy[j];
                v.z = u.z + A->dz[j];
                q = v.x + scn->tby[v.y] + scn->tbz[v.z];
                if ( (ValidVoxel(scn,v.x,v.y,v.z)) && ( ( mask == NULL ) || ( mask->data[q] != 0 ) ) )
                {
                    sg->node[i].feat[j]=(float)scn->data[q]/sg->Imax;
                }
                else
                {
                    sg->node[i].feat[j]=(float)scn->data[p]/sg->Imax;
                }
            }
        }
        DestroyAdjRel3(&A);
    }
    else
    {
        sg->dm=0.0;
        sg->nfeats = 1;
        for (i=0; i < sg->nnodes; i++)
            sg->node[i].feat = AllocFloatArray(sg->nfeats);

        for (i=0; i < sg->nnodes; i++)
        {
            p   = sg->node[i].pixel;
            sg->node[i].feat[0]=(float)scn->data[p]/sg->Imax;
        }
    }
}


/*--------- Training functions compute an optimum path forest for
  --------- further classification ----------------------------- */

// Compute unsupervised training for the pre-computed best k

///void UnsupTrain(Subgraph *sg)
//{
//    Set *adj_i,*adj_j;
//    char insert_i;
//    int i,j;
//
//    //   Add arcs to guarantee symmetry on plateaus
//
//    for (i=0; i < sg->nnodes; i++)
//    {
//      adj_i = sg->node[i].adj;
//      while (adj_i != NULL)
//        {
//	  j        = adj_i->elem;
//	  if (sg->node[i].dens==sg->node[j].dens)
//            {
//	      // insert i in the adjacency of j if it is not there.
//	      adj_j    = sg->node[j].adj;
//	      insert_i = 1;
//	      while (adj_j != NULL)
//                {
//		  if (i == adj_j->elem)
//                    {
//		      insert_i=0;
//		      break;
//                    }
//		  adj_j=adj_j->next;
//                }
//	      if (insert_i)
//		InsertSet(&(sg->node[j].adj),i);
//            }
//	  adj_i=adj_i->next;
//        }
//    }
//    UnsupOPF(sg);
//}

// Supervised training for complete graph

///void SupTrainCompGraph(Subgraph *sg)
//{
//    int p,q,i;
//    int tmp,weight;
//    GQueue *Q = NULL;
//    int *pathval = NULL;
//
//    // compute optimum prototypes
//
//    opf_MSTPrototypes(sg);
//
//    // initialization
//
//    pathval = AllocIntArray(sg->nnodes);
//
//    Q=CreateGQueue(opf_ArcWeight(sg->df)+1,sg->nnodes, pathval);
//
//    for (p = 0; p < sg->nnodes; p++)
//    {
//        if (sg->node[p].status==opf_PROTOTYPE)
//        {
//            pathval[p]         = 0;
//            sg->node[p].label  = sg->node[p].truelabel;
//            sg->node[p].pred   = NIL;
//            InsertGQueue(&Q, p);
//        }
//        else  // non-prototypes
//        {
//            pathval[p]  = INT_MAX;
//        }
//    }
//
//    // IFT with fmax
//
//
//    i=0;
//    while ( !EmptyGQueue(Q) )
//    {
//        p=RemoveGQueue(Q);
//	sg->ordered_list_of_nodes[i]=p; i++;
//
//        sg->node[p].pathval = pathval[p];
//
//        for (q=0; q < sg->nnodes; q++)
//        {
//            if (p!=q)
//            {
//                if (pathval[p] < pathval[q])
//                {
//                    if (sg->node[p].truelabel!=sg->node[q].truelabel)
//                        weight=INT_MAX;
//                    else
//                        if (!opf_PrecomputedDistance)
//                            weight = (int)opf_ArcWeight(DistFun(sg->node[p].feat,sg->node[q].feat,sg->nfeats));
//                        else
//                            weight =  (int)opf_ArcWeight(opf_DistanceValue[sg->node[p].pixel][sg->node[q].pixel]);
//                    tmp  = MAX(pathval[p],weight);
//                    if ( tmp < pathval[ q ] )
//                    {
//                        if (Q->L.elem[q].color==GRAY)
//                            RemoveGQueueElem(Q,q);
//                        sg->node[q].pred  = p;
//                        sg->node[q].label = sg->node[p].label;
//                        pathval[q] = tmp;
//                        InsertGQueue(&Q, q);
//                    }
//                }
//            }
//        }
//    }
//
//    // Verify if error is really zero in training set
//    for (p=0; p < sg->nnodes; p++)
//    {
//        if (sg->node[p].label!=sg->node[p].truelabel)
//            Warning("Class. error in the training set","SupTrainCompGraph");
//    }
//
//    DestroyGQueue( &Q );
//    free( pathval );
//}

// Supervised training for knn-graph

void SupTrainKnnGraph(Subgraph *sg)
{
    Set *adj_i,*adj_j;
    char insert_i;
    int i,j;

    //   Add arcs to guarantee symmetry on plateaus

    for (i=0; i < sg->nnodes; i++)
    {
        adj_i = sg->node[i].adj;
        while (adj_i != NULL)
        {
            j        = adj_i->elem;
            if (sg->node[i].dens==sg->node[j].dens)
            {
                // insert i in the adjacency of j if it is not there.
                adj_j    = sg->node[j].adj;
                insert_i = 1;
                while (adj_j != NULL)
                {
                    if (i == adj_j->elem)
                    {
                        insert_i=0;
                        break;
                    }
                    adj_j=adj_j->next;
                }
                if (insert_i)
                    InsertSet(&(sg->node[j].adj),i);
            }
            adj_i=adj_i->next;
        }
    }

    SupOPF(sg);
}

void SemiSupTrainKnnGraph(Subgraph *sg)
{
    Set *adj_i,*adj_j;
    char insert_i;
    int i,j;

    //   Add arcs to guarantee symmetry on plateaus

    for (i=0; i < sg->nnodes; i++)
    {
        adj_i = sg->node[i].adj;
        while (adj_i != NULL)
        {
            j        = adj_i->elem;
            if (sg->node[i].dens==sg->node[j].dens)
            {
                // insert i in the adjacency of j if it is not there.
                adj_j    = sg->node[j].adj;
                insert_i = 1;
                while (adj_j != NULL)
                {
                    if (i == adj_j->elem)
                    {
                        insert_i=0;
                        break;
                    }
                    adj_j=adj_j->next;
                }
                if (insert_i)
                    InsertSet(&(sg->node[j].adj),i);
            }
            adj_i=adj_i->next;
        }
    }

    SemiSupOPF(sg);
}

/*--------- Classification functions ----------------------------- */

Image *ImagePDF(Subgraph *sg, Features *f)
{
    int     i,p,k;
    Image  *pdf=CreateImage(f->ncols,f->nrows);
    float dist;

    for (p=0; p < f->nelems; p++)
    {
      for (i = 0; i < sg->nnodes; i++){
	k = sg->ordered_list_of_nodes[i];
	dist = opf_ArcWeight(sg->node[k].feat,f->elem[p].feat,sg->nfeats);

	if (dist <= sg->node[k].radius){
	  pdf->val[p]=sg->node[k].dens;

	  break;
	}
      }
    }
    return(pdf);
}

/* Image *ImagePDF(Subgraph *sg, Features *f, int label) */
/* { */
/*     CNode  *node=CreateCNode(sg); */
/*     int     i,p; */
/*     Image  *pdf=CreateImage(f->ncols,f->nrows); */
/*     float *prob=AllocFloatArray(sg->nlabels); */

/*     for (i=0; i < sg->nnodes; i++) */
/*         prob[sg->node[i].truelabel]+=1.0/sg->nnodes; */

/*     for (p=0; p < f->nelems; p++) */
/*     { */
/*         for (i=0; i < sg->nfeats; i++) */
/*             node->feat[i]=f->elem[p].feat[i]; */
/*         NodeArcsByTrueLabel(sg, node, label); */
/*         NodePD(sg,node); */
/*         pdf->val[p] = (int)(prob[label]*node->dens); */
/*     } */

/*     DestroyCNode(&node); */
/*     free(prob); */
/*     return(pdf); */
/* } */

Image *ImageLabel(Subgraph *sg, Image *img, int vol)
{
    int i;
    Image *label=CreateImage(img->ncols,img->nrows);

    SPDF(sg,img);
    opf_ElimMaxBelowVolume(sg,vol);
    opf_OPFClustering(sg);
    for (i=0; i < sg->nnodes; i++)
        label->val[i]=sg->node[i].label;

    return(label);
}

void SceneLabel( Subgraph *sg, Scene *mask ) {
  SPDF3( sg, mask );
  SpatialUnsupOPF3( sg, mask );
}

Image *ImageClassKnnGraph(Subgraph *sg, Features *f)
{
    int     i,p,k;
    Image  *label=CreateImage(f->ncols,f->nrows);
    float dist;

    for (p=0; p < f->nelems; p++)
    {
      for (i = 0; i < sg->nnodes; i++){
	k = sg->ordered_list_of_nodes[i];
	dist = opf_ArcWeight(sg->node[k].feat,f->elem[p].feat,sg->nfeats);
	if (dist <= sg->node[k].radius){
	  label->val[p]=sg->node[k].label+1;

	  break;
	}
      }
    }
    return(label);
}

Scene *SceneClassKnnGraph( Subgraph *sg, Scene *mask, Features3 *f ) {
  //CNode  *node = CreateCNode( sg );
  //int   pred;
  float dist;
  int   i,p,k;
  Scene *label = CreateScene( f->xsize, f->ysize, f->zsize );

  for( p = 0; p < f->nelems; p++ ) {
    if( ( mask == NULL ) || ( mask->data[ p ] != 0 ) ) {
      for( i = 0; i < sg->nnodes; i++ ) {
      	k = sg->ordered_list_of_nodes[ i ];
	dist = opf_ArcWeight( sg->node[ k ].feat, f->elem[ p ].feat, sg->nfeats );
	if( dist <= sg->node[ k ].radius ) {
	  label->data[ p ] = sg->node[ k ].label;
	  break;
	}
      //for( i = 0; i < sg->nfeats; i++ )
      //  node->feat[ i ] = f->elem[ p ].feat[ i ];
      //NodeArcs( sg, node );
      //NodePD( sg, node );
      //pred = PredBestPath( sg, node );
      //label->data[ p ] = sg->node[ pred ].label;
      }
    }
  }

  //DestroyCNode( &node );
  return( label );
}

// Classify nodes of evaluation/test set for KnnGraph
//void ClassifyKnnGraph(Subgraph *sgtrain, Subgraph *sg)
//{
//    CNode  *node=CreateCNode(sgtrain);
//    int     i,p,pred;
//
//    for (p=0; p < sg->nnodes; p++)
//    {
//        for (i=0; i < sg->nfeats; i++)
//            node->feat[i]=sg->node[p].feat[i];
//        node->pixel = sg->node[p].pixel;
//        NodeArcs(sgtrain,node);
//        NodePD(sgtrain,node);
//        pred=PredBestPath(sgtrain,node);
//        //pred=PredBestPathBayesTie(sgtrain,node);
//        //pred = PredBestPathMeanTie(sg, node);
//        sg->node[p].label=sgtrain->node[pred].label;
//    }
//
//    DestroyCNode(&node);
//}

// Classify nodes of evaluation/test set By bayes
void ClassifyBayes( Subgraph *sgTrain, Subgraph *sg )
{
    CNode  *node = CreateCNode( sgTrain );
    int     i, label, bestlabel, p;
    float  maxval;

    for ( p = 0; p < sg->nnodes; p++ )
    {
        for ( i = 0; i < sg->nfeats; i++ ) node->feat[ i ] = sg->node[ p ].feat[ i ];
        node->pixel = sg->node[ p ].pixel;
        bestlabel = NIL;
        maxval = INT_MIN;
        NodeArcs(sgTrain, node);
        for ( label = 0; label < sgTrain->nlabels; label++ )
        {
            NodePDByTrueLabel( sgTrain, node, label );
            if ( node->dens > maxval )
            {
                maxval    = node->dens;
                bestlabel = label;
            }
        }
        sg->node[ p ].label = bestlabel;
    }

    DestroyCNode( &node );
}

// Classify nodes of evaluation/test set for CompGraph
//void ClassifyCompGraph(Subgraph *sgtrain, Subgraph *sg)
//{
//  int i, j, k, l, label = -1;
//  int tmp, weight, minCost;
//
//  for (i = 0; i < sg->nnodes; i++)
//  {
//    j       = 0;
//    k       = sgtrain->ordered_list_of_nodes[j];
//    if (!opf_PrecomputedDistance)
//      weight = (int)opf_ArcWeight(sgtrain->node[k].feat,sg->node[i].feat,sg->nfeats);
//    else
//      weight =  (int)opf_DistanceValue[sgtrain->node[k].pixel][sg->node[i].pixel];
//
//
//
//    minCost = MAX(sgtrain->node[k].pathval, weight);
//    label   = sgtrain->node[k].label;
//
//    while((j < sgtrain->nnodes-1)&&
//    (minCost > sgtrain->node[sgtrain->ordered_list_of_nodes[j+1]].pathval)){
//
//      l  = sgtrain->ordered_list_of_nodes[j+1];
//
//    if (!opf_PrecomputedDistance)
//      weight = (int)opf_ArcWeight(sgtrain->node[l].feat,sg->node[i].feat,sg->nfeats);
//    else
//      weight =  (int)opf_DistanceValue[sgtrain->node[l].pixel][sg->node[i].pixel];
//
//
//      tmp = MAX(sgtrain->node[l].pathval, weight);
//      if(tmp < minCost){
//	minCost = tmp;
//	label = sgtrain->node[l].label;
//      }
//      j++;
//      k  = l;
//    }
//    sg->node[i].label = label;
//  }
//}

Image *ImageClassCompGraph(Subgraph *sg, Features *f)
{
    Image *label=CreateImage(f->ncols,f->nrows);
    int p,q,n=f->nelems;
    int tmp,minval,weight;


    for (q=0; q < n; q++)
    {
        if (sg->node[0].pixel!=q)
        {
            weight = opf_ArcWeight(f->elem[q].feat,sg->node[0].feat,sg->nfeats);
            minval = MAX(sg->node[0].pathval,weight);
            label->val[q] = sg->node[0].label;
            for (p=1; p < sg->nnodes; p++)
            {
                if (sg->node[p].pixel!=q)
                {
                    weight = opf_ArcWeight(f->elem[q].feat,sg->node[p].feat,sg->nfeats);
                    tmp  = MAX(sg->node[p].pathval,weight);
                    if (tmp < minval)
                    {
                        label->val[q]=sg->node[p].label;
                    }
                }
                else
                {
                    label->val[q]=sg->node[p].label;
                    break;
                }
            }
        }
        else
        {
            label->val[q]=sg->node[0].label;
        }
    }
    return(label);
}

Scene *FasterSceneCluster( Subgraph *sg, Features3 *f, Scene *mask, int tiebreak ) {
  CNode  *node = CreateCNode( sg );
  int     p, pred, nvoxels = 0, *index;
  float   *dens, dist;
  Scene  *label = CreateScene( mask->xsize, mask->ysize, mask->zsize );

  dens = AllocFloatArray( sg->nnodes );
  index = AllocIntArray( sg->nnodes );
  for( p = 0; p < sg->nnodes; p++ )
    dens[ p ] = sg->node[ p ].dens;
  FloatQuickSort( dens, index, 0, sg->nnodes - 1, DECREASING );

  for ( p = 0; p < mask->n; p++ ) {
    if ( mask->data[ p ] ) {
      CopyNodeFeat3( f, node, p );
      pred = 0;
      do{
	dist = opf_ArcWeight( node->feat, sg->node[ pred ].feat, sg->nfeats );
	pred++;
      } while( ( pred < sg->nnodes ) && ( dist > sg->node[ pred ].radius ) );
      if( pred == sg->bestk )
	pred = 0;
      else
	pred--;
      label->data[ p ] = sg->node[ pred ].label;
      nvoxels++;
    }
  }
  free( dens );
  free( index );
  DestroyCNode( &node );
  return( label );
}

Scene *SceneCluster( Subgraph *sg, Features3 *f, Scene *mask, int tiebreak ) {
  CNode  *node = CreateCNode( sg );
  int     p, pred, nvoxels = 0;
  Scene  *label = CreateScene( mask->xsize, mask->ysize, mask->zsize );

  for ( p = 0; p < mask->n; p++ ) {
    if ( mask->data[ p ] ) {
      CopyNodeFeat3( f, node, p );
      NodeArcs( sg, node );
      NodePD( sg, node );
      if ( tiebreak == BAYES_TIE ) pred = PredBestPathBayesTie( sg, node );
      else if ( tiebreak == NO_TIE ) pred = PredBestPath( sg, node );
      else pred = PredBestPathMeanTie( sg, node );
      label->data[ p ] = sg->node[ pred ].label;
      nvoxels++;
    }
  }
  DestroyCNode( &node );
  return( label );
}

Scene *FastSceneCluster( Subgraph *sg, Features3 *f, Scene *mask, int tiebreak ) {
  CNode  *node = CreateCNode( sg );
  int     p, best, maxpathval, i;
  Scene  *label = CreateScene( mask->xsize, mask->ysize, mask->zsize );

  for ( p = 0; p < mask->n; p++ ) {
    if ( mask->data[ p ] ) {
      CopyNodeFeat3( f, node, p );
      NodeArcs( sg, node );
      best = 0;
      maxpathval = sg->node[ node->adj[ 0 ] ].pathval;
      for ( i = 1; i < sg->bestk; i++ ) {
	if ( sg->node[ node->adj[ i ] ].pathval > maxpathval ) {
	  best = node->adj[ i ];
	  maxpathval = sg->node[ node->adj[ i ] ].pathval;
        }
      }
      label->data[ p ] = sg->node[ best ].label;
    }
  }
  DestroyCNode( &node );
  return( label );
}

Scene **SceneClusterforMajorityVote( Subgraph *sg, Features3 *f, Scene *mask, int tiebreak )
{
  CNode  *node = CreateCNode( sg );
  int     p, pred, nvoxels = 0;
  Scene **result;
  float   dist;

  result = ( Scene** ) calloc( 3, sizeof( Scene* ) );
  result[ 0 ] = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  result[ 1 ] = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  result[ 2 ] = CreateScene( mask->xsize, mask->ysize, mask->zsize );

  for ( p = 0; p < mask->n; p++ ) {
    if ( mask->data[ p ] ) {
      CopyNodeFeat3( f, node, p );
      NodeArcs( sg, node );
      NodePD( sg, node );
      if ( tiebreak == BAYES_TIE ) pred = PredBestPathBayesTie( sg, node );
      else if ( tiebreak == NO_TIE ) pred = PredBestPath( sg, node );
      else pred = PredBestPathMeanTie( sg, node );
      dist = opf_ArcWeight( sg->node[ pred ].feat, node->feat, sg->nfeats );
      ( result[ 0 ] )->data[ p ] = sg->node[ pred ].label;
      ( result[ 1 ] )->data[ p ] = node->dens;
      ( result[ 2 ] )->data[ p ] = sg->node[ pred ].dens / ( 20 * dist );
      nvoxels++;
    }
  }
  DestroyCNode( &node );

  return( result );
}

// Classify nodes of evaluation/test set By bayes
Scene *SceneClassifyBayes( Subgraph *sg, Features3 *f, Scene *mask ) {
  CNode  *node = CreateCNode( sg );
  int     label, bestlabel, p;
  float   maxval;
  Scene *result;
  result = CreateScene( mask->xsize, mask->ysize, mask->zsize );

  for ( p = 0; p < mask->n; p++ ) {
    if( mask->data[ p ] == 1 ) {
      CopyNodeFeat3( f, node, p );
      node->pixel = p;
      bestlabel = NIL;
      maxval = INT_MIN;
      NodeArcs( sg, node );
      for ( label = 0; label <= sg->nlabels; label++ ) {
	NodePDByTrueLabel( sg, node, label );
	if ( node->dens > maxval ) {
	  maxval    = node->dens;
	  bestlabel = label;
	}
      }
      result->data[ p ] = bestlabel;
    }
  }

  DestroyCNode( &node );
  return( result );
}

/* ----------  Auxiliary functions ------------------------------- */

// Normalized cut

///float NormalizedCut( Subgraph *sg )
//{
//    int l, p, q;
//    Set *Saux;
//    float ncut, dist;
//    float *acumIC; //acumulate weights inside each class
//    float *acumEC; //acumulate weights between the class and a distinct one
//
//    ncut = 0.0;
//    acumIC = AllocFloatArray( sg->nlabels );
//    acumEC = AllocFloatArray( sg->nlabels );
//
//    for ( p = 0; p < sg->nnodes; p++ )
//    {
//        for ( Saux = sg->node[ p ].adj; Saux != NULL; Saux = Saux->next )
//        {
//            q = Saux->elem;
//            dist = (float) DistFun(sg->node[p].feat,sg->node[q].feat,sg->nfeats);
//            dist = sqrtf( dist );
//            if ( dist > 0.0 )
//            {
//                if ( sg->node[ p ].label == sg->node[ q ].label )
//                {
//                    acumIC[ sg->node[ p ].label ] += 1.0 / dist; // intra-class weight
//                }
//                else   // inter - class weight
//                {
//                    acumEC[ sg->node[ p ].label ] += 1.0 / dist; // inter-class weight
//                }
//            }
//        }
//    }
//
//    for ( l = 0; l < sg->nlabels; l++ )
//    {
//        if ( acumIC[ l ] + acumEC[ l ]  > 0.0 ) ncut += (float) acumEC[ l ] / ( acumIC[ l ] + acumEC[ l ] );
//    }
//    free( acumEC );
//    free( acumIC );
//    return( ncut );
//}

// Influence zones of the maxima: Nao eh mais necessario unir zonas de
// influencia, pois o grafo esta simetrico nos plateaus.

///void UnsupOPF(Subgraph *sg)
//{
//  int p, q, l, i ;
//    float tmp,*pathval=NULL;
//    RealHeap *Q=NULL;
//    Set *Saux=NULL;
//    int *minlevel=NULL;
//
//    pathval = AllocFloatArray(sg->nnodes);
////    minlevel = AllocIntArray(sg->nnodes);
//    Q = CreateRealHeap(sg->nnodes, pathval);
//    SetRemovalPolicyRealHeap(Q, MAXVALUE);
//
//    for (p = 0; p < sg->nnodes; p++)
//    {
////        sg->node[ p ].volume = 0.0;
//        pathval[ p ] = sg->node[ p ].pathval;
//        sg->node[ p ].pred  = NIL;
//        sg->node[ p ].root  = p;
//        InsertRealHeap(Q, p);
//    }
//
//    l = 0; i = 0;
//
//    while ( !IsEmptyRealHeap(Q) )
//    {
//      RemoveRealHeap(Q,&p);
//
//      sg->ordered_list_of_nodes[i]=p; i++;
//
//      if ( sg->node[ p ].pred == NIL )
//        {
//	  pathval[ p ] = sg->node[ p ].dens;
//	  sg->node[p].label=l;
//	  l++;
//	  minlevel[ p ] = sg->node[ p ].dens;
//        }
////      else
////        {
////	  if (sg->node[ p ].dens < minlevel[ sg->node[p].root ])
////	    minlevel[ sg->node[p].root ] = sg->node[ p ].dens;
////        }
//
//      sg->node[ p ].pathval = pathval[ p ];
//
//      for ( Saux = sg->node[ p ].adj; Saux != NULL; Saux = Saux->next )
//        {
//	  q = Saux->elem;
//
//	  if (Q->color[q] != BLACK) {
//
//	    tmp = MIN( pathval[ p ], sg->node[ q ].dens);
//	    if ( tmp > pathval[ q ] )
//	      {
//		UpdateRealHeap(Q,q,tmp);
//		sg->node[ q ].pred  = p;
//		sg->node[ q ].root  = sg->node[ p ].root;
//		sg->node[ q ].label = sg->node[ p ].label;
//	      }
//	  }
//
//        }
//
//    }
//
//    DestroyRealHeap( &Q );
//
////    // compute volume of each cluster
////    for (p=0; p < sg->nnodes; p++)
////        sg->node[ sg->node[p].root ].volume += (sg->node[ p ].dens - minlevel[ sg->node[p].root ]);
//
//    free( pathval );
////    free(minlevel);
//
//    sg->nlabels = l;
//
//}

// Influence zones of selected maxima.

void OPFByMarkers(Subgraph *sg, Set **S)
{

    int  p, q, l;
    float tmp,*pathval;
    RealHeap *H=NULL;
    Set *Saux=NULL;

    //printf("Opf by markers\n");

    if ((*S) == NULL) return;

    pathval=AllocFloatArray(sg->nnodes);
    H = CreateRealHeap(sg->nnodes, pathval);
    SetRemovalPolicyRealHeap(H, MAXVALUE);


    for (p = 0; p < sg->nnodes; p++)
    {
        pathval[ p ] = INT_MIN;
        sg->node[ p ].pathval = INT_MIN;
        sg->node[ p ].label = 0;
    }

    l=1;
    while ((*S) != NULL)
    {
        p=RemoveSet(S);
        pathval[ p ] = sg->node[p].dens;
        sg->node[ p ].pred  = NIL;
        sg->node[ p ].root  = p;
        sg->node[ p ].label = l;
        l++;
        InsertRealHeap(H, p);
        //printf("seed %d\n",p);
    }

    while ( !IsEmptyRealHeap(H) )
    {
        RemoveRealHeap(H,&p);

        sg->node[ p ].pathval = pathval[ p ];

        for ( Saux = sg->node[ p ].adj; Saux != NULL; Saux = Saux->next )
        {
            q = Saux->elem;
            if ( pathval[ p ] > pathval[ q ] )
            {
                tmp = MIN( pathval[ p ], sg->node[ q ].dens);
                if ( tmp > pathval[ q ] )
                {
                    if (pathval[ q ] != INT_MIN)
                        UpdateRealHeap(H,q,tmp);
                    else
                    {
                        pathval[ q ] = tmp;
                        InsertRealHeap(H,q);
                    }
                    sg->node[ q ].pred  = p;
                    sg->node[ q ].root  = sg->node[ p ].root;
                    sg->node[ q ].label = sg->node[ p ].label;
                }
            }
        }
    }

    DestroyRealHeap( &H );
    free( pathval );

    sg->nlabels = l;

    // Classify trivial trees

    CNode  *node;
    int     i,pred;
    char    firsttime=1;
    node = CreateCNode(sg);

    for (p=0; p < sg->nnodes; p++)
    {
        if (sg->node[p].label==0)
        {
            if (firsttime)
            {
	      //printf("Classifying trivial tree\n");
                firsttime=0;
            }
            for (i=0; i < sg->nfeats; i++)
                node->feat[i]=sg->node[p].feat[i];
            NodeArcsInSubforest(sg,node);
            NodePD(sg,node);
            pred=PredBestPath(sg,node);
            sg->node[p].root = sg->node[pred].root;
            sg->node[p].pred = pred;
        }
    }
    for (p=0; p < sg->nnodes; p++)
    {
        if (sg->node[p].label==0)
        {
            sg->node[p].label=sg->node[sg->node[p].root].label;
        }
        //if (sg->node[p].root==p) printf("root %d\n",p);
    }
    DestroyCNode(&node);
}

// Influence zones of the maxima: Nao eh mais necessario unir zonas de
// influencia, pois o grafo esta simetrico nos plateaus.
/*
void UnsupOPF(Subgraph *sg) {
  int p, q, l;
  int rp, rq;
  float tmp,*pathval=NULL;
  RealHeap *Q=NULL;
  Set *Saux=NULL;
  char flatzone_p,flatzone_q;

  pathval = AllocFloatArray(sg->nnodes);
  Q = CreateRealHeap(sg->nnodes, pathval);
  SetRemovalPolicyRealHeap(Q, MAXVALUE);

  for (p = 0; p < sg->nnodes; p++) {
    pathval[ p ] = sg->node[ p ].pathval;
    sg->node[ p ].pred  = NIL;
    InsertRealHeap(Q, p);
  }

  while ( !IsEmptyRealHeap(Q) ) {
    RemoveRealHeap(Q,&p);
    rp = RootNode(sg,p,&flatzone_p);

    if ( sg->node[ p ].pred == NIL ) {
      pathval[ p ] = sg->node[ p ].dens;
    }

    sg->node[ p ].pathval = pathval[ p ];

    for ( Saux = sg->node[ p ].adj; Saux != NULL; Saux = Saux->next ) {
      q = Saux->elem;
      if ( pathval[ p ] > pathval[ q ] ) {
	tmp = MIN( pathval[ p ], sg->node[ q ].dens);
	if ( tmp > pathval[ q ] ) {
	  UpdateRealHeap(Q,q,tmp);
	  sg->node[ q ].pred  = p;
	}
      }
      else {
      if ( pathval[ p ] == pathval[ q ] ) {
      rq = RootNode(sg,q,&flatzone_q);
      if ((rp != rq)&&
      (flatzone_p)&&
      (flatzone_q)&&
      (sg->node[rp].dens==sg->node[rq].dens)){
      sg->node[rq].pred = rp;
      }
      }
      }
    }
  }
  DestroyRealHeap( &Q );
  free( pathval );

  l = 1;
  for (p=0; p < sg->nnodes; p++)
    if (sg->node[p].pred==NIL){
      sg->node[p].label = l;
      l++;
    }

  for (p=0; p < sg->nnodes; p++){ // label propagation
    rp = RootNode(sg,p,&flatzone_p);
    sg->node[p].label = sg->node[rp].label;
  }
  sg->nlabels = l;
}
*/

// Unsupervised OPF with spatial constrains
void SpatialUnsupOPF3( Subgraph *sg, Scene *mask ) {
  Voxel u,v;
  double dist;
  int k, p, q, l;
  RealHeap *Q = NULL;
  float tmp, *pathval = NULL;
  AdjRel3 *A = Spheric( sg->di );
  
  pathval = AllocFloatArray( sg->nnodes );
  Q = CreateRealHeap( sg->nnodes, pathval );
  SetRemovalPolicyRealHeap( Q, MAXVALUE );
  
  for( p = 0; p < sg->nnodes; p++ ) {
    if( mask->data[ p ] != 0 ) {
      pathval[ p ] = sg->node[ p ].pathval;
      sg->node[ p ].pred  = NIL;
      InsertRealHeap( Q, p );
    }
    else {
      pathval[ p ] = 0;
      sg->node[ p ].pred  = NIL;
    }
  }
  l = 1;
  while( !IsEmptyRealHeap( Q ) ) {
    RemoveRealHeap( Q, &p );
    u.x = VoxelX( mask, p );
    u.y = VoxelY( mask, p );
    u.z = VoxelZ( mask, p );
    if ( sg->node[ p ].pred == NIL ) {
      pathval[ p ] = sg->node[ p ].dens;
      sg->node[ p ].label = l;
      l++;
    }
    sg->node[ p ].pathval = pathval[ p ];
    for ( k = 1; k < A->n; k++ ) {  // image space restriction
      v.x = u.x + A->dx[ k ];
      v.y = u.y + A->dy[ k ];
      v.z = u.z + A->dz[ k ];
      if( ValidVoxel( mask, v.x, v.y, v.z ) ) {
	q = VoxelAddress( mask, v.x, v.y, v.z );
	if( mask->data[ q ] != 0 ) {
	  dist = opf_ArcWeight( sg->node[ p ].feat, sg->node[ q ].feat, sg->nfeats);
	  if ( ( dist <= sg->df ) && ( pathval[ p ] > pathval[ q ] ) ) {
	    tmp = MIN( pathval[ p ], sg->node[ q ].dens );
	    if ( tmp > pathval[ q ] ) {
	      UpdateRealHeap( Q, q, tmp );
	      sg->node[ q ].pred  = p;
	      sg->node[ q ].label = sg->node[ p ].label;
	    }
	  }
	}
      }
    }
  }
  DestroyRealHeap( &Q );
  free( pathval );
  
  sg->nlabels = l;
  DestroyAdjRel3( &A );
}

// Influence zones of the maxima: Nao eh mais necessario unir zonas de
// influencia, pois o grafo esta simetrico nos plateaus.
/*
// Unsupervised OPF with spatial constrains
void SpatialUnsupOPF3( Subgraph *sg, Scene *scn ) {
  Voxel u,v;
  double dist;
  int k, p, q, l, rp, rq;
  RealHeap *Q=NULL;
  float tmp, *pathval = NULL;
  char flatzone_p,flatzone_q;
  AdjRel3 *A=Spheric(sg->di);

  pathval = AllocFloatArray(sg->nnodes);
  Q = CreateRealHeap(sg->nnodes, pathval);
  SetRemovalPolicyRealHeap(Q, MAXVALUE);

  for ( p = 0; p < sg->nnodes; p++ ) {
    if ( scn->data[ p ] != 0 ) {
      pathval[ p ] = sg->node[ p ].pathval;
      sg->node[ p ].pred  = NIL;
      InsertRealHeap(Q, p);
    }
    else {
      pathval[ p ] = 0;
      sg->node[ p ].pred  = NIL;
    }
  }
  while ( !IsEmptyRealHeap( Q ) ) {
    RemoveRealHeap( Q, &p );
    u.x = VoxelX( scn, p );
    u.y = VoxelY( scn, p );
    u.z = VoxelZ( scn, p );
    rp = RootNode( sg, p, &flatzone_p );
    if ( sg->node[ p ].pred == NIL ) {
      pathval[ p ] = sg->node[ p ].dens;
    }
    sg->node[ p ].pathval = pathval[ p ];
    for ( k = 1; k < A->n; k++ ) { // image space restriction
      v.x = u.x + A->dx[k];
      v.y = u.y + A->dy[k];
      v.z = u.z + A->dz[k];
      q = VoxelAddress( scn, v.x, v.y, v.z );
      if ( ( ValidVoxel( scn, v.x, v.y, v.z ) ) && ( scn->data[ q ] != 0 ) ) {
	dist = DistFun(sg->node[p].feat,sg->node[q].feat,sg->nfeats);
	if ( ( dist <= sg->df ) && ( pathval[ p ] > pathval[ q ] ) ) {
	  tmp = MIN( pathval[ p ], sg->node[ q ].dens);
	  if ( tmp > pathval[ q ] ) {
	    UpdateRealHeap(Q,q,tmp);
	    sg->node[ q ].pred  = p;
	  }
	} else {
	  if ( pathval[ p ] == pathval[ q ] ) {
	    rq = RootNode(sg,q,&flatzone_q);
	    if ((rp != rq)&&
		(flatzone_p)&&
		(flatzone_q)&&
		(sg->node[rp].dens==sg->node[rq].dens)){
	      sg->node[rq].pred = rp;
	    }
	  }
	}
      }
    }
  }
  DestroyRealHeap( &Q );
  free( pathval );

  l = 1;
  for ( p = 0; p < sg->nnodes; p++ ) {
    if ( ( sg->node[p].pred == NIL ) && ( scn->data[ p ] != 0 ) ) {
      sg->node[p].label = l;
      l++;
    }
  }
  for ( p = 0; p < sg->nnodes; p++ ) { // label propagation
    rp = RootNode(sg,p,&flatzone_p);
    sg->node[p].label = sg->node[rp].label;
  }
  sg->nlabels = l;
  DestroyAdjRel3( &A );
}
*/

// Influence zones of the maxima with their true labels
void SupOPF(Subgraph *sg)
{
    int p, q;
    RealHeap *Q=NULL;
    Set *Saux=NULL;
    float tmp, *pathval = NULL;

    pathval = AllocFloatArray(sg->nnodes);
    Q = CreateRealHeap(sg->nnodes, pathval);
    SetRemovalPolicyRealHeap(Q, MAXVALUE);

    for (p = 0; p < sg->nnodes; p++)
    {
        pathval[ p ] = sg->node[ p ].pathval;
        sg->node[ p ].pred    = NIL;
        InsertRealHeap(Q, p);
    }

    while ( !IsEmptyRealHeap(Q) )
    {
        RemoveRealHeap(Q,&p);

        if ( sg->node[ p ].pred == NIL )
        {
            pathval[ p ]  = sg->node[ p ].dens;
            sg->node[p].label = sg->node[p].truelabel;
        }

        sg->node[ p ].pathval = pathval[ p ];

        for ( Saux = sg->node[ p ].adj; Saux != NULL; Saux = Saux->next )
        {
            q = Saux->elem;
            if ( pathval[ p ] > pathval[ q ] )
            {
                if (sg->node[p].truelabel!=sg->node[q].truelabel)
                    tmp = INT_MIN;
                else
                    tmp = MIN( pathval[ p ], sg->node[ q ].dens );
                if ( tmp > pathval[ q ] )
                {
                    UpdateRealHeap(Q,q,tmp);
                    sg->node[ q ].pred   = p;
                    sg->node[q].label = sg->node[p].label;
                }
            }
        }
    }
    DestroyRealHeap( &Q );
    free( pathval );

    // Verify if error is really zero in training set

    for (p=0; p < sg->nnodes; p++)
        if (sg->node[p].label!=sg->node[p].truelabel)
        {
            Warning("Class. error in the training set","SupOPF");
        }

}

// Influence zones of the maxima with their true labels

void SupOPFwithErrors(Subgraph *sg)
{
    int p, q;
    RealHeap *Q=NULL;
    Set *Saux=NULL;
    float tmp, *pathval = NULL;

    pathval = AllocFloatArray(sg->nnodes);
    Q = CreateRealHeap(sg->nnodes, pathval);
    SetRemovalPolicyRealHeap(Q, MAXVALUE);

    for (p = 0; p < sg->nnodes; p++)
    {
        pathval[ p ] = sg->node[ p ].pathval;
        sg->node[ p ].pred    = NIL;
        InsertRealHeap(Q, p);
    }

    while ( !IsEmptyRealHeap(Q) )
    {
        RemoveRealHeap(Q,&p);

        if ( sg->node[ p ].pred == NIL )
        {
            pathval[ p ]  = sg->node[ p ].dens;
            sg->node[p].label = sg->node[p].truelabel;
        }

        sg->node[ p ].pathval = pathval[ p ];

        for ( Saux = sg->node[ p ].adj; Saux != NULL; Saux = Saux->next )
        {
            q = Saux->elem;
            if ( pathval[ p ] > pathval[ q ] )
            {
                tmp = MIN( pathval[ p ], sg->node[ q ].dens );
                if ( tmp > pathval[ q ] )
                {
                    UpdateRealHeap(Q,q,tmp);
                    sg->node[ q ].pred   = p;
                    sg->node[q].label = sg->node[p].label;
                }
            }
        }
    }
    DestroyRealHeap( &Q );
    free( pathval );
}

// Influence zones of the prototypes

void SemiSupOPF(Subgraph *sg)
{
    int p, q;
    RealHeap *Q = NULL;
    Set *Saux=NULL;
    float tmp, *pathval = NULL;

    pathval = AllocFloatArray(sg->nnodes);
    Q = CreateRealHeap(sg->nnodes, pathval);
    SetRemovalPolicyRealHeap(Q, MAXVALUE);

    for (p = 0; p < sg->nnodes; p++)
    {
        if (sg->node[p].status==opf_PROTOTYPE)
        {
            pathval[ p ] = sg->node[p].dens;
            sg->node[ p ].pred    = NIL;
            sg->node[ p ].label   = sg->node[p].truelabel;
            InsertRealHeap(Q, p);
        }
        else
            sg->node[p].pathval = pathval[p]= INT_MIN;
    }

    while ( !IsEmptyRealHeap(Q) )
    {
        RemoveRealHeap(Q,&p);
        sg->node[ p ].pathval = pathval[ p ];

        for ( Saux = sg->node[ p ].adj; Saux != NULL; Saux = Saux->next )
        {
            q = Saux->elem;
            if ( pathval[ p ] > pathval[ q ] )
            {
                tmp = MIN( pathval[ p ], sg->node[ q ].dens );
                if ( tmp > pathval[ q ] )
                {
                    UpdateRealHeap(Q,q,tmp);
                    sg->node[ q ].pred   = p;
                    sg->node[ q ].label   = sg->node[p].label;
                }
            }
        }
    }

    for (p = 0; p < sg->nnodes; p++)
    {
        if ( pathval[ p ] == INT_MIN )
        {
            sg->node[p].pathval = 0;
            sg->node[p].label   = NIL;
            sg->node[p].pred    = NIL;
        }
    }

    DestroyRealHeap( &Q );
    free( pathval );
}

// Find root node and identify if the path is on a flatzone

int RootNode(Subgraph *sg, int p, char *flatzone)
{
    float density=sg->node[p].dens;

    *flatzone=1;
    while (sg->node[p].pred!=NIL)
    {
        if (sg->node[p].dens!=density)
            *flatzone=0;
        p = sg->node[p].pred;
    }
    return(p);
}

// Compute prototypes by setting their label as true label. Other
// nodes have their label set to NIL. It also computes maximum arc
// weight in sg->df for the subsequent IFT.

//void MSTPrototypes(Subgraph *sg)
//{
//    int p,q;
//    int weight;
//    GQueue *Q=NULL;
//    int *pathval = NULL;
//    int  pred;
//    float nproto, dist;
//
//    // Compute maximum distance to avoid growing the queue
//
//    sg->df=0.0;
//    for (p=0; p < sg->nnodes; p++)
//    {
//        for (q=0; q < sg->nnodes; q++)
//        {
//            if (p!=q)
//            {
//                if (!opf_PrecomputedDistance)
//                    dist = DistFun(sg->node[p].feat,sg->node[q].feat,sg->nfeats);
//                else
//                    dist = opf_DistanceValue[sg->node[p].pixel][sg->node[q].pixel];
//                if (dist>sg->df)
//                    sg->df=dist;
//            }
//        }
//    }
//
//    // initialization
//    pathval = AllocIntArray(sg->nnodes);
//    Q = CreateGQueue(opf_ArcWeight(sg->df, sg->df)+1,sg->nnodes, pathval);
//
//    for (p = 0; p < sg->nnodes; p++)
//    {
//        pathval[ p ] = INT_MAX;
//        sg->node[p].status=0;
//    }
//
//    pathval[0]  = 0;
//    sg->node[0].pred = NIL;
//    InsertGQueue(&Q, 0);
//
//    nproto=0.0;
//    // Prim's algorithm for Minimum Spanning Tree
//    while ( !EmptyGQueue(Q) )
//    {
//        p = RemoveGQueue(Q);
//
//        sg->node[p].pathval = pathval[p];
//
//        pred=sg->node[p].pred;
//        if (pred!=NIL)
//            if (sg->node[p].truelabel != sg->node[pred].truelabel)
//            {
//                if (sg->node[p].status!=opf_PROTOTYPE)
//                {
//                    sg->node[p].status=opf_PROTOTYPE;
//                    nproto++;
//                }
//                if (sg->node[pred].status!=opf_PROTOTYPE)
//                {
//                    sg->node[pred].status=opf_PROTOTYPE;
//                    nproto++;
//                }
//            }
//
//        for (q=0; q < sg->nnodes; q++)
//        {
//            if (Q->L.elem[q].color!=BLACK)
//            {
//                if (p!=q)
//                {
//                    if (!opf_PrecomputedDistance)
//                        weight = (int)opf_ArcWeight(DistFun(sg->node[p].feat,sg->node[q].feat,sg->nfeats), sg->df);
//                    else
//                        weight = (int)opf_ArcWeight(opf_DistanceValue[sg->node[p].pixel][sg->node[q].pixel], sg->df);
//                    if ( weight < pathval[ q ] )
//                    {
//                        if (pathval[q]!=INT_MAX)
//                            RemoveGQueueElem(Q,q);
//                        sg->node[q].pred = p;
//                        pathval[q]=weight;
//                        InsertGQueue(&Q, q);
//                    }
//                }
//            }
//        }
//    }
//    DestroyGQueue(&Q);
//    free( pathval );
//
//    //printf("Perc of prototypes %f\n",nproto/sg->nnodes);
//
//}

// Compute Euclidean distance between feature vectors

///float EuclDist(float *f1, float *f2, int n)
//{
//    int i;
//    float dist=0.0f;
//
//    for (i=0; i < n; i++)
//        dist += (f1[i]-f2[i])*(f1[i]-f2[i]);
//    return(dist);
//}

// Compute Jaccard distance between feature vectors

float JaccDist(float *f1, float *f2, int n)
{
    int i;
    float dist=0.0f;

    for (i=0; i < n; i++)
    {
        if (f1[i] == f2[i]);
        dist += 1.0;
    }
    dist = 1.0 - (dist/((float)(2*n) - dist));

    return dist;
}

// Compute Extended Jaccard (Tanimoto) distance between feature vectors

float ExtJaccDist(float *f1, float *f2, int n)
{
    int i;
    float dist=0.0f, dot = 0.0f, sf1 = 0.0f, sf2 = 0.0f;

    for (i=0; i < n; i++)
    {
        dot += f1[i] + f2[i];
        sf1 += f1[i] * f1[i];
        sf2 += f2[i] * f2[i];
    }
    dist = dot / (sf1 + sf2 - dot);

    return dist;
}


// Compute Jaccard distance between feature vectors

float CosineDist(float *f1, float *f2, int n)
{
    int i;
    float  dist=0.0f, dot = 0.0f, sf1 = 0.0f, sf2 = 0.0f;

    for (i=0; i < n; i++)
    {
        dot += f1[i] * f2[i];
        sf1 += f1[i] * f1[i];
        sf2 += f2[i] * f2[i];
    }
    dist = dot/sqrt(sf1*sf2);
    return dist;

}



// Compute Pearson's correlation distance between feature vectors
// Warning: must redefine f1_mean, f2_mean, f1_stddev and f2_stddev

float CorrelationDist(float *f1, float *f2, int n)
{
    /*
    int i, dist = 0.0f, cov = 0.0f;
    if (f1_mean !=NULL && f2_mean != NULL) {
      for (i=0; i < n; i++) {
        cov += (f1[i] - f1_mean[i]) * (f2[i] - f2_mean[i]);
      }
      dist = cov/((float)(n-1)*f1_stddev*f2_stddev);
    } else {
      dist = CosineDist(f1,f2,n);
    }
    */
    return CosineDist(f1,f2,n);
}

// Compute  chi-squared distance between feature vectors

///float ChiSquaredDist(float *f1, float *f2, int n)
//{
//    int i;
//    float dist=0.0f, sf1 = 0.0f, sf2 = 0.0f;
//
//    for (i = 0; i < n; i++)
//    {
//        sf1+=f1[i];
//        sf2+=f2[i];
//    }
//
//    for (i=0; i < n; i++)
//        dist += 1/(f1[i]+f2[i]+0.000000001)*pow(f1[i]/sf1-f2[i]/sf2,2);
//
//    return(sqrtf(dist));
//}

// Compute  Manhattan distance between feature vectors

///float ManhattanDist(float *f1, float *f2, int n)
//{
//    int i;
//    float dist=0.0f;
//
//    for (i=0; i < n; i++)
//        dist += fabs(f1[i]-f2[i]);
//
//    return(dist);
//}

// Compute  Camberra distance between feature vectors

///float CanberraDist(float *f1, float *f2, int n)
//{
//    int i;
//    float dist=0.0f, aux;
//
//    for (i=0; i < n; i++)
//    {
//        aux = fabs(f1[i]+f2[i]);
//        if (aux > 0)
//            dist += (fabs(f1[i]-f2[i])/aux);
//    }
//
//    return(dist);
//}

// Compute  Squared Chord between feature vectors
///float SquaredChordDist(float *f1, float *f2, int n)
//{
//    int i;
//    float dist=0.0f, aux1, aux2;
//
//    for (i=0; i < n; i++)
//    {
//        aux1 = sqrtf(f1[i]);
//        aux2 = sqrtf(f2[i]);
//
//        if ((aux1 >= 0) && (aux2 >=0))
//            dist += pow(aux1-aux2,2);
//    }
//
//    return(dist);
//}

// Compute  Squared Chi-squared between feature vectors
///float SquaredChiSquaredDist(float *f1, float *f2, int n)
//{
//    int i;
//    float dist=0.0f, aux;
//
//    for (i=0; i < n; i++)
//    {
//        aux = fabs(f1[i]+f2[i]);
//        if (aux > 0)
//            dist += (pow(f1[i]-f2[i],2)/aux);
//    }
//
//    return(dist);
//}

// Compute  Bray Curtis distance between feature vectors

///float BrayCurtisDist(float *f1, float *f2, int n)
//{
//    int i;
//    float dist=0.0f, aux;
//
//    for (i=0; i < n; i++)
//    {
//        aux = f1[i]+f2[i];
//        if (aux > 0)
//            dist += (fabs(f1[i]-f2[i])/aux);
//    }
//
//    return(dist);
//}

// Create adjacent list in subgraph: a knn graph
///void CreateArcs(Subgraph *sg, int knn) {
//  int    i,j,l,k;
//  float  dist;
//  int   *nn=AllocIntArray(knn+1);
//  float *d=AllocFloatArray(knn+1);
//
//  /* Create graph with the knn-nearest neighbors */
//
//  sg->df=0.0;
//  for (i=0; i < sg->nnodes; i++) {
//    for (l=0; l < knn; l++)
//      d[l]=REAL_MAX;
//    for (j=0; j < sg->nnodes; j++) {
//      if (j!=i) {
//	if (!opf_PrecomputedDistance)
//	  d[knn] = DistFun(sg->node[i].feat,sg->node[j].feat,sg->nfeats);
//	else
//	  d[knn] = opf_DistanceValue[sg->node[i].pixel][sg->node[j].pixel];
//	nn[knn]= j;
//	k      = knn;
//	while ((k > 0)&&(d[k]<d[k-1])) {
//	  dist    = d[k];
//	  l       = nn[k];
//	  d[k]    = d[k-1];
//	  nn[k]   = nn[k-1];
//	  d[k-1]  = dist;
//	  nn[k-1] = l;
//	  k--;
//	}
//      }
//    }
//    for (l=0; l < knn; l++) {
//      if (d[l]!=REAL_MAX) {
//	if (d[l] > sg->df)
//	  sg->df = d[l];
//	sg->node[i].radius = d[l];
//	InsertSet(&(sg->node[i].adj),nn[l]);
//      }
//    }
//  }
//  free(d);
//  free(nn);
//
//  if (sg->df<0.00001)
//    sg->df = 1.0;
//}

// Create adjacent list in subgraph: a knn graph with arcs between the
// correct classes only.
void CreateArcsByTrueLabel(Subgraph *sg, int knn)
{
    int    i,j,l,k;
    float  dist;
    int   *nn=AllocIntArray(knn+1);
    float *d=AllocFloatArray(knn+1);

    /* Create graph with the knn-nearest neighbors */

    sg->df=0.0;
    for (i=0; i < sg->nnodes; i++)
    {
        for (l=0; l < knn; l++)
            d[l]=REAL_MAX;
        for (j=0; j < sg->nnodes; j++)
        {
            if ((j!=i)&&(sg->node[i].truelabel==sg->node[j].truelabel))
            {
                if (!opf_PrecomputedDistance)
                    d[knn] = opf_ArcWeight(sg->node[i].feat,sg->node[j].feat,sg->nfeats);
                else
                    d[knn] = opf_DistanceValue[sg->node[i].pixel][sg->node[j].pixel];
                nn[knn]= j;
                k      = knn;
                while ((k > 0)&&(d[k]<d[k-1]))
                {
                    dist    = d[k];
                    l       = nn[k];
                    d[k]    = d[k-1];
                    nn[k]   = nn[k-1];
                    d[k-1]  = dist;
                    nn[k-1] = l;
                    k--;
                }
            }
        }
        //sorting the adjacent nodes in non-decreasing order
        for (l=0; l < knn; l++)
        {
            if (d[l]!=REAL_MAX)
            {
                if (d[l] > sg->df)
                    sg->df = d[l];
                //the adjacent nodes are inserted at the beginning of the list
                InsertSet(&(sg->node[i].adj),nn[l]);
            }
        }
    }
    free(d);
    free(nn);

    if (sg->df<0.00001)
        sg->df = 1.0;
}

void CreateSymArcs(Subgraph *sg)
{
    int    i,j;
    float  dist;

    /* Create graph with neighbors closer than sg->df */

    for (i=0; i < sg->nnodes; i++)
    {
        for (j=0; j < sg->nnodes; j++)
        {
            if (i!=j)
            {
                dist = opf_ArcWeight(sg->node[i].feat,sg->node[j].feat,sg->nfeats);
                if (dist <= sg->df)
                    InsertSet(&(sg->node[i].adj),j);
            }
        }
    }
}

//void DestroyArcs(Subgraph *sg)
//{
//    int i;
//
//    for (i=0; i < sg->nnodes; i++)
//        DestroySet(&(sg->node[i].adj));
//}

//void PDF(Subgraph *sg)
//{
//    int     i,nelems;
//    double  dist;
//    float  *value=AllocFloatArray(sg->nnodes);
//    Set    *adj=NULL;
//
//    sg->K    = (2.0*(float)sg->df/9.0);
////    sg->K1    = ((float)1.0/sqrt(2.0*PI*(float)sg->df/9.0));
//
//    sg->mindens = INT_MAX;
//    sg->maxdens = INT_MIN;
//    for (i=0; i < sg->nnodes; i++)
//    {
//        adj=sg->node[i].adj;
//        value[i]=0.0;
//        nelems=1;
//        while (adj != NULL)
//        {
//            if (!opf_PrecomputedDistance)
//                dist = DistFun(sg->node[i].feat,sg->node[adj->elem].feat,sg->nfeats);
//            else
//                dist = opf_DistanceValue[sg->node[i].pixel][sg->node[adj->elem].pixel];
//            value[i] += exp(-dist/sg->K);
//            adj = adj->next;
//            nelems++;
//        }
//
//        value[i] = (value[i]/(float)nelems);
//
//        if (value[i] < sg->mindens)
//            sg->mindens = value[i];
//        if (value[i] > sg->maxdens)
//            sg->maxdens = value[i];
//    }
//
//    //  printf("df=%f,K1=%f,K2=%f,mindens=%f, maxdens=%f\n",sg->df,sg->K1,sg->K2,sg->mindens,sg->maxdens);
//
//    if (sg->mindens==sg->maxdens)
//    {
//        for (i=0; i < sg->nnodes; i++)
//        {
//	  sg->node[i].dens = opf_MAXDENS;
//	  sg->node[i].pathval= opf_MAXDENS-1;
//        }
//    }
//    else
//    {
//        for (i=0; i < sg->nnodes; i++)
//        {
//	  sg->node[i].dens = ((float)(opf_MAXDENS-1)*(value[i]-sg->mindens)/(float)(sg->maxdens-sg->mindens))+1.0;
//	  sg->node[i].pathval=sg->node[i].dens-1;
//        }
//    }
//    free(value);
//}

// Compute PDF for the supervised case

void SupPDF(Subgraph *sg)
{
    int     i,j,nelems;
    double  dist;
    float  *value=AllocFloatArray(sg->nnodes),*r=AllocFloatArray(sg->nnodes);

    // Find the radius for each node as its closest distance to a
    // different class and the normalized factor

    sg->df = 0.0;
    for (i=0; i < sg->nnodes; i++) {
      r[i]=FLT_MAX;
      for (j=0; j < sg->nnodes; j++) {
	if ((i != j)&&(sg->node[i].truelabel != sg->node[j].truelabel)){
	  if (!opf_PrecomputedDistance)
	    dist = opf_ArcWeight(sg->node[i].feat,sg->node[j].feat,sg->nfeats);
	  else
	    dist = opf_DistanceValue[sg->node[i].pixel][sg->node[j].pixel];
	  if (dist < r[i])
	    r[i]=dist;
	}
      }
      if (r[i]>sg->df)
	sg->df = r[i];
    }


    // Compute PDF

    sg->K    = (2.0*(float)sg->df/9.0);
//    sg->K1    = ((float)1.0/sqrt(2.0*PI*(float)sg->df/9.0));

    sg->mindens = INT_MAX;
    sg->maxdens = INT_MIN;

    for (i=0; i < sg->nnodes; i++)
    {
      value[i]=0.0;
      nelems=1;

      for (j=0; j < sg->nnodes; j++)
	{
	  if ((i != j)&&(sg->node[i].truelabel == sg->node[j].truelabel)){
	    if (!opf_PrecomputedDistance)
	      dist = opf_ArcWeight(sg->node[i].feat,sg->node[j].feat,sg->nfeats);
	    else
	      dist = opf_DistanceValue[sg->node[i].pixel][sg->node[j].pixel];
	    if (dist <= r[i]){
	      value[i] += exp(-dist/sg->K);
	      nelems++;
	      InsertSet(&(sg->node[i].adj),j);
	    }
	  }
	}
      value[i] = (value[i]/(float)nelems);

      if (value[i] < sg->mindens)
	sg->mindens = value[i];
      if (value[i] > sg->maxdens)
	sg->maxdens = value[i];
    }

    //    printf("df=%f,K1=%f,K2=%f,mindens=%f, maxdens=%f\n",sg->df,sg->K1,sg->K2,sg->mindens,sg->maxdens);

    if (sg->mindens==sg->maxdens)
    {
        for (i=0; i < sg->nnodes; i++)
        {
	  sg->node[i].dens = opf_MAXDENS;
	  sg->node[i].pathval= opf_MAXDENS-1;
        }
    }
    else
      {
        for (i=0; i < sg->nnodes; i++)
	  {
	    sg->node[i].dens = ((float)(opf_MAXDENS-1)*(value[i]-sg->mindens)/(float)(sg->maxdens-sg->mindens))+1.0;
	    sg->node[i].pathval=sg->node[i].dens-1;
	  }
      }
    free(value);    free(r);
}

// PDF with spatial constraint

void SPDF(Subgraph *sg, Image *img)
{
    int     i,j,k,nelems;
    float   dist;
    float  *value=AllocFloatArray(sg->nnodes);
    Pixel   u,v;
    AdjRel *A=Circular(sg->di);

    sg->K    = (2.0*(float)sg->df/9.0);
//    sg->K1    = ((float)1.0/sqrt(2.0*PI*(float)sg->df/9.0));

    sg->mindens = INT_MAX;
    sg->maxdens = INT_MIN;
    for (i=0; i < sg->nnodes; i++)
    {
        value[i]=0.0;
        nelems=1;
        u.x = i%img->ncols;
        u.y = i/img->ncols;
        for (k=1; k < A->n; k++)   // image space restriction
        {
            v.x = u.x + A->dx[k];
            v.y = u.y + A->dy[k];
            if (ValidPixel(img,v.x,v.y))
            {
                j = v.x+img->tbrow[v.y];
                dist = opf_ArcWeight(sg->node[i].feat,sg->node[j].feat,sg->nfeats);
                if (dist <= sg->df)
                {
                    value[i] += exp(-dist/sg->K);
                    InsertSet(&(sg->node[i].adj),j);
                    nelems++;
                }
            }
        }
        value[i] = value[i]/(float)nelems;
        if (value[i] < sg->mindens)
            sg->mindens = value[i];
        if (value[i] > sg->maxdens)
            sg->maxdens = value[i];
    }

    if (sg->mindens==sg->maxdens)
        sg->mindens=0;

    for (i=0; i < sg->nnodes; i++)
    {
        sg->node[i].dens = ((opf_MAXDENS-1)*(value[i]-sg->mindens)/(sg->maxdens-sg->mindens))+1;
        sg->node[i].pathval=sg->node[i].dens-1;
    }
    DestroyAdjRel(&A);
    free(value);
}

void SPDF3( Subgraph *sg, Scene *mask ) {
  int      i, j, k, nelems;
  float    dist;
  float    *value = AllocFloatArray( sg->nnodes );
  Voxel    u,v;
  AdjRel3  *A = Spheric( sg->di );

  sg->df = 0.0;
  for( i = 0; i < sg->nnodes; i++ ) {
    if( mask->data[ i ] != 0 ) {
      u.x = VoxelX( mask, i );
      u.y = VoxelY( mask, i );
      u.z = VoxelZ( mask, i );
      for( k = 1; k < A->n; k++ ) {
	v.x = u.x + A->dx[ k ];
	v.y = u.y + A->dy[ k ];
	v.z = u.z + A->dz[ k ];
	if( ValidVoxel( mask, v.x, v.y, v.z ) ) {
	  j = VoxelAddress( mask, v.x, v.y, v.z );
	  if( mask->data[ j ] != 0 ) {
	    dist = opf_ArcWeight( sg->node[ i ].feat, sg->node[ j ].feat, sg->nfeats );
	    if( dist > sg->df )
	      sg->df = dist;
	  }
	}
      }
    }
  }

  sg->K = ( 2.0 * ( float ) sg->df / 9.0 );
  
  sg->mindens = INT_MAX;
  sg->maxdens = INT_MIN;
  for( i = 0; i < sg->nnodes; i++ ) {
    sg->node[ i ].dens = 0.0;
    if( mask->data[ i ] != 0 ) {
      nelems = 1;
      value[ i ] = 0.0;
      u.x = VoxelX( mask, i );
      u.y = VoxelY( mask, i );
      u.z = VoxelZ( mask, i );
      for( k = 1; k < A->n; k++ ) {  // image space restriction
	v.x = u.x + A->dx[ k ];
	v.y = u.y + A->dy[ k ];
	v.z = u.z + A->dz[ k ];
	j = VoxelAddress( mask, v.x, v.y, v.z );
	if( ( ValidVoxel( mask, v.x, v.y, v.z ) ) && ( mask->data[ j ] != 0 ) ) {
	  dist = opf_ArcWeight( sg->node[ i ].feat, sg->node[ j ].feat, sg->nfeats );
	  if( dist <= sg->df ) {
	    nelems++;
	    value[ i ] += exp( -dist / sg->K );
	  }
	}
      }
      value[ i ] = value[ i ] / ( float ) nelems;
      if ( value[ i ] < sg->mindens )
	sg->mindens = value[ i ];
      if ( value[ i ] > sg->maxdens )
	sg->maxdens = value[ i ];
    }
  }
  
  if( sg->mindens == sg->maxdens )
    sg->mindens = 0;
  
  printf( "maxdens:%f, mindens:%f\n", sg->maxdens, sg->mindens );
  for ( i = 0; i < sg->nnodes; i++ ) {
    sg->node[ i ].dens = ( ( opf_MAXDENS - 1 ) * value[ i ] - sg->mindens / ( sg->maxdens - sg->mindens ) ) + 1;
    sg->node[ i ].pathval = sg->node[ i ].dens - 1;
  }

  DestroyAdjRel3( &A );
  free( value );
}

CNode *CreateCNode(Subgraph *sg)
{
    CNode *node=(CNode *)calloc(1,sizeof(CNode));

    node->feat = AllocFloatArray(sg->nfeats);
    node->adj  = AllocIntArray(sg->bestk+1);
    node->dist = AllocFloatArray(sg->bestk+1);

    return(node);
}

void DestroyCNode(CNode **node)
{
    if ((*node)!=NULL)
    {
        free((*node)->feat);
        free((*node)->adj);
        free((*node)->dist);
        free((*node));
        *node = NULL;
    }
}

void MarkovNodeFeat3(Scene *scn, Scene *mask, CNode *node, int p, float dm, int Imax)
{
    int     q,j,xysize=scn->xsize*scn->ysize,aux;
    Voxel   u,v;
    AdjRel3 *A;

    node->pixel = p;

    if (dm==0.0)
        node->feat[0]=scn->data[p]/(float)Imax;
    else
    {
        A = Spheric(dm);
        aux = p%xysize;
        u.x = aux%scn->xsize;
        u.y = aux/scn->xsize;
        u.z = p/xysize;
        for (j=0; j < A->n; j++)
        {
            v.x = u.x + A->dx[j];
            v.y = u.y + A->dy[j];
            v.z = u.z + A->dz[j];
            q = v.x + scn->tby[v.y] + scn->tbz[v.z];
            if ( ( ValidVoxel( scn, v.x, v.y, v.z ) ) && ( ( mask == NULL ) || ( mask->data[ q ] != 0 ) ) )
            {
                node->feat[j]=scn->data[q]/(float)Imax;
            }
            else
            {
                node->feat[j]=scn->data[p]/(float)Imax;
            }
        }
        DestroyAdjRel3(&A);
    }
}

void CopyNodeFeat3( Features3 *f, CNode *node, int p )
{
    int i;
    node->pixel = p;
    for ( i = 0; i < f->nfeats; i++ )
    {
        node->feat[ i ] = f->elem[ p ].feat[ i ];
    }
}

void NodeArcs(Subgraph *sg, CNode *node)
{
    int    j,l,k;
    float  dist;

    for (l=0; l <= sg->bestk; l++)
        node->dist[l]=REAL_MAX;

    for (j=0; j < sg->nnodes; j++)
    {
        if (!opf_PrecomputedDistance)
            node->dist[sg->bestk] = opf_ArcWeight(node->feat,sg->node[j].feat,sg->nfeats);
        else
            node->dist[sg->bestk] = opf_DistanceValue[node->pixel][sg->node[j].pixel];
        node->adj[sg->bestk]  = j;
        k                     = sg->bestk;
        while ((k > 0)&&(node->dist[k]<node->dist[k-1]))
        {
            dist            = node->dist[k];
            l               = node->adj[k];
            node->dist[k]   = node->dist[k-1];
            node->adj[k]    = node->adj[k-1];
            node->dist[k-1] = dist;
            node->adj[k-1]   = l;
            k--;
        }
    }
}

void NodeArcsInSubforest(Subgraph *sg, CNode *node)
{
    int    j,l,k;
    float  dist;

    for (l=0; l <= sg->bestk; l++)
        node->dist[l]=REAL_MAX;

    for (j=0; j < sg->nnodes; j++)
    {

        if (sg->node[j].label!=0)
        {

            if (!opf_PrecomputedDistance)
                node->dist[sg->bestk] = opf_ArcWeight(node->feat,sg->node[j].feat,sg->nfeats);
            else
                node->dist[sg->bestk] = opf_DistanceValue[node->pixel][sg->node[j].pixel];
            node->adj[sg->bestk]  = j;
            k                     = sg->bestk;
            while ((k > 0)&&(node->dist[k]<node->dist[k-1]))
            {
                dist            = node->dist[k];
                l               = node->adj[k];
                node->dist[k]   = node->dist[k-1];
                node->adj[k]    = node->adj[k-1];
                node->dist[k-1] = dist;
                node->adj[k-1]   = l;
                k--;
            }
        }
    }
}

void NodeArcsByLabel(Subgraph *sg, CNode *node, int label)
{
    int    j,l,k;
    float  dist;

    for (l=0; l <= sg->bestk; l++)
        node->dist[l]=REAL_MAX;

    for (j=0; j < sg->nnodes; j++)
    {
        if (sg->node[j].label != label)
            continue;

        if (!opf_PrecomputedDistance)
            node->dist[sg->bestk] = opf_ArcWeight(node->feat,sg->node[j].feat,sg->nfeats);
        else
            node->dist[sg->bestk] = opf_DistanceValue[node->pixel][sg->node[j].pixel];

        node->adj[sg->bestk]  = j;
        k                     = sg->bestk;
        while ((k > 0)&&(node->dist[k]<node->dist[k-1]))
        {
            dist            = node->dist[k];
            l               = node->adj[k];
            node->dist[k]   = node->dist[k-1];
            node->adj[k]    = node->adj[k-1];
            node->dist[k-1] = dist;
            node->adj[k-1]   = l;
            k--;
        }
    }
}


void NodeArcsByTrueLabel(Subgraph *sg, CNode *node, int label)
{
    int    j,l,k;
    float  dist;

    for (l=0; l <= sg->bestk; l++)
    {
        node->dist[l]=REAL_MAX;
    }
    for (j=0; j < sg->nnodes; j++)
    {
        if (sg->node[j].truelabel != label) continue;
        if (!opf_PrecomputedDistance) node->dist[sg->bestk] = opf_ArcWeight(node->feat,sg->node[j].feat,sg->nfeats);
        else node->dist[sg->bestk] = opf_DistanceValue[node->pixel][sg->node[j].pixel];
        node->adj[sg->bestk]  = j;
        k                     = sg->bestk;
        while ((k > 0)&&(node->dist[k]<node->dist[k-1]))
        {
            dist            = node->dist[k];
            l               = node->adj[k];
            node->dist[k]   = node->dist[k-1];
            node->adj[k]    = node->adj[k-1];
            node->dist[k-1] = dist;
            node->adj[k-1]   = l;
            k--;
        }
    }
}

void NodePD(Subgraph *sg, CNode *node)
{
    int     i,nelems;
    float   dist;
    double  value;

    value=0.0;
    nelems=1;
    for (i=0; i < sg->bestk; i++)
    {
        if (!opf_PrecomputedDistance)
            dist = opf_ArcWeight(node->feat,sg->node[node->adj[i]].feat,sg->nfeats);
        else
            dist = opf_DistanceValue[node->pixel][sg->node[node->adj[i]].pixel];
        value += exp(-dist/sg->K);
        nelems++;
    }

    value = value/(float)nelems;

    if (value>sg->maxdens)
        value=sg->maxdens;
    if (value<sg->mindens)
        value=sg->mindens;

    node->dens = (opf_MAXDENS-1)*(value-sg->mindens)/(sg->maxdens-sg->mindens)+1;

}

void NodePDByLabel(Subgraph *sg, CNode *node, int label)
{
    int     i,nelems;
    float   dist;
    double  value;

    value=0.0;
    nelems=1;
    for (i=0; i < sg->bestk; i++)
    {
        if (sg->node[node->adj[i]].label==label)
        {
            if (!opf_PrecomputedDistance)
                dist = opf_ArcWeight(node->feat,sg->node[node->adj[i]].feat,sg->nfeats);
            else
                dist = opf_DistanceValue[node->pixel][sg->node[node->adj[i]].pixel];

            value += exp(-dist/sg->K);
            nelems++;
        }
    }

    value = value/(float)nelems;
    if (value>sg->maxdens)
        value=sg->maxdens;
    if (value<sg->mindens)
        value=sg->mindens;

    node->dens = ((opf_MAXDENS-1)*(value-sg->mindens)/(sg->maxdens-sg->mindens))+1;
}

void NodePDByTrueLabel(Subgraph *sg, CNode *node, int label)
{
    int     i,nelems;
    float   dist;
    double  value;

    value=0.0;
    nelems=1;
    for (i=0; i < sg->bestk; i++)
    {
        if (sg->node[node->adj[i]].truelabel==label)
        {
            if (!opf_PrecomputedDistance)
                dist = opf_ArcWeight(node->feat,sg->node[node->adj[i]].feat,sg->nfeats);
            else
                dist = opf_DistanceValue[node->pixel][sg->node[node->adj[i]].pixel];
            value += exp(-dist/sg->K);
            nelems++;
        }
    }

    value = value/(float)nelems;
    if (value>sg->maxdens)
        value=sg->maxdens;
    if (value<sg->mindens)
        value=sg->mindens;
    node->dens = ((opf_MAXDENS-1)*(value-sg->mindens)/(sg->maxdens-sg->mindens))+1;

}

int ValueBestPath(Subgraph *sg, CNode *node)
{
    int maxpathval, i, tmp;

    maxpathval = MIN( node->dens, sg->node[ node->adj[ 0 ] ].pathval );
    for ( i = 1; i < sg->bestk; i++ )
    {
        tmp = MIN( node->dens, sg->node[ node->adj[ i ] ].pathval );
        if ( tmp > maxpathval )
        {
            maxpathval = tmp;
        }
    }
    return(maxpathval);
}

int PredBestPath(Subgraph *sg, CNode *node)
{
    int maxpathval, i, pred, tmp;

    maxpathval = MIN( node->dens, sg->node[ node->adj[ 0 ] ].pathval );
    pred       = node->adj[ 0 ];
    for ( i = 1; i < sg->bestk; i++ )
    {
        tmp = MIN( node->dens, sg->node[ node->adj[ i ] ].pathval );
        if ( tmp > maxpathval )
        {
            pred = node->adj[ i ];
            maxpathval = tmp;
        }
    }
    return(pred);
}

int PredBestPathBayesTie(Subgraph *sg, CNode *node)
{
    int maxpathval, i, pred, tmp, label;
    float P[sg->nlabels+1], p[sg->nlabels+1];

    for (i=0; i < sg->nlabels; i++)
    {
        P[ i ] = 0.0;
        p[ i ] = 0.0;
    }

    for (i=0; i < sg->nnodes; i++)
    {
        P[ sg->node[i].label ] += 1.0;
    }

    for (i=0; i < sg->bestk; i++)
    {
        p[ sg->node[ node->adj[ i ] ].label ] += exp( -node->dist[ i ] / sg->K );
    }

    // find knn node with minimum path-cost using Fmax function
    maxpathval = MIN( node->dens, sg->node[ node->adj[ 0 ] ].pathval );
    pred       = node->adj[ 0 ];
    label      = sg->node[ node->adj[ 0 ] ].label;
    for ( i = 1; i < sg->bestk; i++ )
    {
        tmp = MIN( node->dens, sg->node[ node->adj[ i ] ].pathval );
        if ( tmp > maxpathval )
        {
            pred = node->adj[ i ];
            label = sg->node[ node->adj[ i ] ].label;
            maxpathval = tmp;
        }
        else if ( ( tmp == maxpathval ) && ( P[ sg->node[ node->adj[ i ] ].label ] * p[ sg->node[ node->adj[ i ] ].label ] > P[ label ] * p[ label ] ) )
        {
            pred  = node->adj[ i ];
            label = sg->node[ node->adj[ i ] ].label;
        }
    }

    return( pred );
}

int PredBestPathMeanTie(Subgraph *sg, CNode *node)
{
    int maxpathval, i, pred, tmp, label;
    float P[sg->nlabels+1], p[sg->nlabels+1];

    for (i=0; i < sg->nlabels; i++)
    {
        P[ i ] = 0.0;
        p[ i ] = 0.0;
    }

    for ( i = 0; i < sg->nnodes; i++ )
    {
        P[ sg->node[ i ].label ] += 1.0;
    }

    for (i=0; i < sg->bestk; i++)
    {
        p[ sg->node[ node->adj[ i ] ].label ] += node->dist[ i ];
    }

    for (i=0; i < sg->nlabels + 1; i++)
    {
        if ( P[ i ] == 0.0 ) p[ i ] = 0.0;
        else p[ i ] /= P[ i ];
    }

    // find knn node with minimum path-cost using Fmax function
    maxpathval = MIN( node->dens, sg->node[ node->adj[ 0 ] ].pathval );
    pred       = node->adj[ 0 ];
    label      = sg->node[ node->adj[ 0 ] ].label;
    for ( i = 1; i < sg->bestk; i++ )
    {
        tmp = MIN( node->dens, sg->node[ node->adj[ i ] ].pathval );
        if ( tmp > maxpathval )
        {
            pred = node->adj[ i ];
            label = sg->node[ node->adj[ i ] ].label;
            maxpathval = tmp;
        }
        else if ( ( tmp == maxpathval ) && ( p[ sg->node[ node->adj[ i ] ].label ] < p[ label ] ) )
        {
            pred  = node->adj[ i ];
            label = sg->node[ node->adj[ i ] ].label;
        }
    }

    return( pred );
}

void SetSubgraphFeatures(Subgraph *sg, Features *f)
{
    int i,j;

    sg->nfeats = f->nfeats;
    for (i=0; i < sg->nnodes; i++)
    {
        sg->node[i].feat = AllocFloatArray(sg->nfeats);

        for (j=0; j < sg->nfeats; j++)
        {
            sg->node[i].feat[j] = f->elem[sg->node[i].pixel].feat[j];
        }
    }
    sg->Imax=f->Imax;
}

void SetSubgraphFeatures3(Subgraph *sg, Features3 *f)
{
    int i,j;

    sg->nfeats = f->nfeats;
    for (i=0; i < sg->nnodes; i++)
    {
        sg->node[i].feat = AllocFloatArray(sg->nfeats);
        for (j=0; j < sg->nfeats; j++)
            sg->node[i].feat[j] = f->elem[sg->node[i].pixel].feat[j];
    }
    sg->Imax=f->Imax;
}

///int ArcWeight(float dist, float maxdist)
//{
//    return((int)((float)opf_MAXARCW*dist/maxdist));
//}

// Copy subgraph (does not copy Arcs)
///Subgraph *CopySubgraph(Subgraph *g)
//{
//    Subgraph *clone = NULL;
//    int i;
//
//    if (g != NULL)
//    {
//        clone = CreateSubgraph(g->nnodes);
//
//        clone->bestk = g->bestk;
//        clone->df = g->df;
//        clone->dm = g->dm;
//        clone->di = g->di;
//        clone->nlabels = g->nlabels;
//        clone->nfeats = g->nfeats;
//        clone->mindens = g->mindens;
//        clone->maxdens = g->maxdens;
////        clone->K1 = g->K1;
////        clone->K2 = g->K2;
//        clone->K = g->K;
//        clone->Imax = g->Imax;
//
//        for (i=0; i< g->nnodes; i++)
//        {
//            clone->node[i].feat = (float *)malloc(g->nfeats*sizeof(float));
//            CopySNode(&clone->node[i], &g->node[i], g->nfeats);
//        }
//
//        return clone;
//    }
//    else return NULL;
//}

// Split subgraph into two parts such that the size of the first part
// is given by a percentual of samples.
//
///void SplitSubgraph(Subgraph *sg, Subgraph **sg1, Subgraph **sg2, float perc1)
//{
//    int *label=AllocIntArray(sg->nlabels),i,j,i1,i2;
//    int *nelems=AllocIntArray(sg->nlabels),totelems;
//    srand((int)time(NULL));
//
//    for (i=0; i < sg->nnodes; i++)
//    {
//        label[sg->node[i].truelabel]++;
//    }
//
//    for (i=0; i < sg->nnodes; i++)
//    {
//        nelems[sg->node[i].truelabel]=MAX((int)(perc1*label[sg->node[i].truelabel]),1);
//    }
//
//    free(label);
//
//    totelems=0;
//    for (j=0; j < sg->nlabels; j++)
//        totelems += nelems[j];
//
//    *sg1 = CreateSubgraph(totelems);
//    *sg2 = CreateSubgraph(sg->nnodes-totelems);
//    (*sg1)->nfeats = sg->nfeats;
//    (*sg2)->nfeats = sg->nfeats;
//
//    for (i1=0; i1 < (*sg1)->nnodes; i1++)
//        (*sg1)->node[i1].feat = AllocFloatArray((*sg1)->nfeats);
//    for (i2=0; i2 < (*sg2)->nnodes; i2++)
//        (*sg2)->node[i2].feat = AllocFloatArray((*sg2)->nfeats);
//
//    (*sg1)->nlabels = sg->nlabels;
//    (*sg2)->nlabels = sg->nlabels;
//
//    i1=0;
//    while (totelems > 0)
//    {
//        i = RandomInteger(0,sg->nnodes-1);
//        if (sg->node[i].status!=NIL)
//        {
//            if (nelems[sg->node[i].truelabel]>0)  // copy node to sg1
//            {
//                (*sg1)->node[i1].pixel = sg->node[i].pixel;
//                for (j=0; j < (*sg1)->nfeats; j++)
//                    (*sg1)->node[i1].feat[j]=sg->node[i].feat[j];
//                (*sg1)->node[i1].truelabel = sg->node[i].truelabel;
//                i1++;
//                nelems[sg->node[i].truelabel] = nelems[sg->node[i].truelabel] - 1;
//                sg->node[i].status = NIL;
//                totelems--;
//            }
//        }
//    }
//
//    i2=0;
//    for (i=0; i < sg->nnodes; i++)
//    {
//        if (sg->node[i].status!=NIL)
//        {
//            (*sg2)->node[i2].pixel = sg->node[i].pixel;
//            for (j=0; j < (*sg2)->nfeats; j++)
//                (*sg2)->node[i2].feat[j]=sg->node[i].feat[j];
//            (*sg2)->node[i2].truelabel = sg->node[i].truelabel;
//            i2++;
//        }
//    }
//
//    free(nelems);
//}

Subgraph *SubSampleLabeledSubgraph( Subgraph *sg, int nsamples )
{
    int i, j, l, i1;
    Subgraph *samples = NULL;
    srand( ( int ) time( NULL ) );

    if ( nsamples > sg->nnodes )
    {
        Error("Sampling more nodes than subgraph supports!","SubSampleLabeledSubgraph");
    }

    samples = CreateSubgraph( nsamples );
    samples->nfeats = sg->nfeats;

    for (i=0; i < samples->nnodes; i++)
        samples->node[ i ].feat = AllocFloatArray( samples->nfeats );
    samples->nlabels = sg->nlabels;

    l = 0;
    i1 = 0;
    while ( nsamples > 0 )
    {
        i = RandomInteger( 0, sg->nnodes - 1 );
        if ( sg->node[ i ].status != NIL )
        {
            if (sg->node[ i ].truelabel == l )   // copy node to sg1
            {
                samples->node[ i1 ].pixel = sg->node[ i ].pixel;
                samples->node[ i1 ].position = sg->node[ i ].position;
                for (j=0; j < samples->nfeats; j++)
                    samples->node[ i1 ].feat[ j ] = sg->node[ i ].feat[ j ];
                samples->node[ i1 ].truelabel = sg->node[ i ].truelabel;
                i1++;
                sg->node[ i ].status = NIL;
                nsamples--;
                l = ( l + 1 ) % sg->nlabels;
            }
        }
    }

    for ( i = 0; i < sg->nnodes; i++ )
    {
        sg->node[i].status = 0;
    }

    return( samples );
}

// Move errors above max_abs_error from the evaluation set to the
// training set
int MoveErrorsToTrainingSet(Subgraph **sgtrain, Subgraph **sgeval, int max_abs_error)
{
    int i,j,k,l,nerrors;
    Subgraph *sgtrain_aux, *sgeval_aux;


    nerrors=0;
    for (i=0; i < (*sgeval)->nnodes; i++)
        if (abs((*sgeval)->node[i].label-(*sgeval)->node[i].truelabel)>max_abs_error)
            nerrors++;

    if (nerrors==0)
        return(0);

    sgtrain_aux = CreateSubgraph((*sgtrain)->nnodes+nerrors);
    sgeval_aux  = CreateSubgraph((*sgeval)->nnodes-nerrors);

    sgtrain_aux->nfeats=(*sgtrain)->nfeats;
    sgeval_aux->nfeats=(*sgeval)->nfeats;


    sgtrain_aux->nlabels = (*sgtrain)->nlabels;
    sgeval_aux->nlabels = (*sgeval)->nlabels;

    for (i=0; i < (*sgtrain)->nnodes; i++)  // Copy training set
    {
        sgtrain_aux->node[i].feat = AllocFloatArray(sgtrain_aux->nfeats);
        for (j=0; j < sgtrain_aux->nfeats; j++)
            sgtrain_aux->node[i].feat[j] = (*sgtrain)->node[i].feat[j];
        sgtrain_aux->node[i].truelabel = (*sgtrain)->node[i].truelabel;
        sgtrain_aux->node[i].pixel = (*sgtrain)->node[i].pixel;
        sgtrain_aux->node[i].position = (*sgtrain)->node[i].position;
    }

    k=i;
    l=0;
    for (i=0; i < (*sgeval)->nnodes; i++)  // Copy errors
    {
        if (abs((*sgeval)->node[i].label-(*sgeval)->node[i].truelabel)>max_abs_error)
        {
            sgtrain_aux->node[k].feat = AllocFloatArray(sgtrain_aux->nfeats);
            for (j=0; j < sgtrain_aux->nfeats; j++)
                sgtrain_aux->node[k].feat[j] = (*sgeval)->node[i].feat[j];
            sgtrain_aux->node[k].truelabel = (*sgeval)->node[i].truelabel;
            sgtrain_aux->node[k].pixel  = (*sgeval)->node[i].pixel;
            sgtrain_aux->node[k].position  = (*sgeval)->node[i].position;
            k++;
        }
        else  // Copy non-errors
        {
            sgeval_aux->node[l].feat = AllocFloatArray(sgeval_aux->nfeats);
            for (j=0; j < sgtrain_aux->nfeats; j++)
                sgeval_aux->node[l].feat[j] = (*sgeval)->node[i].feat[j];
            sgeval_aux->node[l].truelabel = (*sgeval)->node[i].truelabel;
            sgeval_aux->node[l].pixel  = (*sgeval)->node[i].pixel;
            sgeval_aux->node[l].position  = (*sgeval)->node[i].position;
            l++;
        }
    }

    DestroySubgraph(sgtrain);
    DestroySubgraph(sgeval);

    *sgtrain = sgtrain_aux;
    *sgeval  = sgeval_aux;

    //printf("Training set size is %d\n",(*sgtrain)->nnodes);

    return(nerrors);
}

// Remove nodes with distinct labels and distance=0
void RemoveInconsistentNodes(Subgraph **sg)
{
    Subgraph *newsg;
    int i,j,k,num_of_inconsistents=0;

    for (i=0; i < (*sg)->nnodes; i++)
        (*sg)->node[i].status=0;

    for (i=0; i < (*sg)->nnodes; i++)
        if ((*sg)->node[i].status!=NIL)  // consistent node
        {
            for (j=0; j < (*sg)->nnodes; j++)
                if ((i!=j)&&((*sg)->node[j].status!=NIL))  // consistent node
                {
                    if ((*sg)->node[i].truelabel!=(*sg)->node[j].truelabel)
                        if (opf_ArcWeight((*sg)->node[i].feat,(*sg)->node[j].feat,(*sg)->nfeats)==0.0) // i is not consistent with j, then mark j as inconsistent node
                        {
                            (*sg)->node[j].status=NIL;
                            num_of_inconsistents++;
                        }
                }
        }

    if (num_of_inconsistents>0)
    {
        printf("There are %d inconsistent nodes among %d nodes\n",num_of_inconsistents,(*sg)->nnodes);
        newsg = CreateSubgraph((*sg)->nnodes - num_of_inconsistents);
        newsg->nfeats = (*sg)->nfeats;
        for (i=0; i < newsg->nnodes; i++)
            newsg->node[i].feat = AllocFloatArray(newsg->nfeats);

        k=0;
        newsg->nlabels = (*sg)->nlabels;
        for (i=0; i < (*sg)->nnodes; i++)
        {
            if ((*sg)->node[i].status!=NIL)  // consistent node
            {
                newsg->node[k].pixel  = (*sg)->node[i].pixel;
                newsg->node[k].position  = (*sg)->node[i].position;
                newsg->node[k].truelabel = (*sg)->node[i].truelabel;
                for (j=0; j < newsg->nfeats; j++)
                    newsg->node[k].feat[j] = (*sg)->node[i].feat[j];
                k++;
            }
        }
        newsg->nlabels=(*sg)->nlabels;
        DestroySubgraph(sg);
        *sg=newsg;
    }
}

// Merge disjoint subgraphs with no arcs
Subgraph *MergeSubgraphs(Subgraph *sg1, Subgraph *sg2)
{
    Subgraph *sg3;
    int i1,i2,i3,nnodes;

    nnodes=sg1->nnodes+sg2->nnodes;
    sg3=CreateSubgraph(nnodes);
    sg3->nfeats=sg1->nfeats;

    i3=0;
    for (i1=0; i1 < sg1->nnodes; i1++)
    {
//        sg3->node[i3].pixel  = sg1->node[i1].pixel;
//        sg3->node[i3].position  = sg1->node[i1].position;
//        sg3->node[i3].status    = sg1->node[i1].status;
//        sg3->node[i3].truelabel = sg1->node[i1].truelabel;
//        sg3->node[i3].label     = sg1->node[i1].label;
//        sg3->node[i3].feat      = AllocFloatArray(sg3->nfeats);
//        for (j=0; j < sg3->nfeats; j++)
//            sg3->node[i3].feat[j]=sg1->node[i1].feat[j];
//        sg3->node[i3].adj=NULL;
        CopySNode(&(sg3->node[i3]), &(sg1->node[i1]), sg3->nfeats);
        i3++;
    }
    for (i2=0; i2 < sg2->nnodes; i2++)
    {
//        sg3->node[i3].pixel  = sg2->node[i2].pixel;
//        sg3->node[i3].position  = sg2->node[i2].position;
//        sg3->node[i3].status    = sg2->node[i2].status;
//        sg3->node[i3].truelabel = sg2->node[i2].truelabel;
//        sg3->node[i3].label     = sg2->node[i2].label;
//        sg3->node[i3].feat      = AllocFloatArray(sg3->nfeats);
//        for (j=0; j < sg3->nfeats; j++)
//            sg3->node[i3].feat[j]=sg2->node[i2].feat[j];
//        sg3->node[i3].adj=NULL;
        CopySNode(&(sg3->node[i3]), &(sg2->node[i2]), sg3->nfeats);
        i3++;
    }
    return(sg3);
}

//Compute accuracy of classification
///float Accuracy(Subgraph *sg)
//{
//    float Acc = 0.0f, **error_matrix = NULL, error = 0.0f;
//    int i, *nclass = NULL, nlabels=0;
//
//    error_matrix = (float **)calloc(sg->nlabels, sizeof(float *));
//    for (i=0; i< sg->nlabels; i++)
//        error_matrix[i] = (float *)calloc(2, sizeof(float));
//
//    nclass = AllocIntArray(sg->nlabels);
//
//    for (i = 0; i < sg->nnodes; i++)
//    {
//        nclass[sg->node[i].truelabel]++;
//    }
//
//    for (i = 0; i < sg->nnodes; i++)
//    {
//        if (sg->node[i].truelabel != sg->node[i].label)
//        {
//            error_matrix[sg->node[i].truelabel][1]++;
//            error_matrix[sg->node[i].label][0]++;
//        }
//    }
//
//    for (i=0; i < sg->nlabels; i++)
//    {
//        if (nclass[i]!=0)
//        {
//            //printf("\nerror_matrix[%d][1]: %d",i,(int)error_matrix[i][1]);
//            //printf("\nnclass[%d]: %d",i,nclass[i]);
//            //printf("\n# of test samples with label %d: %d (%2.2f%%)\n",i, nclass[i], (1-error_matrix[i][1]/nclass[i])*100);
//            error_matrix[i][1] /= (float)nclass[i];
//            error_matrix[i][0] /= (float)(sg->nnodes - nclass[i]);
//            nlabels++;
//        }
//    }
//
//    for (i=0; i < sg->nlabels; i++)
//    {
//        if (nclass[i]!=0)
//            error += (error_matrix[i][0]+error_matrix[i][1]);
//    }
//
//
//    Acc = 1.0-(error/(2.0*nlabels));
//
//    for (i=0; i < sg->nlabels; i++)
//        free(error_matrix[i]);
//    free(error_matrix);
//    free(nclass);
//
//    return(Acc);
//}

//Compute accuracy per class
float *AccuracyByLabel(Subgraph *sg)
{
    float *Acc = NULL;
    int i, *nclass = NULL;

    nclass = AllocIntArray(sg->nlabels);
    Acc = AllocFloatArray(sg->nlabels);

    for (i = 0; i < sg->nnodes; i++)
    {
        nclass[sg->node[i].truelabel]++;
    }

    for (i = 0; i < sg->nnodes; i++)
    {
        if (sg->node[i].truelabel != sg->node[i].label)
        {
            Acc[sg->node[i].truelabel]++;
        }
    }

    for (i=0; i < sg->nlabels; i++)
    {
        if (nclass[i]!=0)
        {
            Acc[i] = 1 - Acc[i]/nclass[i];
        }
    }

    free(nclass);

    return(Acc);
}

//Executes the learning procedure for CompGraph replacing the
//missclassified samples in the evaluation set by non prototypes from
//training set
///void LearningCompGraph(Subgraph **sgtrain, Subgraph **sgeval, int iterations)
//{
//    int i;
//    float Acc,MaxAcc=INT_MIN;
//    Subgraph *sg=NULL;
//    FILE *fp = fopen("AccZ2.opf.txt", "w");
//
//    for (i = 1; i <= iterations && MaxAcc != 1.0; i++)
//    {
//        fprintf(stdout, "\nrunning iteration ... %d ", i);
//        ResetCompGraph(*sgtrain);
//        SupTrainCompGraph(*sgtrain);
//        ClassifyCompGraph(*sgtrain, *sgeval);
//        Acc = opf_Accuracy(*sgeval);
//        if (Acc > MaxAcc)
//        {
//            MaxAcc = Acc;
//            if (sg!=NULL) DestroySubgraph(&sg);
//            sg = CopySubgraph(*sgtrain);
//        }
//        opf_SwapErrorsbyNonPrototypes(&(*sgtrain), &(*sgeval));
//        fprintf(stdout,"Acc: %f\n", Acc);
//        fprintf(fp, "%f\n", Acc);
//    }
//    fclose(fp);
//    DestroySubgraph(&(*sgtrain));
//    *sgtrain = sg;
//    ResetCompGraph(*sgtrain);
//    SupTrainCompGraph(*sgtrain);
//    printf("Best accuracy %f\n",MaxAcc);
//}

//Executes the learning procedure for KnnGraph replacing the
//missclassified samples in the evaluation set by non prototypes from
//training set

void LearningKnnGraph(Subgraph **sgtrain, Subgraph **sgeval, int iterations, int kmax)
{
    int i;
    float Acc,MaxAcc=INT_MIN;
    Subgraph *sg=NULL;
    FILE *fp = fopen("AccZ2.opf-knn.txt", "w");

    for (i = 1; i <= iterations; i++)
    {
        fprintf(stdout, "\nrunning iteration ... %d ", i);
        opf_DestroyArcs(*sgtrain);
        BestkMinError(*sgtrain,*sgeval,kmax);
        SupTrainKnnGraph(*sgtrain);
        opf_OPFKNNClassify(*sgtrain, *sgeval);
//        ClassifyKnnGraph(*sgtrain, *sgeval);
        Acc = opf_Accuracy(*sgeval);
        fprintf(fp, "%f\n", Acc);
        if (Acc > MaxAcc)
        {
            MaxAcc = Acc;
            if (sg!=NULL) DestroySubgraph(&sg);
            sg = CopySubgraph(*sgtrain);
        }
        opf_SwapErrorsbyNonPrototypes(&(*sgtrain), &(*sgeval));
        fprintf(stdout,"Acc: %f\n", Acc);
    }
    fclose(fp);

    opf_DestroyArcs(*sgtrain);
    DestroySubgraph(&(*sgtrain));
    *sgtrain = sg;
    opf_CreateArcs(*sgtrain,(*sgtrain)->bestk);
    SupTrainKnnGraph(*sgtrain);
    printf("bestk %d MaxAcc %f\n",(*sgtrain)->bestk,MaxAcc);
}

// Baysian learning
void LearningBayes(Subgraph **sgtrain, Subgraph **sgeval, int iterations, int kmax)
{
    int i;
    float Acc,MaxAcc=INT_MIN;
    Subgraph *sg=NULL;

    for (i = 1; (i <= iterations)&&(MaxAcc != 1.0); i++)
    {
        fprintf(stdout, "\n running iteration ... %d \n ", i);
        BestkMinErrorBayes(*sgtrain,*sgeval, kmax);
        ClassifyBayes(*sgtrain, *sgeval);
        Acc = opf_Accuracy(*sgeval);
        fprintf(stdout,"Acc: %f\n", Acc);
        if (Acc > MaxAcc)
        {
            MaxAcc = Acc;
            if (sg!=NULL) DestroySubgraph(&sg);
            sg = CopySubgraph(*sgtrain);
        }
        opf_DestroyArcs(*sgtrain);
        SwapErrorsbySamples(&(*sgtrain), &(*sgeval));
    }

    DestroySubgraph(&(*sgtrain));
    *sgtrain = sg;
    CreateArcsByTrueLabel(*sgtrain,(*sgtrain)->bestk);
    opf_PDF(*sgtrain);
    printf("bestk %d MaxAcc %f\n",(*sgtrain)->bestk,MaxAcc);
}

// Estimate the best k by minimum cut
//void BestkMinCut(Subgraph *sg, int kmax)
//{
//    int k;
//    float mincut=REAL_MAX,nc;
//
//    // Find the best k
//    for (k=1; (k <= kmax)&&(mincut != 0.0); k++)
//    {
//
//        opf_CreateArcs(sg,k);
//        opf_PDF(sg);
//        opf_OPFClustering(sg);
//
//        nc = opf_NormalizedCut(sg);
//
//        if (nc < mincut)
//        {
//            mincut=nc;
//            sg->bestk =k;
//        }
//        opf_DestroyArcs(sg);
//    }
//    opf_CreateArcs(sg,sg->bestk);
//    opf_PDF(sg);
//    //printf("best k %d\n",sg->bestk);
//}

///void BrainBestkMinCut(Subgraph *sg, int kmin, int kmax)
//{
//    int k;
//    float mincut=REAL_MAX,nc;
//
//    // Find the best k
//    for (k=kmin; (k <= kmax)&&(mincut != 0.0); k++)
//    {
//
//        opf_CreateArcs(sg,k);
//
//        opf_PDF(sg);
//
//        opf_OPFClustering(sg);
//
//        nc = opf_NormalizedCut(sg);
//
//        if (nc < mincut)
//        {
//            mincut=nc;
//            sg->bestk =k;
//        }
//        opf_DestroyArcs(sg);
//    }
//    opf_CreateArcs(sg,sg->bestk);
//    opf_PDF(sg);
//    //printf("best k %d\n",sg->bestk);
//}

void ComputeEntropy(Subgraph *sg)
{
    float *P1=AllocFloatArray(sg->nlabels),*P2=AllocFloatArray(sg->nlabels);
    int i;

    for (i=0; i < sg->nnodes; i++)
        P1[sg->node[i].label]+=1.0;

    for (i=0; i < sg->nnodes; i++)
        P2[sg->node[i].label]+= (-sg->node[i].dens/1000.0*log(sg->node[i].dens/1000.0));

    sg->entropy=0.0;
    for (i=0; i < sg->nlabels; i++)
    {
        if (P1[i]!=0.0)
            sg->entropy += (-log(P1[i]/sg->nnodes))*P2[i]/sg->nnodes;
    }

    free(P1);
    free(P2);
}


// Estimate the best k by minimum entropy

void BestkMinEntropy(Subgraph *sg, int kmax)
{
    int k;
    float minent=REAL_MAX;

    // Find the best k
    for (k=1; (k <= kmax)&&(minent != 0.0); k++)
    {

        opf_CreateArcs(sg,k);

        opf_PDF(sg);
        opf_OPFClustering(sg);
        ComputeEntropy(sg);
        if (sg->entropy < minent)
        {
            minent=sg->entropy;
            sg->bestk =k;
        }
        opf_DestroyArcs(sg);
    }
    opf_CreateArcs(sg,sg->bestk);
    opf_PDF(sg);
    //printf("best k %d and entropy %f\n",sg->bestk,sg->entropy);
}

// Estimate the best k by Bayes
void BestkMinErrorBayes(Subgraph *sgTrain, Subgraph *sgEval, int kmax)
{
    int k;
    float maxacc=INT_MIN,Acc;

    // Find the best k
    for (k=1; (k <= kmax); k++)
    {
        CreateArcsByTrueLabel(sgTrain,k);
        opf_PDF(sgTrain);
        sgTrain->bestk=k;
        ClassifyBayes(sgTrain,sgEval);
        Acc = opf_Accuracy(sgEval);
        if (Acc >= maxacc)
        {
            maxacc=Acc;
            sgEval->bestk =k;
        }
        opf_DestroyArcs(sgTrain);
    }
    sgTrain->bestk=sgEval->bestk;
    CreateArcsByTrueLabel(sgTrain,sgTrain->bestk);
    opf_PDF(sgTrain);
}

// Estimate the best k by minimum error
void BestkMinError(Subgraph *sgTrain, Subgraph *sgEval, int kmax)
{
    int k;
    float maxacc=INT_MIN,Acc;

    // Find the best k
    for (k=1; (k <= kmax); k++)
    {
        opf_CreateArcs(sgTrain,k);
        opf_PDF(sgTrain);
        SupOPFwithErrors(sgTrain);
        sgTrain->bestk =k;
        opf_OPFKNNClassify(sgTrain, sgEval);
//        ClassifyKnnGraph(sgTrain, sgEval);
        Acc = opf_Accuracy(sgEval);
        if (Acc >= maxacc)
        {
            maxacc=Acc;
            sgEval->bestk =k;
        }
        opf_DestroyArcs(sgTrain);
    }

    sgTrain->bestk=sgEval->bestk;
    opf_CreateArcs(sgTrain,sgTrain->bestk);
    opf_PDF(sgTrain);
}

// Find the best k with minimum cut to create n clusters
void BestkMinCutNClusters(Subgraph *sg, int nclusters, int kmax)
{

    int k, *nlabels=AllocIntArray(kmax+1);
    float  mincut=REAL_MAX, *nclist=AllocFloatArray(kmax+1);

    // Compute cuts for all values of k

    for (k=1; (k <= kmax); k++)
    {
        opf_CreateArcs(sg,k);
        opf_PDF(sg);
        opf_OPFClustering(sg);
        nclist[k] = opf_NormalizedCut(sg);
        nlabels[k] = sg->nlabels;
        if (nclist[k] < mincut)
        {
            sg->bestk=k;
            mincut=nclist[k];
        }
        opf_DestroyArcs(sg);
    }

    if (nlabels[sg->bestk]!=nclusters)  // mincut does not give the
    {
        // desired number of clusters

        mincut=REAL_MAX;
        sg->bestk=0;

        for (k=1; (k <= kmax); k++)   // select k which gives nclusters
        {
            // and lowest cut
            if (nlabels[k]==nclusters)
            {
                if (nclist[k] < mincut)
                {
                    sg->bestk=k;
                    mincut=nclist[k];
                }
            }
        }
        if (sg->bestk == 0)  // there is no such a k, then select k which
        {
            // gives more than nclusters and lowest cut.


            for (k=1; (k <= kmax); k++)
            {
                if (nlabels[k]>nclusters)
                {
                    if (nclist[k] < mincut)
                    {
                        sg->bestk=k;
                        mincut=nclist[k];
                    }
                }
            }

            if (sg->bestk == 0)   // there is no such a k, then select k=1.
            {
                sg->bestk = 1;
            }
        }
    }
    free(nclist);
    free(nlabels);

    opf_CreateArcs(sg,sg->bestk);
    opf_PDF(sg);

}

/** Eliminate maxima in the graph with pdf below H */

//void ElimMaxBelowH(Subgraph *sg, float H)
//{
//    int i;
//
//    if (H>0.0)
//    {
//        for (i=0; i < sg->nnodes; i++)
//            sg->node[i].pathval = MAX(sg->node[i].dens - H,0);
//    }
//}

/** Eliminate maxima in the graph with area below A */

//void ElimMaxBelowArea(Subgraph *sg, int A)
//{
//    int i, *area;
//
//    area = SgAreaOpen(sg,A);
//    for (i=0; i < sg->nnodes; i++)
//    {
//        sg->node[i].pathval    = MAX(area[i] - 1,0);
//    }
//
//    free(area);
//}

/** Eliminate maxima in the graph with volume below V */

//void ElimMaxBelowVolume(Subgraph *sg, int V)
//{
//    int i, *volume=NULL;
//
//    volume = SgVolumeOpen(sg,V);
//    for (i=0; i < sg->nnodes; i++)
//    {
//        sg->node[i].pathval  = MAX(volume[i] - 1,0);
//    }
//
//    free(volume);
//}

////read subgraph from opf format file
//Subgraph *ReadSubgraph(char *file)
//{
//    Subgraph *g = NULL;
//    FILE *fp = NULL;
//    int nnodes, *nclass = NULL, i, j, l, k = 0;
//
//    if ((fp = fopen(file, "rb")) == NULL)
//    {
//        fprintf(stderr, "\nUnable to open file %s in ReadSubGraph", file);
//        return NULL;
//    }
//
//    /*reading # of nodes, classes and feats*/
//    fread(&nnodes, sizeof(int), 1, fp);
//    g = CreateSubgraph(nnodes);
//    fread(&g->nlabels, sizeof(int), 1, fp);
//    fread(&g->nfeats, sizeof(int), 1, fp);
//    g->nlabels++; //vector starts from 0 to n-1
//
//    /*reading # of elements from each class*/
//    nclass = (int *)malloc((g->nlabels-1)*sizeof(int));
//    for (i = 0; i < g->nlabels-1; i++)
//        fread(&nclass[i], sizeof(int), 1, fp);
//
//    /*reading features*/
//    for (i = 0; i < g->nlabels-1; i++)
//    {
//        for (j = 1; j <= nclass[i]; j++)
//        {
//            g->node[k].feat = (float *)malloc(g->nfeats*sizeof(float));
//            g->node[k].truelabel = i+1;
//            fread(&g->node[k].position, sizeof(int), 1, fp);
//            for (l = 0; l < g->nfeats; l++)
//                fread(&g->node[k].feat[l], sizeof(float), 1, fp);
//            k++;
//        }
//    }
//
//    free(nclass);
//    fclose(fp);
//
//    return g;
//}

////write subgraph to disk
//void WriteSubgraph(Subgraph *g, char *file)
//{
//    FILE *fp = NULL;
//    int nlabels = g->nlabels-1, *nclasses = NULL, i, j;
//
//    nclasses = (int *)calloc(g->nlabels+1,sizeof(int));
//
//    fp = fopen(file, "wb");
//    fwrite(&g->nnodes, sizeof(int), 1, fp);
//    fwrite(&nlabels, sizeof(int), 1, fp);
//    fwrite(&g->nfeats, sizeof(int), 1, fp);
//
//    /*loading #of elements from each class*/
//    for (i = 0; i < g->nnodes; i++)
//        nclasses[g->node[i].truelabel]++;
//    for (i = 1; i <= g->nlabels-1; i++)
//    {
//        fwrite(&nclasses[i], sizeof(int), 1, fp);
//    }
//
//    /*writing position(id) and features*/
//    for (i = 0; i < g->nnodes; i++)
//    {
//        fwrite(&g->node[i].position, sizeof(int), 1, fp);
//        for (j = 0; j < g->nfeats; j++)
//            fwrite(&g->node[i].feat[j], sizeof(float), 1, fp);
//    }
//
//    fclose(fp);
//    free(nclasses);
//}

void WriteLabelnode(LNode *ln, char *file)
{
    FILE *fp = NULL;
    Set *aux;

    fp = fopen( file, "a" );
    fprintf( fp, "size=%d, label=%d, mean=%f, ref=%d\n", ln->size, ln->label, ln->mean, ln->rep );
    fprintf( fp, "adjacents:" );
    for ( aux = ln->adj; aux != NULL; aux = aux->next ) fprintf( fp,"%d ", aux->elem );
    fprintf( fp,"\n" );
    fprintf( fp, "voxels:" );
    for ( aux = ln->voxels; aux != NULL; aux = aux->next ) fprintf( fp,"%d ", aux->elem );
    fprintf( fp,"\n" );
    fclose( fp );
}


//Swap nodes
//void SwapSNode(SNode *a, SNode *b)
//{
//    SNode tmp;
//
//    tmp = *a;
//    *a = *b;
//    *b = tmp;
//}

////Copy nodes
//void CopySNode(SNode *dest, SNode *src, int nfeats)
//{
//    memcpy(dest->feat, src->feat, nfeats*sizeof(float));
//    dest->dens = src->dens;
//    dest->pathval = src->pathval;
//    dest->label  = src->label;
//    dest->root  = src->root;
//    dest->volume  = src->volume;
//    dest->pred  = src->pred;
//    dest->truelabel = src->truelabel;
//    dest->position = src->position;
//    dest->status = src->status;
//}


//Resets subgraph fields (pred)
void ResetCompGraph(Subgraph *sg)
{
    int i;

    for (i = 0; i < sg->nnodes; i++)
    {
        sg->node[i].pred    = NIL;
    }
}

//Replace errors from evaluating set by non prototypes from training set
//void SwapErrorsbyNonPrototypes(Subgraph **sgtrain, Subgraph **sgeval){
//  int i, j, counter, nonprototypes = 0, nerrors = 0;
//
//  for (i = 0; i < (*sgtrain)->nnodes; i++){
//    if((*sgtrain)->node[i].pred != NIL){ // non prototype
//      (*sgtrain)->node[i].status = 0;
//      nonprototypes++;
//    }else
//      (*sgtrain)->node[i].status = NIL;
//  }
//
//  for (i = 0; i < (*sgeval)->nnodes; i++)
//    if((*sgeval)->node[i].label != (*sgeval)->node[i].truelabel) nerrors++;
//
//  for (i = 0; i < (*sgeval)->nnodes && nonprototypes >0 && nerrors > 0; i++){
//    if((*sgeval)->node[i].label != (*sgeval)->node[i].truelabel){
//      counter = nonprototypes;
//      while(counter > 0){
//	j = RandomInteger(0,(*sgtrain)->nnodes-1);
//	if ((*sgtrain)->node[j].status==0)
//	  {
//	    SwapSNode(&((*sgtrain)->node[j]), &((*sgeval)->node[i]));
//	    (*sgtrain)->node[j].status = NIL;
//	    nonprototypes--;
//	    nerrors--;
//	    counter = 0;
//	  }
//	else counter--;
//      }
//    }
//  }
//}

//Replace errors from evaluating set by randomly samples from training set
void SwapErrorsbySamples(Subgraph **sgtrain, Subgraph **sgeval)
{
    int i, j, counter, ntrainsamples = (*sgtrain)->nnodes, nerrors = 0;

    for (i = 0; i < (*sgeval)->nnodes; i++)
        if ((*sgeval)->node[i].label != (*sgeval)->node[i].truelabel){
	  (*sgtrain)->node[i].status = 0;
	  nerrors++;
	}else
          (*sgtrain)->node[i].status = NIL;

    for (i = 0; i < (*sgeval)->nnodes && ntrainsamples >0 && nerrors > 0; i++)
    {
        if ((*sgeval)->node[i].label != (*sgeval)->node[i].truelabel)
        {
            counter = ntrainsamples;
            while (counter > 0)
            {
                j = RandomInteger(0,(*sgtrain)->nnodes-1);
                if ((*sgtrain)->node[j].status == 0)
                {
                    SwapSNode(&((*sgtrain)->node[j]), &((*sgeval)->node[i]));
                    (*sgtrain)->node[j].status = NIL;
                    ntrainsamples--;
                    nerrors--;
                    counter = 0;
                }
                else counter--;
            }
        }
    }
}

//sort subgraph: order = INCREASING or DECREASING
void SortSubgraphByDensity( Subgraph **cg, char order )
{
    Subgraph *cgAux = NULL, *cgSort = NULL;
    cgAux = *cg;
    Curve *curve = NULL;
    int i;

    if (cgAux != NULL)
    {

        curve = CreateCurve(cgAux->nnodes+1);

        for (i = 0; i < cgAux->nnodes; i++)
        {
            curve->X[i] = i;
            curve->Y[i] = cgAux->node[i].dens;
        }

        SortCurve(curve, 0,(curve->n)-2,order);
        cgSort = CreateSubgraph(cgAux->nnodes);
        cgSort->nfeats  = cgAux->nfeats;
        cgSort->nlabels = cgAux->nlabels;
        cgSort->df      = cgAux->df;
        cgSort->dm      = cgAux->dm;
        cgSort->di      = cgAux->di;
	cgSort->bestk   = cgAux->bestk;
	cgSort->entropy = cgAux->entropy;
        cgSort->mindens = cgAux->mindens;
        cgSort->maxdens = cgAux->maxdens;
//        cgSort->K1      = cgAux->K1;
//        cgSort->K2      = cgAux->K2;
        cgSort->K      = cgAux->K;
        cgSort->Imax    = cgAux->Imax;

	for (i = 0; i< cgSort->nnodes; i++)
        {
//            cgSort->node[i].feat = (float *)malloc(cgSort->nfeats*sizeof(float));
            CopySNode(&cgSort->node[i], &cgAux->node[(int)curve->X[i]], cgAux->nfeats);
//	    cgSort->node[i].adj = NULL;
//	    cgSort->node[i].adj = CloneSet(cgAux->node[(int)curve->X[i]].adj);
	}

        DestroyCurve(&curve);
        *cg = cgSort;
        DestroySubgraph(&cgAux);
    }
}


//sort subgraph: order = INCREASING or DECREASING
void SortSubgraph(Subgraph **cg, char order)
{
    Subgraph *cgAux = NULL, *cgSort = NULL;
    cgAux = *cg;
    Curve *curve = NULL;
    int i;

    if (cgAux != NULL)
    {

        curve = CreateCurve(cgAux->nnodes+1);

        for (i = 0; i < cgAux->nnodes; i++)
        {
            curve->X[i] = i;
            curve->Y[i] = cgAux->node[i].truelabel;
        }

        SortCurve(curve, 0,(curve->n)-2,order);
        cgSort = CreateSubgraph(cgAux->nnodes);
        cgSort->nfeats = cgAux->nfeats;
        cgSort->nlabels = cgAux->nlabels;
        cgSort->df = cgAux->df;

        for (i = 0; i< cgSort->nnodes; i++)
        {
//            cgSort->node[i].feat = (float *)malloc(cgSort->nfeats*sizeof(float));
            CopySNode(&cgSort->node[i], &cgAux->node[(int)curve->X[i]], cgAux->nfeats);
        }

        DestroyCurve(&curve);
        *cg = cgSort;
        DestroySubgraph(&cgAux);
    }
}

/*read precomputed distances*/
//float **ReadDistances(char *fileName)
//{
//    int nsamples, i;
//    FILE *fp = NULL;
//    float **M = NULL;
//
//    fp = fopen(fileName,"rb");
//    if (fp == NULL)
//    {
//        fprintf(stderr,"\nunable to open file %s",fileName);
//        exit(-1);
//    }
//
//    fread(&nsamples, sizeof(int), 1, fp);
//    M = (float **)malloc(nsamples*sizeof(float *));
//
//    for (i = 0; i < nsamples; i++)
//    {
//        M[i] = (float *)malloc(nsamples*sizeof(float));
//        fread(M[i], sizeof(float), nsamples, fp);
//    }
//
//    fclose(fp);
//
//    return M;
//}

/**normalize features*/
//void NormalizeFeatures(Subgraph *sg)
//{
//    float *mean = (float *)calloc(sg->nfeats,sizeof(float)), *std = (float *)calloc(sg->nfeats, sizeof(int));
//    int i,j;
//
//    for (i = 0; i < sg->nfeats; i++)
//    {
//        for (j = 0; j < sg->nnodes; j++)
//            mean[i]+=sg->node[j].feat[i]/sg->nnodes;
//        for (j = 0; j < sg->nnodes; j++)
//            std[i]+=pow(sg->node[j].feat[i]-mean[i],2)/sg->nnodes;
//        std[i]=sqrt(std[i]);
//    }
//
//    for (i = 0; i < sg->nfeats; i++)
//    {
//        for (j = 0; j < sg->nnodes; j++)
//        {
//            float v =(sg->node[j].feat[i]-mean[i]);
//            /// Preventing division by zero
//            if (std[i] != 0.0)
//                sg->node[j].feat[i] = v/std[i];
//            else
//                sg->node[j].feat[i] = v;
//        }
//    }
//
//    free(mean);
//    free(std);
//}

//-------------------- PDF  Filters ---------------------------- //


Set *SelectLargestDomes(Subgraph *sg, int nclusters)
{
    int   i,p;
    RealHeap *H=NULL;
    Set *S=NULL;

    float* minlevel = (float *) AllocFloatArray(sg->nnodes);
    float* initvolume = (float *) AllocFloatArray(sg->nnodes);
    float* volume = (float *) AllocFloatArray(sg->nnodes);
    H = CreateRealHeap(sg->nnodes,volume);
    SetRemovalPolicyRealHeap(H,MAXVALUE);

    for(p=0; p < sg->nnodes; p++)
    {
		minlevel[ p ] = sg->node[ p ].dens;
    }

    for(p=0; p < sg->nnodes; p++)
    {
		if (sg->node[ p ].dens < minlevel[ sg->node[p].root ])
			minlevel[ sg->node[p].root ] = sg->node[ p ].dens;
    }
    // compute volume of each cluster
    for (p=0; p < sg->nnodes; p++)
        initvolume[ sg->node[p].root ] += (sg->node[ p ].dens - minlevel[ sg->node[p].root ]);

    for (p=0; p < sg->nnodes; p++)
        if (sg->node[p].root == p)
        {
            volume[p]=initvolume[p];
            InsertRealHeap(H,p);
        }

    i=0;
    while ((!IsEmptyRealHeap(H))&&(i < nclusters))
    {
        RemoveRealHeap(H,&p);
        i++;
        InsertSet(&S,p);
    }

    DestroyRealHeap(&H);
    free(volume);
    free(initvolume);
    free(minlevel);

    return(S);
}
//
//Set *SelectLargestDomes(Subgraph *sg, int nclusters)
//{
//    int   i,p;
//    float *volume=NULL;
//    RealHeap *H=NULL;
//    Set *S=NULL;
//
//    volume = (float *) AllocFloatArray(sg->nnodes);
//    H = CreateRealHeap(sg->nnodes,volume);
//    SetRemovalPolicyRealHeap(H,MAXVALUE);
//
//    for (p=0; p < sg->nnodes; p++)
//        if (sg->node[p].root == p)
//        {
//            volume[p]=sg->node[p].volume;
//            InsertRealHeap(H,p);
//        }
//
//    i=0;
//    while ((!IsEmptyRealHeap(H))&&(i < nclusters))
//    {
//        RemoveRealHeap(H,&p);
//        i++;
//        InsertSet(&S,p);
//    }
//
//    DestroyRealHeap(&H);
//    free(volume);
//
//    return(S);
//}

//SgCTree *CreateSgMaxTree(Subgraph *g)
//{
//    SgCTree *ctree=(SgCTree *)calloc(1,sizeof(SgCTree));
//    int *dad,*cmap, *tmp, *level, Imax, *val;
//    int i,r,p,q,rp,rq,n;
//    GQueue *Q;
//    int *nsons=NULL;
//    int *size=NULL;
//    Set *adj;
//
//    n           = g->nnodes;
//    level       = AllocIntArray(n);
//    val         = AllocIntArray(n);
//    Imax        = INT_MIN;
//
//    for (p=0; p < n; p++)
//    {
//        val[p] = (int)g->node[p].dens;
//        if (val[p]>Imax)
//            Imax=val[p];
//    }
//
//    ctree->cmap = AllocIntArray(n);
//    cmap        = ctree->cmap;
//    ctree->root = NIL; /* Tree is empty */
//    dad         = AllocIntArray(n);
//    size        = AllocIntArray(n);
//    Q           = CreateGQueue(Imax+1,n,level);
//    SetTieBreak(Q,LIFOBREAK);
//
//    for (p=0; p < n; p++)
//    {
//        dad[p]  =NIL;
//        cmap[p] =p;
//        level[p]=Imax-val[p];
//        size[p]=1;
//        InsertGQueue(&Q,p);
//    }
//
//    while (!EmptyGQueue(Q))
//    {
//        p  = RemoveGQueue(Q);
//        rp = SgRepresentative(cmap,p);
//
//        adj = g->node[p].adj;
//        while (adj != NULL)
//        {
//            q = adj->elem;
//            if (val[p]==val[q])  /* propagate component */
//            {
//                if (Q->L.elem[q].color==GRAY)
//                {
//                    cmap[q]=rp;
//                    if (p==rp) size[rp]=size[rp]+1;
//                    UpdateGQueue(&Q,q,level[p]);
//                }
//            }
//            else
//            {
//                if (val[p] < val[q]) /* find current dad of rq */
//                {
//                    rq = SgRepresentative(cmap,q);
//                    r  = SgAncestor(dad,cmap,rq);
//                    if (r == NIL)   /* rp is dad of the rq */
//                    {
//                        dad[rq]=rp;
//                    }
//                    else
//                    {
//                        if (val[r]==val[rp])  /* merge components */
//                        {
//                            if (r != rp)
//                            {
//                                if (size[rp] <= size[r])
//                                {
//                                    cmap[rp] = r;
//                                    size[r]  = size[r] + size[rp];
//                                    rp = r;
//                                }
//                                else
//                                {
//                                    cmap[r]  = rp;
//                                    size[rp] = size[rp] + size[r];
//                                }
//                            }
//                        }
//                        else   /* val[r] > val[rp] */
//                        {
//                            dad[r] = rp; /* rp is dad of r */
//                        }
//                    }
//                }
//            }
//            adj = adj->next;
//        }
//    }
//
//    free(size);
//    DestroyGQueue(&Q);
//    free(level);
//
//    /* Compress cmap map and count number of nodes */
//
//    ctree->numnodes = 0;
//    for (p=0; p < n; p++)
//    {
//        if (dad[cmap[p]]!=NIL)
//            r = cmap[p];
//        cmap[p] = SgRepresentative(cmap,p);
//
//        if (cmap[p]==p)
//            ctree->numnodes++;
//    }
//
//    /* Create and initialize nodes of the MaxTree. */
//
//    ctree->node = (SgCTNode *)calloc(ctree->numnodes,sizeof(SgCTNode));
//    tmp         = AllocIntArray(n);
//    for (p=0; p < n; p++)
//    {
//        tmp[p]=NIL;
//    }
//
//    i = 0;
//    for (p=0; p < n; p++)
//    {
//        if (cmap[p]==p)
//        {
//            ctree->node[i].level = val[p];
//            ctree->node[i].comp  = p;
//            tmp[p]               = i;
//            ctree->node[i].dad   = NIL;
//            ctree->node[i].son   = NULL;
//            ctree->node[i].numsons = 0;
//            ctree->node[i].size  = 0;
//            i++;
//        }
//    }
//
//    free(val);
//
//    /* Make the component map to point back to the maxtree. */
//
//    for (p=0; p < n; p++)
//    {
//        if (tmp[p] == NIL)
//            tmp[p] = tmp[cmap[p]];
//    }
//
//    for (p=0; p < n; p++)
//    {
//        cmap[p] = tmp[p];
//    }
//    free(tmp);
//
//    /* Copy dad information to the maxtree and find its root */
//
//    for (i=0; i < ctree->numnodes; i++)
//    {
//        if (dad[ctree->node[i].comp]!=NIL)
//            ctree->node[i].dad = cmap[dad[ctree->node[i].comp]];
//        else
//        {
//            ctree->node[i].dad = NIL;
//            ctree->root = i;
//        }
//    }
//    free(dad);
//
//    /* Copy son information to the maxtree */
//
//    nsons = AllocIntArray(ctree->numnodes);
//    for (i=0; i < ctree->numnodes; i++)
//    {
//        p = ctree->node[i].dad;
//        if (p != NIL)
//        {
//            nsons[p]++;
//        }
//    }
//    for (i=0; i < ctree->numnodes; i++)
//    {
//        if (nsons[i] != 0)
//        {
//            ctree->node[i].son = AllocIntArray(nsons[i]);
//        }
//    }
//    free(nsons);
//
//    for (i=0; i < ctree->numnodes; i++)
//    {
//        p = ctree->node[i].dad;
//        if (p != NIL)
//        {
//            ctree->node[p].son[ctree->node[p].numsons]=i;
//            ctree->node[p].numsons++;
//        }
//    }
//
//    /* Compute size of each node */
//
//    for (p=0; p < n; p++)
//        ctree->node[cmap[p]].size++;
//
//    return(ctree);
//}
//
//void DestroySgCTree(SgCTree **ctree)
//{
//    SgCTree *tmp=*ctree;
//    int i;
//
//    if (tmp != NULL)
//    {
//        free(tmp->cmap);
//        for (i=0; i < tmp->numnodes; i++)
//        {
//            if (tmp->node[i].numsons!=0)
//                free(tmp->node[i].son);
//        }
//        free(tmp->node);
//        free(tmp);
//        *ctree = NULL;
//    }
//}
//
//void SgCumSize(SgCTree *ctree, int i)
//{
//    int s,j;
//
//    for (j=0; j < ctree->node[i].numsons; j++)
//    {
//        s = ctree->node[i].son[j];
//        SgCumSize(ctree,s);
//        ctree->node[i].size = ctree->node[i].size + ctree->node[s].size;
//    }
//    return;
//}
//
//int SgAreaLevel(SgCTree *ctree, int *level, int i, int thres)
//{
//
//    if (i==-1)// passou o pai da raiz
//        return(0);
//
//    if ((ctree->node[i].size > thres)||(i==ctree->root))
//        return(ctree->node[i].level);
//    else
//        return(level[i]=SgAreaLevel(ctree,level,ctree->node[i].dad,thres));
//}
//
//int *SgAreaOpen(Subgraph *g, int thres)
//{
//    SgCTree *ctree=NULL;
//    int i,p;
//    int *fval;
//    int *level=NULL;
//
//    ctree = CreateSgMaxTree(g);
//    SgCumSize(ctree,ctree->root);
//    level = AllocIntArray(ctree->numnodes);
//    for (i=0; i < ctree->numnodes; i++)
//        level[i]=ctree->node[i].level;
//
//    for (i=0; i < ctree->numnodes; i++)
//        if (ctree->node[i].numsons==0)
//            level[i]=SgAreaLevel(ctree,level,i,thres);
//    fval = AllocIntArray(g->nnodes);
//    for (p=0; p < g->nnodes; p++)
//        fval[p]=level[ctree->cmap[p]];
//    DestroySgCTree(&ctree);
//    free(level);
//    return(fval);
//}
//
//int SgVolumeLevel(SgCTree *ctree, int *level, int i, int thres, int cumvol)
//{
//    int dad,vol=cumvol;
//
//    if (i==-1)// passou o pai da raiz
//        return(0);
//
//    dad = ctree->node[i].dad;
//    if (dad != NIL)
//        vol = cumvol+
//              abs(ctree->node[i].level-ctree->node[dad].level)*ctree->node[i].size;
//
//    if ((vol > thres)||(i==ctree->root))
//        return(ctree->node[i].level);
//    else
//        return(level[i]=SgVolumeLevel(ctree,level,dad,thres,vol));
//}
//
//int *SgVolumeOpen(Subgraph *g, int thres)
//{
//    SgCTree *ctree=NULL;
//    int i,p;
//    int *fval=NULL;
//    int *level=NULL;
//
//    ctree = CreateSgMaxTree(g);
//    SgCumSize(ctree,ctree->root);
//    level = AllocIntArray(ctree->numnodes);
//    for (i=0; i < ctree->numnodes; i++)
//        level[i]=ctree->node[i].level;
//    for (i=0; i < ctree->numnodes; i++)
//        if (ctree->node[i].numsons==0)
//            level[i]=SgVolumeLevel(ctree,level,i,thres,0);
//    fval = AllocIntArray(g->nnodes);
//    for (p=0; p < g->nnodes; p++)
//        fval[p]=level[ctree->cmap[p]];
//    DestroySgCTree(&ctree);
//    free(level);
//    return(fval);
//}
//
//int SgRepresentative(int *cmap, int p)
//{
//    if (cmap[p]==p)
//        return(p);
//    else
//        return(cmap[p]=SgRepresentative(cmap,cmap[p]));
//}
//
//int SgAncestor(int *dad, int *cmap, int rq)
//{
//    int r,ro;
//
//    ro = r  = dad[rq];
//    while (r != NIL)
//    {
//        ro = r = SgRepresentative(cmap,r);
//        r  = dad[r];
//    }
//    return(ro);
//}


Subgraph* SplitSubgraphByTrueLabel(Subgraph* sg, int label)
{
    if(sg == NULL)
    {
        Error("Subgraph NULL ", "SplitSubgraphByTrueLabel");
    }

    int nnodes = CountNodesByTrueLabel(sg, label);

    Subgraph* sgresult = CreateSubgraph(nnodes);

    int i, i0 = 0;

    sgresult->nfeats = sg->nfeats;
    sgresult->df = sg->df;

    for(i = 0; i < sg->nnodes; i++)
    {
        if(sg->node[i].truelabel == label)
        {
            CopySNode(&sgresult->node[i0++], &sg->node[i], sg->nfeats);
        }
    }

    return sgresult;
}

int CountNodesByTrueLabel(Subgraph* sg, int label)
{
    int i;
    int count = 0;

    for(i = 0; i < sg->nnodes; i++)
        if(sg->node[i].truelabel == label) count++;

    return count;
}

