#include "oldift.h"
#include <pthread.h>
#include "orderlist.h"
#include "inhomogeneity.h"
#include "braincluster_spectral.h"
#include "rigidregistration3.h"
#include "filelist.h"
#include "featmap.h"
#include "gradient.h"
#include "braincluster_opf.h"

BCArcWeightFun BCArcWeight = BCEuclDist;

BCSubgraph *CreateBCSubgraph( int nnodes ) {
  BCSubgraph *sg = ( BCSubgraph * ) calloc( 1, sizeof( BCSubgraph ) );
  int i;
  
  sg->nnodes = nnodes;
  sg->node   = ( BCNode * ) calloc( nnodes, sizeof( BCNode ) );
  sg->ordered_list_of_nodes = ( int * ) calloc( nnodes, sizeof( int ) );
  sg->feat = NULL;
  
  if (sg->node == NULL)
    Error("Cannot allocate nodes","CreateBCSubgraph");
  
  for (i=0; i < sg->nnodes; i++) {
    sg->node[ i ].plt = NULL;
    sg->node[ i ].adj = NULL;
    sg->node[ i ].feat = NULL;
  }
  return( sg );
}


BCSubgraph *BCRandomSampl3( Scene *scn, Scene *mask, int nnodes ) {
  BCSubgraph *sg = NULL;
  Scene *used = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  int i, n, p;

  /* Obtain nnodes random samples with similar histogram */
  n = mask->n;
  sg = CreateBCSubgraph( nnodes );
  i = 0;
  while( i < nnodes ) {
    p = RandomInteger( 0, n - 1 );
    if( ( mask == NULL ) || ( mask->data[ p ] ) ) {
      if (used->data[ p ] == 0 ) {
	sg->node[ i ].pixel = p;
	used->data[ p ] = 1;
	i++;
      }
    }
  }
  
  DestroyScene( &used );
  return( sg );
}

BCSubgraph *BCUnifSampl3( Scene *scn, Scene *mask, int dx, int dy, int dz ) {
  BCSubgraph *sg = NULL;
  int x, y, z, nnodes, i, p;

  if( mask != NULL ) {
    nnodes = 0;
    for( z = 0; z < scn->zsize; z = z + dz )
      for( y = 0; y < scn->ysize; y = y + dy ) {
	for( x = 0; x < scn->xsize; x = x + dx ) {
	  p = x + scn->tby[ y ] + scn->tbz[ z ];
	  if( mask->data[ p ] )
	    nnodes++;
	}
      }
    sg = CreateBCSubgraph( nnodes );
    i = 0;
    for( z = 0; z < scn->zsize; z = z + dz ) {
      for( y = 0; y < scn->ysize; y = y + dy ) {
	for( x = 0; x < scn->xsize; x = x + dx ) {
	  p = x + scn->tby[ y ] + scn->tbz[ z ];
	  if( mask->data[ p ] ) {
	    sg->node[ i ].pixel = p;
	    i++;
	  }
	}
      }
    }
  }
  else {
    nnodes = 0;
    for( z = 0; z < scn->zsize; z = z + dz ) {
      for( y = 0; y < scn->ysize; y = y + dy ) {
	for( x = 0; x < scn->xsize; x = x + dx ) {
	  p = x + scn->tby[ y ] + scn->tbz[ z ];
	  nnodes++;
	}
      }
    }
    sg = CreateBCSubgraph( nnodes );
    i = 0;
    for( z = 0; z < scn->zsize; z = z + dz ) {
      for( y = 0; y < scn->ysize; y = y + dy ) {
	for( x = 0; x < scn->xsize; x = x + dx ) {
	  p = x + scn->tby[ y ] + scn->tbz[ z ];
	  sg->node[ i ].pixel = p;
	  i++;
	}
      }
    }
  }
  return( sg );
}

/* Compute Euclidean distance between feature vectors */
float BCEuclDist( float *f1, float *f2, int n ) {
  int i;
  float dist = 0.0f;
  for( i = 0; i < n; i++ )
    dist += ( f1[ i ] - f2[ i ] ) * ( f1[ i ] - f2[ i ] );

  return( dist );
}

/* Discretizes original distance */
float BCEuclDistLog( float *f1, float *f2, int n ) {
  return( ( ( float ) BCMAXARCW * log( BCEuclDist( f1, f2, n ) + 1 ) ) );
}

float* BCCreateArcs( BCSubgraph *sg, int kmax ) {
  int i, j, l, k;
  float dist;
  int *nn;
  float *d;
  float *maxdists;
  
  nn = AllocIntArray( kmax + 1 );
  d = AllocFloatArray( kmax + 1 );
  maxdists = AllocFloatArray( kmax );

  sg->df = 0.0;
  sg->node[ 0 ].adj = AllocIntArray( kmax * sg->nnodes );
  for( i = 0; i < sg->nnodes; i++ ) {
    /* sg->node[ i ].adj = AllocIntArray( sg->nnodes ); */
    sg->node[ i ].adj = &( sg->node[ 0 ].adj[ i * kmax ] );
    for( l = 0; l < kmax; l++ )
      d[ l ] = FLT_MAX;
    for( j = 0; j < sg->nnodes; j++ ) {
      if( j != i ) {
	d[ kmax ] = BCArcWeight( sg->node[ i ].feat, sg->node[ j ].feat, sg->nfeats );
	nn[ kmax ] = j;
	k = kmax;
	while( ( k > 0 ) && ( d[ k ] < d[ k - 1 ] ) ) {
	  dist = d[ k ];
	  l = nn[ k ];
	  d[ k ] = d[ k - 1 ];
	  nn[ k ] = nn[ k - 1 ];
	  d[ k - 1 ] = dist;
	  nn[ k - 1 ] = l;
	  k--;
	}
      }
    }
    sg->node[ i ].radius = 0.0;
    /* making sure that the adjacent nodes be sorted in non-decreasing order */
    for( l = kmax - 1; l >= 0; l-- ) {
      if( d[ l ] != FLT_MAX ) {
	if( d[ l ] > sg->df )
	  sg->df = d[ l ];
	if( d[ l ] > sg->node[ i ].radius )
	  sg->node[ i ].radius = d[ l ];
	if( d[ l ] > maxdists[ l ] )
	  maxdists[ l ] = d[ l ];
	/* adding the current neighbor at the beginnig of the list */
	sg->node[ i ].adj[ l ] = nn[ l ];
      }
    }
  }
  free( d );
  free( nn );

  if( sg->df < 0.00001 )
    sg->df = 1.0;
  return maxdists;
}

void BCPDF( BCSubgraph *sg ) {
  int i,nelems;
  int k;
    int kmax = sg->bestk;
  double dist;
  float *value;
  int *adj;
  
  value = AllocFloatArray( sg->nnodes );
  sg->K = ( 2.0 * ( float ) sg->df / 9.0 );
  sg->mindens = FLT_MAX;
  sg->maxdens = -FLT_MAX;
  for( i = 0; i < sg->nnodes; i++ ) {
    adj = sg->node[ i ].adj;
    value[ i ] = 0.0;
    nelems = 1;
    for( k = 0; k < kmax; k++ ) {
      dist = BCArcWeight( sg->node[ i ].feat, sg->node[ adj[ k ] ].feat, sg->nfeats );
      value[ i ] += exp( -dist / sg->K );
      nelems++;
    }
    value[ i ] = ( value[ i ] / ( float ) nelems );
    if( value[ i ] < sg->mindens )
      sg->mindens = value[ i ];
    if( value[ i ] > sg->maxdens )
      sg->maxdens = value[ i ];
  }
  if( sg->mindens == sg->maxdens ) {
    for( i = 0; i < sg->nnodes; i++ )
      sg->node[ i ].dens = BCMAXDENS;
  }
  else {
    for( i = 0; i < sg->nnodes; i++ )
      sg->node[ i ].dens = ( ( float ) ( BCMAXDENS - 1 ) * ( value[ i ] - sg->mindens ) / ( float ) ( sg->maxdens - sg->mindens ) ) + 1.0;
  }
  free(value);
}

void BCSPDF( BCSubgraph *sg, Scene *mask, int first_voxel, int last_voxel ) {
  float dist;
  int i, j, k;
  int nelems;
  Voxel u,v;
  AdjRel3 *A = NULL;

  sg->K = ( 2.0 * ( float ) sg->df / 9.0 );
  sg->mindens = INT_MAX;
  sg->maxdens = INT_MIN;

  A = Ellipsoid( sg->di / mask->dx, sg->di / mask->dy, sg->di / mask->dz );
  for( i = first_voxel; i <= last_voxel; i++ ) {
    sg->node[ i ].dens = 0.0;
    if( mask->data[ i ] != 0 ) {
      nelems = 1;
      sg->node[ i ].dens = 0.0;
      u.x = VoxelX( mask, i );
      u.y = VoxelY( mask, i );
      u.z = VoxelZ( mask, i );
      for( k = 1; k < A->n; k++ ) {  /* image space restriction */
	v.x = u.x + A->dx[ k ];
	v.y = u.y + A->dy[ k ];
	v.z = u.z + A->dz[ k ];
	if( ValidVoxel( mask, v.x, v.y, v.z ) ) {
	  j = VoxelAddress( mask, v.x, v.y, v.z );
	  if( mask->data[ j ] != 0 ) {
	    dist = BCArcWeight( sg->node[ i ].feat, sg->node[ j ].feat, sg->nfeats );
	    if( dist <= sg->df ) {
	      nelems++;
	      sg->node[ i ].dens += exp( -dist / sg->K );
	    }
	  }
	}
      }
      sg->node[ i ].dens = sg->node[ i ].dens / ( float ) nelems;
      if ( sg->node[ i ].dens < sg->mindens )
	sg->mindens = sg->node[ i ].dens;
      if ( sg->node[ i ].dens > sg->maxdens )
	sg->maxdens = sg->node[ i ].dens;
    }
  }
  for( i = first_voxel; i <= last_voxel; i++ )
    sg->node[ i ].dens = ( ( BCMAXDENS - 1 ) * sg->node[ i ].dens - sg->mindens / ( sg->maxdens - sg->mindens ) ) + 1;
  DestroyAdjRel3( &A );

  /*   printf( "sg->K:%f, maxdens:%f, mindens:%f\n", sg->K, sg->maxdens, sg->mindens ); */
}

 /* OPFClustering computation only for sg->bestk neighbors */
void BCOPFClustering( BCSubgraph *sg ) {
  int *adj_i, *adj_j;
  int *plt_i;
  char insert_i;
  int i, j;
  int nadj;
  int p, q, l, ki, kj;
    int kmax = sg->bestk;
  float tmp, *pathval = NULL;
  RealHeap *Q = NULL;

  /* zeroing amount of nodes on plateaus */
  for( i = 0; i < sg->nnodes; i++ )
    sg->node[ i ].nplatadj = 0;
  /* Add arcs to guarantee symmetry on plateaus */
  for( i = 0; i < sg->nnodes; i++ ) {
    adj_i = sg->node[ i ].adj;
    ki = 0;
    while( ki < kmax ) {
      j = adj_i[ ki ];
      if( sg->node[ i ].dens == sg->node[ j ].dens ) {
	 /* insert i in the adjacency of j if it is not there. */
	adj_j = sg->node[ j ].adj;
	insert_i = 1;
	kj = 0;
	while( kj < kmax ) {
	  if( i == adj_j[ kj ] ) {
	    insert_i = 0;
	    break;
	  }
	  kj++;
	}
	if( insert_i ) {
	  if( sg->node[ j ].nplatadj % kmax == 0 ) {
	    /* printf( "kmax = %d, Realloc with %d nodes.\n", kmax, sg->node[ j ].nplatadj + kmax ); */
	    sg->node[ j ].plt = ( int* ) realloc( sg->node[ j ].plt, ( sg->node[ j ].nplatadj + kmax ) * sizeof( int ) );
	  }
	  /* printf( "sg->node[ %d ].nplatadj:%d\n", j, sg->node[ j ].nplatadj ); */
	  sg->node[ j ].plt[ sg->node[ j ].nplatadj ] = i;
	  sg->node[ j ].nplatadj++; /* number of adjacent nodes on plateaus */
	}
      }
      ki++;
    }
  }
  
  /* Compute clustering */
  pathval = AllocFloatArray( sg->nnodes );
  Q = CreateRealHeap( sg->nnodes, pathval );
  SetRemovalPolicyRealHeap( Q, MAXVALUE );
  
  for( p = 0; p < sg->nnodes; p++ ) {
    pathval[ p ] = sg->node[ p ].dens - 1;
    sg->node[ p ].pred = NIL;
    InsertRealHeap( Q, p );
  }
  
  l = 0;
  i = 0;
  while( !IsEmptyRealHeap( Q ) ) {
    RemoveRealHeap( Q,&p );
    sg->ordered_list_of_nodes[i] = p;
    i++;
    
    if( sg->node[ p ].pred == NIL ) {
      pathval[ p ] = sg->node[ p ].dens;
      sg->node[ p ].label = l;
      l++;
    }
    
    nadj = sg->node[ p ].nplatadj + kmax; /* total amount of neighbors */
    for( adj_i = sg->node[ p ].adj, plt_i = sg->node[ p ].plt, ki = 0; ki < nadj; ki++ ) {
      if( ki < kmax )
	q = adj_i[ ki ];
      else
	q = plt_i[ ki - kmax ];
      if( Q->color[ q ] != BLACK ) {
	tmp = MIN( pathval[ p ], sg->node[ q ].dens );
	if( tmp > pathval[ q ] ) {
	  UpdateRealHeap( Q, q, tmp );
	  sg->node[ q ].pred = p;
	  sg->node[ q ].label = sg->node[ p ].label;
	}
      }
    }
  }
  
  sg->nlabels = l;
  
  DestroyRealHeap( &Q );
  free( pathval );
}

void BCSpatialOPFClustering( BCSubgraph *sg, Scene *mask ) {
  Voxel u,v;
  double dist;
  int k, p, q, l;
  RealHeap *Q = NULL;
  float tmp, *pathval = NULL;
  AdjRel3 *A = NULL;

  A = Ellipsoid( sg->di / mask->dx, sg->di / mask->dy, sg->di / mask->dz );
  
  pathval = AllocFloatArray( sg->nnodes );
  Q = CreateRealHeap( sg->nnodes, pathval );
  SetRemovalPolicyRealHeap( Q, MAXVALUE );
  
  for( p = 0; p < sg->nnodes; p++ ) {
    if( mask->data[ p ] != 0 ) {
      pathval[ p ] = sg->node[ p ].dens - 1;
      sg->node[ p ].pred  = NIL;
      InsertRealHeap( Q, p );
    }
    else {
      pathval[ p ] = 0;
      sg->node[ p ].pred  = NIL;
    }
  }
  l = 0;
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
    for ( k = 1; k < A->n; k++ ) {  /* image space restriction */
      v.x = u.x + A->dx[ k ];
      v.y = u.y + A->dy[ k ];
      v.z = u.z + A->dz[ k ];
      if( ValidVoxel( mask, v.x, v.y, v.z ) ) {
	q = VoxelAddress( mask, v.x, v.y, v.z );
	if( mask->data[ q ] != 0 ) {
	  dist = BCArcWeight( sg->node[ p ].feat, sg->node[ q ].feat, sg->nfeats);
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

 /* Normalized cut computed only for sg->bestk neighbors */
float BCNormalizedCut( BCSubgraph *sg ) {
  int l, p, q, k;
    int kmax = sg->bestk;
  float ncut, dist;
  float *acumIC; /* acumulate weights inside each class */
  float *acumEC; /* acumulate weights between the class and a distinct one */
  int nadj;
  int *adj_p;
  int *plt_p;
  
  ncut = 0.0;
  acumIC = AllocFloatArray( sg->nlabels );
  acumEC = AllocFloatArray( sg->nlabels );
  
  for( p = 0; p < sg->nnodes; p++ ) {
    nadj = sg->node[ p ].nplatadj + kmax; /* for plateaus the number of adjacent nodes will be greater than the current kmax, but they should be considered */
    for( adj_p = sg->node[ p ].adj, plt_p = sg->node[ p ].plt, k = 0; k < nadj; k++ ) {
      if( k < kmax )
	q = adj_p[ k ];
      else
	q = plt_p[ k - kmax ];
	dist = BCArcWeight( sg->node[ p ].feat, sg->node[ q ].feat, sg->nfeats );
      if( dist > 0.0 ) {
	if( sg->node[ p ].label == sg->node[ q ].label )
	  acumIC[ sg->node[ p ].label ] += 1.0 / dist;  /* intra-class weight */
	else  /* inter - class weight */
	  acumEC[ sg->node[ p ].label ] += 1.0 / dist;  /* inter-class weight */
      }
    }
  }

  for ( l = 0; l < sg->nlabels; l++ )
    if ( acumIC[ l ] + acumEC[ l ]  > 0.0 ) ncut += (float) acumEC[ l ] / ( acumIC[ l ] + acumEC[ l ] );
  free( acumEC );
  free( acumIC );
  return( ncut );
}

void BCRemovePlateauNeighbors( BCSubgraph *sg ) {
  int i;

  for( i = 0; i < sg->nnodes; i++ ) {
    if( sg->node[ i ].nplatadj != 0 ) {
      free( sg->node[ i ].plt );
      sg->node[ i ].plt = NULL;
    }
    sg->node[ i ].nplatadj = 0;
  }
}

 /* Destroy Arcs */
void BCDestroyArcs( BCSubgraph *sg ) {
  int i;

  for( i = 0; i < sg->nnodes; i++ ) {
    if( sg->node[ i ].nplatadj != 0 ) {
      free( sg->node[ i ].plt );
      sg->node[ i ].plt = NULL;
    }
    sg->node[ i ].nplatadj = 0;
    /* if( sg->node[ i ].adj != NULL ) */
    /*   free( sg->node[ i ].adj ); */
  }
  if( sg->node[ 0 ].adj != NULL )
    free( sg->node[ 0 ].adj );
}

 /* Estimate the best k by minimum cut */
void BCBestkClustering( BCSubgraph *sg, int kmin, int kmax) {
  int k, bestk = kmax;
  float mincut = FLT_MAX;
  float nc;
  float* maxdists;

  maxdists = BCCreateArcs( sg, kmax );
    
   /* Find the best k */
  for( k = kmin; ( k <= kmax ) && ( mincut != 0.0 ); k++ ) {
    sg->df = maxdists[ k - 1 ];
    sg->bestk = k;
    
    BCPDF( sg );
    BCOPFClustering( sg );
    nc = BCNormalizedCut( sg );
    
    if( nc < mincut ) {
      mincut=nc;
      bestk = k;
    }
    BCRemovePlateauNeighbors( sg );
  }
  free( maxdists );
  sg->bestk = bestk;
  BCPDF( sg );
  BCOPFClustering( sg );
}

BCFeatures* CreateBCFeatures( Scene *scn, Scene *mask, int nfeats ) {
  int i, j;
  int max;
  int size;
  BCFeatures *f = ( BCFeatures* ) calloc( 1, sizeof( BCFeatures ) );

  if( mask == NULL ) {
    f->Imax = MaximumValue3( scn );
    f->nelems = scn->n;
    f->nfeats = nfeats;
    f->feat = ( float* ) calloc( f->nelems * nfeats, sizeof( float ) );
    f->nodefeat = ( float** ) calloc( f->nelems, sizeof( float* ) );
    for( i = 0; i < f->nelems; i++ )
      f->nodefeat[ i ] = &( f->feat[ i * nfeats ] );
  }
  else {
    max = 0;
    size = 0;
    for( i = 0; i < scn->n; i++ ) {
      if( mask->data[ i ] != 0 ) {
	size++;
	if( max < scn->data[ i ] )
	  max = scn->data[ i ];
      }
    }
    f->Imax = max;
    f->nelems = size;
    f->nfeats = nfeats;
    f->feat = ( float* ) calloc( size * nfeats, sizeof( float ) );
    f->nodefeat = ( float** ) calloc( scn->n, sizeof( float* ) );
    for( i = 0, j = 0; j < size; i++ ) {
      if( mask->data[ i ] != 0 ) {
	f->nodefeat[ i ] = &( f->feat[ j * nfeats ] );
	j++;
      }
    }
  }
  return( f );
}

void DestroyBCFeatures( BCFeatures **f ) {
  if( ( *f ) != NULL ) {
    if( ( *f )->nodefeat != NULL )
      free( ( *f )->nodefeat );
    if( ( *f )->feat != NULL )
      free( ( *f )->feat );
    free( *f );
    ( *f ) = NULL;
  }
}

void WriteFeats( BCFeatures* feats, Scene *mask, char name[ ] ) {
  FILE *fio = NULL;
  int p, i, j;

  fio = fopen( name, "w" );
  for( p = 0, j = 0; p < mask->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      fprintf( fio, "node %d(%d): ", p, j );
      j++;
      for( i = 0; i < feats->nfeats; i++ )
	fprintf( fio, "%.4f, ", feats->nodefeat[ p ][ i ] );
      fprintf( fio, "\n" );
    }
  }
}

void *MedianSceneFeaturesThread( void *arg ) {
  Scene *scn = NULL;
  Scene *mask = NULL;
  AdjRel3 *A = NULL;
  BCFeatures *feats = NULL;
  int p, q, i, j, k;
  Voxel u, v;
  int *val = NULL;
  int pn, p0, p1;
  int t_threads;
  int first_voxel;
  int last_voxel;
  int valid_range;
  int nfeats;
  float fmax;

  pn = ( ( pointer_type* ) arg )[ 0 ];
  t_threads = ( ( pointer_type* ) arg )[ 1 ];
  first_voxel = ( ( pointer_type* ) arg )[ 2 ];
  last_voxel = ( ( pointer_type* ) arg )[ 3 ];
  scn = ( ( Scene** ) arg )[ 4 ];
  mask = ( ( Scene** ) arg )[ 5 ];
  A = ( ( AdjRel3** ) arg )[ 6 ];
  feats = ( ( BCFeatures** ) arg )[ 7 ];
  nfeats = feats->nfeats;
  valid_range = last_voxel - first_voxel + 1;
  
  if( t_threads == 1 ) {
    p0 = first_voxel;
    p1 = last_voxel + 1;
  }
  else if( ( pn > 0 ) && ( pn < t_threads - 1 ) ) {
    p0 = first_voxel + ( 2 * pn + 1 ) * ( valid_range / ( 2 * t_threads + 2 ) );
    p1 = first_voxel + ( 2 * pn + 3 ) * ( valid_range / ( 2 * t_threads + 2 ) );
  }
  else if( pn == 0 ) {
    p0 = first_voxel;
    p1 = first_voxel + 3              * ( valid_range / ( 2 * t_threads + 2 ) );
  }
  else {
    p0 = first_voxel + ( 2 * pn + 1 ) * ( valid_range / ( 2 * t_threads + 2 ) );
    p1 = last_voxel + 1;
  }

  val = AllocIntArray( A->n );
  fmax = ( float ) feats->Imax;

  for( p = p0; p < p1; p++ ) {
    if( mask->data[ p ] != 0 ) {
      u.x = VoxelX( scn, p );
      u.y = VoxelY( scn, p );
      u.z = VoxelZ( scn, p );
      for( j = 0; j < A->n; j++ ) {
	v.x = u.x + A->dx[ j ];
	v.y = u.y + A->dy[ j ];
	v.z = u.z + A->dz[ j ];
	if( ValidVoxel( scn, v.x, v.y, v.z ) ) {
	  q = VoxelAddress( scn, v.x, v.y, v.z );
	  if ( mask->data[ q ] != 0 )
	    val[ j ] = scn->data[ q ];
	}
	else
	  val[ j ] = scn->data[ p ];
      }
      SelectionSort( val, A->n, INCREASING );
      k = -( nfeats / 2 );
      for( i = 0; i < nfeats; i++, k++ ) {
	feats->nodefeat[ p ][ i ] = ( float ) val[ A->n / 2 + k ] / fmax;
      }
    }
  }

  free( val );
  return( NULL );
}

BCFeatures *MedianSceneFeatures( Scene *scn, Scene *mask, int first_voxel, int last_voxel, float r, int t_threads ) {
  BCFeatures *feats = NULL;
  AdjRel3   *A = NULL;
  pointer_type pars[ t_threads ][ 15 ];
  pthread_t tid[ t_threads ];
  int       i;

  A = Ellipsoid( r / scn->dx, r / scn->dy, r / scn->dz );
  feats = CreateBCFeatures( scn, mask, MAX( A->n / 3, MIN( A->n, 3 ) ) );

  for( i = 0; i < t_threads; i++ ) {
    pars[ i ][ 0 ] = i;
    pars[ i ][ 1 ] = t_threads;
    pars[ i ][ 2 ] = first_voxel;
    pars[ i ][ 3 ] = last_voxel;
    pars[ i ][ 4 ] = ( pointer_type ) scn;
    pars[ i ][ 5 ] = ( pointer_type ) mask;
    pars[ i ][ 6 ] = ( pointer_type ) A;
    pars[ i ][ 7 ] = ( pointer_type ) feats;
  }

  for( i = 0; i < t_threads; i++ )
    pthread_create( &tid[ i ], NULL, MedianSceneFeaturesThread, &pars[ i ][ 0 ] );
  for( i = 0; i < t_threads; i++ )
    pthread_join( tid[ i ], NULL );

  DestroyAdjRel3( &A );
  return( feats );
}

BCFeatures *MedianSceneFeaturesAtlas( Scene *scn, Scene *mask, Scene *dark_atlas, Scene *light_atlas, int first_voxel, int last_voxel, float r, int t_threads ) {
  BCFeatures *feats = NULL;
  AdjRel3   *A = NULL;
  pointer_type pars[ t_threads ][ 15 ];
  pthread_t tid[ t_threads ];
  int       i;

  A = Ellipsoid( r / scn->dx, r / scn->dy, r / scn->dz );
  feats = CreateBCFeatures( scn, mask, MAX( A->n / 3, MIN( A->n, 3 ) ) + 2 );
  feats->nfeats -= 2;

  for( i = 0; i < t_threads; i++ ) {
    pars[ i ][ 0 ] = i;
    pars[ i ][ 1 ] = t_threads;
    pars[ i ][ 2 ] = first_voxel;
    pars[ i ][ 3 ] = last_voxel;
    pars[ i ][ 4 ] = ( pointer_type ) scn;
    pars[ i ][ 5 ] = ( pointer_type ) mask;
    pars[ i ][ 6 ] = ( pointer_type ) A;
    pars[ i ][ 7 ] = ( pointer_type ) feats;
  }

  for( i = 0; i < t_threads; i++ )
    pthread_create( &tid[ i ], NULL, MedianSceneFeaturesThread, &pars[ i ][ 0 ] );
  for( i = 0; i < t_threads; i++ )
    pthread_join( tid[ i ], NULL );

  feats->nfeats += 2;
  for( i = 0; i < feats->nelems; i++ ) {
    if( mask->data[ i ] != 0 ) {
      feats->nodefeat[ i ][ feats->nfeats - 2 ] = ( float ) 0.5 * dark_atlas->data[ i ] / ( ( float ) dark_atlas->maxval );
      feats->nodefeat[ i ][ feats->nfeats - 1 ] = ( float ) 0.5 * light_atlas->data[ i ] / ( ( float ) light_atlas->maxval );
    }
  }
  
  DestroyAdjRel3( &A );
  return( feats );
}

void *CoOccurFeaturesThread( void *arg ) {
  BCFeatures *feats;
  Scene *scn = NULL;
  Scene *mask = NULL;
  AdjRel3 *A = NULL;
  int     i, j, p, q;
  int     dst; 
  int     *qmin, *dmin;
  Voxel   u,v;
  int pn, p0, p1;
  int t_threads;
  int nfeats, vfeats;
  int first_voxel;
  int last_voxel;
  int valid_range;
  float fmax;

  pn = ( ( pointer_type* ) arg )[ 0 ];
  t_threads = ( ( pointer_type* ) arg )[ 1 ];
  first_voxel = ( ( pointer_type* ) arg )[ 2 ];
  last_voxel = ( ( pointer_type* ) arg )[ 3 ];
  scn = ( ( Scene** ) arg )[ 4 ];
  mask = ( ( Scene** ) arg )[ 5 ];
  A = ( ( AdjRel3** ) arg )[ 6 ];
  feats = ( ( BCFeatures** ) arg )[ 7 ];
  nfeats = feats->nfeats;
  valid_range = last_voxel - first_voxel + 1;
  dmin = AllocIntArray( nfeats );
  qmin = AllocIntArray( nfeats );

  /* picking the part of the brain, according to the number of the thread. */
  if( t_threads == 1 ) {
    p0 = first_voxel;
    p1 = last_voxel + 1;
  }
  else if( ( pn > 0 ) && ( pn < t_threads - 1 ) ) {
    p0 = first_voxel + ( 2 * pn + 1 ) * ( valid_range / ( 2 * t_threads + 2 ) );
    p1 = first_voxel + ( 2 * pn + 3 ) * ( valid_range / ( 2 * t_threads + 2 ) );
  }
  else if( pn == 0 ) {
    p0 = first_voxel;
    p1 = first_voxel + 3              * ( valid_range / ( 2 * t_threads + 2 ) );
  }
  else {
    p0 = first_voxel + ( 2 * pn + 1 ) * ( valid_range / ( 2 * t_threads + 2 ) );
    p1 = last_voxel + 1;
  }

  fmax = ( double ) feats->Imax;

  for( p = p0; p < p1; p++ ) {
    /* Compute voxel intensities that are closest to p among p neighbors. */
    if( mask->data[ p ] != 0 ) {
      vfeats = 0;
      u.x  = VoxelX( scn, p );
      u.y  = VoxelY( scn, p );
      u.z  = VoxelZ( scn, p );
      for( i = 0; i < nfeats; i++ )
	dmin[ i ] = INT_MAX;
      for( i = 0; i < A->n; i++ ) {
	v.x = u.x + A->dx[ i ];
	v.y = u.y + A->dy[ i ];
	v.z = u.z + A->dz[ i ];
	if( ValidVoxel( scn, v.x, v.y, v.z ) ) {
	  q = VoxelAddress( scn, v.x, v.y, v.z );
	  if( mask->data[ q ] != 0 ) {
	    vfeats++;
	    dst = ( scn->data[ p ] - scn->data[ q ] ) * ( scn->data[ p ] - scn->data[ q ] );
	    for( j = 0; j < nfeats; j++ ) {
	      if( dst < dmin[ j ] ) {
		Change( &( dmin[ j ] ), &dst );
		Change( &( qmin[ j ] ), &q );
	      }
	    }
	  }
	}
      }
      /* Ensure nfeats features if p has not sufficient neighbors. */
      for( i = vfeats; i < nfeats; i++ ) {
	dmin[ i ] = dmin[ i % vfeats ];
	qmin[ i ] = qmin[ i % vfeats ];
      }
      /* Compute and sort features by intensity. */
      for( i = 0; i < nfeats; i++ ) { 
	feats->nodefeat[ p ][ i ] = ( float ) scn->data[ qmin[ i ] ] / fmax;
	for( j = i; j > 0; j-- ) {
	  if( feats->nodefeat[ p ][ j ] < feats->nodefeat[ p ][ j - 1 ] )
	    FChange( &( feats->nodefeat[ p ][ j - 1 ] ), &( feats->nodefeat[ p ][ j ] ) );
	}
      }
    }
  }
  free( dmin );
  free( qmin );
  return( NULL );
}

BCFeatures *CoOccurFeatures( Scene *scn, Scene *mask, int first_voxel, int last_voxel, int nfeats, int t_threads ) {
  BCFeatures *feats = NULL;
  AdjRel3 *A = NULL;
  pointer_type pars[ t_threads ][ 15 ];
  pthread_t tid[ t_threads ];
  int i;

  A = Ellipsoid( 1.1 / mask->dx, 1.1 / mask->dy, 1.1 / mask->dz );
  feats = CreateBCFeatures( scn, mask, nfeats );

  for( i = 0; i < t_threads; i++ ) {
    pars[ i ][ 0 ] = i;
    pars[ i ][ 1 ] = t_threads;
    pars[ i ][ 2 ] = first_voxel;
    pars[ i ][ 3 ] = last_voxel;
    pars[ i ][ 4 ] = ( pointer_type ) scn;
    pars[ i ][ 5 ] = ( pointer_type ) mask;
    pars[ i ][ 6 ] = ( pointer_type ) A;
    pars[ i ][ 7 ] = ( pointer_type ) feats;
  }

  for( i = 0; i < t_threads; i++ )
    pthread_create( &tid[ i ], NULL, CoOccurFeaturesThread, &pars[ i ][ 0 ] );
  for( i = 0; i < t_threads; i++ )
    pthread_join( tid[ i ], NULL );

  DestroyAdjRel3( &A );
  return( feats );
}

BCFeatures *CoOccurFeaturesAtlas( Scene *scn, Scene *mask, Scene *dark_atlas, Scene *light_atlas, int first_voxel, int last_voxel, int nfeats, int t_threads ) {
  BCFeatures *feats = NULL;
  AdjRel3 *A = NULL;
  pointer_type pars[ t_threads ][ 15 ];
  pthread_t tid[ t_threads ];
  int i;
  float min, max;

  A = Ellipsoid( 1.1 / mask->dx, 1.1 / mask->dy, 1.1 / mask->dz );
  feats = CreateBCFeatures( scn, mask, nfeats + 2 );
  feats->nfeats -= 2;

  for( i = 0; i < t_threads; i++ ) {
    pars[ i ][ 0 ] = i;
    pars[ i ][ 1 ] = t_threads;
    pars[ i ][ 2 ] = first_voxel;
    pars[ i ][ 3 ] = last_voxel;
    pars[ i ][ 4 ] = ( pointer_type ) scn;
    pars[ i ][ 5 ] = ( pointer_type ) mask;
    pars[ i ][ 6 ] = ( pointer_type ) A;
    pars[ i ][ 7 ] = ( pointer_type ) feats;
  }

  for( i = 0; i < t_threads; i++ )
    pthread_create( &tid[ i ], NULL, CoOccurFeaturesThread, &pars[ i ][ 0 ] );
  for( i = 0; i < t_threads; i++ )
    pthread_join( tid[ i ], NULL );

  feats->nfeats += 2;
  min = 1.0;
  max = 0.0;
  for( i = 0; i < mask->n; i++ ) {
    if( mask->data[ i ] != 0 ) {
      feats->nodefeat[ i ][ feats->nfeats - 2 ] = ( float ) 0.05 * dark_atlas->data[ i ] / ( ( float ) dark_atlas->maxval );
      if( min > feats->nodefeat[ i ][ feats->nfeats - 2 ] )
	min = feats->nodefeat[ i ][ feats->nfeats - 2 ];
      if( max < feats->nodefeat[ i ][ feats->nfeats - 2 ] )
	max = feats->nodefeat[ i ][ feats->nfeats - 2 ];
    }
  }
  for( i = 0; i < mask->n; i++ ) {
    if( mask->data[ i ] != 0 )
      feats->nodefeat[ i ][ feats->nfeats - 2 ] = ( feats->nodefeat[ i ][ feats->nfeats - 2 ] - min ) / ( max - min );
  }
  min = 1.0;
  max = 0.0;
  for( i = 0; i < mask->n; i++ ) {
    if( mask->data[ i ] != 0 ) {
      feats->nodefeat[ i ][ feats->nfeats - 1 ] = ( float ) 0.05 * light_atlas->data[ i ] / ( ( float ) light_atlas->maxval );
      if( min > feats->nodefeat[ i ][ feats->nfeats - 1 ] )
	min = feats->nodefeat[ i ][ feats->nfeats - 1 ];
      if( max < feats->nodefeat[ i ][ feats->nfeats - 1 ] )
	max = feats->nodefeat[ i ][ feats->nfeats - 1 ];
    }
  }
  for( i = 0; i < mask->n; i++ ) {
    if( mask->data[ i ] != 0 )
      feats->nodefeat[ i ][ feats->nfeats - 1 ] = ( feats->nodefeat[ i ][ feats->nfeats - 1 ] - min ) / ( max - min );
  }
  DestroyAdjRel3( &A );
  return( feats );
}

BCFeatures *AtlasFeatures( Scene *scn, Scene *mask, Scene *dark_atlas, Scene *light_atlas, int first_voxel, int last_voxel ) {
  BCFeatures *feats = NULL;
  int p;
  
  feats = CreateBCFeatures( scn, mask, 2 );
  for( p = first_voxel; p <= last_voxel; p++ ) {
    if( mask->data[ p ] != 0 ) {
      feats->nodefeat[ p ][ 0 ] = ( float ) dark_atlas->data[ p ] / ( ( float ) dark_atlas->maxval );
      feats->nodefeat[ p ][ 1 ] = ( float ) light_atlas->data[ p ] / ( ( float ) light_atlas->maxval );
    }
  }
  return( feats );
}

 /* Compute maximal standard deviation in flat regions. */
float FlatRegionStdDev( Scene *scn, Scene *mask, TissueMarker *tm, AdjRel3 *A, int nfeats, int protocol, int tissue ) {
  float std_dev;
  float max_std_dev;
  int min_range;
  int p, q, i;
  int total_pixels;
  int flat_pixels;
  float mean;
  
  if( tissue == WMGM_SEG )
     /* minimal range among GM and WM tissues is the range considered to get the maximal standard deviation. */
    min_range = MIN( abs( tm->gm - tm->gm_wm ), abs( tm->gm_wm - tm->wm ) );
  else
    min_range = MIN( abs( tm->csf_gm - tm->gm ), abs( tm->csf - tm->csf_gm ) );
   /* verificar se limites MIN/CSF/CSF_GM/GM/GM_WM/WM/MAX formam intervalos. Não pode haver valores que coincidem. Tratar isso. */
  printf( "min_range=%d\n", min_range );
  max_std_dev = 0;

  for( p = 0; p < scn->n; p++ ) {
    if( mask->data[ p ] != 0 ) { 
      /* Computing mean intensity of adjacency of p and checking if it is a flat region */
      flat_pixels = 0;
      total_pixels = 0;
      mean = 0;
      for( i = 0; i < A->n; i++ ) {
	q = ValidAdjacentAddressMask( mask, A, p, i );
	if( q != -1 ) {
	  total_pixels++;
	  mean += scn->data[ q ];
	  if( abs( scn->data[ p ] - scn->data[ q ] ) < min_range )
	    flat_pixels++;
	}
      }
      /* If it is part of a flat region that is not in the border of the mask, compute standard deviation. */
      if( ( total_pixels == A->n ) && ( flat_pixels == total_pixels ) ) { 
	mean /= total_pixels;
	std_dev = 0;
	total_pixels = 0;
	for( i = 0; i < A->n; i++ ) {
	  q = ValidAdjacentAddressMask( mask, A, p, i );
	  if( q != -1 ) {
	    total_pixels++;
	    std_dev += abs( scn->data[ q ] - mean );
	  }
	}
	std_dev /= total_pixels;
	if( max_std_dev < std_dev ) 
	  max_std_dev = std_dev;
      }
    }
  }

  printf( "max_std_dev=%f\n", max_std_dev );
  return( max_std_dev );
}

void *AdaptiveSceneFeaturesThread( void *arg ) {
  BCFeatures *feats;
  Scene *scn = NULL;
  Scene *mask = NULL;
  AdjRel3 *A = NULL;
  int     i, j, p, q;
  int     dst; 
  int     *qmin, *dmin;
  int     *val;
  int pn, p0, p1;
  int t_threads;
  int nfeats, vfeats;
  int first_voxel;
  int last_voxel;
  int valid_range;
  float max_std_dev;
  float std_dev;
  float mean;
  float fmax;

  //Scene *tmp;


  pn = ( ( pointer_type* ) arg )[ 0 ];
  t_threads = ( ( pointer_type* ) arg )[ 1 ];
  first_voxel = ( ( pointer_type* ) arg )[ 2 ];
  last_voxel = ( ( pointer_type* ) arg )[ 3 ];
  scn = ( ( Scene** ) arg )[ 4 ];
  mask = ( ( Scene** ) arg )[ 5 ];
  A = ( ( AdjRel3** ) arg )[ 6 ];
  feats = ( ( BCFeatures** ) arg )[ 7 ];
  max_std_dev = ( ( pointer_type* ) arg )[ 8 ];
  //tmp = ( ( Scene** ) arg )[ 9 ];
  nfeats = feats->nfeats;
  valid_range = last_voxel - first_voxel + 1;
  dmin = AllocIntArray( nfeats );
  qmin = AllocIntArray( nfeats );
  val = AllocIntArray( A->n );

  /* picking the part of the brain, according to the number of the thread. */
  if( t_threads == 1 ) {
    p0 = first_voxel;
    p1 = last_voxel + 1;
  }
  else if( ( pn > 0 ) && ( pn < t_threads - 1 ) ) {
    p0 = first_voxel + ( 2 * pn + 1 ) * ( valid_range / ( 2 * t_threads + 2 ) );
    p1 = first_voxel + ( 2 * pn + 3 ) * ( valid_range / ( 2 * t_threads + 2 ) );
  }
  else if( pn == 0 ) {
    p0 = first_voxel;
    p1 = first_voxel + 3              * ( valid_range / ( 2 * t_threads + 2 ) );
  }
  else {
    p0 = first_voxel + ( 2 * pn + 1 ) * ( valid_range / ( 2 * t_threads + 2 ) );
    p1 = last_voxel + 1;
  }

  fmax = ( double ) feats->Imax;

  int means, cooccurs;
  means = cooccurs = 0;
  
  for( p = p0; p < p1; p++ ) {
    /* Compute cooccurrence features, median features, and standard deviation. */
    if( mask->data[ p ] != 0 ) {
      for( i = 0; i < nfeats; i++ )
	dmin[ i ] = INT_MAX;
      vfeats = 0;
      mean = 0;
      std_dev = 0;
      for( i = 0; i < A->n; i++ ) {
	q = ValidAdjacentAddressMask( mask, A, p, i );
	if( q != -1 ) {
	  mean += scn->data[ q ];
	  val[ i ] = scn->data[ q ];
	  vfeats++;
	  dst = ( scn->data[ p ] - scn->data[ q ] ) * ( scn->data[ p ] - scn->data[ q ] );
	  for( j = 0; j < nfeats; j++ ) {
	    if( dst < dmin[ j ] ) {
	      Change( &( dmin[ j ] ), &dst );
	      Change( &( qmin[ j ] ), &q );
	    }
	  }
	}
	else {
	  val[ i ] = scn->data[ p ];
	}
      }
      /* standard deviation */
      mean /= vfeats;
      for( i = 0; i < A->n; i++ ) {
	q = ValidAdjacentAddressMask( mask, A, p, i );
	if( q != -1 ) 
	  std_dev += abs( scn->data[ q ] - mean );
      }
      std_dev /= vfeats;
      /* If standard deviation is small, use median features */
      if( std_dev < max_std_dev ) {
	means++;
	//tmp->data[ p ] = 1;
	SelectionSort( val, A->n, INCREASING );
	j = -( nfeats / 2 );
	for( i = 0; i < nfeats; i++, j++ ) {
	  feats->nodefeat[ p ][ i ] = ( float ) val[ A->n / 2 + j ] / fmax;
	}
      }
      /* If standard deviation is large, use co-occurrence features */
      else {
	cooccurs++;
	//tmp->data[ p ] = 2;
	/* Ensure nfeats features if p has not sufficient neighbors. */
	for( i = vfeats; i < nfeats; i++ ) {
	  dmin[ i ] = dmin[ i % vfeats ];
	  qmin[ i ] = qmin[ i % vfeats ];
	}
	/* Compute and sort features by intensity. */
	for( i = 0; i < nfeats; i++ ) { 
	  feats->nodefeat[ p ][ i ] = ( float ) scn->data[ qmin[ i ] ] / fmax;
	  for( j = i; j > 0; j-- ) {
	    if( feats->nodefeat[ p ][ j ] < feats->nodefeat[ p ][ j - 1 ] )
	      FChange( &( feats->nodefeat[ p ][ j - 1 ] ), &( feats->nodefeat[ p ][ j ] ) );
	  }
	}
      }
    }
  }
  printf( "means:%d, cooccurs:%d\n", means, cooccurs );

  free( val );
  free( dmin );
  free( qmin );
  return( NULL );
}

BCFeatures *AdaptiveSceneFeatures( Scene *scn, Scene *mask, int first_voxel, int last_voxel, int protocol, int tissue, float r, int t_threads ) {
  BCFeatures *feats = NULL;
  AdjRel3   *A = NULL;
  pointer_type pars[ t_threads ][ 15 ];
  pthread_t tid[ t_threads ];
  int       i;
  float std_dev;
  TissueMarker *tm;
  //Scene *tmp = CopyScene( mask );

  A = Ellipsoid( r / scn->dx, r / scn->dy, r / scn->dz );
  feats = CreateBCFeatures( scn, mask, MAX( A->n / 3, 1 ) );

  tm = BrainTissueThresholds( scn, protocol );
  std_dev = FlatRegionStdDev( scn, mask, tm, A, feats->nfeats, protocol, tissue );
  free( tm );

  for( i = 0; i < t_threads; i++ ) {
    pars[ i ][ 0 ] = i;
    pars[ i ][ 1 ] = t_threads;
    pars[ i ][ 2 ] = first_voxel;
    pars[ i ][ 3 ] = last_voxel;
    pars[ i ][ 4 ] = ( pointer_type ) scn;
    pars[ i ][ 5 ] = ( pointer_type ) mask;
    pars[ i ][ 6 ] = ( pointer_type ) A;
    pars[ i ][ 7 ] = ( pointer_type ) feats;
    pars[ i ][ 8 ] = std_dev;
    //pars[ i ][ 9 ] = ( pointer_type ) tmp;
  }

  for( i = 0; i < t_threads; i++ )
    pthread_create( &tid[ i ], NULL, AdaptiveSceneFeaturesThread, &pars[ i ][ 0 ] );
  for( i = 0; i < t_threads; i++ )
    pthread_join( tid[ i ], NULL );

  //WriteScene( tmp, "adaptive_feats.scn.bz2" );
  //DestroyScene( &tmp );


  DestroyAdjRel3( &A );
  return( feats );
}

/* Copy features from vector to subgraph. */
void SetBCFeatures( BCSubgraph *sg, BCFeatures *f ) {
  int i;
  
  sg->nfeats = f->nfeats;
  sg->Imax = f->Imax;
  sg->feat = f->feat;
  for( i = 0; i < sg->nnodes; i++ )
    sg->node[ i ].feat = f->nodefeat[ sg->node[ i ].pixel ];
}

/* Deallocate memory for subgraph. */
void FreeSubgraph( BCSubgraph **sg ) {
  
  if( ( *sg ) != NULL ) {
    BCDestroyArcs( *sg );
    free( ( *sg )->node );
    free( ( *sg )->ordered_list_of_nodes );
    free( ( *sg ) );
    *sg = NULL;
  }
}

void *SceneClassifyKnnGraphThread( void *arg ) {
  BCSubgraph *sg;
  Scene *mask;
  Scene *segmentation;
  BCFeatures *f;
  int t_threads;
  int pn, p0, p1;
  int first_voxel;
  int last_voxel;
  int valid_range;
  float dist;
  int i,p,k;
  
  pn = ( ( pointer_type* ) arg )[ 0 ];
  t_threads = ( ( pointer_type* ) arg )[ 1 ];
  first_voxel = ( ( pointer_type* ) arg )[ 2 ];
  last_voxel = ( ( pointer_type* ) arg )[ 3 ];
  sg = ( ( BCSubgraph** ) arg )[ 4 ];
  mask = ( ( Scene** ) arg )[ 5 ];
  f = ( ( BCFeatures** ) arg )[ 6 ];
  segmentation = ( ( Scene** ) arg )[ 7 ];
  valid_range = last_voxel - first_voxel + 1;

  /* picking the part of the brain, according to the number of the thread. */
  if( t_threads == 1 ) {
    p0 = first_voxel;
    p1 = last_voxel + 1;
  }
  else if( ( pn > 0 ) && ( pn < t_threads - 1 ) ) {
    p0 = first_voxel + ( 2 * pn + 1 ) * ( valid_range / ( 2 * t_threads + 2 ) );
    p1 = first_voxel + ( 2 * pn + 3 ) * ( valid_range / ( 2 * t_threads + 2 ) );
  }
  else if( pn == 0 ) {
    p0 = first_voxel;
    p1 = first_voxel + 3              * ( valid_range / ( 2 * t_threads + 2 ) );
  }
  else {
    p0 = first_voxel + ( 2 * pn + 1 ) * ( valid_range / ( 2 * t_threads + 2 ) );
    p1 = last_voxel + 1;
  }
  
  for( p = p0; p < p1; p++ ) {
    if( ( mask == NULL ) || ( mask->data[ p ] != 0 ) ) {
      for( i = 0; i < sg->nnodes; i++ ) {
      	k = sg->ordered_list_of_nodes[ i ];
	dist = BCArcWeight( sg->node[ k ].feat, f->nodefeat[ p ], sg->nfeats );
	if( dist <= sg->node[ k ].radius ) {
	  segmentation->data[ p ] = sg->node[ k ].label;
	  break;
	}
      }
    }
  }
  return( NULL );
}

Scene *SceneClassifyKnnGraph( BCSubgraph *sg, Scene *mask, int first_voxel, int last_voxel, BCFeatures *f, int t_threads ) {
  pointer_type pars[ t_threads ][ 15 ];
  pthread_t tid[ t_threads ];
  int i;
  Scene *segmentation = NULL;
  
  segmentation = CreateScene( mask->xsize, mask->ysize, mask->zsize );

  for( i = 0; i < t_threads; i++ ) {
    pars[ i ][ 0 ] = i;
    pars[ i ][ 1 ] = t_threads;
    pars[ i ][ 2 ] = first_voxel;
    pars[ i ][ 3 ] = last_voxel;
    pars[ i ][ 4 ] = ( pointer_type ) sg;
    pars[ i ][ 5 ] = ( pointer_type ) mask;
    pars[ i ][ 6 ] = ( pointer_type ) f;
    pars[ i ][ 7 ] = ( pointer_type ) segmentation;
  }

  for( i = 0; i < t_threads; i++ )
    pthread_create( &tid[ i ], NULL, SceneClassifyKnnGraphThread, &pars[ i ][ 0 ] );
  for( i = 0; i < t_threads; i++ )
    pthread_join( tid[ i ], NULL );
  return( segmentation );
}

