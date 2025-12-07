#include "oldift.h"
#include "scene.h"
#include <pthread.h>
#include "filelist.h"
#include "arraylist.h"
#include "sgctree.h"
#include "orderlist.h"
#include "braincluster_opf.h"
#include "braincluster_spectral.h"
#include "mri.h"

bool verbose;
Scene *subint;
Scene *corrected;

Scene *InhomogeneityCorrection1(Scene *scn, Scene *mask, Scene *emask)
{
  Scene *inho=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  Scene *root=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  Scene *cost=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  GQueue *Q = NULL;
  Voxel  u, v;
  int    i, p, q, n, s, tmp;
  AdjRel3 *A=Spheric(5.0);

  inho->dx = scn->dx; inho->dy = scn->dy; inho->dz = scn->dz;
  s = scn->xsize * scn->ysize;
  n = s * scn->zsize;
  Q = CreateGQueue(MaximumValue3(scn)+2,n,cost->data);
  SetRemovalPolicy(Q,MAXVALUE);

  for(p = 0; p < n; p++) {
    if (emask->data[p]==1){
      cost->data[p]   = MAX(scn->data[p]-5,0);
      root->data[p]   = p;
      InsertGQueue(&Q, p);
    }
    else if(mask->data[p]==1){
      cost->data[p]   = 0;
      root->data[p]   = p;
      InsertGQueue(&Q, p);
    }
  }

  while (!EmptyGQueue(Q)) {
    p = RemoveGQueue(Q);
    if (root->data[p]==p){
      cost->data[p]=scn->data[p];
    }

    i   = (p % s);
    u.x =  i % scn->xsize;
    u.y =  i / scn->xsize;
    u.z = p / s;
    for (i = 1; i < A->n; i++) {
      v.z = u.z + A->dz[i];
      v.y = u.y + A->dy[i];
      v.x = u.x + A->dx[i];
      if (ValidVoxel(scn, v.x, v.y, v.z)) {
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (mask->data[q]==1){
	  if (cost->data[p] > cost->data[q]) {
	    //tmp = MAX( MIN(scn->data[q],cost->data[p]), cost->data[p] - Imax );
	    tmp = MIN(scn->data[q],cost->data[p]);
	    if (tmp > cost->data[q]){
	      RemoveGQueueElem(Q,q);
	      cost->data[q] = tmp;
	      root->data[q] = root->data[p];
	      InsertGQueue(&Q, q);	      
	    }
	  }
	}
      }
    }  	
  }

  //WriteScene( root, "root.scn.bz2" );

  DestroyGQueue(&Q);
  DestroyAdjRel3(&A);

  for(p=0; p < n; p++) { 
    if (mask->data[p]==1)
      inho->data[p] = scn->data[root->data[p]];
  }
  
  DestroyScene(&cost);
  DestroyScene(&root);

  return (inho);
}

float *ParseInput( int argc, char **argv ) {
  float *result;
  int i;
  result = ( float* ) calloc( 10, sizeof( float ) );

  result[ 0 ] = ( float ) 1.0;
  result[ 1 ] = ( float ) 15.3;
  result[ 2 ] = ( float ) 1.6;
  result[ 3 ] = ( float ) 2;
  result[ 4 ] = T1_PROTOCOL;
  result[ 5 ] = 1;
  verbose = false;
  for( i = 2; i < argc; i++ ) {
    if( strcmp( argv[ i ], "-function" ) == 0 ) 
      result[ 0 ] = atof( argv[ ++i ] );
    if( strcmp( argv[ i ], "-radius" ) == 0 ) 
      result[ 1 ] = atof( argv[ ++i ] );
    if( strcmp( argv[ i ], "-rgrowth" ) == 0 ) 
      result[ 2 ] = atof( argv[ ++i ] );
    if( strcmp( argv[ i ], "-compression" ) == 0 ) 
      result[ 3 ] = atoi( argv[ ++i ] );
    if( strcmp( argv[ i ], "-protocol" ) == 0 ) 
      result[ 4 ] = atoi( argv[ ++i ] );
    if( strcmp( argv[ i ], "-threads" ) == 0 ) 
      result[ 5 ] = atoi( argv[ ++i ] );
    if( strcmp( argv[ i ], "-v" ) == 0 ) 
      verbose = true;
  }
  return result;
}

// dir: 0 - High, 1- Low;
int ClippingThreshold( Scene *scn, Scene *mask, float T, int dir ) {
  Curve *h = NULL;
  Curve *m = NULL;
  Scene *brain = NULL;
  int ymax;
  int p;

  brain = Mult3( scn, mask );

  h = Histogram3( brain );
  m = MedianFilterHistogram( h );

  // Compute amplitude.
  ymax = 0;
  for( p = 1; p < m->n; p++ ) {
    if( m->Y[ p ] > ymax )
      ymax = m->Y[ p ];
  }
  
  if( dir == T1_PROTOCOL ) {
    for( p = m->n - 2; m->Y[ p ] <= 0.2  * ymax; p-- );
    for(             ; ( m->Y[ p ] >= T * ymax ) && ( p < m->n ); p++ );
    if( p == m->n )
      p = m->n - 1;
  }
  else {
    for( p = 1; m->Y[ p ] <= 0.2  * ymax; p++ );
    for(      ; ( m->Y[ p ] >= T * ymax ) && ( p >= 0 ); p-- );
    if( p < 0 )
      p = 0;
  }
  p = m->X[ p ];

  DestroyScene( &brain );
  DestroyCurve( &h );
  DestroyCurve( &m );

  return( p );
}

Scene *ErodeBinScene3(Scene *bin, float radius) {
  Scene *cost = NULL, *root = NULL, *ero = NULL;
  GQueue *Q = NULL;
  int i, p, q, sz;
  Voxel u, v, w;
  float *sqx = NULL, *sqy = NULL, *sqz = NULL;
  int tmp, dx, dy, dz;
  float dx2, dy2, dz2;
  float dist;
  AdjRel3 *A = NULL;
  Set *seed = NULL;

  A = Spheric( 1.0 );
  for( u.z = 0; u.z < bin->zsize; u.z++ ) {
    for( u.y = 0; u.y < bin->ysize; u.y++ ) {
      for( u.x = 0; u.x < bin->xsize; u.x++ ) {
	p = u.x + bin->tby[ u.y ] + bin->tbz[ u.z ];
	if( bin->data[ p ] == 0 ) {
	  for( i = 1; i < A->n; i++ ) {
	    v.x = u.x + A->dx[ i ];
	    v.y = u.y + A->dy[ i ];
	    v.z = u.z + A->dz[ i ];
	    if( ValidVoxel( bin, v.x, v.y, v.z ) ) {
	      q = v.x + bin->tby[ v.y ] + bin->tbz[ v.z ];
	      if( bin->data[ q ] == 1 ) {
		InsertSet( &seed, p );
		break;
	      }
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel3( &A );

  ero  = CopyScene( bin );
  dist = ( radius * radius );
  A  = Ellipsoid( 1.8 * ero->dx, 1.8 * ero->dy, 1.8 * ero->dz );
  sqx = AllocFloatArray( ero->xsize );
  sqy = AllocFloatArray( ero->ysize );
  sqz = AllocFloatArray( ero->zsize );
  dx2 = ero->dx * ero->dx;
  dy2 = ero->dy * ero->dy;
  dz2 = ero->dz * ero->dz;
  for( i = 0; i < ero->xsize; i++ )
    sqx[ i ] = i * i * dx2;
  for( i = 0; i < ero->ysize; i++ )
    sqy[ i ] = i * i * dy2;
  for( i = 0; i < ero->zsize; i++ )
    sqz[ i ] = i * i * dz2;
  cost = CreateScene( ero->xsize, ero->ysize, ero->zsize );
  root = CreateScene( ero->xsize, ero->ysize, ero->zsize );
  sz = FrameSize3( A );
  Q  = CreateGQueue( 2 * sz * ( ero->xsize / ero->dx + ero->ysize / ero->dy + ero->zsize / ero->dz ) + 3 * sz * sz, ero->n, cost->data );
  for( p = 0; p < ero->n; p++ )
    cost->data[ p ] = INT_MAX;

  while( seed != NULL ) {
    p = RemoveSet( &seed );
    cost->data[ p ] = 0;
    root->data[ p ] = p;
    InsertGQueue( &Q, p );
  }
  while( !EmptyGQueue( Q ) ) {
    p = RemoveGQueue( Q );
    if( cost->data[ p ] <= dist ) {

      ero->data[ p ] = 0;
      
      u.x = VoxelX( cost, p );
      u.y = VoxelY( cost, p );
      u.z = VoxelZ( cost, p );

      w.x = VoxelX( cost, root->data[ p ] );
      w.y = VoxelY( cost, root->data[ p ] );
      w.z = VoxelZ( cost, root->data[ p ] );

      for( i = 1; i < A->n; i++ ) {
	v.x = u.x + A->dx[ i ];
	v.y = u.y + A->dy[ i ];
	v.z = u.z + A->dz[ i ];
	if( ValidVoxel( ero, v.x, v.y, v.z ) ) {
	  q = v.x + ero->tby[ v.y ] + ero->tbz[ v.z ];
	  if( ( cost->data[ p ] < cost->data[ q ] ) && ( ero->data[ q ] == 1 ) ) {
	    dx  = abs( v.x - w.x );
	    dy  = abs( v.y - w.y );
	    dz  = abs( v.z - w.z );
	    tmp = ( sqx[ dx ] + sqy[ dy ] + sqz[ dz ] ) + 0.001;
	    if( tmp < cost->data[ q ] ) {
	      if( cost->data[ q ] == INT_MAX ) {
		cost->data[ q ] = tmp;
		InsertGQueue( &Q, q );
	      }
	      else
		UpdateGQueue( &Q, q, tmp );
	      root->data[ q ] = root->data[ p ];
	    }
	  }
	}
      }
    } 
  }
  free( sqx );
  free( sqy );
  free( sqz );
  DestroyGQueue( &Q );
  DestroyScene( &root );
  DestroyScene( &cost );
  DestroyAdjRel3( &A );

  return( ero );
}

Scene *DilateBinScene3(Scene *bin, float radius) {
  Scene *cost = NULL, *root = NULL, *dil = NULL;
  GQueue *Q = NULL;
  int i, p, q, sz;
  Voxel u, v, w;
  float *sqx = NULL, *sqy = NULL, *sqz = NULL;
  int tmp, dx, dy, dz;
  float dx2, dy2, dz2;
  float dist;
  AdjRel3 *A = NULL;
  Set *seed = NULL;

  A = Spheric( 1.0 );
  for( u.z = 0; u.z < bin->zsize; u.z++ ) {
    for( u.y = 0; u.y < bin->ysize; u.y++ ) {
      for( u.x = 0; u.x < bin->xsize; u.x++ ) {
	p = u.x + bin->tby[ u.y ] + bin->tbz[ u.z ];
	if( bin->data[ p ] == 0 ) {
	  for( i = 1; i < A->n; i++ ) {
	    v.x = u.x + A->dx[ i ];
	    v.y = u.y + A->dy[ i ];
	    v.z = u.z + A->dz[ i ];
	    if( ValidVoxel( bin, v.x, v.y, v.z ) ) {
	      q = v.x + bin->tby[ v.y ] + bin->tbz[ v.z ];
	      if( bin->data[ q ] == 1 ) {
		InsertSet( &seed, p );
		break;
	      }
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel3( &A );

  dil  = CopyScene( bin );
  dist = ( radius * radius );
  A  = Ellipsoid( 1.8 * dil->dx, 1.8 * dil->dy, 1.8 * dil->dz );
  sqx = AllocFloatArray( dil->xsize );
  sqy = AllocFloatArray( dil->ysize );
  sqz = AllocFloatArray( dil->zsize );
  dx2 = dil->dx * dil->dx;
  dy2 = dil->dy * dil->dy;
  dz2 = dil->dz * dil->dz;
  for( i = 0; i < dil->xsize; i++ )
    sqx[ i ] = i * i * dx2;
  for( i = 0; i < dil->ysize; i++ )
    sqy[ i ] = i * i * dy2;
  for( i = 0; i < dil->zsize; i++ )
    sqz[ i ] = i * i * dz2;
  cost = CreateScene( dil->xsize, dil->ysize, dil->zsize );
  root = CreateScene( dil->xsize, dil->ysize, dil->zsize );
  sz = FrameSize3( A );
  Q  = CreateGQueue( 2 * sz * ( dil->xsize / dil->dx + dil->ysize / dil->dy + dil->zsize / dil->dz ) + 3 * sz * sz, dil->n, cost->data );
  for( p = 0; p < dil->n; p++ )
    cost->data[ p ] = INT_MAX;

  while( seed != NULL ) {
    p = RemoveSet( &seed );
    cost->data[ p ] = 0;
    root->data[ p ] = p;
    InsertGQueue( &Q, p );
  }
  while( !EmptyGQueue( Q ) ) {
    p = RemoveGQueue( Q );
    if( cost->data[ p ] <= dist ) {

      dil->data[ p ] = 1;
      
      u.x = VoxelX( cost, p );
      u.y = VoxelY( cost, p );
      u.z = VoxelZ( cost, p );

      w.x = VoxelX( cost, root->data[ p ] );
      w.y = VoxelY( cost, root->data[ p ] );
      w.z = VoxelZ( cost, root->data[ p ] );

      for( i = 1; i < A->n; i++ ) {
	v.x = u.x + A->dx[ i ];
	v.y = u.y + A->dy[ i ];
	v.z = u.z + A->dz[ i ];
	if( ValidVoxel( dil, v.x, v.y, v.z ) ) {
	  q = v.x + dil->tby[ v.y ] + dil->tbz[ v.z ];
	  if( ( cost->data[ p ] < cost->data[ q ] ) && ( dil->data[ q ] == 0 ) ) {
	    dx  = abs( v.x - w.x );
	    dy  = abs( v.y - w.y );
	    dz  = abs( v.z - w.z );
	    tmp = ( sqx[ dx ] + sqy[ dy ] + sqz[ dz ] ) + 0.001;
	    if( tmp < cost->data[ q ] ) {
	      if( cost->data[ q ] == INT_MAX ) {
		cost->data[ q ] = tmp;
		InsertGQueue( &Q, q );
	      }
	      else
		UpdateGQueue( &Q, q, tmp );
	      root->data[ q ] = root->data[ p ];
	    }
	  }
	}
      }
    } 
  }
  free( sqx );
  free( sqy );
  free( sqz );
  DestroyGQueue( &Q );
  DestroyScene( &root );
  DestroyScene( &cost );
  DestroyAdjRel3( &A );

  return( dil );
}

Scene *RemoveBrainExtremeOutliers( Scene *scn, Scene *mask, int protocol ) {
  int p, T;
  Scene *emask;
  Scene *aux;
  
  T = ClippingThreshold( scn, mask, 0.1, protocol );
  aux = CopyScene( mask );
  
  if( protocol == T1_PROTOCOL ) {
    for( p = 0; p < scn->n; p++ ) {
      if( ( mask->data[ p ] != 0 ) && ( scn->data[ p ] > T ) )
	aux->data[ p ] = 0;
    }
  }
  else { //( protocol == T2_PROTOCOL ) {
    for( p = 0; p < scn->n; p++ ) {
      if( ( mask->data[ p ] != 0 ) && ( scn->data[ p ] < T ) )
	aux->data[ p ] = 0;
    }
  }

  emask = ErodeBinScene3( aux, 2.5 );

  DestroyScene( &aux );
  if( verbose )
    printf( "X Cut:%d\n", T );
  
  return( emask );
}

void LChange(long long *a, long long *b){ /* It changes content between longs a and b */
  long long c;
  c  = *a;
  *a = *b;
  *b = c;
}

void LongQuickSort( long long *value, int *index, int i0, int i1, char order ) {
  int m, d;
  
  if( i0 < i1 ) {
    // número aleatório para evitar pivo ruim.
    d = ( i1 + i0 ) / 2; //RandomInteger( i0, i1 );
    LChange( &value[ d ], &value[ i0 ] );
    Change( &index[ d ], &index[ i0 ] );
    m = i0;

    if( order == INCREASING ) {
      for( d = i0 + 1; d <= i1; d++ ) {
	if( value[ d ] <= value[ m ] ) {
	  m++;
	  LChange( &value[ d ], &value[ m ] );
	  Change( &index[ d ], &index[ m ] );
	}
      }
    }
    else {
      for( d = i0 + 1; d <= i1; d++ ) {
	if( value[ d ] >= value[ m ] ) {
	  m++;
	  LChange( &value[ d ], &value[ m ] );
	  Change( &index[ d ], &index[ m ] );
	}
      }
    }
    LChange( &value[ m ], &value[ i0 ] );
    Change( &index[ m ], &index[ i0 ] );
    LongQuickSort( value, index, i0, m - 1, order );
    LongQuickSort( value, index, m + 1, i1, order );
  }
}

AdjRel3 *GrowingEllipsoid( float rx, float ry, float rz ) {
  
  AdjRel3 *A=NULL;
  AdjRel3 *B=NULL;
  int *index;
  long long *dist;
  int i;

  A = Ellipsoid( rx, ry, rz );
  B = CreateAdjRel3( A->n );
  index = AllocIntArray( A->n );
  dist = ( long long* ) calloc( A->n, sizeof( long long ) );

  // Sorting A elements.
  for( i = 0; i < A->n; i++ ) {
    index[ i ] = i;
    dist[ i ] = ( A->dx[ i ] * A->dx[ i ] + A->dy[ i ] * A->dy[ i ] + A->dz[ i ] * A->dz[ i ] );
  }
  LongQuickSort( dist, index, 0, A->n - 1, INCREASING );

  // Setting B elements with ordered elements of A.
  for( i = 0; i < A->n; i++ ) {
    B->dx[ i ] = A->dx[ index[ i ] ];
    B->dy[ i ] = A->dy[ index[ i ] ];
    B->dz[ i ] = A->dz[ index[ i ] ];
  }

  DestroyAdjRel3( &A );
  free( index );
  free( dist );

  return( B );
}

Scene *SubSampleScene( Scene *scn, Scene *mask, int compression ) {
  Scene *result;
  int x, y, z;
  int dx, dy, dz;
  int sum, p, div;

  result = CreateScene( scn->xsize / compression, scn->ysize / compression, scn->zsize / compression );
  result->dx = scn->dx * compression;
  result->dy = scn->dy * compression;
  result->dz = scn->dz * compression;

  for( x = 0; x < result->xsize; x++ ) {
    for( y = 0; y < result->ysize; y++ ) {
      for( z = 0; z < result->zsize; z++ ) {
	sum = 0;
	div = 0;
	for( dx = 0; dx < compression; dx++ ) {
	  for( dy = 0; dy < compression; dy++ ) {
	    for( dz = 0; dz < compression; dz++ ) {
	      p = VoxelAddress( scn, x * compression + dx, y * compression + dy, z * compression + dz );
	      if( mask->data[ p ] != 0 ) {
		sum += scn->data[ p ];
		div++;
	      }
	    }
	  }
	}
	p = VoxelAddress( result, x, y, z );
	if( div != 0 )
	  result->data[ p ] = sum / div;
      }
    }
  }

  return( result );

}

Scene *SuperSampleScene( Scene *scn, Scene *mask ) {
  Scene *result;
  int x, y, z;
  int dx, dy, dz;
  int sum, p, div;
  int compression;

  result = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  compression = mask->xsize / scn->xsize;
  div = compression * compression * compression;
  //printf( "Compression: %d\n", compression );

  for( x = 0; x < scn->xsize; x++ ) {
    for( y = 0; y < scn->ysize; y++ ) {
      for( z = 0; z < scn->zsize; z++ ) {
	p = VoxelAddress( scn, x, y, z );
	sum = scn->data[ p ];
	for( dx = 0; dx < compression; dx++ ) {
	  for( dy = 0; dy < compression; dy++ ) {
	    for( dz = 0; dz < compression; dz++ ) {
	      p = VoxelAddress( result, x * compression + dx, y * compression + dy, z * compression + dz );
	      if( mask->data[ p ] != 0 )
		result->data[ p ] = sum;
	    }
	  }
	}
      }
    }
  }

  return( result );
}

void BiasSurfaceClipping( Scene *absint, Scene *mask, int protocol ) {
  int T, p;
  
  if( protocol == T1_PROTOCOL ) {
    T = ClippingThreshold( absint, mask, 0.01, T2_PROTOCOL );
    for( p = 0; p < mask->n; p++ ) {
      if( ( mask->data[ p ] != 0 ) && ( absint->data[ p ] < T ) )
	absint->data[ p ] = T;
    }
  }
  else {
    T = ClippingThreshold( absint, mask, 0.01, T1_PROTOCOL );
    for( p = 0; p < mask->n; p++ ) {
      if( ( mask->data[ p ] != 0 ) && ( absint->data[ p ] > T ) )
	absint->data[ p ] = T;
    }
  }
  if( verbose )
    printf( "Bias surface clipping:%d\n", T );
}

void *BiasSurfaceEstimationThread( void *arg ) {
  Scene *subemask;
  Scene *submask;
  Scene *subscn;
  AdjRel3 *A = NULL;
  AdjRel3 *Amax = NULL;
  int nv;
  int p, q, i, t;
  int val, total_voxels;
  int pn, p0, p1;
  int max_int, min_int;
  int abs_val;
  int first_voxel;
  int last_voxel;
  int valid_range;
  int protocol;
  int t_threads;
  
  pn = ( ( pointer_type* ) arg )[ 0 ];
  t_threads = ( ( pointer_type* ) arg )[ 1 ];
  first_voxel = ( ( pointer_type* ) arg )[ 2 ];
  last_voxel = ( ( pointer_type* ) arg )[ 3 ];
  valid_range = ( ( pointer_type* ) arg )[ 4 ];
  subscn = ( ( Scene** ) arg )[ 5 ];
  submask = ( ( Scene** ) arg )[ 6 ];
  subemask = ( ( Scene** ) arg )[ 7 ];
  A = ( ( AdjRel3** ) arg )[ 8 ];
  Amax = ( ( AdjRel3** ) arg )[ 9 ];
  abs_val = ( ( pointer_type* ) arg )[ 10 ];
  nv = ( ( pointer_type* ) arg )[ 11 ];
  protocol = ( ( pointer_type* ) arg )[ 12 ];

  int vec[ nv + 1 ];
  int init_vec[ nv + 1 ];
  
  // picking the part of the brain, according to the number of the thread.
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

  if( verbose )
    printf( "Thread %d running sector(%d-%d)...\n", pn, p0, p1 );
  
  // Start vector values with minimal and maximal values.
  for( t = 0; t <= nv; t++ )
    init_vec[ t ] = abs_val;

  // Computing the bias estimation scene.
  max_int = MaximumValue3( subscn );
  min_int = MinimumValueMask3( subscn, subemask );
  for( p = p0; p < p1; p++ ) {
    if( submask->data[ p ] != 0 ) {
      total_voxels = 0;
      memcpy( &vec, init_vec, ( nv + 1 ) * sizeof( int ) );
      subint->data[ p ] = 0;
      // For border voxels.
      for( total_voxels = 0, i = 0; ( total_voxels < A->n ) && ( i < Amax->n ); i++ ) {
	q = p + Amax->dx[ i ] + subemask->xsize * Amax->dy[ i ] + subemask->xsize * subemask->ysize * Amax->dz[ i ];
	if( subemask->data[ q ] != 0 ) {
	  total_voxels++;
	  val = subscn->data[ q ];
	  for( t = nv - 1; t >= 0; t-- ) {
	    if ( ( ( protocol == T1_PROTOCOL ) && ( val > vec[ t ] ) ) || ( ( protocol != T1_PROTOCOL ) && ( val < vec[ t ] ) ) )
	      vec[ t + 1 ] = vec[ t ];
	    else
	      break;
	  }
	  vec[ t + 1 ] = val;
	}
      }
      if( total_voxels == 0 ) {
	if( protocol == T1_PROTOCOL )
	  subint->data[ p ] = max_int;
	else
	  subint->data[ p ] = min_int;
      }
      else if( total_voxels < nv * 2 )
	subint->data[ p ] = vec[ MIN( total_voxels, nv / 2 ) - 1 ];
      else
	subint->data[ p ] = vec[ MIN( nv, total_voxels ) - 1 ];
    }
  }
  return( NULL );
}

Scene *BiasSurfaceEstimation( Scene *scn, Scene *mask, Scene *emask, float radius, float radius_increase, 
			      int nv, float compression, int protocol, int t_threads ) {
  Scene *subemask = NULL;
  Scene *submask = NULL;
  Scene *subscn = NULL;
  Scene *absint = NULL;
  AdjRel3 *A = NULL;
  AdjRel3 *Amax = NULL;
  pointer_type pars[ t_threads ][ 15 ];
  pthread_t tid[ 32 ];
  float subradius;
  float subradius_max;
  int p, abs_val;
  int last_voxel, first_voxel, valid_range;

  // Preparing subimages for inhomogeity surface computing.
  subscn = SubSampleScene( scn, mask, compression );
  submask = SubSampleScene( mask, mask, compression );
  subemask = SubSampleScene( emask, emask, compression );
  subradius = radius / compression;
  subradius_max = radius * radius_increase / compression;

  // Computing attributes to speed up surface computing.
  first_voxel = INT_MAX;
  last_voxel = 0;
  for ( p = 0; p < submask->n; p++ ) {
    if ( submask->data[ p ] != 0 ) {
      if( first_voxel > p )
	first_voxel = p;
      if( last_voxel < p )
	last_voxel = p;
    }
  }
  valid_range = last_voxel - first_voxel + 1;
  if( protocol == T1_PROTOCOL )
    abs_val = MinimumValueMask3( scn, emask );
  else
    abs_val = MaximumValueMask3( scn, emask );

  if( verbose )
    printf( "abs_val = %d, first_voxel = %d, last_voxel = %d, valid_range = %d\n", abs_val, first_voxel, last_voxel, valid_range );

  // Adjacency relation computing and result scene creating. Must be done here because media_val function is multi-thread.
  A = Ellipsoid( subradius / scn->dx, subradius / scn->dy, subradius / scn->dz  );
  Amax = GrowingEllipsoid( subradius_max / scn->dx, subradius_max / scn->dy, subradius_max / scn->dz  );
  subint  = CreateScene( subscn->xsize, subscn->ysize, subscn->zsize );

  // Computing inhomogenity surface ------------------

  for( p = 0; p < t_threads; p++ ) {
    pars[ p ][ 0 ] = p;
    pars[ p ][ 1 ] = t_threads;
    pars[ p ][ 2 ] = first_voxel;
    pars[ p ][ 3 ] = last_voxel;
    pars[ p ][ 4 ] = valid_range;
    pars[ p ][ 5 ] = ( pointer_type ) subscn;
    pars[ p ][ 6 ] = ( pointer_type ) submask;
    pars[ p ][ 7 ] = ( pointer_type ) subemask;
    pars[ p ][ 8 ] = ( pointer_type ) A;
    pars[ p ][ 9 ] = ( pointer_type ) Amax;
    pars[ p ][ 10 ] = abs_val;
    pars[ p ][ 11 ] = nv;
    pars[ p ][ 12 ] = protocol;
  }

  for( p = 0; p < t_threads; p++ )
    pthread_create( &tid[ p ], NULL, BiasSurfaceEstimationThread, &pars[ p ][ 0 ] );
  for( p = 0; p < t_threads; p++ )
    pthread_join( tid[ p ], NULL );

  // Finishing bias surface computation.
  absint = SuperSampleScene( subint, mask );

  BiasSurfaceClipping( absint, mask, protocol );

  DestroyAdjRel3( &A );
  DestroyAdjRel3( &Amax );
  DestroyScene( &subscn );
  DestroyScene( &submask );
  DestroyScene( &subemask );
  DestroyScene( &subint );

  return( absint );
}

void RemoveVentricles( Scene *scn, Scene *mask, Scene *emask, int protocol ) {
  int p;
  Scene *aux;
  Scene *aux2;
  TissueMarker *tm;

  aux = Mult3( scn, mask );
  tm = BrainTissueThresholds( aux, protocol );
  DestroyScene( &aux );
  aux = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  if( protocol == T1_PROTOCOL ) {
    for( p = 0; p < scn->n; p++ ) {
      if( ( mask->data[ p ] != 0 ) && ( scn->data[ p ] < tm->csf_gm ) )
	aux->data[ p ] = 1;
    }
  }
  else {
    for( p = 0; p < scn->n; p++ ) {
      if( ( mask->data[ p ] != 0 ) && ( scn->data[ p ] > tm->csf_gm ) )
	aux->data[ p ] = 1;
    }
  }
  aux2 = ErodeBinScene3( aux, 2.5 );
  for( p = 0; p < scn->n; p++ ) {
    if( aux2->data[ p ] == 1 )
      emask->data[ p ] = 0;
  }
  if( verbose )
    printf( "CSF Cut:%d\n", tm->csf_gm );
  free( tm );
  DestroyScene( &aux );
  DestroyScene( &aux2 );
}

void *InhomogeneityThread( void *arg ) {
  int p;
  int first_voxel;
  int last_voxel;
  int valid_range;
  int pn, p0, p1;
  Scene *scn = NULL;
  Scene *mask = NULL;
  Scene *absint = NULL;
  int abs_val;
  float pf;
  int t_threads;

  pn = ( ( pointer_type* ) arg )[ 0 ];
  t_threads = ( ( pointer_type* ) arg )[ 1 ];
  first_voxel = ( ( pointer_type* ) arg )[ 2 ];
  last_voxel = ( ( pointer_type* ) arg )[ 3 ];
  valid_range = ( ( pointer_type* ) arg )[ 4 ];
  scn = ( ( Scene** ) arg )[ 5 ];
  mask = ( ( Scene** ) arg )[ 6 ];
  absint = ( ( Scene** ) arg )[ 7 ];
  abs_val = ( ( pointer_type* ) arg )[ 8 ];
  pf = ( ( pointer_type* ) arg )[ 9 ] * 0.001;

  // picking the part of the brain, according to the number of the thread.
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

  if( verbose )
    printf( "Thread %d running sector(%d-%d)...\n", pn, p0, p1 );
  
  for( p = p0; p < p1; p++ ) {
    if( mask->data[ p ] != 0 )
      corrected->data[ p ] = scn->data[ p ] * powf( ( float ) abs_val / absint->data[ p ], pf );
  }
  return( NULL );
}

void Inhomogeneity( Scene *scn, Scene *mask, Scene *absint, float pf, int protocol, int t_threads ) {
  pointer_type pars[ t_threads ][ 15 ];
  pthread_t tid[ 32 ];
  int first_voxel, last_voxel, valid_range;
  int abs_val;
  int i;

  // Computing attributes used by Inhomogeneity function. Must be done here because Inhomogeneity function is multi-thread.
  first_voxel = INT_MAX;
  last_voxel = 0;
  for ( i = 0; i < mask->n; i++ ) {
    if ( mask->data[ i ] != 0 ) {
      if( first_voxel > i )
	first_voxel = i;
      if( last_voxel < i )
	last_voxel = i;
    }
  }
  valid_range = last_voxel - first_voxel + 1;
  if( protocol == T1_PROTOCOL )
    abs_val = MaximumValueMask3( absint, mask );
  else
    abs_val = MinimumValueMask3( absint, mask );

  for( i = 0; i < t_threads; i++ ) {
    pars[ i ][ 0 ] = i;
    pars[ i ][ 1 ] = t_threads;
    pars[ i ][ 2 ] = first_voxel;
    pars[ i ][ 3 ] = last_voxel;
    pars[ i ][ 4 ] = valid_range;
    pars[ i ][ 5 ] = ( pointer_type ) scn;
    pars[ i ][ 6 ] = ( pointer_type ) mask;
    pars[ i ][ 7 ] = ( pointer_type ) absint;
    pars[ i ][ 8 ] = abs_val;
    pars[ i ][ 9 ] = pf * 1000;
  }

  // Inhomogeneity correction
  corrected = CreateScene( scn->xsize, scn->ysize, scn->zsize );
  corrected->dx = scn->dx;
  corrected->dy = scn->dy;
  corrected->dz = scn->dz;

  for( i = 0; i < t_threads; i++ )
    pthread_create( &tid[ i ], NULL, InhomogeneityThread, &pars[ i ][ 0 ] );
  for( i = 0; i < t_threads; i++ )
    pthread_join( tid[ i ], NULL );
}

Scene* ROIVolume(Scene *in, int *xi, int *yi, int *zi, int *xf, int *yf, int *zf ) {
  Scene *scn = NULL;
  int x, y, z, p, q;
  
  if( ( *xi == -1 ) || ( *yi == -1 ) || ( *zi == -1 ) ) {
    //printf( "in->xsize:%d, in->ysize:%d, in->zsize:%d\n", in->xsize, in->ysize, in->zsize );
    *xi = *yi = *zi = -1;
    for( x = 0; x < in->xsize; x++ ) {
      for( y = 0; y < in->ysize; y++ ) {
	for( z = 0; z < in->zsize; z++ ) {
	  p = VoxelAddress( in, x, y, z );
	  if( in->data[ p ] != 0 ) {
	    if( *xi == -1 ) {
	      *xi = x;
	      *xf = x;
	    }
	    else {
	      *xf = x;
	      y = in->ysize - 1;
	      z = in->zsize - 1;
	    }
	  }
	}
      }
    }
    *xf = in->xsize - *xf - 1;

    for( y = 0; y < in->ysize; y++ ) {
      for( x = 0; x < in->xsize; x++ ) {
	for( z = 0; z < in->zsize; z++ ) {
	  p = VoxelAddress( in, x, y, z );
	  if( in->data[ p ] != 0 ) {
	    if( *yi == -1 ) {
	      *yi = y;
	      *yf = y;
	    }
	    else {
	      *yf = y;
	      x = in->xsize - 1;
	      z = in->zsize - 1;
	    }
	  }
	}
      }
    }
    *yf = in->ysize - *yf - 1;

    for( z = 0; z < in->zsize; z++ ) {
      for( y = 0; y < in->ysize; y++ ) {
	for( x = 0; x < in->xsize; x++ ) {
	  p = VoxelAddress( in, x, y, z );
	  if( in->data[ p ] != 0 ) {
	    if( *zi == -1 ) {
	      *zi = z;
	      *zf = z;	      
	    }
	    else {
	      *zf = z;
	      x = in->xsize - 1;
	      y = in->ysize - 1;
	    }
	  }
	}
      }
    }
    *zf = in->zsize - *zf - 1;
    //printf( "*xi:%d, *xf:%d, *yi:%d, *yf:%d, *zi:%d, *zf:%d\n", *xi, *xf, *yi, *yf, *zi, *zf ); 
  }
  scn = CreateScene( in->xsize - *xf - *xi, in->ysize - *yf - *yi, in->zsize - *zf - *zi );
  scn->dx = in->dx;
  scn->dy = in->dy;
  scn->dz = in->dz;
  for( x = *xi; x < in->xsize - *xf; x++ ) {
    for( y = *yi; y < in->ysize - *yf; y++ ) {
      for( z = *zi; z < in->zsize - *zf; z++ ) {
	p = VoxelAddress( in, x, y, z );
	q = VoxelAddress( scn, x - *xi, y - *yi, z - *zi );
	scn->data[ q ] = in->data[ p ];
      }
    }
  }
  return( scn );
}
 
Scene* UndoROIVolume( Scene *scn, int *xi, int *yi, int *zi, int *xf, int *yf, int *zf ) {
  Scene *aux = NULL;
  int x, y, z, p, q;
  
  aux = CreateScene( scn->xsize + *xi + *xf, scn->ysize + *yi + *yf, scn->zsize + *zi + *zf );
  for( x = *xi; x < aux->xsize - *xf; x++ ) {
    for( y = *yi; y < aux->ysize - *yf; y++ ) {
      for( z = *zi; z < aux->zsize - *zf; z++ ) {
	p = VoxelAddress( aux, x, y, z );
	q = VoxelAddress( scn, x - *xi, y - *yi, z - *zi );
	aux->data[ p ] = scn->data[ q ];
      }
    }
  }
  aux->dx = scn->dx;
  aux->dy = scn->dy;
  aux->dz = scn->dz;
  return aux;
}

void GetFileExtension( char file[ ], char extension[ ] ) {
  int i = 0, j = 0;
  bool enable = false;
  while( i <= strlen( file ) ) {
    if( enable == true ) {
      extension[ j ] = file[ i ];
      i++;
      j++;
    }
    else {
      if( ( file[ i ] == '.' ) && ( file[ i - 1 ] == '.' ) )
	i++;
      else if( ( i < strlen( file ) ) && ( file[ i ] == '.' ) && ( file[ i + 1 ] != '.' ) )
	enable = true;
      else
	i++;
    }
  }
}

Scene* InhomogeneityCorrection(Scene *par_scn, Scene *par_mask, float par_function, float par_radius, float par_rgrowth, 
			       int par_compression, int par_protocol, int par_threads, int par_verbose) {
/* Parameters:
  -function <f>. f: corretion factor of GM and CSF tissues, from 0 to 1. Default 0.
  -radius <f>. f: from 7.0 to 28.0. Default 15.3mm.
  -rgrowth <f>. f: from 1.0 to 2.0. Default 1.6.
  -compression <i>. i: (best and slower) 1 to 4 (worst and faster). Default 2.
  -protocol <i>. i: 0-T1, 1-T2 or PD. Default 0.
  -threads <i>. i: 1 to 8.
  -v: for inho_verbose mode
*/
  timer *t1 = NULL, *t2 = NULL;
  Scene *scn = NULL;
  Scene *mask = NULL;
  Scene *emask = NULL;
  Scene *aux = NULL;
  Scene *absint = NULL;
  Scene *final = NULL;
  int scnxi, scnxf, scnyi, scnyf, scnzi, scnzf;
  int compression, nv;
  int t_threads, protocol;
  float radius, rgrowth;
  float pf;
  
  t1 = Tic();
  srand( ( int ) t1->tv_usec );

  pf = par_function;
  radius = par_radius;
  rgrowth = par_rgrowth;
  nv = 7;
  compression = par_compression;
  protocol = par_protocol;
  t_threads = par_threads;
  verbose = par_verbose;

  if( ( pf <= -0.0001 ) || ( ( radius > 28.0 ) || ( radius < 4.0 ) ) || ( ( rgrowth > 2.0 ) || ( rgrowth < 1.0 ) ) 
      || ( ( compression > 4 ) || ( compression < 1 ) ) ) {
    printf( "Invalid function or radius or compression. rgrowth:%f\n", rgrowth);
    return NULL;
  }

  if( verbose ) {
    printf( "multiply function %f, radius %f, rgrowth %f, compression %d,", pf, radius, rgrowth, compression );
    if( protocol == T1_PROTOCOL )
      printf( " protocol T1.\n" );
    else
      printf( " protocol T2 or PD.\n" );
  }

  // ROI of imagens with a security border for adjacency relation free access.
  scnxi = -1;
  aux = ROIVolume( par_mask, &scnxi, &scnyi, &scnzi, &scnxf, &scnyf, &scnzf );
  mask = AddFrame3( aux, radius * rgrowth + ( float ) compression, 0 );
  DestroyScene( &aux );
  aux = ROIVolume( par_scn , &scnxi, &scnyi, &scnzi, &scnxf, &scnyf, &scnzf );
  scn = AddFrame3( aux, radius * rgrowth + ( float ) compression, 0 );
  DestroyScene( &aux );
  scn->maxval = MaximumValue3( scn );
  scn->dx = par_scn->dx;
  scn->dy = par_scn->dy;
  scn->dz = par_scn->dz;
  mask->dx = scn->dx;
  mask->dy = scn->dy;
  mask->dz = scn->dz;
  emask = RemoveBrainExtremeOutliers( scn, mask, protocol );

  // Compute Bias surface.
  //absint = InhomogeneityCorrection1( scn, mask, emask );
  absint = BiasSurfaceEstimation( scn, mask, emask, radius, rgrowth, nv, compression, protocol, t_threads );

  // Compute corrected image.
  Inhomogeneity( scn, mask, absint, pf, protocol, t_threads );

  // Mapping corrected to the original scene format.
  aux = RemFrame3( corrected, radius * rgrowth + ( float ) compression );
  final = UndoROIVolume( aux, &scnxi, &scnyi, &scnzi, &scnxf, &scnyf, &scnzf );
  DestroyScene( &aux );
  
  DestroyScene( &corrected );
  DestroyScene( &absint );

  DestroyScene( &scn );
  DestroyScene( &mask );
  DestroyScene( &emask );
    
  t2 = Toc();
  if( verbose )
    fprintf(stdout,"\ninhomogeneity correction in %f ms\n",CTime(t1,t2));

  return( final );
}


#ifdef _INHOMOGENEITY_BRAIN_STANDALONE_
int main(int argc, char **argv){
  Scene *scn = NULL;
  Scene *msk = NULL;
  /* Scene *bias = NULL; */
  Scene *result = NULL;
  float *inputs;
  /* char filename[ 1024 ]; */
  /* char fileextension[ 1024 ]; */
  /* char basename[ 1024 ]; */
  float pf;
  float radius;
  float rgrowth;
  int compression;
  int protocol;
  int threads;
  /* int p; */

  //------- check number of parameters ----------
  if ( argc < 4 ) {
    printf( "Usage must be: %s <scene filename> <mask filename> <output filename> [options]\n", argv[ 0 ] );
    printf( "options:\n" );
    printf( "  -function <f>. f: corretion factor of GM and CSF tissues, from 0 to 2. Default 1.0\n" );
    printf( "  -radius <f>. f: from 7.0 to 28.0. Default 15.3mm.\n" );
    printf( "  -rgrowth <f>. f: from 1.0 to 2.0. Default 1.6.\n" );
    printf( "  -compression <i>. i: (best and slower) 1 to 4 (worst and faster). Default 2.\n" );
    printf( "  -protocol <i>. i: 0-T1, 1-T2 or PD. Default 0.\n" );
    printf( "  -threads <i>. i: 1 to 8.\n" );
    printf( "  -v: for verbose mode\n" );
    exit( 0 );
  }

  //------- read and check parameters -----------
  inputs = ParseInput( argc, argv );
  pf = inputs[ 0 ];
  radius = inputs[ 1 ];
  rgrowth = inputs[ 2 ];
  compression = ( int ) inputs[ 3 ];
  protocol = ( int ) inputs[ 4 ];
  threads = ( int ) inputs[ 5 ];
  scn = ReadVolume( argv[ 1 ] );
  msk = ReadVolume( argv[ 2 ] );

  result = InhomogeneityCorrection( scn, msk, pf, radius, rgrowth, compression, protocol, threads, verbose );
  /* sprintf( filename, "%s", argv[ 1 ] ); */
  /* sprintf( basename, "%s", filename ); */
  /* GetFileExtension( filename, fileextension ); */
  /* RemoveFileExtension( basename ); */
  /* RemoveFileExtension( basename ); */
  /* sprintf( filename, "%s-nobias%s", basename, fileextension ); */
  CopySceneHeader( scn, result );
  WriteVolume( result, argv[ 3 ] );

  /* bias = CopyScene( result ); */
  /* for( p = 0; p < bias->n; p++ ) */
  /*   bias->data[ p ] = abs( bias->data[ p ] - scn->data[ p ] ); */
  /* sprintf( filename, "%s-bias%s", basename, fileextension ); */
  /* CopySceneHeader( scn, bias ); */
  /* WriteVolume( bias, filename ); */

  free( inputs );
  DestroyScene( &scn );
  DestroyScene( &msk );
  /* DestroyScene( &bias ); */
  DestroyScene( &result );

  return( 0 );
}
#endif
