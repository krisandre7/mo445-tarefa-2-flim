#include "oldift.h"
#include <pthread.h>
#include "orderlist.h"
#include "inhomogeneity.h"
#include "braincluster_spectral.h"
#include "braincluster_opf.h"
#include "rigidregistration3.h"
#include "filelist.h"
#include "featmap.h"
#include "gradient.h"

bool verbose;

float *ParseBrainclusterAtlasInput( int argc, char **argv ) {
  float *result;
  int i;

  result = ( float* ) calloc( 20, sizeof( float ) );
  result[ 0 ] = ( float ) 1000; /* samples */
  result[ 1 ] = 0.00; /* mean proportion */
  result[ 2 ] = T1_PROTOCOL; /* protocol type */
  result[ 3 ] = 1.0; /* auto proportion flag */
  result[ 4 ] = 1; /* number of threads */
  result[ 5 ] = 0.0; /* delta proportion */
  verbose = false;
  for( i = 6; i < argc; i++ ) {
    if( ( strcmp( argv[ i ], "-samples" ) == 0 ) || ( strcmp( argv[ i ], "-s" ) == 0 ) ) 
      result[ 0 ] = atoi( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-mean_prop" ) == 0 ) || ( strcmp( argv[ i ], "-mp" ) == 0 ) ) {
      result[ 1 ] = atof( argv[ ++i ] );
      result[ 3 ] = 0.0;
    }
    if( ( strcmp( argv[ i ], "-protocol" ) == 0 ) || ( strcmp( argv[ i ], "-p" ) == 0 ) )
      result[ 2 ] = atof( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-threads" ) == 0 ) ||( strcmp( argv[ i ], "-th" ) == 0 ) )
      result[ 4 ] = atoi( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-delta_p" ) == 0 ) ||( strcmp( argv[ i ], "-dp" ) == 0 ) )
      result[ 5 ] = atof( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-verbose" ) == 0 ) || ( strcmp( argv[ i ], "-v" ) == 0 ) )
      verbose = true;
  }
  return result;
}

void BrainExtremities( Scene *mask, int vx[ ], int vy[ ], int vz[ ] ){
  int x, y, z, p;

  vx[ 0 ] = mask->xsize - 1;
  vy[ 0 ] = mask->ysize - 1;
  vz[ 0 ] = mask->zsize - 1;
  vx[ 1 ] = 0;
  vy[ 1 ] = 0;
  vz[ 1 ] = 0;
  
  for( x = 0; x < mask->xsize - 1; x++ ) {
    for( y = 0; y < mask->ysize - 1; y++ ) {
      for( z = 0; z < mask->zsize - 1; z++ ) {
	p = VoxelAddress( mask, x, y, z );
	if( mask->data[ p ] == 1 ) {
	  if( vx[ 0 ] > x )
	    vx[ 0 ] = x;
	  if( vy[ 0 ] > y )
	    vy[ 0 ] = y;
	  if( vz[ 0 ] > z )
	    vz[ 0 ] = z;
	  if( vx[ 1 ] < x )
	    vx[ 1 ] = x;
	  if( vy[ 1 ] < y )
	    vy[ 1 ] = y;
	  if( vz[ 1 ] < z )
	    vz[ 1 ] = z;
	}
      }
    }
  }
  if( vx[ 0 ] > vx[ 1 ] )
    printf( "Error: BrainExtremities - mask is blank!\n" );
}

Scene *CreateSubCorticalMask( Scene *mask ) {
  int p, q;
  int diameter;
  int x, y, z;
  int i, j, k;
  int vx[ 2 ], vy[ 2 ], vz[ 2 ];
  float mi, mj, mk;
  float si, sj, sk;
  Scene *subcortical = NULL;
  AdjRel3 *A = NULL;
  /* Scene *erode = NULL; */
  /* Set *S = NULL; */
  
  /* erode = ErodeBin3( mask, &S, 2.0 ); */
  /* DestroySet( &S ); */

  /* Get mask dimensions */
  BrainExtremities( mask, vx, vy, vz );
  diameter = MAX( vx[ 1 ] - vx[ 0 ], MAX( vy[ 1 ] - vy[ 0 ], vz[ 1 ] - vz[ 0 ] ) );

  /* Checking input orientation. */
  if( mask->nii_hdr != NULL ) {
    nifti_mat44_to_orientation( mask->nii_hdr->qto_xyz, &i, &j, &k );
  }
  else {
    i = NIFTI_R2L;
    j = NIFTI_A2P;
    k = NIFTI_I2S;
  }
  mi = 0.25;
  mj = 0.18;
  mk = 0.18;
  si = 0.0;
  sj = 0.0;
  sk = 0.03;      
  if( ( i == NIFTI_L2R ) || ( i == NIFTI_R2L ) ) {
    if( ( j == NIFTI_I2S ) || ( j == NIFTI_S2I ) ) {
      FChange( &mj, &mk );
      FChange( &sj, &sk );
      if( j == NIFTI_I2S )
	sj = -sj;
    }
    if( k == NIFTI_I2S )
      sk = -sk;
  }
  else if( ( i == NIFTI_A2P ) || ( i == NIFTI_P2A ) ) {
    FChange( &mi, &mj );
    FChange( &si, &sj );
    if( ( j == NIFTI_S2I ) || ( j == NIFTI_I2S ) ) {
      FChange( &mj, &mk );
      FChange( &sj, &sk );
      if( j == NIFTI_I2S )
	sj = -sj;
    }
    if( k == NIFTI_I2S )
      sk = -sk;
  }
  else {
    FChange( &mi, &mk );
    FChange( &si, &sk );
    if( ( j == NIFTI_L2R ) || ( j == NIFTI_R2L ) ) {
      FChange( &mj, &mk );
      FChange( &sj, &sk );
    }
    if( i == NIFTI_I2S )
      si = -si;
  }

  /* Computing the ellipsoid to be set in the center of the brain */
  A = Ellipsoid( mi * ( vx[ 1 ] - vx[ 0 ] ), mj * ( vy[ 1 ] - vy[ 0 ] ), mk * ( vz[ 1 ] - vz[ 0 ] ) );
  x = ( ( vx[ 1 ] + vx[ 0 ] ) / 2 ) + ( ( vx[ 1 ] - vx[ 0 ] ) * si );
  y = ( ( vy[ 1 ] + vy[ 0 ] ) / 2 ) + ( ( vy[ 1 ] - vy[ 0 ] ) * sj );
  z = ( ( vz[ 1 ] + vz[ 0 ] ) / 2 ) + ( ( vz[ 1 ] - vz[ 0 ] ) * sk );

  /* Computing subcortical mask */
  subcortical = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  CopySceneHeader( subcortical, mask );
  for( q = 0; q < A->n; q++ ) {
    if( ValidVoxel( mask, x + A->dx[ q ], y + A->dy[ q ], z + A->dz[ q ] ) ) {
      p = VoxelAddress( mask, x + A->dx[ q ], y + A->dy[ q ], z + A->dz[ q ] );
      subcortical->data[ p ] = mask->data[ p ];
    }
  }
  
  for( p = 0; p < mask->n; p++ ) {
    if( subcortical->data[ p ] == 1 )
      mask->data[ p ] = 0;
  }
  
  /* DestroyScene( &erode ); */
  DestroyAdjRel3( &A );
  return( subcortical );
}

void AtlasLabeling( Scene *brain, Scene *mask, Scene *dark_atlas, Scene *light_atlas, Scene **dark, Scene **light, BCSubgraph *sg, int protocol, float T ) {
  int p, dark_limit;
  unsigned long long *mean;
  int *position;
  uint *size;
  float *prob;
  int *psize;
  float dmin, dist;
  float dark_size;
  
  mean = ( unsigned long long* ) calloc( sg->nlabels, sizeof( unsigned long long ) );
  size = AllocUIntArray( sg->nlabels );
  prob = AllocFloatArray( sg->nlabels );
  psize = AllocIntArray( sg->nlabels );

  /* compute mean brightness and size of every label region */
  for( p = 0; p < (*dark)->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      mean[ sg->node[ p ].label ] += brain->data[ p ];
      size[ sg->node[ p ].label ] += 1;
    }
  }
  for( p = 0; p < sg->nlabels; p++ ) {
    if( size[ p ] != 0 ) 
      mean[ p ] = mean[ p ] / size[ p ];
    else mean[ p ] = 0;
  }
  /* sorting mean vector  */
  position = SortClusters( mean, size, sg->nlabels );
  
  /* compute intensity probability values. */
  dark_size = 0;
  dmin = 1.0;
  dark_limit = 0;
  for( p = 0; p < sg->nlabels; p++ ) {
    dist = fabs( T - ( dark_size / sg->nnodes ) );
    if( dist < dmin ) {
      dmin = dist;
      dark_limit = p;
    }
    dark_size += size[ p ];
  }

  /* compute atlas probability values. */
  for( p = 0; p < mask->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      if( dark_atlas->data[ p ] + light_atlas->data[ p ] != 0 ) {
	prob[ sg->node[ p ].label ] += ( ( ( float ) dark_atlas->data[ p ] - light_atlas->data[ p ] ) / 
					 ( ( float ) dark_atlas->data[ p ] + light_atlas->data[ p ] ) );
	psize[ sg->node[ p ].label ]++;
      }
    }
  }
  for( p = 0; p < sg->nlabels; p++ ) {
    if( psize[ p ] != 0 )
      prob[ p ] = prob[ p ] / psize[ p ];
    else
      prob[ p ] = 0;
  }
  
  /* Labeling samples into tissue classes. */
  for( p = 0; p < brain->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      if( ( light_atlas->data[ p ] != 0 ) || ( dark_atlas->data[ p ] != 0 ) ) {
	if( ( ( protocol ==  T1_PROTOCOL ) && ( prob[ sg->node[ p ].label ] > 0.0 ) )
	    || ( ( protocol !=  T1_PROTOCOL ) && ( prob[ sg->node[ p ].label ] < 0.0 ) ) )
	  ( *dark )->data[ p ] = 1; /* GM */
	else if( ( ( protocol ==  T1_PROTOCOL ) && ( prob[ sg->node[ p ].label ] < 0.0 ) )
		 || ( ( protocol !=  T1_PROTOCOL ) && ( prob[ sg->node[ p ].label ] > 0.0 ) ) )
	  ( *light )->data[ p ] = 1; /* WM */
      }
      else {
	if( ( ( protocol ==  T1_PROTOCOL ) && ( position[ sg->node[ p ].label ] <= dark_limit ) )
	    || ( ( protocol !=  T1_PROTOCOL ) && ( position[ sg->node[ p ].label ] >= dark_limit ) ) )
	  ( *dark )->data[ p ] = 1; /* GM */
	else
	  ( *light )->data[ p ] = 1; /* WM */
      }
    }
  }

  /* Printing for correctness verification. */
  if( verbose ) {
    printf( "Last dark cluster:%d\n", dark_limit );
    printf( "\nFinal image with proportion: dark:%5.4f, light:%5.4f\n", 
	    ( float ) dark_size / ( float ) ( sg->nnodes ), ( float ) ( sg->nnodes - dark_size ) / ( float ) ( sg->nnodes ) );
  }

  free( position );
  free( mean );
  free( size );
  free( prob );
  free( psize );
}

int BrainClusterAtlasSegmentation( Scene *scn, Scene *mask, Scene *dark_atlas, Scene *light_atlas, Scene **gm, Scene **wm, 
				   int samples, float mean_prop, int protocol, int auto_prop, float delta_prop, int nthreads ) {
  Scene *brain = NULL;
  Scene *submask = NULL;
  Scene *gm_cortical = NULL;
  Scene *wm_cortical = NULL;
  BCSubgraph *sg = NULL;
  BCFeatures *feats = NULL;
  int first_voxel;
  int last_voxel;
  int p;
  float df;

  if( verbose )
    printf( "Verificar máscara elíptica. Aprender a ler matrizes nii para rodar o método nas coordenadas corretas!\n");
  /* Create subcortical mask */
  submask = CreateSubCorticalMask( mask );
  /* WriteScene( submask, "subcortical.scn.bz2" ); */
  /* WriteScene( mask, "cortical.scn.bz2" ); */

  /* Run knn clustering over the cortical region */
  BrainClusterSegmentation( scn, mask, &gm_cortical, &wm_cortical, samples / 2, 0, protocol, WMGM_SEG, 1, delta_prop, 10, 20, nthreads );

  /* Brain restricted to the mask */
  brain = GetBrain( scn, submask );

  /* Computing submask limits for multiple thread execution. */
  MaskLimits( submask, &first_voxel, &last_voxel );
  
  /* GM, and WM proportions computation */
  TissueProportions( brain, submask, protocol, WMGM_SEG, auto_prop, delta_prop, &mean_prop );

  /* Computing feature vectors. */
  /* Needs to adjust the range of intensities and atlas, although they are already normalized. */
  /* feats = MedianSceneFeaturesAtlas( brain, submask, dark_atlas, light_atlas, first_voxel, last_voxel, 1.5, nthreads ); */
  /* feats = AtlasFeatures( brain, submask, dark_atlas, light_atlas, first_voxel, last_voxel ); */
  feats = CoOccurFeaturesAtlas( brain, submask, dark_atlas, light_atlas, first_voxel, last_voxel, 2, nthreads );
  /* feats = MedianSceneFeatures( brain, submask, first_voxel, last_voxel, 1.0, nthreads ); */
  
  // WriteFeats( feats, submask, "features.txt" );

  /* Clustering by k-nn to find best df. */
  sg = BCRandomSampl3( brain, submask, samples );
  SetBCFeatures( sg, feats );
  BCBestkClustering( sg, 10, 20 );
  /* Clustering by complete graph. */
  df = sg->df;
  FreeSubgraph( &sg );
  sg = BCUnifSampl3( brain, NULL, 1, 1, 1 );
  sg->df = df;
  sg->di = 3.0;
  SetBCFeatures( sg, feats );
  BCSPDF( sg, submask, first_voxel, last_voxel );
  BCSpatialOPFClustering( sg, submask );
  /* Setting labels */
  *gm = CreateScene( submask->xsize, submask->ysize, submask->zsize );
  CopySceneHeader( submask, *gm );
  *wm = CreateScene( submask->xsize, submask->ysize, submask->zsize );
  CopySceneHeader( submask, *wm );
  AtlasLabeling( brain, submask, dark_atlas, light_atlas, gm, wm, sg, protocol, mean_prop );

  /* Compose final image with cortical and subcortical regions */
  for( p = 0; p < mask->n; p++ ) {
    if( gm_cortical->data[ p ] == 1 )
      ( *gm )->data[ p ] = 1;
    if( wm_cortical->data[ p ] == 1 )
      ( *wm )->data[ p ] = 1;
  }
  CopySceneHeader( mask, *gm );
  CopySceneHeader( mask, *wm );
  /* Finishing processing. */  
  if( verbose )
    printf( "best k: %d, clusters: %d\n", sg->bestk, sg->nlabels );
  FreeSubgraph( &sg );
  DestroyBCFeatures( &feats );
  DestroyScene( &gm_cortical );
  DestroyScene( &wm_cortical );
  DestroyScene( &submask );
  DestroyScene( &brain );
  return( 0 );
}

#ifdef _BRAINCLUSTER_ATLAS_STANDALONE_
int main( int argc, char **argv ) {
  float *inputs=NULL;
  char filename[ 200 ];
  char fileextension[ 200 ];
  Scene *brain = NULL;
  Scene *mask = NULL;
  Scene *obj1 = NULL;
  Scene *obj2 = NULL;
  Scene *dark_atlas = NULL;
  Scene *light_atlas = NULL;
  int samples;
  float mean_prop;
  float delta_prop;
  int protocol;
  int auto_prop;
  int nthreads;
  
  /* Usage instructions. */
  if( argc < 5 ) {
    printf( "Usage must be: %s <scene name> <wmgm_mask_name> <gm_atlas> <wm_atlas> <output_basename> [<options>]\n", argv[ 0 ] );
    printf( "<options>\n\t-samples or -s <i>: i = number of samples (300 to 2000). Default: 1000.\n" );
    printf( "\t-mean_prop or -mp <f>: f = mean float value of GM partitions size(0.01 to 0.99). Default: Automatic.\n");
    printf( "\t-delta_prop or -dp <f>: f = delta to be added to the mean value of CSF or GM partitions size (-0.2 to 0.2). Default: 0.0.\n");
    printf( "\t-protocol or -p <i>: i = 0 - T1 image, 1 - T2 image, 2 - PD image. Default: 0.\n" );
    printf( "\t-threads or -th <i>: i: Number of threads.\n" );
    printf( "\t-v: Verbose. Default: off.\n" );
    return 0;
  }

  /* Reading and validating input images */
  sprintf( filename, "%s", argv[ 1 ] );
  if( !FileExists( filename ) ) {
    printf( "Input scene 1 not found!\n" );
    return 0;
  }
  brain = ReadVolume( filename );
  sprintf( filename, "%s", argv[ 3 ] );
  if( !FileExists( filename ) ) {
    printf( "Input dark atlas not found!\n" );
    return 0;
  } 
  dark_atlas = ReadVolume( filename );
  sprintf( filename, "%s", argv[ 4 ] );
  if( !FileExists( filename ) ) {
    printf( "Input light atlas not found!\n" );
    return 0;
  } 
  light_atlas = ReadVolume( filename );
  sprintf( filename, "%s", argv[ 2 ] );
  if( !FileExists( filename ) ) {
    printf( "Input mask not found!\n" );
    return 0;
  } 
  mask = ReadVolume( filename );

  /* Reading input flags */
  inputs = ParseBrainclusterAtlasInput( argc, argv );
  samples = ( int ) inputs[ 0 ];
  mean_prop = inputs[ 1 ];
  protocol = ( int ) inputs[ 2 ];
  auto_prop = ( int ) inputs[ 3 ];
  nthreads = ( int ) inputs[ 4 ];
  delta_prop = ( float ) inputs[ 5 ];
  free( inputs );

  /* Validating input flags */
  if( ( samples < 300.0 )  || ( samples > 2000.0 ) ) {
    printf( "Invalid number of samples: Expected value is from 300 to 2000.\n" );
    return 0;
  }
  if( ( auto_prop == 0 ) && ( ( mean_prop < 0.01 ) || ( mean_prop > 0.99 ) ) ) {
    printf( "Invalid mean proportion! Expected value is from 0.01 to 0.99.\n" );
    return 0;
  }
  if( ( delta_prop < -0.2 ) || ( delta_prop > 0.2 ) ) {
    printf( "Invalid delta proportion! Expected value is from -0.2 to 0.2.\n" );
    return 0;
  }
  if( ( protocol < T1_PROTOCOL ) || ( protocol > PD_PROTOCOL ) ) {
    printf( "Invalid protocol! Expected 0 - T1 image, 1 - T2 image, 2 - PD image.\n" );
    return 0;
  }

  /* Running the algorithm. */
  BrainClusterAtlasSegmentation( brain, mask, dark_atlas, light_atlas, &obj1, &obj2, samples, mean_prop, protocol, auto_prop, delta_prop, nthreads );

  /* Saving results. */
  GetFileExtension( filename, fileextension );
  sprintf( filename, "%s-gm%s", argv[ 5 ], fileextension );
  WriteVolume( obj1, filename );
  sprintf( filename, "%s-wm%s", argv[ 5 ], fileextension );
  WriteVolume( obj2, filename );
  
  /* Cleaning memory. */
  DestroyScene( &obj1 );
  DestroyScene( &obj2 );
  DestroyScene( &brain );
  DestroyScene( &mask );
  DestroyScene( &dark_atlas );
  DestroyScene( &light_atlas );
  
  return( 0 );
}
#endif
