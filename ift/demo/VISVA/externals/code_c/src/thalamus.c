#include "oldift.h"
#include "orderlist.h"
#include "inhomogeneity.h"
#include "braincluster.h"
#include "rigidregistration3.h"
#include "filelist.h"
#include "featmap.h"
#include "gradient.h"

//extern int opf_MAXDENS;
extern bool verbose, err;
extern Features3 *feats;
extern Scene *segmentation;

float *ParseThalamusInput( int argc, char **argv ) {
  float *result;
  int i;
  bool auto_k = true;

  result = ( float* ) calloc( 20, sizeof( float ) );

  result[ 0 ] = T1_PROTOCOL;
  result[ 1 ] = 10.0; // kmin
  result[ 2 ] = 20.0; // kmax
  result[ 3 ] = 1; // number of threads
  verbose = false;
  err = false;
  for( i = 3; i < argc; i++ ) {
    if( ( strcmp( argv[ i ], "-protocol" ) == 0 ) || ( strcmp( argv[ i ], "-p" ) == 0 ) )
      result[ 0 ] = atof( argv[ ++i ] );
    if( strcmp( argv[ i ], "-k" ) == 0 ) {
      result[ 1 ] = atof( argv[ ++i ] );
      result[ 2 ] = atof( argv[ ++i ] );
      auto_k = false;
    }
    if( ( strcmp( argv[ i ], "-threads" ) == 0 ) ||( strcmp( argv[ i ], "-th" ) == 0 ) )
      result[ 3 ] = atoi( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-verbose" ) == 0 ) || ( strcmp( argv[ i ], "-v" ) == 0 ) )
      verbose = true;
    if( strcmp( argv[ i ], "-e" ) == 0 ) 
      err = true;
  }
  return result;
}

void SetMinMaxThreshold( float auto_prop, float *mean_prop, float prop_int, float auto_move, float *Tmin, float *Tmax );

int *sortClusters( unsigned long long *mean, unsigned long long *std_dev, uint *size, int nlabels );

void GnuplotClusters( char *fname, char *gname, char *cbasename, unsigned long long *mean, 
		      unsigned long long *std_dev, uint *size, int T, int n );

void MedianSceneFeaturesAtlas( Scene *scn, Scene *mask, Scene *dark_atlas, Scene *light_atlas, float r, int t_threads );

void SetSubgraphFeaturesMask3( Subgraph *sg, Features3 *f, Scene *mask );

void SpatialPDF3( Subgraph *sg, Scene *mask, int t_threads );

// Unsupervised OPF with spatial constrains
void SceneSpatialUnsupOPF3( Subgraph *sg, Scene *mask );

void LabelSetting( Subgraph *sg, Scene *mask );

void AtlasMaskLabeling( Scene *brain, Scene *mask, Scene *result, Scene *dark_atlas, Scene *light_atlas, Scene *thalamus_atlas, int nlabels, int protocol ) {
  float *prob;
  int *size;
  int p;
  prob = AllocFloatArray( nlabels );
  size = AllocIntArray( nlabels );

  // compute probabilities and size of every label region
  for( p = 0; p < result->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      if( dark_atlas->data[ p ] + light_atlas->data[ p ] != 0 ) {
	prob[ result->data[ p ] ] +=  ( ( float ) dark_atlas->data[ p ] - light_atlas->data[ p ] ) /
	  ( ( float ) dark_atlas->data[ p ] + light_atlas->data[ p ] );
	size[ result->data[ p ] ]++;
      }
    }
  }
  for( p = 0; p < nlabels; p++ ) {
    if( size[ p ] != 0 )
      prob[ p ] = prob[ p ] / size[ p ];
    else
      prob[ p ] = 0;
  }

  // Removing WM clusters from mask.
  for( p = 0; p < brain->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      if( ( ( protocol ==  T1_PROTOCOL ) && ( prob[ result->data[ p ] ] < 0.0 ) )
	       || ( ( protocol !=  T1_PROTOCOL ) && ( prob[ result->data[ p ] ] > 0.0 ) ) ) {
	mask->data[ p ] = 0;
      }
    }
  }

  // Picking thalamus clusters.
  for( p = 0; p < nlabels; p++ ) {
    size[ p ] = 0;
    prob[ p ] = 0;
  }
  for( p = 0; p < brain->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      size[ result->data[ p ] ]++;
      prob[ result->data[ p ] ] += thalamus_atlas->data[ p ];
    }
  }

  for( p = 0; p < brain->n; p++ ) {
    if( mask->data[ p ] == 0 )
      result->data[ p ] = 0;
    else if( prob[ result->data[ p ] ] / size[ result->data[ p ] ] >= 0.5 )
      result->data[ p ] = 1;
    else
      result->data[ p ] = 0;
  }
}

void BrightnessMaskLabeling( Scene *brain, Scene *mask, Scene *result, int nlabels, int protocol, 
			    float T, int nvoxels, int dvoxels ) {
  int p, dark_limit, light_limit;
  unsigned long long *mean;
  unsigned long long *std_dev;
  int *position;
  uint *size;
  float dark_size, light_size;
  
  mean = ( unsigned long long* ) calloc( nlabels, sizeof( unsigned long long ) );
  size = AllocUIntArray( nlabels );
  std_dev = ( unsigned long long* ) calloc( nlabels, sizeof( unsigned long long ) );

  // compute mean brightness and size of every label region
  for( p = 0; p < result->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      mean[ result->data[ p ] ] += brain->data[ p ];
      size[ result->data[ p ] ] += 1;
    }
  }
  for( p = 0; p < nlabels; p++ ) {
    if( size[ p ] != 0 )
      mean[ p ] /= size[ p ];
    else mean[ p ] = 0;
  }

  // Compute std_dev for gnuplot graphics.
  for( p = 0; p < brain->n; p++ )
    if( mask->data[ p ] != 0 )
      std_dev[ result->data[ p ] ] += abs( brain->data[ p ] - mean[ result->data[ p ] ] );
  
  for( p = 0; p < nlabels; p++ ) {
    if( size[ p ] != 0 )
      std_dev[ p ] /= size[ p ];
  }
  
  // sorting mean vector
  position = sortClusters( mean, std_dev, size, nlabels );

  // Labeling clusters in dark and bright classes
  for( p = 0; size[ p ] == 0; p++ ); // find first non-zero size cluster.
  dark_size  = dvoxels + size[ p ];
  dark_limit = p;
  for( p++; ( p < nlabels - 1 ) && ( T - ( ( dark_size + size[ p ] ) / nvoxels ) > 0.16 ); p++ ){
    dark_size += size[ p ];
    dark_limit = p;
  }
  light_limit = p;
  light_size = nvoxels - dark_size;
  for( ; ( p < nlabels - 1 ) && ( ( 1.0 - T ) - ( ( light_size - size[ p ] ) / nvoxels ) < 0.16 ); p++ ){
    light_size -= size[ p ];
    light_limit = p;
  }

  // Removing WM clusters from mask.
  for( p = 0; p < brain->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      if( ( ( protocol ==  T1_PROTOCOL ) && ( position[ result->data[ p ] ] >= light_limit ) )
	       || ( ( protocol !=  T1_PROTOCOL ) && ( position[ result->data[ p ] ] <= dark_limit ) ) ) {
	mask->data[ p ] = 0; // WM
      }
    }
  }

  if( err ) {
    GnuplotClusters( "clusters_wmgm.gnp", "clusters_wmgm.png", "cluster_wmgm", mean, std_dev, size, dark_limit, nlabels );
    free( std_dev );
  }

  free( position );
  free( mean );
  free( size );
}

int ThalamusSegmentation( Scene *scn, Scene *par_mask, Scene *dark_atlas , Scene *light_atlas, Scene *thalamus_atlas, 
			      Scene **obj, float p_protocol, float p_kmin, float p_kmax, float threads ) {

  timer *t1 = NULL, *t2 = NULL;
  Scene *mask = NULL;
  Scene *cluster_mask = NULL;
  Scene *brain = NULL;
  Subgraph *sg = NULL;
  int p;
  float Tmin, Tmax;
  float mean_prop, prop_int;
  int nvoxels;
  int kmin, kmax;
  int protocol;
  int t_threads;

  if( verbose ) {
    t1 = Tic();
    srand( ( int ) t1->tv_usec );
  }

  protocol            = ROUND( p_protocol );
  kmin                = ROUND( p_kmin );
  kmax                = ROUND( p_kmax );
  t_threads           = ( int ) threads;

  mask = par_mask;
  brain = CopyScene( scn );
  nvoxels = 0;
  for( p = 0; p < brain->n; p++ ) {
    if( mask->data[ p ] == 0 )
      brain->data[ p ] = 0;
    else
      nvoxels++;
  }

  // proportions computation
  ProportionsBrain( brain, mask, protocol, WMGM_SEG, &mean_prop, &prop_int );
  SetMinMaxThreshold( 1.0, &mean_prop, prop_int, 0.0, &Tmin, &Tmax );

  if( verbose ) {
    if( protocol == T1_PROTOCOL )
      printf( "Protocol: T1, " );
    else if( protocol == T2_PROTOCOL )
      printf( "Protocol: T2, " );
    else if( protocol == PD_PROTOCOL )
      printf( "Protocol: PD, " );
    else
      printf( "Protocol: PD and T2, " );
  }

  cluster_mask = CopyScene( mask );
  MedianSceneFeaturesAtlas( brain, cluster_mask, dark_atlas, light_atlas, 1.5, t_threads );
    
  sg = CreateSubgraph( scn->n );
  sg->di = 3.0;
  SetSubgraphFeaturesMask3( sg, feats, cluster_mask );
  SpatialPDF3( sg, cluster_mask, t_threads );
  SceneSpatialUnsupOPF3( sg, cluster_mask );
  LabelSetting( sg, cluster_mask );
  err = true;
  if( err )
    WriteScene( segmentation, "labels.scn.bz2" );
  err = false;
  
  BrightnessMaskLabeling( brain, cluster_mask, segmentation, sg->nlabels, protocol, mean_prop, nvoxels, 0 );
  if( err )
    WriteScene( cluster_mask, "atlas_mask.scn.bz2" );
  AtlasMaskLabeling( brain, cluster_mask, segmentation, dark_atlas, light_atlas, thalamus_atlas, sg->nlabels, protocol );
  
  DestroySubgraph( &sg );
  DestroyScene( &cluster_mask );
  DestroyFeatures3( &feats );

  //cluster_mask = OpenBin3( segmentation, 3.3 );
  cluster_mask = CopyScene( segmentation );
  DestroyScene( &segmentation );
  cluster_mask->dx  = mask->dx;
  cluster_mask->dy  = mask->dy;
  cluster_mask->dz  = mask->dz;

  *obj = cluster_mask;

  if( verbose ) {
    t2 = Toc();
    printf( "Elapsed time: %f ms\n", CTime( t1, t2 ) );
  }

  DestroyScene( &brain );

  return( 0 );
}

int main( int argc, char **argv ) {
  float *inputs=NULL;
  char basename[ 200 ], filename[ 200 ];
  char fileextension[ 200 ];
  Scene *brain = NULL;
  Scene *mask = NULL;
  Scene *gm_atlas = NULL;
  Scene *wm_atlas = NULL;
  Scene *thalamus_atlas = NULL;
  Scene *obj = NULL;
  
  if( argc <= 5 ) {
    printf( "Usage must be: %s <scene_name> <mask_name> <gm_atlas_name> <wm_atlas_name> <thalamus_atlas_name> [<options>]\n", argv[ 0 ] );
    printf( "<options>\n" );
    printf( "\t-k <f> <f>: f = lower and upper limits to k-nn. Defaults: 18 and 20 for CSF/WMGM and 38 and 40 for WM/GM segmentation.\n" );
    printf( "\t-protocol or -p <i>: i = 0 - T1 image, 1 - T2 image, 2 - PD image, 3 - PD and T2 images. Default: 0.\n" );
    printf( "\t-threads or -th <i>: i: Number of threads.\n" );
    printf( "\t-v: Verbose. Default: off.\n" );
    return 0;
  }

  sprintf( filename, "%s", argv[ 1 ] );
  if( !FileExists( filename ) ) {
    printf( "Input scene 1 not found!\n" );
    return 0;
  }
  else 
    brain = ReadVolume( filename );

  sprintf( basename, "%s", filename );
  GetFileExtension( filename, fileextension );
  //printf( "%s, %s\n", filename, fileextension );
  RemoveFileExtension( basename );
  RemoveFileExtension( basename );

  sprintf( filename, "%s", argv[ 2 ] );
  if( !FileExists( filename ) ) {
    printf( "Input mask not found!\n" );
    return 0;
  } 
  else
    mask = ReadVolume( filename );

  sprintf( filename, "%s", argv[ 3 ] );
  if( !FileExists( filename ) ) {
    printf( "Input mask not found!\n" );
    return 0;
  } 
  else
    gm_atlas = ReadVolume( filename );

  sprintf( filename, "%s", argv[ 4 ] );
  if( !FileExists( filename ) ) {
    printf( "Input mask not found!\n" );
    return 0;
  } 
  else
    wm_atlas = ReadVolume( filename );

  sprintf( filename, "%s", argv[ 5 ] );
  if( !FileExists( filename ) ) {
    printf( "Input mask not found!\n" );
    return 0;
  } 
  else
    thalamus_atlas = ReadVolume( filename );

  inputs      = ParseThalamusInput( argc, argv );
  
  if( inputs[ 1 ] > inputs[ 2 ] ) {
    printf( "kmin must be greater than kmax!\n" );
    return 0;
  }

  ThalamusSegmentation( brain, mask, gm_atlas, wm_atlas, thalamus_atlas, &obj, inputs[ 0 ], inputs[ 1 ], inputs[ 2 ], inputs[ 3 ] );

  sprintf( filename, "%s-thalamus%s", basename, fileextension );
  WriteVolume( obj, filename );
  
  DestroyScene( &obj );
  DestroyScene( &brain );
  DestroyScene( &gm_atlas );
  DestroyScene( &wm_atlas );
  DestroyScene( &thalamus_atlas );
  DestroyScene( &mask );
  free( inputs );
  
  return( 0 );
}
