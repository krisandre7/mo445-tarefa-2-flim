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

bool verbose;

float *ParseBrainclusterInput( int argc, char **argv ) {
  float *result;
  int i;

  result = ( float* ) calloc( 20, sizeof( float ) );
  result[ 0 ] = ( float ) 1000; /* samples */
  result[ 1 ] = 0.00; /* mean proportion */
  result[ 2 ] = T1_PROTOCOL; /* protocol type */
  result[ 3 ] = WMGM_SEG; /* tissue type */
  result[ 4 ] = 1.0; /* auto proportion flag */
  result[ 5 ] = 10.0; /* kmin */
  result[ 6 ] = 20.0; /* kmax */
  result[ 7 ] = 1; /* number of threads */
  result[ 8 ] = 0.0; /* delta proportion */
  verbose = false;
  for( i = 4; i < argc; i++ ) {
    if( ( strcmp( argv[ i ], "-samples" ) == 0 ) || ( strcmp( argv[ i ], "-s" ) == 0 ) ) 
      result[ 0 ] = atoi( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-mean_prop" ) == 0 ) || ( strcmp( argv[ i ], "-mp" ) == 0 ) ) {
      result[ 1 ] = atof( argv[ ++i ] );
      result[ 4 ] = 0.0;
    }
    if( ( strcmp( argv[ i ], "-protocol" ) == 0 ) || ( strcmp( argv[ i ], "-p" ) == 0 ) )
      result[ 2 ] = atof( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-tissue_type" ) == 0 ) || ( strcmp( argv[ i ], "-tt" ) == 0 ) )
      result[ 3 ] = atoi( argv[ ++i ] );
    if( strcmp( argv[ i ], "-k" ) == 0 ) {
      result[ 5 ] = atof( argv[ ++i ] );
      result[ 6 ] = atof( argv[ ++i ] );
    }
    if( ( strcmp( argv[ i ], "-threads" ) == 0 ) ||( strcmp( argv[ i ], "-th" ) == 0 ) )
      result[ 7 ] = atoi( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-delta_p" ) == 0 ) ||( strcmp( argv[ i ], "-dp" ) == 0 ) )
      result[ 8 ] = atof( argv[ ++i ] );
    if( ( strcmp( argv[ i ], "-verbose" ) == 0 ) || ( strcmp( argv[ i ], "-v" ) == 0 ) )
      verbose = true;
  }
  return result;
}

/* Brain restricted to the mask */
Scene *GetBrain( Scene *scn, Scene *mask ) {
  Scene *result = NULL;
  int p;

  result = CopyScene( scn );
  for( p = 0; p < scn->n; p++ ) {
    if( mask->data[ p ] == 0 )
      result->data[ p ] = 0;
  }
  return( result );
}

/* Auxiliar function to complement T2- and PD-images for automatic threshold computing. */
Scene  *NonZeroComplement3( Scene *scn ) {
  Scene *cscn = NULL;
  Curve *h = NULL;
  int p; 
  int Imax;

  /* Compute accumulated histogram. */
  h = Histogram3( scn );
  for( p = 2; p < h->n; p++ )
    h->Y[ p ] += h->Y[ p - 1 ];
  
  /* Smooth 0.1% of maximal. */
  for( p = h->n - 1; h->Y[ p ] >= h->Y[ h->n - 1 ] * 0.999; p-- );
  Imax = p;
  
  /* Compute complement scene. */
  cscn = CreateScene( scn->xsize, scn->ysize, scn->zsize );
  for ( p=0; p < scn->n; p++ ) {
    if( scn->data != 0 )
      cscn->data[ p ] = MAX( Imax - scn->data[ p ], 0 );
  }
  
  DestroyCurve( &h );
  return( cscn );
}

void AbnormalHistogramCorrection( Curve *m, int xmin, int xmax, int xmean, int *dpeak, int *valley, int *bpeak ) {
  int p, p0;
  int q, q0;

  /* Detect abnormal histograms with more than two peaks. */
  for( p0 = *dpeak; m->Y[ p0 ] >= 0.95 * m->Y[ *dpeak ]; p0++ ); /* Move to the first slope after dark peak. */
  for( q0 = *bpeak; m->Y[ q0 ] >= 0.95 * m->Y[ *bpeak ]; q0-- ); /* Move to the first slope before light peak. */
  if( p0 < q0 ) { /* If dark and light peaks are not part of a dome, there may be some valleys and peaks between slopes p0 and q0. */
    p = GetFirstValley( m, p0, q0 ); /* Move to the first valley after slope p0. */
    q = GetLastValley( m, p0, q0 ); /* Move to the first valley before slope q0. */
    for( p0 = p; ( m->Y[ p ] >= 0.95 * m->Y[ p0 ] ) && ( p0 < xmax ); p0++ ); /* Move to the first acclivity after the valley p. */
    for( q0 = q; ( m->Y[ q ] >= 0.95 * m->Y[ q0 ] ) && ( q0 > xmin ); q0-- ); /* Move to the first acclivity before the valley q. */
    if( p0 < q0 ) { /* There is at least another peak. */
      *valley = xmean;
      *dpeak = GetPeak( m, xmin, *valley );
      *bpeak = GetPeak( m, *valley, xmax );
      if( verbose )
	printf( "Could not detect dark and/or light peaks! Mean intensity value taken.\n" );
    }
    else { /* Good histogram. */
      if( verbose )
	printf( "Good histogram.\n" );
      for( q0 = *valley; m->Y[ *valley ] * 1.05 >= m->Y[ q0 ]; q0-- ); /* Move to the first acclivity before the valley dark/light. */
      for( p0 = *valley; m->Y[ *valley ] * 1.05 >= m->Y[ p0 ]; p0++ ); /* Move to the first acclivity after the valley p. */
      *valley = ( q0 + p0 ) / 2;
    }
  }
  else if( m->Y[ *dpeak ] > m->Y[ *bpeak ] * 1.06 ) { /* There are a dark peak and a large light dome including dark/light transition. */
    if( verbose )
      printf( "There are a dark peak and a large light dome including dark/light transition.\n" );
    /* Refining dark peak: */
    p0 = GetPlateauBegining( m, *dpeak, 0.05 );
    q0 = GetPlateauEnding( m, *dpeak, 0.05 );
    *dpeak = ( p0 + q0 ) / 2;
    /* Refining light peak: */
    p0 = GetPlateauBegining( m, *bpeak, 0.05 );
    q0 = GetPlateauEnding( m, *bpeak, 0.05 );
    *valley = p0 + ( q0 - p0 ) / 2;
    *bpeak = q0 - ( q0 - p0 ) / 4;
  }
  else if( m->Y[ *bpeak ] > m->Y[ *dpeak ] * 1.06 ) { /* There are a light peak and a large dark dome including dark/light transition. */
    if( verbose )
      printf( "There are a light peak and a large dark dome including dark/light transition.\n" );
    /* Refining light peak: */
    p0 = GetPlateauBegining( m, *bpeak, 0.05 );
    q0 = GetPlateauEnding( m, *bpeak, 0.05 );
    *bpeak = ( p0 + q0 ) / 2;
    /* Refining dark peak: */
    p0 = GetPlateauBegining( m, *dpeak, 0.05 );
    q0 = GetPlateauEnding( m, *dpeak, 0.05 );
    *dpeak = p0 + ( q0 - p0 ) / 4;
    *valley = q0 - ( q0 - p0 ) / 2;
  }
  else { /* If light and dark peaks compose an unique dome. Set gm peak to a reasonable position. */
    if( verbose )
      printf( "No WM/GM valley. Valey X,Y:%d,%d. It is a dome!", ( int ) m->X[ *valley ], ( int ) m->Y[ *valley ] );
    for( p = *dpeak; m->Y[ p ] >= 0.85 * m->Y[ *dpeak ]; p-- ); /* get begining of dome */
    for( q = *bpeak; m->Y[ q ] >= 0.85 * m->Y[ *bpeak ]; q++ ); /* get end of dome */
    if( verbose )
      printf( " Dome min X,Y:%d,%d. Dome max X,Y:%d,%d\n", ( int ) m->X[ p ], ( int ) m->Y[ p ], ( int ) m->X[ q ], ( int ) m->Y[ q ] );
    *dpeak = p + ( q - p ) / 4;
    *bpeak = q - ( q - p ) / 4;
    *valley = ( p + q ) / 2;
  }
}

TissueMarker *BrainTissueThresholds( Scene *brain, int protocol ) {
  Scene *aux = NULL;
  Curve *h=NULL;
  Curve *m=NULL;
  float ymax;
  int p, nv;
  int step;
  unsigned long long sum;
  TissueMarker *tm = NULL;

  tm = ( TissueMarker* ) calloc( 1, sizeof( TissueMarker ) );
  h = Histogram3( brain );
  m = MedianFilterHistogram( h );
  DestroyCurve( &h );

  tm->min = INT_MAX;
  tm->max = INT_MIN;
  ymax = 0;

  /* Compute amplitude. */
  for( p = 1; p < m->n; p++ ) {
    if( m->Y[ p ] > ymax )
      ymax = m->Y[ p ];
  }

  if( protocol == T1_PROTOCOL ) {
    /* Compute CSF and WM extremities. */
    for( p = GetFirstValley( m, 1, m->n - 1 ); m->Y[ p ] <= 0.05 * ymax; p++ );
    tm->min = p;
    for( p = m->n - 2; m->Y[ p ] <= 0.3 * ymax; p-- );
    tm->max = p;
    /* Find otsu threshold coordinates */
    tm->otsu = MyOtsu( brain );

    /* Compute histogram mean for GM and WM voxels, greater than Otsu threshold */
    sum = 0;
    nv = 0;
    for( p = 0; p < brain->n; p++ ) {
      if( brain->data[ p ] > tm->otsu ) {
	sum += brain->data[ p ];
	nv++;
      }
    }
    tm->mean = ROUND( sum / ( float ) nv );
    for( p = 0; tm->otsu > m->X[ p ]; p++ );
    tm->otsu = p;
    for( p = 0; tm->mean > m->X[ p ]; p++ );
    tm->mean = p;

    /* Set step size depending on the size of the histogram. */
    step = ( tm->max - tm->min ) / 100;
    if( step == 0 )
      step = 1;
    /* Tries to jump a small CSF peak. */
    p = tm->otsu;
    if( m->Y[ p ] >= 0.2 * ymax ) {
      while( m->Y[ p ] >= 0.2 * ymax )
	p++;
    }
    /* Ensure that did not jump all peaks and reaches the begining of the slope. */
    if( p >= tm->max - step )
      p = tm->otsu;
    else {
      while( m->Y[ p ] <= 0.2 * ymax )
	p++;
    }
    /* Ensure that did not jump GM peak and if so, reset it to the maximum between the begining of the histogram and Otsu. */
    if( p > tm->mean )
      p = MAX( tm->min, tm->otsu );
    /* Compute GM peak. */
    tm->gm = GetFirstPeak( m, p, tm->mean );
    /* Find WM peak. */
    tm->wm = GetLastPeak( m, tm->gm, tm->max );
    /* Find GM_WM valley */
    tm->gm_wm = GetValley( m, tm->gm, tm->wm );

    if( verbose )
      printf( "Pré-thesholds: GM:%2.1f, GMWM:%2.1f, WM:%2.1f\n", m->X[ tm->gm ], m->X[ tm->gm_wm ], m->X[ tm->wm ] );

    /* Test for abnormal histograms with more or less than two peaks. */
    AbnormalHistogramCorrection( m, tm->min, tm->max, tm->mean, &tm->gm, &tm->gm_wm, &tm->wm );
    /* Find CSF peak and CSF_GM valley */
    tm->csf_gm = ( tm->min + tm->gm ) / 2;
    tm->csf = ( tm->min + tm->csf_gm ) / 2;
  }
  else { /* T2 or PD protocols. */
    /* Compute CSF and WM extremities. */
    for( p = 1; m->Y[ p ] <= 0.3 * ymax; p++ );
    tm->min = p;
    for( p = m->n - 2; m->Y[ p ] <= 0.05 * ymax; p-- );
    tm->max = p;
    /* complementing image for otsu computation. */
    aux = NonZeroComplement3( brain );
    /* Find otsu threshold coordinates. */
    tm->otsu = MyOtsu( aux );
    DestroyScene( &aux );

    /* Compute histogram mean for GM and WM voxels, lower than Otsu threshold */
    sum = 0;
    nv = 0;
    for( p = 0; p < brain->n; p++ ) {
      if( ( brain->data[ p ] < tm->otsu ) && ( brain->data[ p ] != 0 ) ) {
	sum += brain->data[ p ];
	nv++;
      }
    }
    tm->mean = ROUND( sum / ( float ) nv );
    for( p = 0; tm->otsu > m->X[ p ]; p++ );
    tm->otsu = p;
    for( p = 0; tm->mean > m->X[ p ]; p++ );
    tm->mean = p;

    /* Find GM peak. */
    step = ( tm->max - tm->min ) / 100;
    p = tm->otsu;
    if( m->Y[ p ] >= 0.25 ) {
      while( m->Y[ p ] >= 0.25 * ymax )
	p--;
    }
    while( m->Y[ p ] <= 0.25 * ymax )
      p--;
    if( p < tm->mean )
      p = tm->max;
    tm->gm = GetLastPeak( m, tm->min, p );
    /* Find WM peak. */
    tm->wm = GetFirstPeak( m, tm->min, tm->gm );
    /* Find GM_WM valley */
    tm->gm_wm = GetValley( m, tm->wm, tm->gm );

    if( verbose )
      printf( "Pré-thesholds:GM:%2.1f, GMWM:%2.1f, WM:%2.1f\n", m->X[ tm->gm ], m->X[ tm->gm_wm ], m->X[ tm->wm ] );
    
    /* Test for abnormal histograms with more or less than two peaks. */
    AbnormalHistogramCorrection( m, tm->min, tm->max, tm->mean, &tm->wm, &tm->gm_wm, &tm->gm );
    /* Find CSF peak and CSF_WM valley */
    tm->csf_gm = ( tm->gm + tm->max ) / 2;
    tm->csf = ( tm->max + tm->csf_gm ) / 2;
  }

  if( verbose )
    printf( "xmin:%d, xmax:%d, otsu:%d, mean:%d, CSF_GM valley:%d, GM peak:%d, GM_WM valley:%d, WM peak:%d\n", 
	    ( int ) m->X[ tm->min ], ( int ) m->X[ tm->max ], ( int ) m->X[ tm->otsu ], ( int ) m->X[ tm->mean ],
	    ( int ) m->X[ tm->csf_gm ], ( int ) m->X[ tm->gm ], ( int ) m->X[ tm->gm_wm ], ( int ) ( int ) m->X[ tm->wm ] );

  /* Get the frequency of marks. */
  tm->min = m->X[ tm->min ];
  tm->max = m->X[ tm->max ];
  tm->mean = m->X[ tm->mean ];
  tm->otsu = m->X[ tm->otsu ];
  tm->csf = m->X[ tm->csf ];
  tm->csf_gm = m->X[ tm->csf_gm ];
  tm->gm = m->X[ tm->gm ];
  tm->gm_wm = m->X[ tm->gm_wm ];
  tm->wm = m->X[ tm->wm ];

  DestroyCurve( &m );
  return( tm );
}

int WMGMThreshold( Scene *brain ) {
  Curve *m = NULL;
  Curve *h = NULL;
  unsigned long long sum;
  float ymax;
  int p, nv;
  int xmin, xmax;
  int gm, wm, step;
  int mean, xmean, gm_wm;
  
  sum = 0;
  nv = 0;
  xmin = INT_MAX;
  xmax = INT_MIN;
  ymax = 0;

  /* Filter histogram from high noise. */
  h = Histogram3( brain );
  m = MedianFilterHistogram( h );

  /* Compute amplitude, and GM and WM extremities. */
  for( p = 1; p < m->n; p++ ) {
    if( m->Y[ p ] > ymax )
      ymax = m->Y[ p ];
  }
  for( p = 1; m->Y[ p ] <= 0.3 * ymax; p++ );
  xmin = p;
  for( p = m->n - 1; m->Y[ p ] <= 0.3 * ymax; p-- );
  xmax = p;
  
  /* Compute histogram mean */
  for( p = 0; p < brain->n; p++ ) {
    if( brain->data[ p ] != 0 ) {
      sum += brain->data[ p ];
      nv++;
    }
  }
  mean = ROUND( sum / ( float ) nv );
  for( xmean = 0; m->X[ xmean ] < mean; xmean++ );

  /* Detect WM and GM peaks and WMGM valley. */
  step = MAX( ( xmax - xmin ) / 100, 1 );
  gm = GetFirstPeak( m, xmin + step, xmean );
  wm = GetLastPeak( m, xmean, xmax - step );
  gm_wm = GetValley( m, gm, wm );

  if( verbose )
    printf( "Pré-thesholds:GM:%2.1f, GMWM:%2.1f, WM:%2.1f\n", m->X[ gm ], m->X[ gm_wm ], m->X[ wm ] );

  /* Test for abnormal histograms with more or less than two peaks. */
  AbnormalHistogramCorrection( m, xmin, xmax, xmean, &gm, &gm_wm, &wm );

  if( verbose ) {
    printf( "xmin:%d, xmax:%d, ymax:%f, mean:%d, Dark peak:%d, light. peak:%d, GM_WM:%d\n", ( int ) m->X[ xmin ], ( int ) m->X[ xmax ], ymax, mean, ( int ) m->X[ gm ], ( int ) m->X[ wm ], ( int ) m->X[ gm_wm ] );
  }

  gm_wm = m->X[ gm_wm ];
  DestroyCurve( &h );
  DestroyCurve( &m );
  return( gm_wm );
}

void BrainPreProportions( Scene *brain, Scene *mask, int protocol, float *T_CSF_GM, float *T_GM_WM ) {
  int p, n;
  int csf_sum, gm_sum;
  TissueMarker *tm;

  tm = BrainTissueThresholds( brain, protocol );
  n = 0;
  csf_sum = 0;
  gm_sum = 0;
  for( p = 0; p < brain->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      n++;
      if( brain->data[ p ] <= tm->csf_gm )
	csf_sum++;
      else if( brain->data[ p ] <= tm->gm_wm )
	gm_sum++;
    }
  }
  *T_CSF_GM = ( float ) csf_sum / ( float ) n;
  *T_GM_WM = ( float ) gm_sum / ( ( float ) n - ( float ) csf_sum );
  if( verbose )
    printf( "Proportions: CSF/GM:%0.2f, GM/WM:%0.2f\n", *T_CSF_GM, *T_GM_WM );
  free( tm );
}

/* Computes segmentation thresholds for expected volume of each class. */
void TissueProportions( Scene *brain, Scene *mask, int protocol, int tissue, int auto_prop, float delta_prop, float *Tmean ) {
  int p, n;
  int T;
  int sum;
  TissueMarker *tm;

  if( auto_prop != 0 ) {
    if( tissue == WMGM_SEG )
      T = WMGMThreshold( brain );
    else {
      tm = BrainTissueThresholds( brain, protocol );
      T = tm->csf_gm;
      free( tm );
    }
    n = 0;
    sum = 0;
    for( p = 0; p < brain->n; p++ ) {
      if( mask->data[ p ] != 0 )
	n++;
      if( ( mask->data[ p ] != 0 ) && ( brain->data[ p ] <= T ) )
	sum++;
    }
    *Tmean = ( ( float ) sum / ( float ) n ) + delta_prop;
    if( *Tmean < 0.01 )
      *Tmean = 0.01;
    if( *Tmean > 0.99 )
      *Tmean = 0.99;
  }
}

/* Computing attributes to speed up image classification. */
void MaskLimits( Scene *mask, int *first_voxel, int *last_voxel ) {
  int i;
  
  *first_voxel = INT_MAX;
  *last_voxel = 0;
  for ( i = 0; i < mask->n; i++ ) {
    if ( mask->data[ i ] != 0 ) {
      if( *first_voxel > i )
	*first_voxel = i;
      if( *last_voxel < i )
	*last_voxel = i;
    }
  }
}

int *SortClusters( unsigned long long *mean, uint *size, int nlabels ) {
  int i, j;
  int *position;
  uint *order;
  order = AllocUIntArray( nlabels );
  position = AllocIntArray( nlabels );
  
  for( i = 0; i < nlabels; i++ ) {
    order[ i ] = i;
    for( j = i; j > 0; j-- ) {
      if( mean[ j ] < mean[ j - 1 ] ) {
	ULChange( &( mean[ j ] ), &( mean[ j - 1 ] ) );
	UChange( &( size[ j ] ), &( size[ j - 1 ] ) );
	UChange( &( order[ j ] ), &( order[ j - 1 ] ) );
      }
    }
  }
  
  for( i = 0; i < nlabels; i++ )
    position[ order[ i ] ] = i;
  free( order );

  return( position );
}

void BrightnessLabeling( Scene *brain, Scene *mask, Scene *dark, Scene *light, int nlabels, int protocol, float T ) {
  int p, i, dark_limit, light_limit, middle = 0;
  unsigned long long *mean;
  int *position, *sort, *index;
  uint *size;
  float dark_size, dist, dmin;
  int nvoxels;
  
  mean = ( unsigned long long* ) calloc( nlabels, sizeof( unsigned long long ) );
  size = AllocUIntArray( nlabels );

  /* compute mean brightness and size of every label region */
  nvoxels = 0;
  for( p = 0; p < mask->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      mean[ dark->data[ p ] ] += brain->data[ p ];
      size[ dark->data[ p ] ] += 1;
      nvoxels++;
    }
  }
  for( p = 0; p < nlabels; p++ ) {
    if( size[ p ] != 0 ) 
      mean[ p ] /= size[ p ];
    else mean[ p ] = 0;
  }

  /* sorting mean vector  */
  position = SortClusters( mean, size, nlabels );

  /* Labeling clusters in dark and bright classes */
  for( p = 0; size[ p ] == 0; p++ ); /* find first non-zero size cluster. */
  dark_size  = size[ p ];
  dark_limit = p;
  dmin = fabs( T - ( dark_size / nvoxels ) );
  for( p++; ( p < nlabels - 1 ) && ( ( dist = fabs( T - (( dark_size + size[ p ] ) / nvoxels ) ) ) < dmin ); p++ ) {
    dmin = dist;
    dark_size += size[ p ];
    dark_limit = p;
  }

  /* Verify if middle cluster is mixed. */
  dmin = T - ( dark_size / nvoxels );
  if( dmin <= MIN( -0.02, MAX( -T * 0.05, -( 1 - T ) * 0.05 ) ) ) {
    if( verbose )
      printf( "Middle cluster %d is mixed! ", dark_limit );
    dark_size -= size[ dark_limit ];
    dark_limit--;
    light_limit = dark_limit + 2;
    dmin = T - ( dark_size / nvoxels );
  }
  else if( dmin >= MAX( 0.02, MIN( T * 0.05, ( 1 - T ) * 0.05 ) ) ) {
    if( verbose )
      printf( "Middle cluster %d is mixed! ", dark_limit + 1 );
    light_limit = dark_limit + 2;
  }
  else {
    if( verbose )
      printf( "Middle cluster belongs to a single class! " );
    light_limit = dark_limit + 1;
  }

  /* Computing middle cluster thresholding. */
  if( dark_limit == light_limit - 2 ) {
    sort = ( int* ) calloc( size[ dark_limit + 1 ], sizeof( int ) );
    index = ( int* ) calloc( size[ dark_limit + 1 ], sizeof( int ) );
    for( p = 0, i = 0; ( p < brain->n ) && ( i < size[ dark_limit + 1 ] ); p++ ) {
      if( ( mask->data[ p ] != 0 ) && ( position[ dark->data[ p ] ] == dark_limit + 1 ) ) {
	sort[ i ] = brain->data[ p ];
	i++;
      }
    }
    IndexQuickSort( sort, index, 0, size[ dark_limit + 1 ] - 1, INCREASING );
    middle = sort[ ( int ) ( dmin * nvoxels ) ];
    if( verbose )
      printf( "middle threshold:%d,", middle );
    free( sort );
    free( index );
  }

  /* Labeling samples into tissue classes. */
  for( p = 0; p < brain->n; p++ ) {
    if( mask->data[ p ] != 0 ) {
      if( ( ( protocol ==  T1_PROTOCOL ) && ( position[ dark->data[ p ] ] <= dark_limit ) )
	  || ( ( protocol !=  T1_PROTOCOL ) && ( position[ dark->data[ p ] ] >= light_limit ) ) )
	dark->data[ p ] = 1; /* GM, or CSF */
      else if( ( ( protocol ==  T1_PROTOCOL ) && ( position[ dark->data[ p ] ] >= light_limit ) )
	       || ( ( protocol !=  T1_PROTOCOL ) && ( position[ dark->data[ p ] ] <= dark_limit ) ) ) {
	light->data[ p ] = 1; /* WM or GM */
	dark->data[ p ] = 0;
      }
      else if( ( ( protocol ==  T1_PROTOCOL ) && ( brain->data[ p ] <= middle ) ) 
	       || ( ( protocol !=  T1_PROTOCOL ) && ( brain->data[ p ] >= middle ) ) ) {
	dark->data[ p ] = 1; /* GM, or CSF */
	dark_size++;
      }
      else {
	light->data[ p ] = 1; /* WM, or GM */
	dark->data[ p ] = 0;
      }
    }
  }

  /* Printing for correctness verification. */
  if( verbose ) {
    for( p = 0; p < nlabels; p++ ) {
      printf( " mean[%d]=%llu", p, mean[p]);
    }
    printf( "\n" );
    for( p = 0; p < nlabels; p++ ) {
      printf( " size[%d]=%u", p, size[p]);
    }
    printf( "\n" );
    printf( "Last dark cluster:%d, first light cluster:%d\n", dark_limit, light_limit );
    printf( "\nFinal image with proportion: dark:%5.4f, light:%5.4f\n", 
	    ( float ) dark_size / ( float ) ( nvoxels ), ( float ) ( nvoxels - dark_size ) / ( float ) ( nvoxels ) );
  }

  free( position );
  free( mean );
  free( size );
}

int BrainClusterSegmentation( Scene *scn, Scene *mask, Scene **obj1, Scene **obj2, int samples, float mean_prop, 
			      int protocol, int tissue, int auto_prop, float delta_prop, int kmin, int kmax, int nthreads ) {
  Scene *brain=NULL;
  BCSubgraph *sg = NULL;
  BCFeatures *feats = NULL;
  int first_voxel;
  int last_voxel;

  //verbose = true;
  /* Brain restricted to the mask */
  brain = GetBrain( scn, mask );

  /* Computing mask limits for multiple thread execution. */
  MaskLimits( mask, &first_voxel, &last_voxel );
  
  /* CSF, GM, and WM proportions computation */
  TissueProportions( brain, mask, protocol, tissue, auto_prop, delta_prop, &mean_prop );

  /* Computing feature vectors. */
  if( tissue == WMGM_SEG ) {
    /* feats = CoOccurFeatures( scn, mask, first_voxel, last_voxel, 3, nthreads ); */
    feats = MedianSceneFeatures( brain, mask, first_voxel, last_voxel, 1.8, nthreads );
    /* feats = AdaptiveSceneFeatures( brain, mask, first_voxel, last_voxel, protocol, tissue, 1.1, nthreads ); */
  }
  else {
    feats = CoOccurFeatures( brain, mask, first_voxel, last_voxel, 3, nthreads );
    /* feats = AdaptiveSceneFeatures( brain, mask, first_voxel, last_voxel, protocol, tissue, 2.1, nthreads ); */
  }

  /* WriteFeats( feats, mask, "features.txt" ); */
    
  /* Clustering */
  sg = BCRandomSampl3( brain, mask, samples );
  SetBCFeatures( sg, feats );

  BCBestkClustering( sg, kmin, kmax );

  *obj1 = SceneClassifyKnnGraph( sg, mask, first_voxel, last_voxel, feats, nthreads );
  CopySceneHeader( mask, *obj1 );
  *obj2 = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  CopySceneHeader( mask, *obj2 );
  BrightnessLabeling( brain, mask, *obj1, *obj2, sg->nlabels, protocol, mean_prop );

  /* Finishing processing. */  
  if( verbose )
    printf( "best k: %d, clusters: %d\n", sg->bestk, sg->nlabels );
  FreeSubgraph( &sg );
  DestroyBCFeatures( &feats );
  DestroyScene( &brain );

  return( 0 );
}

#ifdef _BRAINCLUSTER_STANDALONE_
int main( int argc, char **argv ) {
  float *inputs=NULL;
  char filename[ 200 ];
  char fileextension[ 200 ];
  Scene *brain = NULL;
  Scene *mask = NULL;
  Scene *obj1 = NULL;
  Scene *obj2 = NULL;
  int samples;
  float mean_prop;
  int protocol;
  int tissue;
  int auto_prop;
  float delta_prop;
  int kmin;
  int kmax;
  int nthreads;
  
  /* Usage instructions. */
  if( argc < 3 ) {
    printf( "Usage must be: %s <scene name> <mask_name> <output_basename> [<options>]\n", argv[ 0 ] );
    printf( "<options>\n\t-samples or -s <i>: i = number of samples (300 to 2000). Default: 1000.\n" );
    printf( "\t-mean_prop or -mp <f>: f = mean float value of CSF or GM partitions size, depending on tissue_type (0.01 to 0.99). Default: Automatic.\n");
    printf( "\t-delta_prop or -dp <f>: f = delta to be added to the mean value of CSF or GM partitions size (-0.2 to 0.2). Default: 0.0.\n");
    printf( "\t-k <f> <f>: f = lower and upper limits to k-nn (0 to 30). Defaults: 10 and 20.\n" );
    printf( "\t-protocol or -p <i>: i = 0 - T1 image, 1 - T2 image, 2 - PD image. Default: 0.\n" );
    printf( "\t-tissue_type or -tt <i>: i = 1 - Segment CSF/WMGM, 2 - Segment GM/WM. Default 2.\n" );
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
  sprintf( filename, "%s", argv[ 2 ] );
  if( !FileExists( filename ) ) {
    printf( "Input mask not found!\n" );
    return 0;
  } 
  mask = ReadVolume( filename );

  /* Reading input flags */
  inputs = ParseBrainclusterInput( argc, argv );
  samples = ( int ) inputs[ 0 ];
  mean_prop = inputs[ 1 ];
  protocol = ( int ) inputs[ 2 ];
  tissue = ( int ) inputs[ 3 ];
  auto_prop = ( int ) inputs[ 4 ];
  kmin = ( int ) inputs[ 5 ];
  kmax = ( int ) inputs[ 6 ];
  nthreads = ( int ) inputs[ 7 ];
  delta_prop = inputs[ 8 ];
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
  if( ( tissue < CSF_SEG ) || ( tissue > WMGM_SEG ) ) {
    printf( "Invalid tissue type! Expected 1 - Segment CSF/WMGM, 2 - Segment GM/WM.\n" );
    return 0;
  }
  if( ( kmin > 30 ) || ( kmin < 0 ) || ( kmax > 30 ) || ( kmax < 0 ) ) {
    printf( "Invalid k values! Expected values are from 0 to 30.\n" );
    return 0;
  }
  if( kmin > kmax ) {
    printf( "Invalid k values! kmin must be smaller than kmax!\n" );
    return 0;
  }

  /* Running the algorithm. */
  BrainClusterSegmentation( brain, mask, &obj1, &obj2, samples, mean_prop, protocol, tissue, auto_prop, delta_prop, kmin, kmax, nthreads );

  /* Saving results. */
  GetFileExtension( filename, fileextension );
  if( tissue == WMGM_SEG ) {
    sprintf( filename, "%s-gm%s", argv[ 3 ], fileextension );
    WriteVolume( obj1, filename );
    sprintf( filename, "%s-wm%s", argv[ 3 ], fileextension );
    WriteVolume( obj2, filename );
  }
  else {
    sprintf( filename, "%s-csf%s", argv[ 3 ], fileextension );
    WriteVolume( obj1, filename );
    sprintf( filename, "%s-wmgm%s", argv[ 3 ], fileextension );
    WriteVolume( obj2, filename );
  }
  
  /* Cleaning memory. */
  DestroyScene( &obj1 );
  DestroyScene( &obj2 );
  DestroyScene( &brain );
  DestroyScene( &mask );
  
  return( 0 );
}
#endif
