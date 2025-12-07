#include "oldift.h"
#include "compressed.h"
//#include "braincluster.h"

// Insere ou remove fatias nas três orientações da imagem, nos dois sentidos.
int main(int argc, char **argv) {
  Scene *scn = NULL;
  Scene *result = NULL;
  int xmin, xmax, ymin, ymax, zmin, zmax;
  int x, y, z;
  int p, q;

  if( argc < 3 ) {
    printf( "Usage must be: %s <input> <output> [-xmin <diff>] [-xmax <diff>] [-ymin <diff>] [-ymax <diff>] [-zmin <diff>] [-zmax <diff>]\nExample: %s in.scn.bz2 out.scn.bz2 -xmin +10 -zmax -20\n\tThis command will create an image with the same dimensions\n\tand content of the input image, adding 10\n\tslices to the begining of x orientation and removing 20\n\tslices from the end of z orientation.\n", argv[ 0 ], argv[ 0 ] );
    return 0;
  }

  scn = ReadVolume( argv[ 1 ] );
  xmin = 0;
  ymin = 0;
  zmin = 0;
  xmax = scn->xsize;
  ymax = scn->ysize;
  zmax = scn->zsize;
  p = 3;
  while( p < argc ) {
    if( strcmp( argv[ p++ ], "-xmin" ) == 0 )
      xmin = xmin - atoi( argv[ p++ ] );
    if( strcmp( argv[ p++ ], "-xmax" ) == 0 )
      xmax += atoi( argv[ p++ ] );
    if( strcmp( argv[ p++ ], "-ymin" ) == 0 )
      ymin -= atoi( argv[ p++ ] );
    if( strcmp( argv[ p++ ], "-ymax" ) == 0 )
      ymax += atoi( argv[ p++ ] );
    if( strcmp( argv[ p++ ], "-zmin" ) == 0 )
      zmin -= atoi( argv[ p++ ] );
    if( strcmp( argv[ p++ ], "-zmax" ) == 0 )
      zmax += atoi( argv[ p++ ] );
  }
  printf( "xmin:%d, xmax:%d, ymin:%d, ymax:%d, zmin:%d, zmax:%d\n", xmin, xmax, ymin, ymax, zmin, zmax );
    
  result = CreateScene( xmax - xmin, ymax - ymin, zmax - zmin );
  result->dx = scn->dx;
  result->dy = scn->dy;
  result->dz = scn->dz;

  for( x = MAX( 0, xmin ); x < MIN( scn->xsize, xmax ); x++ ) {
    for( y = MAX( 0, ymin ); y < MIN( scn->ysize, ymax ); y++ ) {
      for( z = MAX( 0, zmin ); z < MIN( scn->zsize, zmax ); z++ ) {
	p = VoxelAddress( scn, x, y, z );
	q = VoxelAddress( result, x - xmin, y - ymin, z - zmin );
	result->data[ q ] = scn->data[ p ];
      }
    }
  }

  WriteVolume( result, argv[ 2 ] );
  DestroyScene( &scn );
  DestroyScene( &result );
  return( 0 );
}
