// Calcular proporções para quadrantes e octantes

#include "oldift.h"
#include "compressed.h"
//#include "braincluster_spectral.h"

int main(int argc, char **argv) {
  Scene *scn = NULL;
  Scene *result = NULL;
  int x, y, z, p, q;
  char axis;

  if( argc != 4 ) {
    printf( "Usage must be: %s <input> <output> <axis>\n<axis>: x, y, or z\n", argv[ 0 ] );
    return 0;
  }

  scn = ReadVolume( argv[ 1 ] );
  result = CreateScene( scn->xsize, scn->ysize, scn->zsize );
  result->dx = scn->dx;
  result->dy = scn->dy;
  result->dz = scn->dz;

  axis = argv[ 3 ][ 0 ];

  for( x = 0; x < scn->xsize; x++ ) {
    for( y = 0; y < scn->ysize; y++ ) {
      for( z = 0; z < scn->zsize; z++ ) {
	p = VoxelAddress( scn, x, y, z );
	if( axis == 'x' )
	  q = VoxelAddress( scn, scn->xsize - x - 1, y, z );
	else if( axis == 'y' )
	  q = VoxelAddress( scn, x, scn->ysize - y - 1, z );
	else
	  q = VoxelAddress( scn, x, y, scn->zsize - z - 1 );
	result->data[ q ] = scn->data[ p ];
      }
    }
  }

  WriteVolume( result, argv[ 2 ] );
  DestroyScene( &scn );
  DestroyScene( &result );

  return( 0 );
}
