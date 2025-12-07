#include "oldift.h"

int main( int argc, char **argv ) {
  if( argc < 5 ) {
    printf( "Usage: %s <input scene> <output image> <slice> <orientation>\n", argv[ 0 ] );
    printf( "  <orientation>:x, y, or z\n" );
    return 0;
  }
  Scene *scn = NULL;
  Image *slice = NULL;
  char ori = argv[ 4 ][ 0 ];
  int num = atoi( argv[ 3 ] );

  scn = ReadVolume( argv[ 1 ] );

  if( ori == 'x' )
    slice = GetXSlice( scn, num );
  else if( ori == 'y' )
    slice = GetYSlice( scn, num );
  else
    slice = GetSlice( scn, num );

  WriteImage( slice, argv[ 2 ] );
  DestroyImage( &slice );
  DestroyScene( &scn );

  return 0;
}
