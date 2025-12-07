#include "oldift.h"
#include "scene.h"
#include "compressed.h"
#include "scene_addons.h"
#include "mri.h"
  
int main( int argc, char **argv ) {
  
  Scene *orig = NULL;
  Scene *res = NULL;
  char x, y, z;
  char forbidden[ 2 ];
  char s[ 4 ];
  
  if ( argc < 5 ) {
    printf( "Usage must be: %s <input filename> <output filename> <z coordinate> <x coordinate> <y coordinate>\n", argv[ 0 ] );
    printf( "For the coordenates, the valid letters are:\n" );
    printf( "  L, R, P, A, I, and S. You must pick one between L and R, P and A,\n" );
    printf( "  and I and S to be set as the three input Cartesian coordinates.\n" );
    printf( "  The first coordinate determines the direction of increasing slice.\n" );
    printf( "  The second coordinate determines the direction of increasing column.\n" );
    printf( "  The third coordinate determines the direction of increasing row.\n" );
    exit( 0 );
  }
  
  orig = ReadVolume( argv[ 1 ] );
  z = argv[ 3 ][ 0 ];
  x = argv[ 4 ][ 0 ];
  y = argv[ 5 ][ 0 ];
  
  switch( x ) {
  case 'L':
    forbidden[ 0 ] = 'L';
    forbidden[ 1 ] = 'R';
    s[ 0 ] = 'L';
    break;
  case 'R':
    forbidden[ 0 ] = 'L';
    forbidden[ 1 ] = 'R';
    s[ 0 ] = 'R';
    break;
  case 'S':
    forbidden[ 0 ] = 'S';
    forbidden[ 1 ] = 'I';
    s[ 0 ] = 'S';
    break;
  case 'I':
    forbidden[ 0 ] = 'S';
    forbidden[ 1 ] = 'I';
    s[ 0 ] = 'I';
    break;
  case 'A':
    forbidden[ 0 ] = 'A';
    forbidden[ 1 ] = 'P';
    s[ 0 ] = 'A';
    break;
  case 'P':
    forbidden[ 0 ] = 'A';
    forbidden[ 1 ] = 'P';
    s[ 0 ] = 'P';
    break;
  default:
    printf( "Invalid option for x cordinate: %c.\n", x );
    exit( 0 );
    break;
  }
  
  if( ( y == forbidden[ 0 ] ) || ( y == forbidden[ 1 ] ) ||
      ( z == forbidden[ 0 ] ) || ( z == forbidden[ 1 ] ) ) {
    printf( "Invalid combination: %c%c%c.\n", x, y, z );
    exit( 0 );
  }
  
  switch( y ) {
  case 'L':
    forbidden[ 0 ] = 'L';
    forbidden[ 1 ] = 'R';
    s[ 1 ] = 'L';
    break;
  case 'R':
    forbidden[ 0 ] = 'L';
    forbidden[ 1 ] = 'R';
    s[ 1 ] = 'R';
    break;
  case 'S':
    forbidden[ 0 ] = 'S';
    forbidden[ 1 ] = 'I';
    s[ 1 ] = 'S';
    break;
  case 'I':
    forbidden[ 0 ] = 'S';
    forbidden[ 1 ] = 'I';
    s[ 1 ] = 'I';
    break;
  case 'A':
    forbidden[ 0 ] = 'A';
    forbidden[ 1 ] = 'P';
    s[ 1 ] = 'A';
    break;
  case 'P':
    forbidden[ 0 ] = 'A';
    forbidden[ 1 ] = 'P';
    s[ 1 ] = 'P';
    break;
  default:
    printf( "Invalid option for y cordinate: %c.\n", y );
    exit( 1 );
    break;
  }
  
  if( ( z == forbidden[ 0 ] ) || ( z == forbidden[ 1 ] ) ) {
    printf( "Invalid combination: %c%c%c.\n", x, y, z );
    exit( 0 );
  }
  
  switch( z ) {
  case 'L':
    forbidden[ 0 ] = 'L';
    forbidden[ 1 ] = 'R';
    s[ 2 ] = 'L';
    break;
  case 'R':
    forbidden[ 0 ] = 'L';
    forbidden[ 1 ] = 'R';
    s[ 2 ] = 'R';
    break;
  case 'S':
    forbidden[ 0 ] = 'S';
    forbidden[ 1 ] = 'I';
    s[ 2 ] = 'S';
    break;
  case 'I':
    forbidden[ 0 ] = 'S';
    forbidden[ 1 ] = 'I';
    s[ 2 ] = 'I';
    break;
  case 'A':
    forbidden[ 0 ] = 'A';
    forbidden[ 1 ] = 'P';
    s[ 2 ] = 'A';
    break;
  case 'P':
    forbidden[ 0 ] = 'A';
    forbidden[ 1 ] = 'P';
    s[ 2 ] = 'P';
    break;
  default:
    printf( "Invalid option for z cordinate: %c.\n", z );
    exit( 1 );
    break;
  }
  
  s[ 3 ] = '\0';    
  
  res = ChangeOrientationToLPS( orig, s );
  WriteVolume( res, argv[ 2 ] );
  
  DestroyScene( &orig );
  DestroyScene( &res );
  
  return( 0 );
}
