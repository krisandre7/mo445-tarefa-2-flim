#include "oldift.h"
#include "compressed.h"
#include "inhomogeneity.h"
#include "nifti1_io.h"

Scene *HighBrightness( Scene *scn, float f ) {
  int i, size, max;
  Curve *c = NULL;
  Scene *result = NULL;

  // Initializing
  result = CreateScene( scn->xsize, scn->ysize, scn->zsize );
  c = Histogram3( scn );

  // Defining number of voxels of the fraction f. 
  size = 0;
  for( i = 0; i < scn->n; i++ ) {
    if( scn->data[ i ] != 0 ) size++;
  }
  f *= size;
  
  // Defining bin that returns proportion.
  max = MaximumValue3( scn );
  for( i = c->Y[ max ]; i < f; max-- ) i += c->Y[ max ];

  // Writting result scene.
  for( i = 0; i < scn->n; i++ ) {
    if( scn->data[ i ] > max ) result->data[ i ] = scn->data[ i ];
  }
  
  // Finishing
  DestroyCurve( &c );
  return( result );
}

Scene *EDT3( Scene *bin_border, Scene *mask ) {
  GQueue *Q=NULL;
  Scene *cost = NULL;
  Scene *root = NULL;
  AdjRel3 *A = NULL;
  int i, p, q, *sq, sqsize;
  float tmp;
  Voxel r, u, v;

  root = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  cost = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  A = Spheric( 1.8 );  
  sqsize = MAX( MAX( mask->xsize, mask->ysize ), mask->zsize );
  sq     = AllocIntArray( sqsize );
  for( i = 0; i < sqsize; i++ ) sq[ i ] = i * i;
  Q = CreateGQueue( sq[ sqsize - 1 ], mask->n, cost->data );
  SetRemovalPolicy( Q, MINVALUE );
  
  for( p = 0; p < mask->n; p++ ) { // Seed pixels
    if ( mask->data[ p ] == 0 ) cost->data[ p ] = 0;
    else cost->data[ p ] = INT_MAX;
    if( bin_border->data[ p ] == 1 ) {
      root->data[ p ] = p;
      cost->data[ p ] = 0;
      InsertGQueue( &Q, p );
    }
  }

  while( !EmptyGQueue( Q ) ) {
    p = RemoveGQueue( Q );
    u.x = VoxelX( cost, p );//(p % xysize) % cost->xsize;
    u.y = VoxelY( cost, p );//(p % xysize) / cost->xsize;
    u.z = VoxelZ( cost, p );//p / xysize;
    r.x = VoxelX( cost, root->data[p] );//(root->data[p] % xysize) % cost->xsize;
    r.y = VoxelY( cost, root->data[p] );//(root->data[p] % xysize) / cost->xsize;
    r.z = VoxelZ( cost, root->data[p] );//root->data[p] / xysize;
    for( i = 1; i < A->n; i++ ){
      v.x = u.x + A->dx[ i ];
      v.y = u.y + A->dy[ i ];
      v.z = u.z + A->dz[ i ];
      if( ValidVoxel( cost, v.x, v.y, v.z ) ) {
	q = VoxelAddress( cost, v.x, v.y, v.z ); //v.x + cost->tby[v.y] + cost->tbz[v.z];
	if( mask->data[ q ] ) {
	  if( cost->data[ p ] < cost->data[ q ] ) {
	    tmp = sqrt( (float) sq[ abs( v.x - r.x ) ] + sq[ abs( v.y - r.y ) ] + sq[ abs( v.z - r.z ) ] );
	    if( ROUND( tmp ) < cost->data[ q ] ) {
	      if( Q->L.elem[ q ].color == GRAY ) RemoveGQueueElem( Q, q );
	      cost->data[ q ] = ROUND( tmp );
	      root->data[ q ] = root->data[ p ];
	      InsertGQueue( &Q, q );	      
	    }
	  }
	}
      }
    }
  }

  DestroyAdjRel3( &A );
  DestroyGQueue( &Q );
  DestroyScene( &root );
  free( sq );
  return( cost );
}

Scene *EDT_mm3( Scene *bin_border, Scene *mask ) {
  GQueue *Q=NULL;
  Scene *cost = NULL;
  Scene *root = NULL;
  AdjRel3 *A = NULL;
  int i, p, q, df_size;
  float tmp;
  Voxel r, u, v;

  root = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  cost = CreateScene( mask->xsize, mask->ysize, mask->zsize );
  A = Spheric( 1.8 );  
  df_size = MIN( MIN( mask->dx, mask->dy ), mask->dz );
  Q = CreateGQueue( 3000000, mask->n, cost->data );
  SetRemovalPolicy( Q, MINVALUE );
  
  for( p = 0; p < mask->n; p++ ) { // Seed pixels
    if ( mask->data[ p ] == 0 ) cost->data[ p ] = 0;
    else cost->data[ p ] = INT_MAX;
    if( bin_border->data[ p ] == 1 ) {
      root->data[ p ] = p;
      cost->data[ p ] = 0;
      InsertGQueue( &Q, p );
    }
  }

  while( !EmptyGQueue( Q ) ) {
    p = RemoveGQueue( Q );
    u.x = VoxelX( cost, p );//(p % xysize) % cost->xsize;
    u.y = VoxelY( cost, p );//(p % xysize) / cost->xsize;
    u.z = VoxelZ( cost, p );//p / xysize;
    r.x = VoxelX( cost, root->data[p] );//(root->data[p] % xysize) % cost->xsize;
    r.y = VoxelY( cost, root->data[p] );//(root->data[p] % xysize) / cost->xsize;
    r.z = VoxelZ( cost, root->data[p] );//root->data[p] / xysize;
    for( i = 1; i < A->n; i++ ){
      v.x = u.x + A->dx[ i ];
      v.y = u.y + A->dy[ i ];
      v.z = u.z + A->dz[ i ];
      if( ValidVoxel( cost, v.x, v.y, v.z ) ) {
	q = VoxelAddress( cost, v.x, v.y, v.z ); //v.x + cost->tby[v.y] + cost->tbz[v.z];
	if( mask->data[ q ] ) {
	  if( cost->data[ p ] < cost->data[ q ] ) {
	    tmp = sqrt( ( float ) 1000000 * ( pow( ( v.x - r.x ) * mask->dx, 2 ) + pow( ( v.y - r.y ) * mask->dy, 2 ) +
					      pow( ( v.z - r.z ) * mask->dz, 2 ) ) );
	    if( ROUND( tmp ) < cost->data[ q ] ) {
	      if( Q->L.elem[ q ].color == GRAY ) RemoveGQueueElem( Q, q );
	      cost->data[ q ] = ROUND( tmp );
	      root->data[ q ] = root->data[ p ];
	      InsertGQueue( &Q, q );	      
	    }
	  }
	}
      }
    }
  }
  
  for( p = 0; p < mask->n; p++ )
    cost->data[ p ] = ROUND( cost->data[ p ] / 1000 );

  DestroyAdjRel3( &A );
  DestroyGQueue( &Q );
  DestroyScene( &root );
  return( cost );
}

Scene *RemoveLabel( Scene *scn, int l ) {
  int i;
  Scene *result = CreateScene( scn->xsize, scn->ysize, scn->zsize );
  for ( i = 0; i < scn->n; i++ ) {
    if ( scn->data[ i ] != l ) result->data[ i ] = scn->data[ i ];
  }
  return( result );
}

Scene *GetALabel( Scene *scn, int l ) {
  int i;
  Scene *result = CreateScene( scn->xsize, scn->ysize, scn->zsize );
  for ( i = 0; i < scn->n; i++ ) {
    if ( scn->data[ i ] == l ) result->data[ i ] = scn->data[ i ];
  }
  return( result );
}

Scene *SelectLargestComp( Scene *scn1 ) {
  AdjRel3 *A = Spheric( 1.8 );
  Scene  *label = LabelBinComp3( scn1, A );
  int Lmax = MaximumValue3( label );
  int *area = ( int* ) AllocIntArray( Lmax + 1 );
  int imax, i;
  int p, n = scn1->n;

  for( p = 0; p < n; p++ ) {
    if( label->data[ p ] > 0 )
      area[ label->data[ p ] ]++;
  }
  imax = 0;
  for( i = 1; i <= Lmax; i++ ) {
    if( area[ i ] > area[ imax ] )
      imax = i;
  }
  for( p = 0; p < n; p++ ) {
    if( label->data[ p ] != imax )
      label->data[ p ] = 0;
    else label->data[ p ] = 1;
  }
  DestroyAdjRel3( &A );
  free( area );

  return( label );
}

Scene *CreateBin( Scene *scn ) {
  int i;
  Scene *result = CreateScene( scn->xsize, scn->ysize, scn->zsize );
  for ( i = 0; i < scn->n; i++ )
    if ( scn->data[ i ] != 0 ) result->data[ i ] = 1;
  if( scn->nii_hdr  != NULL ) {
    scn->nii_hdr->datatype = NIFTI_TYPE_UINT8;
    scn->nii_hdr->nbyper = 1;
    scn->nii_hdr->swapsize = 0;
    //nifti_datatype_sizes( DT_BINARY, &( scn->nii_hdr->nbyper ), &( scn->nii_hdr->swapsize ) );
  }
  return( result );
}

Scene *Inverse( Scene *scn, Scene *mask ) {
  int i, Imax;
  Scene *result =  CreateScene( scn->xsize, scn->ysize, scn->zsize );
  if( mask == NULL ) Imax = MaximumValue3( scn );
  else Imax = MaximumValueMask3( scn, mask );
  for( i = 0; i < scn->n; i++ ) {
    if( ( mask == NULL ) || ( mask->data[ i ] != 0 ) ) result->data[ i ] = Imax - scn->data[ i ];
  }
  return( result );
}

int main(int argc, char **argv) 
{
  Scene *scn1 = NULL, *scn2 = NULL, *result = NULL;
  int op, i_op = 0;
  AdjRel3 *A = NULL;
  float f_op = 0.0;

  if ( argc < 3 ) {
    printf( "Usage must be: %s <operation> <scene1> [<scene2 or operator>] <result>\n<operation>:    0-mul3(scn1,scn2)\n\t\t1-sum3(scn1,scn2)\n\t\t2-diff3(scn1,scn2)\n\t\t3-add3(scn1,scalar)\n\t\t4-removelabel(scn1,label)\n\t\t5-connectedcomponents(scn1,thres)\n\t\t6-connectedcomponentsbin(scn)\n\t\t7-binary(scn1)\n\t\t8-edt(binborder,mask)\n\t\t9-getborder3(scn,adj_radius)\n\t\t10-invert_mask(scn, mask)\n\t\t11-invert(scn)\n\t\t12-hibright(scn,fraction)\n\t\t13-LargestComponent(bin)\n\t\t14-Close3(scn1,radius)\n\t\t15-CloseBin3(scn1,radius)\n\t\t16-Open3(scn1,radius)\n\t\t17-OpenBin3(scn1,radius)\n\t\t18-And3(scn1,scn2)\n\t\t19-Or3(scn1,scn2)\n\t\t20-AddFrame0(scn1,width)\n\t\t21-RemoveFrame(scn1,width)\n\t\t22-Dilate3(scn1,radius)\n\t\t23-Erode3(scn1,radius)\n\t\t24-GetALabel(scn1,label)\n\t\t25-EDT_mm(binborder,mask)\n\t\t26-MedianFilter(scn1,radius)\n\t\t27-MedianFilter_mm(scn1,radius)\n\t\t28-DilateBin3(bin,radius)\n\t\t29-ErodeBin3(bin,radius)\n", argv[ 0 ] );
    return 0;
  }

  op = atoi( argv[ 1 ] );
  scn1 = ReadVolume( argv[ 2 ] );
  if ( ( op < 3 ) || ( op == 8 ) || ( op == 10 ) || ( op == 18 ) || ( op == 19 ) || ( op == 25 ) ) 
    scn2 = ReadVolume( argv[ 3 ] );
  else if ( ( op < 6 ) || ( op == 20 ) || ( op == 21 ) || ( op == 24 ) ) 
    i_op = atoi( argv [ 3 ] );
  else if( op != 7 )
    f_op = atof( argv[ 3 ] );
  printf( "op=%d\n", op );

  switch ( op ) {
  case 0: 
    result = Mult3( scn1, scn2 );
    DestroyScene( &scn2 );
    break;
  case 1:
    result = Sum3( scn1, scn2 );
    DestroyScene( &scn2 );
    break;
  case 2:
    result = Diff3( scn1, scn2 );
    DestroyScene( &scn2 );
    break;
  case 3:
    result = Add3( scn1, i_op );
    break;
  case 4:
    result = RemoveLabel( scn1, i_op );
    break;
  case 5:
    A = Spheric( 1.8 );
    result = LabelComp3( scn1, A, i_op );
    DestroyAdjRel3( &A );
    break;
  case 6:
    A = Spheric( 1.8 );
    result = LabelBinComp3( scn1, A );
    DestroyAdjRel3( &A );
    break;
  case 7:
    result = CreateBin( scn1 );
    break;
  case 8:
    result = EDT3( scn1, scn2 );
    DestroyScene( &scn2 );
    break;
  case 9:
    A = Spheric( f_op );
    result = GetBorder3( scn1, A );
    DestroyAdjRel3( &A );
    break;
  case 10:
    result = Inverse( scn1, scn2 );
    DestroyScene( &scn2 );
    break;
  case 11:
    result = Inverse( scn1, NULL );
    break;
  case 12:
    result = HighBrightness( scn1, f_op );
    break;
  case 13:
    result = SelectLargestComp( scn1 );
    break;
  case 14:
    A = Spheric( f_op );
    result = Close3( scn1, A );
    DestroyAdjRel3( &A );
    break;
  case 15:
    result = CloseBin3( scn1, f_op );
    break;
  case 16:
    A = Spheric( f_op );
    result = Open3( scn1, A );
    DestroyAdjRel3( &A );
    break;
  case 17:
    result = OpenBin3( scn1, f_op );
    break;
  case 18:
    result = And3( scn1, scn2 );
    DestroyScene( &scn2 );
    break;
  case 19:
    result = Or3( scn1, scn2 );
    DestroyScene( &scn2 );
    break;
  case 20:
    result = AddFrame3( scn1, i_op, 0 );
    break;
  case 21:
    result = RemFrame3( scn1, i_op );
    break;
  case 22:
    A = Spheric( f_op );
    result = Dilate3( scn1, A );
    DestroyAdjRel3( &A );
    break;
  case 23:
    A = Spheric( f_op );
    result = Erode3( scn1, A );
    DestroyAdjRel3( &A );
    break;
  case 24:
    result = GetALabel( scn1, i_op );    
    break;
  case 25:
    result = EDT_mm3( scn1, scn2 );
    DestroyScene( &scn2 );
    break;
  case 26:
    A = Spheric( f_op );
    result = MedianFilter3( scn1, A );
    DestroyAdjRel3( &A );
    break;
  case 27:
    A = Ellipsoid( f_op / scn1->dx, f_op / scn1->dy, f_op / scn1->dz );
    result = MedianFilter3( scn1, A );
    DestroyAdjRel3( &A );
    break;
  case 28:
    result = DilateBinScene3( scn1, f_op );
    break;
  default:
    result = ErodeBinScene3( scn1, f_op );
    break;
  }
  CopySceneHeader( scn1, result );
  /* result->dx = scn1->dx; */
  /* result->dy = scn1->dy; */
  /* result->dz = scn1->dz; */
  if( ( op < 6 ) || ( ( op > 7 ) && ( op  < 11 ) ) || ( op == 12 ) || ( ( op >= 14 ) && ( op <= 29 ) ) )
    WriteScene( result, argv[ 4 ] );
  else
    WriteScene( result, argv[ 3 ] );
  DestroyScene( &scn1 );
  DestroyScene( &result );

  return(0);
}


