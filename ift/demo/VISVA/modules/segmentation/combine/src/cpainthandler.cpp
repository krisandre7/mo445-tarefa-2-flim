#include "cpainthandler.h"

namespace Combine{

  CpaintHandler :: CpaintHandler(char axis,
			       ModuleCombine *mod){
    this->axis = axis;
    this->mod = mod;
  }

  CpaintHandler :: ~CpaintHandler(){}

  void CpaintHandler :: OnEnterWindow(){
    wxCursor *cursor;
    float zoom = APP->GetZoomLevel();

    cursor = mod->GetBrushCursor(ROUND(zoom));
    APP->Set2DViewCursor(cursor, axis);
  }

  void CpaintHandler :: OnLeaveWindow(){
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, axis);
  }

  void CpaintHandler :: OnLeftClick(int p){
    AdjRel *A;
    int o, b;
    cbutton bt = LEFT;

    o = mod->obj_sel;
    if( o < 0 )
      return;
    b = mod->bkg_sel;
    //if( b < 0 )
    //b = 0;

    A = mod->GetBrush();
    CombineBrushCustom( p, o, b, mode, A, axis, bt );
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void CpaintHandler :: OnRightClick(int p){
    AdjRel *A;
    int o, b;
    cbutton bt = RIGHT;
    
    o = mod->obj_sel;
    if(o<0) return;
    b = mod->bkg_sel;
    //if( b < 0 ) 
    //b = 0;

    A = mod->GetBrush();
    CombineBrushCustom( p, o, b, mode, A, axis, bt );

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void CpaintHandler :: OnMiddleClick(int p){
  }

  void CpaintHandler :: OnMouseWheel(int p, int rotation, int delta){
    if(rotation>0) mod->NextBrush();
    else           mod->PrevBrush();

    wxCursor *cursor;
    float zoom = APP->GetZoomLevel();
    
    cursor = mod->GetBrushCursor(ROUND(zoom));
    APP->Set2DViewCursor(cursor, axis);
  }

  void CpaintHandler :: OnLeftDrag(int p, int q){
    AdjRel *A;
    int o, b;
    cbutton bt = LEFT;

    o = mod->obj_sel;
    if(o<0)
      return;
    b = mod->bkg_sel;
    //if(b<0)
    //b = 0;

    A = mod->GetBrush();
    CombineBrushTraceCustom( p, q, o, b, mode, A, axis, bt );
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void CpaintHandler :: OnRightDrag(int p, int q){
    AdjRel *A;
    int o, b;
    cbutton bt = RIGHT;

    o = mod->obj_sel;
    if(o<0) 
      return;
    b = mod->bkg_sel;
    //if(b<0)
    //b = 0;

    A = mod->GetBrush();
    CombineBrushTraceCustom( p, q, o, b, mode, A, axis, bt );

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void CpaintHandler :: OnMouseMotion(int p){
    wxCursor *cursor=NULL;
    static float zoom = -1.0;
    
    if(zoom!=APP->GetZoomLevel()){
      zoom = APP->GetZoomLevel();
      cursor = mod->GetBrushCursor(ROUND(zoom));
      APP->Set2DViewCursor(cursor, axis);
    }

    APP->Window->SetStatusText(_T("Mouse Left: Paint Object, Mouse Right: Paint Background"));
  }
  
  // Set pixels from brush with multiple functions.
  void CpaintHandler::CombineBrushCustom(int p, int label1, int label2, int mode, AdjRel *A, char axis, cbutton bt) {
    Voxel u,v;
    int q,i;
    Scene *scn = APP->Data.label;
    
    if(scn->n != (APP->Data.orig)->n) return;
    v.x = VoxelX(scn, p);
    v.y = VoxelY(scn, p);
    v.z = VoxelZ(scn, p);
    for(i=0;i<A->n;i++){
      if(axis == 'z'){
	u.x = v.x + A->dx[i];
	u.y = v.y + A->dy[i];
	u.z = v.z;
      }
      else if(axis == 'x'){
	u.x = v.x;
	u.y = v.y + A->dy[i];
	u.z = v.z + A->dx[i];
      }
      else if(axis == 'y'){
	u.x = v.x + A->dx[i];
	u.y = v.y;
	u.z = v.z + A->dy[i];
      }
      
      if( ValidVoxel( scn, u.x, u.y, u.z ) ) {
	q = VoxelAddress(scn,u.x,u.y,u.z);
	if( ( label1 != -1 ) && ( label2 != -1 ) ) {
	  switch( mode ) {
	  case 0: // Paint
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 1 );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 0 );
	    }
	    break;
	  case 1: // And
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, _fast_BMapGet( mod->myobj[ label1 ].mask, q ) 
			     & _fast_BMapGet( mod->myobj[ label2 ].mask, q ) );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, ! ( _fast_BMapGet( mod->myobj[ label1 ].mask, q ) 
			     & _fast_BMapGet( mod->myobj[ label2 ].mask, q ) ) );
	    }
	    break;
	  case 2: // Or
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, _fast_BMapGet( mod->myobj[ label1 ].mask, q ) 
			     | _fast_BMapGet( mod->myobj[ label2 ].mask, q ) );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, ! (_fast_BMapGet( mod->myobj[ label1 ].mask, q ) 
			     | _fast_BMapGet( mod->myobj[ label2 ].mask, q ) ) );
	    }
	    break;
	  case 3: // Neg
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, !_fast_BMapGet( mod->myobj[ label2 ].mask, q ) );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, _fast_BMapGet( mod->myobj[ label2 ].mask, q ) );
	    }
	    break;
	  case 4: // Xor
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, _fast_BMapGet( mod->myobj[ label1 ].mask, q ) 
			     ^ _fast_BMapGet( mod->myobj[ label2 ].mask, q ) );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, ! (_fast_BMapGet( mod->myobj[ label1 ].mask, q ) 
			     ^ _fast_BMapGet( mod->myobj[ label2 ].mask, q ) ) );
	    }
	    break;
	  default:
	    break;
	  }
	}
	else if( label1 != -1 ) {
	  switch( mode ) {
	  case 0: // Paint
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 1 );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 0 );
	    }
	    break;
	  case 1: // And
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 0 );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 1 );
	    }
	    break;
	  case 2: // Or
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 1 );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 0 );
	    }
	    break;
	  case 3: // Neg
	    _fast_BMapSet( mod->myobj[ label1 ].mask, q, !_fast_BMapGet( mod->myobj[ label1 ].mask, q ) );
	    break;
	  case 4: // Xor
	    if( bt == LEFT ) {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 0 );
	    }
	    else {
	      _fast_BMapSet( mod->myobj[ label1 ].mask, q, 1 );
	    }
	    break;
	  default:
	    break;
	  }
	}
	if( ( mod->obj_sel != -1 ) && ( mod->bkg_sel != -1 ) )
	  scn->data[ q ] = _fast_BMapGet( mod->myobj[ mod->obj_sel ].mask, q ) + 
	    2 * _fast_BMapGet( mod->myobj[ mod->bkg_sel ].mask, q );
	else if( mod->obj_sel != -1 )
	  scn->data[ q ] = _fast_BMapGet( mod->myobj[ mod->obj_sel ].mask, q );
	else if( mod->bkg_sel != -1 )
	  scn->data[ q ] = 2 * _fast_BMapGet( mod->myobj[ mod->bkg_sel ].mask, q );
      }
    }
  }
  
  // Set pixels in trace with multiple functions.
  void CpaintHandler::CombineBrushTraceCustom(int p, int q, int label1, int label2, int mode, AdjRel *A, char axis, cbutton bt) {
    Voxel v0,v1,v2;
    int t;
    Scene *scn = APP->Data.label;
    
    if(scn->n != (APP->Data.orig)->n) return;
    v0.x = VoxelX(scn, p);
    v0.y = VoxelY(scn, p);
    v0.z = VoxelZ(scn, p);
    v1.x = VoxelX(scn, q);
    v1.y = VoxelY(scn, q);
    v1.z = VoxelZ(scn, q);
    v2.x = (v0.x+v1.x)/2;
    v2.y = (v0.y+v1.y)/2;
    v2.z = (v0.z+v1.z)/2;
    t = VoxelAddress(scn,v2.x,v2.y,v2.z);
    
    if ((v2.x==v1.x && v2.y==v1.y && v2.z==v1.z) || 
	(v2.x==v0.x && v2.y==v0.y && v2.z==v0.z)) {
      CombineBrushCustom( p, label1, label2, mode, A, axis, bt );
      CombineBrushCustom( q, label1, label2, mode, A, axis, bt );
      return;
    }
    CombineBrushTraceCustom(p, t, label1, label2, mode, A, axis, bt );
    CombineBrushTraceCustom(t, q, label1, label2, mode, A, axis, bt );
  }

} //end Combine namespace

