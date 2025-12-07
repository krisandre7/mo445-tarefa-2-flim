
#include "interactionhandler.h"


InteractionHandler :: InteractionHandler(){
}


InteractionHandler :: ~InteractionHandler(){
}


void InteractionHandler :: OnLeftClick(int p){
}

void InteractionHandler :: OnRightClick(int p){
}

void InteractionHandler :: OnMiddleClick(int p){
}
  
void InteractionHandler :: OnLeftDrag(int p, int q){
}

void InteractionHandler :: OnRightDrag(int p, int q){
}

void InteractionHandler :: OnLeftRelease(int p){
}

void InteractionHandler :: OnRightRelease(int p){
}

void InteractionHandler :: OnMiddleRelease(int p){
}


void InteractionHandler :: OnMouseMotion(int p){
}

void InteractionHandler :: OnMouseWheel(int p,
					int rotation,
					int delta){
}


void InteractionHandler :: OnEnterWindow(){
}


void InteractionHandler :: OnLeaveWindow(){
}


//This function is for experts only.
void InteractionHandler :: OnMouseEvent(wxMouseEvent& event,
					int p){
  static int drag_p=0;

  if(p<0) return;

  switch (event.GetButton()) {
  case wxMOUSE_BTN_LEFT : 
    if(event.GetEventType() == wxEVT_LEFT_DOWN){
      drag_p = p;
      OnLeftClick(p);
    }
    else if (event.GetEventType() == wxEVT_LEFT_UP){
      OnLeftRelease(p);
    }
    break;
  case wxMOUSE_BTN_MIDDLE :
    if (event.GetEventType() == wxEVT_MIDDLE_DOWN){
      OnMiddleClick(p);
    }
    else if (event.GetEventType() == wxEVT_MIDDLE_UP){
      OnMiddleRelease(p);
    }
    break;
  case wxMOUSE_BTN_RIGHT : 
    if (event.GetEventType() == wxEVT_RIGHT_DOWN){
      drag_p = p;
      OnRightClick(p);
    }
    else if (event.GetEventType() == wxEVT_RIGHT_UP){
      OnRightRelease(p);
    }
    break;
  case wxMOUSE_BTN_NONE :
    if (event.GetEventType() == wxEVT_ENTER_WINDOW) {
      OnEnterWindow();
    }
    else if (event.GetEventType() == wxEVT_LEAVE_WINDOW){
      OnLeaveWindow();
    }
    else if (event.GetEventType() == wxEVT_MOTION) {
      if (event.Dragging()) {
	if(event.LeftIsDown()) {
	  OnLeftDrag(drag_p, p);
	  drag_p = p;
	}
	else if(event.RightIsDown()) {
	  OnRightDrag(drag_p, p);
	  drag_p = p;
	}
	else if(event.MiddleIsDown()) {}
      }
      OnMouseMotion(p);
    }
    else if (event.GetEventType() == wxEVT_MOUSEWHEEL) {
      OnMouseWheel(p, 
		   event.GetWheelRotation(), 
		   event.GetWheelDelta());
    }
    break;
  }
}


