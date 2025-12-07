
#ifndef _INTERACTIONHANDLER_H_
#define _INTERACTIONHANDLER_H_

#include "wxgui.h"

class InteractionHandler {
public:
  InteractionHandler();
  virtual ~InteractionHandler();

  //This function is for experts only.
  //Do not redefine it unless you really 
  //know what you are doing.
  virtual void OnMouseEvent(wxMouseEvent& event,
			    int p);

protected:
  //You should redefine the functions below.
  virtual void OnLeftClick(int p);
  virtual void OnRightClick(int p);
  virtual void OnMiddleClick(int p);

  virtual void OnLeftDrag(int p, int q);
  virtual void OnRightDrag(int p, int q);

  virtual void OnLeftRelease(int p);
  virtual void OnRightRelease(int p);
  virtual void OnMiddleRelease(int p);
  
  virtual void OnMouseMotion(int p);

  virtual void OnMouseWheel(int p,
			    int rotation,
			    int delta);
  virtual void OnEnterWindow();
  virtual void OnLeaveWindow();
};

#include "gui.h"

#endif


