
#ifndef _RENDERCANVAS_H_
#define _RENDERCANVAS_H_


#include "wxgui.h"
extern "C" {
#include "oldift.h"
}

#include "canvas.h"

class RenderCanvas : public Canvas {
public:
  bool rotChanged;
  RenderCanvas(wxWindow *parent);
  virtual ~RenderCanvas();
  void RotateX(float angle);
  void RotateY(float angle);
  void Rotate(float angle, Vector axis);  

protected:
  Matrix *rot;
};

#endif
