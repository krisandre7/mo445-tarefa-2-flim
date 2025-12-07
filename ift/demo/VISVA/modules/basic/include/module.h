
#ifndef _MODULE_H_
#define _MODULE_H_

#include "gui.h"
extern "C" {
#include "oldift.h"
#include "code_c.h"
}
#include "code_cpp.h"


class Module {
 public:
  typedef enum {PREPROC=0,
		SEGMENTATION, 
		ANALYSIS, 
		VIEW2D,
                VIEW3D,
		REGISTRATION,
                LAST_TYPE} ModuleType;

  Module();
  Module(char *name,
	 char *author);
  Module(  SoftwareVersion& ver,
	 char *name,
	 char *author);
  virtual ~Module();

  virtual void Start();
  virtual bool Stop();
  virtual bool Automatic();

  void SetVersion(  SoftwareVersion& ver);
  void SetName(char *name);
  void SetAuthor(char *author);
  SoftwareVersion *GetVersion();
  void GetName(char *name);
  void GetAuthor(char *author);
  //Funcao virtual pura.
  virtual Module::ModuleType GetType()=0;
 protected:
  SoftwareVersion *version;
  char name[512];
  char author[512];
};


void freeModule(void **mem);


class PreProcModule : public Module {
 public:
  PreProcModule();
  virtual ~PreProcModule();
  Module::ModuleType GetType();
};

class SegmentationModule : public Module {
 public:
  SegmentationModule();
  virtual ~SegmentationModule();
  Module::ModuleType GetType();
};


class RegistrationModule : public Module {
 public:
  RegistrationModule();
  virtual ~RegistrationModule();
  Module::ModuleType GetType();
};


class AnalysisModule : public Module {
 public:
  AnalysisModule();
  virtual ~AnalysisModule();
  Module::ModuleType GetType();
};


class RefreshHandler {
public:
  RefreshHandler();
  virtual ~RefreshHandler();
  virtual void OnRefresh2D(char axis);
protected:
};


class View2DModule : public Module {
 public:
  View2DModule();
  virtual ~View2DModule();
  Module::ModuleType GetType();
  virtual wxPanel *GetViewPanel(wxWindow *parent, 
				char axis)=0;
  virtual wxPanel *GetCustomViewPanel(wxWindow *parent,
				      Scene *scn,
				      char axis)=0;
  virtual void DelViewPanel(wxPanel **view)=0;
  virtual void SetInteractionHandler(InteractionHandler *handler,
				     char axis)=0;
  virtual void  Refresh()=0;
  virtual void  Refresh(char axis)=0;
  virtual void  SetRefreshHandler(RefreshHandler *handler)=0;
  virtual void  Clear()=0;
  virtual void  SetSliceVoxel(Voxel Cut)=0;
  virtual Voxel GetSliceVoxel()=0;
  virtual void  SetZoomLevel(float zoom)=0;
  virtual float GetZoomLevel()=0;
  virtual void  ChangeDrawMarker()=0;
  virtual CImage *CopyAsCImage(char axis)=0;
  virtual void  SetCursor(  wxCursor *cursor, char axis);
};

class View3DModule : public Module {
 public:
  View3DModule();
  virtual ~View3DModule();
  Module::ModuleType GetType();
  virtual wxPanel *GetViewPanel(wxWindow *parent)=0;
  virtual void  SetInteractionHandler(InteractionHandler *handler)=0;
  virtual void  Refresh(bool dataChanged, 
			float quality)=0;
  virtual void  SetZoomLevel(float zoom)=0;
  virtual float GetZoomLevel()=0;
  virtual CImage *CopyAsCImage()=0;
};

#endif

