
#ifndef _BIA_H_
#define _BIA_H_

#include "gui.h"

//This is required since the c++ will include C header files and 'class' is a special word in C++ but not in C.
#define class class_
#ifdef __cplusplus 
extern "C" { 
  #include "ift.h"
  #include "oldift.h"
  #include "code_c.h"
  #include "scene.h"
} 
#endif 
#undef class

#include "code_cpp.h"
#include "module.h"


typedef struct _AppPreferencesType{
  bool IsMaximized;
  char dir_LoadVolume[512];
  char dir_LoadLabel[512];
  char dir_SaveVolume[512];
  char dir_SaveLabel[512];
  char dir_SaveObject[512];
  char dir_LoadMarkers[512];
  char dir_SaveMarkers[512];

  int  ext_LoadVolume;
  int  ext_LoadLabel;
  int  ext_SaveVolume;
  int  ext_SaveLabel;
  int  ext_SaveObject;
  int  ext_SaveObjectMap;
  int  ext_LoadMarkers;
  int  ext_SaveMarkers;

  int color_preproc;
  int color_segmentation;
  int color_view;
  int color_analysis;
} AppPreferencesType;


typedef enum _modalityType {T1_WEIGHTED=0,
			    T2_WEIGHTED,
			    PD_PROTONDENSITY,
			    BIN_BINARYMASK,
			    MLM_MULTILABELMASK,
			    UNKNOWN_PROTOCOL} ModalityType;

//int ModalityToInt(ModalityType m);

typedef struct _appDataType{
  char         volumename[512];
  char         labelname[512];
  char         projectfilename[512];
  int          w,h,nframes;
  Scene        *orig;
  Scene        *label;
  Scene        *marker;
  Plane        *msp;
  Scene        *arcw;
  ScnGradient  *grad;
  Scene        *objmap;
  int          loaded; // is there any image loaded?
  int oriented; // setorientation module 1=yes/0=no
  int aligned;  // MSP alignment module  1=yes/0=no
  int corrected; // inhomogeneity module 1=yes/0=no
  ModalityType modality;
} AppDataType;


//--------------------------------------------------------

class MyFrame: public wxFrame{
public:

  MyFrame(  wxString& title,   wxPoint& pos,   wxSize& size);
  ~MyFrame();
  void SetViewPanel(wxPanel *view, int pos);  
  void UniformView();
  void AppendOptPanel(wxPanel *panel, Module::ModuleType type);
  void PrependOptPanel(wxPanel *panel, Module::ModuleType type);
  bool DetachOptPanel(wxPanel *panel);
  void LoadVolume(char *filename);
  void LoadLabel(char *filename,char *name, int color);
  void LoadMarkers(char *seeds_path);
  void LoadScnGradient(char *filename);
  void LoadArcWeight(char *filename);
  void OnNewProj(wxCommandEvent & event);
  void OnOpenProj(wxCommandEvent & event);
  void OnSaveProj(wxCommandEvent & event);
  void OnSaveProjAs(wxCommandEvent & event);
  void OpenProj(char *filename);
  void SaveProj(char *filename);
  void OnLoadVolume(wxCommandEvent & event);
  void OnLoadLabel(wxCommandEvent & event);
  void OnLoadScnGradient(wxCommandEvent & event);
  void OnLoadArcWeight(wxCommandEvent & event);
  void OnSaveVolume(wxCommandEvent & event);
  void OnSaveLabel(wxCommandEvent & event);
  void OnSaveObject(wxCommandEvent & event);
  void OnSaveObjectMask(wxCommandEvent & event);
  void OnSaveObjectMap(wxCommandEvent &event);
  void OnLoadMarkers(wxCommandEvent & event);
  void OnSaveMarkers(wxCommandEvent & event);
  void OnQuit(wxCommandEvent& event);
  void OnAbout(wxCommandEvent& event);
  void OnVolumeInformation(wxCommandEvent& event);
  void OnParamEstimation(wxCommandEvent& event);
  void OnPreprocessing(wxCommandEvent& event);
  void OnSegmentation(wxCommandEvent& event);
  void OnAnalysis(wxCommandEvent& event);
  void OnMeasurements(wxCommandEvent& event);
  void OnGroupStudy(wxCommandEvent& event);
  void On3DVisualization(wxCommandEvent& event);
  void OnRegistration(wxCommandEvent& event);
  void OnShowObjWindow(wxCommandEvent & event);
  void OnZoomin(wxCommandEvent& event);
  void OnZoomout(wxCommandEvent& event);
  void OnChangeDrawMarker(wxCommandEvent& event);
  void OnChangeBriContr(wxScrollEvent& event);
  void OnShowBriContr(wxCommandEvent& event);
  void OnSelectView(wxCommandEvent & event);
  void EnableObjWindow(bool enable);
  void SetBriContr(int B, int C);
  void GetBriContr(int *B, int *C);

private:
  void SetOptPanelBkgColor(wxPanel *panel,
			   Module::ModuleType type);

  wxToolBar      *toolBar;

  wxPanel        *panel;
  wxBoxSizer     *vsizer;
  wxSplitterWindow *splitter;
  wxSplitterWindow *topWindow;
  wxSplitterWindow *bottomWindow;
  wxPanel *Views[4];

  wxBitmapButton *buteye;
  wxBitmapButton *buZoomin;
  wxBitmapButton *buZoomout;
  wxBitmapButton *buBriContr;
  BriContrDialog *BCDialog;
  float zoom;
  int   B,C;
  
  DECLARE_EVENT_TABLE()
};

//----------------------------------------------------------

class BIAClass{
public:
  AppPreferencesType Preferences;
  AppDataType        Data;
  MyFrame           *Window;
  DynamicID         *idManager;

  BIAClass();
  ~BIAClass();

  //***** OLD IFT to NEW IFT functions ****
  Scene * iftImageToScene(iftImage *img);
  Scene * LabelImageToScene(iftImageForest *fst);
  Scene * IGraphLabelImageToScene(iftIGraph *igraph);
  void SetLabeledSetAsSeeds(iftLabeledSet *seeds);
  iftImage* SceneToiftImage(Scene *scn);


  //***** Busy/Message functions: ****
  void Busy(char *msg);
  void Unbusy();
  void StatusMessage(char *msg);
  int  GetColorDialog();

  //***** Data functions: **********
  void InitData();
  void ResetData();
  void DestroyData();
  void SetDataVolume(Scene *scn);
  void SetDataVolume_NoDestroy(Scene *scn);
  int  ShowImageModalityDialog();


  //***** View functions: *********
  void Set2DViewModule(View2DModule *mod);
  void Set3DViewModule(View3DModule *mod);
  void Refresh2DCanvas();
  void Refresh2DCanvas(char axis);
  void Refresh3DCanvas(bool dataChanged, float quality);
  void SetRefresh2DHandler(RefreshHandler *handler);
  void Clear2DCanvas();
  void ChangeDrawMarker();
  CImage *Copy3DCanvasAsCImage();
  CImage *Copy2DCanvasAsCImage(char axis);

  void DrawSegmObjects();
  void EnableObjWindow(bool enable);
  void Set2DViewCursor(  wxCursor *cursor, char axis);
  void SetDefaultInteractionHandler();
  void Set2DViewInteractionHandler(InteractionHandler *handler,
				   char axis);
  void  Set2DViewSlice(Voxel Cut);
  Voxel Get2DViewSlice();
  void  SetBriContr(int B, int C);
  void  GetBriContr(int *B, int *C);
  void  SetZoomLevel(float zoom);
  float GetZoomLevel();

  //***** OptPanel functions: ********

  void AppendOptPanel(wxPanel *panel,
		      Module::ModuleType type);
  void PrependOptPanel(wxPanel *panel,
		       Module::ModuleType type);
  bool DetachOptPanel(wxPanel *panel);

  //***** Seed functions: ************
  bool  IsSeed(int p);
  int   GetSeedLabel(int p);
  int   GetSeedLabelById(int id);
  int   GetSeedId(int p);

  void  AddSeed(int p, int label, int id);
  void  AddSeedsInSet(Set *S, int label, int id);
  void  AddSeedsInBrush(int p,
			int label, int id,
			AdjRel *A, char axis);
  void  AddSeedsInBrushTrace(int p, int q,
			     int label, int id,
			     AdjRel *A, char axis);
  void  AddSeedsInMarkerList(MarkerList *ML);

  void  DelSeed(int p);
  void  DelSeedsById(int id);
  void  DelSeedsInSet(Set *S);
  void  DelSeedsNotInSet(Set *S);
  void  DelAllSeeds();
  Set  *CopyInternalSeeds();
  Set  *CopyExternalSeeds();
  Set  *CopySeeds();
  MarkerList *CopyMarkerList();
  void  ReallocSeedStructures();

  void  MarkForRemoval(int id);
  void  UnmarkForRemoval(int id);
  void  UnmarkAllForRemoval();
  bool  IsMarkedForRemoval(int p);
  bool  IsIdMarkedForRemoval(int id);
  void  DelMarkedForRemoval();
  Set  *CopyIdsMarkedForRemoval();
  int GetMarkerValue(int id, int lb);

  //**** Label functions: ************

  int   GetLabelColour(int label);
  int   GetLabelAlpha(int label);
  void  SetLabelColour(int label, 
		       int color);
  void  SetLabelColour(int label, 
		       int color,
		       int alpha);
  void  DrawBrush(int p, int label,
		  AdjRel *A, char axis);
  void  DrawBrushTrace(int p, int q,
		       int label,
		       AdjRel *A, char axis);
  void  DrawBrushCustom(int p, int label,
			AdjRel *A, char axis, 
			Scene *scn);
  void  DrawBrushCustom(int p, int label,
			AdjRel *A, char axis, 
			bia::Scene16::Scene16 *scn);
  void  DrawBrushTraceCustom(int p, int q,
			     int label,
			     AdjRel *A, char axis,
			     Scene *scn);
  void  DrawBrushTraceCustom(int p, int q,
			     int label,
			     AdjRel *A, char axis,
			     bia::Scene16::Scene16 *scn);

  Scene *CopyVisibleLabelUnion();
  Scene *CopyDistinctVisibleLabels();

  //**** Object functions: ***********

  void        SetObjVisibility(char *obj_name, bool enable);
  SegmObject *SearchObjByName(char *obj_name);
  SegmObject *GetObjByIndex(int index);
  SegmObject *GetObjByDialog();
  ArrayList  *GetSegmObjsArrayList();
  int         AddObj(char *name, int color);
  int         AddCustomObj(SegmObject *obj);
  void        DelObjByIndex(int index);
  void        DelAllObjs();
  int         GetNumberOfObjs();
  int         ShowNewObjDialog(int *color, char *name);
  int         ShowObjColorDialog(int *color, char *name);

  //**** Dependency functions: ***********
  
  bool CheckDependency(char *name, Module::ModuleType type);
  bool SolveDependency(char *name, Module::ModuleType type);
  void SetDependencyStatus(char *name,
			   Module::ModuleType type,
			   bool solved);

  // masks of all segmented objects.
  ArrayList *segmobjs;   
  Measures  *measures;
  
private:
  //***** Busy/Message data: ****
  wxWindowDisabler *disableAll;
  wxBusyCursor     *busycursor;
  wxBusyInfo       *busyinfo;

  int          *labelcolor;
  int          *labelalpha;
  int           Lmax; //Label maximum.

  Set          *removalIdSet;
  Set          *seedSet;
  Scene        *seedIdLabel;
  int           _GetMarkerID(int mk);
  int           _GetMarkerLabel(int mk);
  int           _GetMarkerValue(int id, int lb);

  View2DModule *mod2D;
  View3DModule *mod3D;
  void InitPreferences();
  void SavePreferences();
  void LoadPreferences();
  void MakePalette();
};

//----------------------------------------------------------

class ObjectDialog : public BaseDialog{
public:
  ObjectDialog(wxWindow *parent);
  ~ObjectDialog();
  void OnCancel(wxCommandEvent& event);
  void OnOk(wxCommandEvent& event);
  void OnDelete(wxCommandEvent& event);
  void OnChange(wxCommandEvent& event);


private:
  wxBoxSizer *sizer;
  int n;
  wxPanel            **v_panel_bkg;
  AlphaColourButton  **v_but_color;
  BitmapToggleButton **v_but_eye;
  wxBitmapButton     **v_but_trash;
  wxTextCtrl         **v_obj_name;
};

//----------------------------------------------------------

class ObjectPickerDialog : public BaseDialog{
public:
  ObjectPickerDialog(wxWindow   *parent,
		     wxWindowID  id,
		     char       *title,
		     const char *choices[], int n,
		     bool        readonly,
		     bool        getcolor,
		     bool        getname);
  ~ObjectPickerDialog();
  void GetName(char *name);
  int  GetColor();

  ColourButton *butColor;
  wxComboBox   *texName;
private:
};

//---------------------------------------------------------

class ModalityPickerDialog : public BaseDialog{
public:
  ModalityPickerDialog(wxWindow   *parent,
		       wxWindowID  id,
		       char       *title);
  ~ModalityPickerDialog();
  ModalityType GetModality();

  wxChoice    *chModality;
private:
};

//---------------------------------------------------------

/* class PopUpView : public BaseDialog{ */
/* public: */
/*   PopUpView(Scene *scn, char *title); */
/*   ~PopUpView(); */
/*   void OnChangeSliceOrientation(wxCommandEvent& event); */

/* private: */
/*   wxPanel *view[3]; */
/*   int current; */
/*   wxChoice *chOrientation; */
/*   int id_ori; */
/*   Scene *view_scn; */
/* }; */



class MeasurementsDialog: public wxDialog{
public:

  MeasurementsDialog();
  ~MeasurementsDialog();
  int GetM1();
  int GetM2();

private:

  wxCheckBox *m1;
  wxCheckBox *m2;

};




#endif

