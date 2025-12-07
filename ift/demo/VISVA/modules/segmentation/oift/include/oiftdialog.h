
#ifndef _OIFTDIALOG_H_
#define _OIFTDIALOG_H_

#include "startnewmodule.h"
#include "moduleoift.h"
#include "addhandleroift.h"
#include "delhandleroift.h"
#include "livehandleroift.h"

namespace OIFT{

    class OIFTParams : public BaseDialog {

    public:

    OIFTParams(wxWindow *parent, ModuleOIFT *mod);
    ~OIFTParams();

    void OnUpdateParams(wxCommandEvent &event);

    private:

    int id_isf_alpha;
    int id_isf_beta;
    int id_isf_gamma;
    int id_isf_niters;

    int id_ow_stop_thresh;
    int id_ow_gamma;
    int id_ow_nmarkers;

    ModuleOIFT *mod;
    BasePanel  *panel;

    //ISF Options:
    wxTextCtrl *tc_ISF_Alpha;
    wxTextCtrl *tc_ISF_Beta;
    wxTextCtrl *tc_ISF_Gamma;
    wxSpinCtrl *sc_ISF_NIters;

    //Oriented Watershed opts
    wxTextCtrl *tc_OW_Gamma;
    wxTextCtrl *tc_OW_StopThresh;
    wxSpinCtrl *sc_OW_NMarkers;
};

class OIFTDialog : public BaseDialog{
    public:
    typedef enum {NAVIGATOR=0,
        ADDMARKER,
        DELMARKER,
        LIVEMARKER} ModeType;

    OIFTDialog(wxWindow *parent,
               ModuleOIFT *mod);
    ~OIFTDialog();

    void OnChangeMode(wxCommandEvent& event);
    void OnReset(wxCommandEvent& event);
    void OnRun(wxCommandEvent& event);

    void OnAddObj(wxCommandEvent& event);
    void OnDelete(wxCommandEvent& event);
    void OnChangeColor(wxCommandEvent& event);
    void OnChangeVisibility(wxCommandEvent& event);

    void OnCancel(wxCommandEvent& event);
    void OnOk(wxCommandEvent& event);
    void OnSize(wxSizeEvent& event);
    void OnUpdateParams(wxCommandEvent &event);

    void ChangeObjSelection(wxPanel *objbkg);


    OIFTDialog::ModeType GetOperationMode();
    AdjRel   *GetBrush();
    wxCursor *GetBrushCursor(int zoom);
    void      NextBrush();
    void      PrevBrush();

    private:
    wxScrolledWindow *CreateObjectPanel();
    void              DestroyObjectPanel();
    void              UnmarkAllForRemoval();

    int id_but;
    int id_bp;
    int id_res;
    int id_run;
    int id_add;

    AddHandlerOIFT         *xhandler;
    AddHandlerOIFT         *yhandler;
    AddHandlerOIFT         *zhandler;
    DelHandlerOIFT         *dhandler;
    LiveHandlerOIFT        *live_xhandler;
    LiveHandlerOIFT        *live_yhandler;
    LiveHandlerOIFT        *live_zhandler;

    ModuleOIFT  *mod;

    BitmapRadioButton  *but;
    BrushPicker        *bPicker;
    wxButton           *run;
    wxButton           *res;
    wxButton           *addobj;

    BasePanel  *panel;

    //ObjectPanel:
    wxScrolledWindow *objPanel;
    wxPanel            **v_panel_bkg;
    AlphaColourButton  **v_but_color;
    BitmapToggleButton **v_but_eye;
    wxBitmapButton     **v_but_trash;


    DECLARE_EVENT_TABLE()
};

//--------------------------------

    class ObjBkgPanel : public wxPanel{
    public:
    ObjBkgPanel(wxWindow *parent,
      wxPoint& pos,
      wxSize& size,
            OIFTDialog  *dialog);
    ~ObjBkgPanel();
    void OnMouseEvent(wxMouseEvent& event);

    private:
    OIFTDialog  *dialog;
    DECLARE_EVENT_TABLE()
};

//--------------------------------

    class ObjLabel : public wxStaticText{
    public:
    ObjLabel(wxWindow *parent,
      wxString& label,
            OIFTDialog  *dialog);
    ~ObjLabel();
    void OnMouseEvent(wxMouseEvent& event);

    private:
    OIFTDialog  *dialog;
    DECLARE_EVENT_TABLE()
};


} //end OIFT namespace

#endif

