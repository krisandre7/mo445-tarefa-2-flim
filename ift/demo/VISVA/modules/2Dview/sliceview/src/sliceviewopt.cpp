
#include "sliceviewopt.h"

namespace SliceView{

  SliceViewOpt :: SliceViewOpt(wxWindow *parent,
			       ModuleSliceView *mod)
    : BasePanel(parent){

    this->Hide();
    this->mod = mod;
    id_high = APP->idManager->AllocID();
    id_data = APP->idManager->AllocID();
    id_mark = APP->idManager->AllocID();

    wxStaticBoxSizer *vosizer = new wxStaticBoxSizer(wxVERTICAL, this, _T("Slice Options"));

    wxStaticText *tHighlight = new wxStaticText(this, -1, _T("Label"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText0"));
    wxStaticText *tData = new wxStaticText(this, -1, _T("Data"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText1"));
    wxStaticText *tMarker = new wxStaticText(this, -1, _T("Marker"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText2"));

    wxString lbChoices[4];
    lbChoices[0] = _T("Fill");
    lbChoices[1] = _T("Border");
    lbChoices[2] = _T("Inside");
    lbChoices[3] = _T("Off");
    chHighlight = new wxChoice(this, id_high, wxDefaultPosition, wxDefaultSize, 4, lbChoices, 0, wxDefaultValidator, _T("choice0"));
    chHighlight->SetSelection(0);

    wxString dtChoices[4];
    dtChoices[0] = _T("Original");
    dtChoices[1] = _T("ArcWeight");
    dtChoices[2] = _T("Fuzzy map");
    dtChoices[3] = _T("Object");
    chData = new wxChoice(this, id_data, wxDefaultPosition, wxDefaultSize, 4, dtChoices, 0, wxDefaultValidator, _T("choice1"));
    chData->SetSelection(0);

    wxString mkChoices[2];
    mkChoices[0] = _T("On");
    mkChoices[1] = _T("Off");
    chMarker = new wxChoice(this, id_mark, wxDefaultPosition, wxDefaultSize, 2, mkChoices, 0, wxDefaultValidator, _T("choice2"));
    chMarker->SetSelection(0);

    wxBoxSizer *hbsHighlight = new wxBoxSizer(wxHORIZONTAL);
    wxBoxSizer *hbsData      = new wxBoxSizer(wxHORIZONTAL);
    wxBoxSizer *hbsMarker    = new wxBoxSizer(wxHORIZONTAL);

    hbsHighlight->Add(tHighlight, 1, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsHighlight->Add(chHighlight, 0, wxALIGN_RIGHT);
    hbsData->Add(tData, 1, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsData->Add(chData, 0, wxALIGN_RIGHT);
    hbsMarker->Add(tMarker, 1, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);   
    hbsMarker->Add(chMarker, 0, wxALIGN_RIGHT);

    vosizer->AddSpacer(5);
    vosizer->Add(hbsHighlight,  0, wxEXPAND);
    vosizer->Add(hbsData,       0, wxEXPAND);
    vosizer->Add(hbsMarker,     0, wxEXPAND);

    sizer->Add(vosizer, 0, wxEXPAND);
    sizer->SetSizeHints(this);
    sizer->Layout();

    Connect( id_high, wxEVT_COMMAND_CHOICE_SELECTED,
	     wxCommandEventHandler(SliceViewOpt::OnChangeChoice),
	     NULL, NULL );
    Connect( id_data, wxEVT_COMMAND_CHOICE_SELECTED,
	     wxCommandEventHandler(SliceViewOpt::OnChangeChoice),
	     NULL, NULL );
    Connect( id_mark, wxEVT_COMMAND_CHOICE_SELECTED,
	     wxCommandEventHandler(SliceViewOpt::OnChangeChoice),
	     NULL, NULL );
  }



  SliceViewOpt::~SliceViewOpt(){ 
    Disconnect( id_high, wxEVT_COMMAND_CHOICE_SELECTED,
		wxCommandEventHandler(SliceViewOpt::OnChangeChoice),
		NULL, NULL );
    Disconnect( id_data, wxEVT_COMMAND_CHOICE_SELECTED,
		wxCommandEventHandler(SliceViewOpt::OnChangeChoice),
		NULL, NULL );
    Disconnect( id_mark, wxEVT_COMMAND_CHOICE_SELECTED,
		wxCommandEventHandler(SliceViewOpt::OnChangeChoice),
		NULL, NULL );

    APP->idManager->FreeID(id_high);
    APP->idManager->FreeID(id_data);
    APP->idManager->FreeID(id_mark);
  }



  void SliceViewOpt::OnChangeChoice(wxCommandEvent& WXUNUSED(event)){
    int highlight, data, marker;

    highlight = chHighlight->GetSelection();
    data   = chData->GetSelection();
    marker = chMarker->GetSelection();
    mod->Set2DViewOptions((HighlightType)highlight, 
			  (DataType)data, 
			  (MarkerType)marker);
  }

} //end SliceView namespace

