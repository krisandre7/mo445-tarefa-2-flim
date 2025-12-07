
#include "modulebasicstats.h"

namespace BasicStats{

  ModuleBasicStats :: ModuleBasicStats()
    : AnalysisModule(){
    SetName((char *)"Basic Statistics");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(1,2,0);
    SetVersion(ver);
  }


  ModuleBasicStats :: ~ModuleBasicStats(){}


  void ModuleBasicStats :: Start(){
    wxGrid *grid;
    wxString *wxstr;
    SegmObject *segm;
    float mean, stdev, vol, dx,dy,dz;
    int size,len,max_len=0;
    int i,n;
    
    BaseDialog dialog(APP->Window, (char *)"Statistics");
    
    grid = new wxGrid(&dialog, wxID_ANY, 
		      wxDefaultPosition, 
		      wxDefaultSize,
		      wxWANTS_CHARS, 
		      _T("StatsGrid"));

    n = APP->GetNumberOfObjs();
    grid->CreateGrid(n, 5, wxGrid::wxGridSelectCells);
    grid->SetDefaultCellAlignment(wxALIGN_RIGHT, wxALIGN_CENTRE);
    grid->SetColLabelValue(0, _T("Volume mm^3"));
    grid->SetColLabelValue(1, _T("Mean"));
    grid->SetColLabelValue(2, _T("Std.Dev."));
    grid->SetColLabelValue(3, _T("Min"));
    grid->SetColLabelValue(4, _T("Max"));
    dx = (APP->Data.orig)->dx;
    dy = (APP->Data.orig)->dy;
    dz = (APP->Data.orig)->dz;
    
    for(i=0; i<n; i++){
      segm = APP->GetObjByIndex(i);

      len = strlen(segm->name);
      if(len>max_len) max_len = len;
      wxstr = new wxString(segm->name, wxConvUTF8);
      grid->SetRowLabelValue(i, *wxstr);
      
      vol = dx*dy*dz*(float)ComputeVolume(segm->mask);
      SetCellValue(grid, i, 0, vol);
      ComputeMeanStdev(segm->mask, &mean, &stdev);
      SetCellValue(grid, i, 1, mean);
      SetCellValue(grid, i, 2, stdev);
      SetCellValue(grid, i, 3, ComputeMin(segm->mask));
      SetCellValue(grid, i, 4, ComputeMax(segm->mask));
    }
    size = grid->GetRowLabelSize();
    size = MAX(max_len*10, size);
    grid->SetRowLabelSize(size);
    grid->AutoSize();

    dialog.AddPanel((wxPanel *)grid);
    dialog.ShowModal();
  }


  void ModuleBasicStats::SetCellValue(wxGrid *grid,
				      int i, int j,
				      int value){
    char str[512];
    sprintf(str,"%d",value);
    SetCellValue(grid, i, j, str);
  }

  void ModuleBasicStats::SetCellValue(wxGrid *grid,
				      int i, int j,
				      float value){
    char str[512];
    sprintf(str,"%.2f",value);
    SetCellValue(grid, i, j, str);
  }

  void ModuleBasicStats::SetCellValue(wxGrid *grid,
				      int i, int j,
				      char *value){
    wxColour wxcolor;
    wxString *wxstr;
    wxstr = new wxString(value, wxConvUTF8);
    grid->SetCellValue(i, j, *wxstr);
    grid->SetReadOnly( i, j, true );
    if(i%2==0) SetColor(&wxcolor, 0xffffff);
    else       SetColor(&wxcolor, 0xffffdd);
    grid->SetCellBackgroundColour(i, j, wxcolor);
  }
  
  int ModuleBasicStats :: ComputeVolume(BMap *mask){
    int p,n,vol=0;
    
    n = mask->N;
    for(p=0; p<n; p++)
      if(_fast_BMapGet(mask, p)>0)
	vol++;
    return vol;
  }

  int ModuleBasicStats :: ComputeMax(BMap *mask){
    int p,n,max=INT_MIN;
    Scene *scn = APP->Data.orig;
    
    n = mask->N;
    for(p=0; p<n; p++)
      if(_fast_BMapGet(mask, p)>0)
	if(scn->data[p]>max)
	  max = scn->data[p];
    return max;
  }

  int ModuleBasicStats :: ComputeMin(BMap *mask){
    int p,n,min=INT_MAX;
    Scene *scn = APP->Data.orig;
    
    n = mask->N;
    for(p=0; p<n; p++)
      if(_fast_BMapGet(mask, p)>0)
	if(scn->data[p]<min)
	  min = scn->data[p];
    return min;
  }


  void ModuleBasicStats::ComputeMeanStdev(BMap *mask,
					  float *mean,
					  float *stdev){
    int p,n,vol=0;
    Scene *scn = APP->Data.orig;
    float m=0.0,s=0.0;
    
    n = mask->N;
    for(p=0; p<n; p++){
      if(_fast_BMapGet(mask, p)>0){
	m += (float)scn->data[p];
	vol++;
      }
    }
    m = m/(float)vol;
    
    for(p=0; p<n; p++)
      if(_fast_BMapGet(mask, p)>0)
	s += (scn->data[p] - m)*(scn->data[p] - m);
    s = sqrt(s)/(float)vol;
    
    *mean  = m;
    *stdev = s;
  }


} //end BasicStats namespace

