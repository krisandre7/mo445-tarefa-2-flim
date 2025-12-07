
#ifndef _MODULEBASICSTATS_H_
#define _MODULEBASICSTATS_H_

#include "startnewmodule.h"

namespace BasicStats{

  class ModuleBasicStats : public AnalysisModule{
  public:
    ModuleBasicStats();
    ~ModuleBasicStats();
    void Start();

  protected:
    void SetCellValue(wxGrid *grid,
		      int i, int j,
		      int value);
    void SetCellValue(wxGrid *grid,
		      int i, int j,
		      float value);
    void SetCellValue(wxGrid *grid,
		      int i, int j,
		      char *value);

    int  ComputeVolume(BMap *mask);
    int  ComputeMax(BMap *mask);
    int  ComputeMin(BMap *mask);
    void ComputeMeanStdev(BMap *mask,
			  float *mean,
			  float *stdev);
  };
}

#endif

