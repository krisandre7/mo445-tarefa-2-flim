#ifndef _BRAINCLUSTER_SPECTRAL_H_
#define _BRAINCLUSTER_SPECTRAL_H_

#define T1_PROTOCOL 0
#define T2_PROTOCOL 1
#define PD_PROTOCOL 2
#define PDT2_PROTOCOL 3

#define BKG_SEG 0
#define CSF_SEG 1
#define WMGM_SEG 2

typedef struct {
  int min;
  int max;
  int mean;
  int otsu;
  int csf;
  int csf_gm;
  int gm;
  int gm_wm;
  int wm;
} TissueMarker;

Scene *GetBrain( Scene *scn, Scene *mask );
Scene *NonZeroComplement3( Scene *scn );
TissueMarker *BrainTissueThresholds( Scene *brain, int protocol );
int WMGMThreshold( Scene *brain );
void BrainPreProportions( Scene *brain, Scene *mask, int protocol, float *T_CSF_GM, float *T_GM_WM );
void TissueProportions( Scene *brain, Scene *mask, int protocol, int tissue, int auto_prop, float delta_prop, float *Tmean );
void MaskLimits( Scene *mask, int *first_voxel, int *last_voxel );
int *SortClusters( unsigned long long *mean, uint *size, int nlabels );
void BrightnessLabeling( Scene *brain, Scene *mask, Scene *dark, Scene *light, int nlabels, int protocol, float T );
int BrainClusterSegmentation( Scene *scn, Scene *mask, Scene **obj1, Scene **obj2, int samples, float mean_prop, 
			      int protocol, int tissue_type, int auto_prop, float delta_prop, int kmin, int kmax, int nthreads );

#endif
