
#ifndef _CLOUDSBRAIN_H_
#define _CLOUDSBRAIN_H_

#ifndef _CLOUDSBRAIN_STANDALONE_
#include "startnewmodule.h"
#else
#include "wx/thread.h"
#endif

#include "bia_common.h"
#include "bia_adjrel3.h"
#include "bia_scene.h"
#include "bia_set.h"
#include "bia_pqueue16.h"
#include "bia_adjregion3.h"
#include "bia_seedmap3.h"

#include "methods_addons.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "oldift.h"
#include "compressed.h"

#include "shared.h"
#include "filelist.h"
#include "gradient.h"
#include "gradient_addons.h"
#include "preproc.h"
#include "posproc.h"
#include "featmap.h"
#include "pyramid3.h"
#include "cloud3.h"
#include "realarray.h"
#include "fuzzycloud3.h"
#include "mri.h"
#include "inhomogeneity.h"
#include "processors.h"
#include <pthread.h>

#ifdef __cplusplus
}
#endif


Scene *CloudsBrainSegmentation(Scene *orig, char *basename);


#endif

