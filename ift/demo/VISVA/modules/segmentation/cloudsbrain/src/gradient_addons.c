
#include "gradient_addons.h"
#include <pthread.h>


ScnGradient *FastSphericalScnGradient(Scene *scn, float r){
  ArgSphericalScnGradient arg;
  ArgSphericalScnGradient args[8];
  int i,dp,nprocs;
  int first,last,nelems,de;
  pthread_t thread_id[8];
  int iret[8];

  nprocs = GetNumberOfProcessors();
  //printf("nprocs: %d\n",nprocs);

  if(nprocs<=1) return SphericalScnGradient(scn, r);
  if(nprocs>=8) nprocs = 8;

  arg.scn = scn;
  arg.A   = SceneSphericalAdjRel3(scn, r);
  arg.N   = AdjVoxels(scn, arg.A);
  arg.mg  = AdjRel3SceneDistance(scn, arg.A);
  FrameSizes3(arg.A, &arg.fx, &arg.fy, &arg.fz);
  arg.grad = CreateScnGradient(scn->xsize,
			       scn->ysize,
			       scn->zsize);
  SetVoxelSize((arg.grad)->Gx, scn->dx, scn->dy, scn->dz);
  SetVoxelSize((arg.grad)->Gy, scn->dx, scn->dy, scn->dz);
  SetVoxelSize((arg.grad)->Gz, scn->dx, scn->dy, scn->dz);

  dp  = (arg.fx)*1;
  dp += (arg.fy)*scn->xsize;
  dp += (arg.fz)*scn->xsize*scn->ysize;

  first  = dp;
  last   = scn->n-dp-1;
  nelems = last-first+1; 
  de     = nelems/nprocs;
  arg.j  = first-1;
  for(i=0; i<nprocs; i++){
    args[i] = arg;

    args[i].i = arg.j+1;
    if(i<nprocs-1) args[i].j = args[i].i+(de-1);
    else           args[i].j = last;

    //Create independent threads each of which will execute function
    iret[i] = pthread_create(&thread_id[i], NULL, 
			     ThreadSphericalScnGradient, 
			     (void*)&args[i]);
    arg = args[i];
  }

  //Wait till threads are complete before main continues.
  for(i=0; i<nprocs; i++){
    pthread_join(thread_id[i], NULL);
  }

  ClearSceneAdjFrame((arg.grad)->Gx, arg.A);
  ClearSceneAdjFrame((arg.grad)->Gy, arg.A);
  ClearSceneAdjFrame((arg.grad)->Gz, arg.A);
  DestroyAdjRel3(&(arg.A));
  DestroyAdjVxl(&(arg.N));
  free(arg.mg);

  return arg.grad;
}


void *ThreadSphericalScnGradient(void *arg){
  ArgSphericalScnGradient *p_arg;
  AdjRel3 *A;
  AdjVxl  *N;
  ScnGradient *grad;
  Scene *scn;
  float *mg;
  int fx,fy,fz;
  int p,q,i,vp,vq;
  float gx,gy,gz,d;

  p_arg = (ArgSphericalScnGradient *)arg;
  grad = p_arg->grad;
  scn  = p_arg->scn;
  A    = p_arg->A;
  N    = p_arg->N;
  mg   = p_arg->mg;
  fx   = p_arg->fx;
  fy   = p_arg->fy;
  fz   = p_arg->fz;

  for(p=p_arg->i; p<=p_arg->j; p++){
    vp = scn->data[p];
    gx = gy = gz = 0.0;
    for(i=1; i<N->n; i++){
      q = p + N->dp[i];
      vq = scn->data[q];
      d = (float)(vq-vp);
	
      gx  += (d*A->dx[i])/(mg[i]);
      gy  += (d*A->dy[i])/(mg[i]);
      gz  += (d*A->dz[i])/(mg[i]);
    }
    (grad->Gx)->data[p] = ROUND(10.0*gx*scn->dx/fx);
    (grad->Gy)->data[p] = ROUND(10.0*gy*scn->dy/fy);
    (grad->Gz)->data[p] = ROUND(10.0*gz*scn->dz/fz);
  }
  return NULL;
}


//---------------------------------------


Scene *FastMRISphericalAccAbsDiff3(Scene *scn, float r, MRI_Info info){
  ArgMRISphericalAccAbsDiff3 arg;
  ArgMRISphericalAccAbsDiff3 args[8];
  int i,dp,nprocs;
  int first,last,nelems,de;
  pthread_t thread_id[8];
  int iret[8];
  AdjRel3 *A;
  int fx,fy,fz;
  int Tgm_dark = ((info.Tcsf+info.Ecsf)+info.Tgm)/2;

  nprocs = GetNumberOfProcessors();
  //printf("nprocs: %d\n",nprocs);

  if(nprocs<=1) return MRI_SphericalAccAbsDiff3(scn, r, info);
  if(nprocs>=8) nprocs = 8;

  arg.scn = scn;
  A = SceneSphericalAdjRel3(scn, r);
  arg.N    = AdjVoxels(scn, A);
  arg.mg   = AdjRel3Distance(A);
  arg.dmax = (info.Twm-info.Tcsf);
  FrameSizes3(A, &fx, &fy, &fz);
  dp  = fx*1;
  dp += fy*scn->xsize;
  dp += fz*scn->xsize*scn->ysize;

  arg.diff = CreateScene(scn->xsize,
			 scn->ysize,
			 scn->zsize);
  SetVoxelSize(arg.diff, scn->dx, scn->dy, scn->dz);

  //---- Likelihood for CSF and GM -----
  arg.weight1 = SceneIntensity2Gaussian(scn, info.Tcsf,
					(float)info.Ecsf,
					(Tgm_dark-info.Tcsf));
  arg.weight2 = SceneIntensity2Gaussian(scn, Tgm_dark,
					(Tgm_dark-info.Tcsf),
					(info.Tgm-Tgm_dark));
  //-----------------------------------
  first  = dp;
  last   = scn->n-dp-1;
  nelems = last-first+1; 
  de     = nelems/nprocs;
  arg.j  = first-1;
  for(i=0; i<nprocs; i++){
    args[i] = arg;

    args[i].i = arg.j+1;
    if(i<nprocs-1) args[i].j = args[i].i+(de-1);
    else           args[i].j = last;

    //Create independent threads each of which will execute function
    iret[i] = pthread_create(&thread_id[i], NULL, 
			     ThreadMRISphericalAccAbsDiff3,
			     (void*)&args[i]);
    arg = args[i];
  }

  //Wait till threads are complete before main continues.
  for(i=0; i<nprocs; i++){
    pthread_join(thread_id[i], NULL);
  }

  ClearSceneAdjFrame(arg.diff, A);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&(arg.N));
  free(arg.weight1);
  free(arg.weight2);
  free(arg.mg);

  return arg.diff;
}


void  *ThreadMRISphericalAccAbsDiff3(void *arg){
  ArgMRISphericalAccAbsDiff3 *p_arg;
  AdjVxl  *N;
  Scene *scn,*diff;
  float *weight1,*weight2;
  float *mg;
  float w,sum;
  int p,q,i,vp,vq;
  int d,dmax;

  p_arg = (ArgMRISphericalAccAbsDiff3 *)arg;
  diff    = p_arg->diff;
  scn     = p_arg->scn;
  N       = p_arg->N;
  mg      = p_arg->mg;
  weight1 = p_arg->weight1;
  weight2 = p_arg->weight2;
  dmax    = p_arg->dmax;

  for(p=p_arg->i; p<=p_arg->j; p++){
    vp = scn->data[p];
    sum = 0.0;
    for(i=1; i<N->n; i++){
      q = p + N->dp[i];
      vq = scn->data[q];

      d = vq-vp;

      //---- Enhance CSF/GM transition ----
      if(d<0){ w = (weight1[vq]+weight2[vp]); d=-d; }  // /2.0;
      else   { w = (weight1[vp]+weight2[vq]);       }  // *2.0;
      //-----------------------------------

      //--- limits the maximum difference ---
      if(d>dmax) d = MAX(0, 2*dmax-d);
      //-------------------------------------

      sum += (w*((float)d)/mg[i]);
    }
    diff->data[p] = ROUND(100.0*(sum/2.0));
  }
  return NULL;
}





