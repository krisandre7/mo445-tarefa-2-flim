
#include "msp_libalign.h"

namespace MSP{

T4 & sagTrans(VolumeOrientation o, Volume<int> *any) {
  T4 a1, a2;
  static T4 r;

  switch(o) {
  case AxialOrientation:
    a1.xrot(90.0, any->W/2, any->H/2, any->D/2);
    a2.yrot(90.0, any->W/2, any->H/2, any->D/2);
    a1 *= a2;
    break;
  case CoronalOrientation:
    a1.yrot(90.0, any->W/2, any->H/2, any->D/2);
    break;
  default:
    a1.identity();
  }

  r = a1;
  return(r);
}

void align(Volume<int> *input, 
	   const char *out, 
	   const char *debugprefix,
	   VolumeOrientation orientation,
	   int &ntrans, 
	   T4 &trans, 
	   int &planez,
	   double &tseg, 
	   double &tali,
	   bool verbose,
	   float scale)
{
  Timestamp T[3];

  // segment brain
  Volume<int>  *enh, *grad, *hsi, *saved_orig=NULL;
  Volume<char> *marker, *tmp, *brain, *hsc;
  char x[512];
  T[0] = Timestamp::now();
  Volume<int> *orig = input->isometricInterpolation();

  if (verbose) {
    cout << "[interp] ";
    fflush(stdout);
  }

  if (scale!=1.0f) {
    hsi = orig->interpolate(orig->dx / scale,
			    orig->dy / scale,
			    orig->dz / scale);
    saved_orig = orig;
    orig = hsi;
    
    if (verbose) {
      cout << "[scale=" << scale << "] ";
      fflush(stdout);
    }
  }

  marker = orig->brainMarkerComp();

  if (verbose) {
    cout << "[markers] ";
    fflush(stdout);
  }

  if (debugprefix!=NULL) {
    sprintf(x,"%s-orig.scn",debugprefix);
    orig->writeSCN(x);
    sprintf(x,"%s-marker.scn",debugprefix);
    marker->writeSCN(x);
  }

  enh    = orig->otsuEnhance();
  grad   = enh->featureGradient();

  if (verbose) {
    cout << "[grad] ";
    fflush(stdout);
  }

  if (debugprefix!=NULL) {
    sprintf(x,"%s-grad.scn",debugprefix);
    grad->writeSCN(x);
  }

  delete enh;
  tmp    = grad->treePruning(marker);

  if (verbose) {
    cout << "[tp] ";
    fflush(stdout);
  }


  if (debugprefix!=NULL) {
    sprintf(x,"%s-seg1.scn",debugprefix);
    tmp->writeSCN(x);
  }

  brain  = tmp->binaryClose(10.0/(orig->dx));

  if (verbose) {
    cout << "[close10] ";
    fflush(stdout);
  }

  if (debugprefix!=NULL) {
    sprintf(x,"%s-seg2.scn",debugprefix);
    brain->writeSCN(x);
  }

  delete tmp;
  delete marker;
  delete grad;

  // remove ventricles and lesions from mask
  Volume<char> *csf, *csf2, *csf3;
  int otsu;
  otsu = orig->computeOtsu();
  csf  = orig->binaryThreshold(otsu); // 0=csf, 1=non-csf
  csf2 = csf->binaryOpen(5.0/(orig->dx));

  if (verbose) {
    cout << "[open5] ";
    fflush(stdout);
  }

  csf3 = csf2->binaryDilate(2.0/(orig->dx));

  if (verbose) {
    cout << "[dil2] ";
    fflush(stdout);
  }

  if (debugprefix!=NULL) {
    sprintf(x,"%s-csf1.scn",debugprefix);
    csf->writeSCN(x);
    sprintf(x,"%s-csf2.scn",debugprefix);
    csf2->writeSCN(x);
    sprintf(x,"%s-csf3.scn",debugprefix);
    csf3->writeSCN(x);
  }

  delete csf;
  delete csf2;
  brain->binaryIntersection(csf3);
  delete csf3;

  if (verbose) {
    cout << "[mask] ";
    fflush(stdout);
  }

  if (scale!=1.0f) {
    if (debugprefix!=NULL) {
      sprintf(x,"%s-hseg.scn",debugprefix);
      brain->writeSCN(x);
    }

    hsc = brain->interpolate(brain->dx * scale,
			     brain->dy * scale,
			     brain->dz * scale);
    delete brain;
    brain = hsc;

    delete orig;
    orig = saved_orig;
    saved_orig = NULL;

    if (verbose) {
      cout << "[scale] ";
      fflush(stdout);
    }
  }

  if (verbose) cout << endl;
  
  if (debugprefix!=NULL) {
    sprintf(x,"%s-seg3.scn",debugprefix);
    brain->writeSCN(x);
  }

  brain->incrementBorder();
  T[1] = Timestamp::now();

  // reorient if needed (broken, to be fixed)

  if (orientation == UnknownOrientation) {
    cout << "WARNING: broken\n";
    orientation = orig->guessOrientation(brain, 10000.0 / (orig->dx * orig->dx));
    if (verbose) {
      cout << "guessed orientation: ";
      switch(orientation) {
      case SagittalOrientation: cout << "sagittal\n"; break;
      case AxialOrientation: cout << "axial\n"; break;
      case CoronalOrientation: cout << "coronal\n"; break;
      default: cout << endl;
      }
    }
  }

  if (orientation!=SagittalOrientation) {
    T4 a1 = sagTrans(orientation, orig);
    orig->transformLI(a1);
    brain->transformNN(a1);
  }

  // align brain
  trans.identity();
  planez = orig->darkerSlice(brain, (int) (10000.0 / (orig->dx * orig->dx)) );
  orig->planeFit( planez, brain, trans, ntrans, out, verbose);

  delete brain;

  // fix Z rotation
  R3 v[3];
  T4 ar;
  float za;
  
  v[0].set(0,0,0);
  v[1].set(0,-1,0);
  v[0] = trans.apply(v[0]);
  v[1] = trans.apply(v[1]);
  v[1] -= v[0];
  v[1].Z = 0.0;
  v[1].normalize();
  v[2].set(0,-1,0);
  za = v[2].inner(v[1]);
  if (za > 1.0) za = 1.0;
  za = 180.0 * acos( za ) / M_PI;
  if (v[0].X > 0.0) za = -za;
  //cout << "za = " << za << endl;

  ar.zrot(za, orig->W/2, orig->H/2, orig->D/2);
  trans *= ar;

  T[2] = Timestamp::now();
  delete orig;
  tseg = T[1]-T[0];
  tali = T[2]-T[1];

  if (verbose) {
    cout << "output plane: Z=" << planez << " ";
    trans.printLine();
  }
}

void align2(Volume<int> *input,
	    const char *out, 
	    const char *debugprefix,
	    VolumeOrientation orientation,
	    int &ntrans,
	    T4 &trans, 
	    int &planez,
	    double &tseg,
	    double &tali,
	    bool verbose,
	    float scale)
{
  Timestamp T[3];

  // segment brain
  Volume<int>  *enh, *grad, *hsi, *saved_orig=NULL;
  Volume<char> *marker, *brain, *tmp, *hsc;
  char x[512];
  T[0] = Timestamp::now();
  Volume<int> *orig = input->isometricInterpolation();

  if (verbose) {
    cout << "[interp] ";
    fflush(stdout);
  }

  if (scale != 1.0f) {
    hsi = orig->interpolate(orig->dx / scale,
			    orig->dy / scale,
			    orig->dz / scale);
    saved_orig = orig;
    orig = hsi;
    
    if (verbose) {
      cout << "[scale=" << scale << "] ";
      fflush(stdout);
    }
  }

  marker = orig->brainMarkerComp();

  if (verbose) {
    cout << "[markers] ";
    fflush(stdout);
  }

  if (debugprefix!=NULL) {
    sprintf(x,"%s-orig.scn",debugprefix);
    orig->writeSCN(x);
    sprintf(x,"%s-marker.scn",debugprefix);
    marker->writeSCN(x);
  }

  enh    = orig->otsuEnhance();
  grad   = enh->featureGradient();

  if (verbose) {
    cout << "[grad] ";
    fflush(stdout);
  }

  if (debugprefix!=NULL) {
    sprintf(x,"%s-grad.scn",debugprefix);
    grad->writeSCN(x);
  }

  delete enh;
  tmp    = grad->treePruning(marker);

  if (verbose) {
    cout << "[tp] ";
    fflush(stdout);
  }

  brain  = tmp->binaryClose(10.0/(orig->dx));


  if (debugprefix!=NULL) {
    sprintf(x,"%s-seg1.scn",debugprefix);
    tmp->writeSCN(x);
    sprintf(x,"%s-seg2.scn",debugprefix);
    brain->writeSCN(x);
  }

  delete tmp;
  delete marker;
  delete grad;

  // compute CSF EDT
  Volume<char> *csf;
  int otsu;
  otsu = orig->computeOtsu();
  csf  = orig->binaryThreshold(otsu,true); // 1=csf, 0=non-csf
  csf->binaryIntersection(brain);

  if (debugprefix!=NULL) {
    sprintf(x,"%s-csf1.scn",debugprefix);
    csf->writeSCN(x);
  }

  if (verbose) {
    cout << "[csfmask] ";
    fflush(stdout);
  }

  if (scale!=1.0f) {
    if (debugprefix!=NULL) {
      sprintf(x,"%s-hseg.scn",debugprefix);
      brain->writeSCN(x);
    }

    hsc = brain->interpolate(brain->dx * scale,
			     brain->dy * scale,
			     brain->dz * scale);
    delete brain;
    brain = hsc;
    hsc = NULL;

    hsc = csf->interpolate(csf->dx * scale,
			   csf->dy * scale,
			   csf->dz * scale);
    delete csf;
    csf = hsc;

    delete orig;
    orig = saved_orig;
    saved_orig = NULL;

    if (verbose) {
      cout << "[scale] ";
      fflush(stdout);
    }
  }

  if (verbose) cout << endl;
  
  if (debugprefix!=NULL) {
    sprintf(x,"%s-seg3.scn",debugprefix);
    brain->writeSCN(x);
    sprintf(x,"%s-csf2.scn",debugprefix);
    csf->writeSCN(x);
  }

  brain->incrementBorder();
  T[1] = Timestamp::now();

  // reorient if needed (broken, to be fixed)

  if (orientation == UnknownOrientation) {
    cout << "WARNING: broken\n";
    orientation = orig->guessOrientation(brain, 10000.0 / (orig->dx * orig->dx));
    if (verbose) {
      cout << "guessed orientation: ";
      switch(orientation) {
      case SagittalOrientation: cout << "sagittal\n"; break;
      case AxialOrientation: cout << "axial\n"; break;
      case CoronalOrientation: cout << "coronal\n"; break;
      default: cout << endl;
      }
    }
  }

  if (orientation!=SagittalOrientation) {
    T4 a1 = sagTrans(orientation, orig);
    orig->transformLI(a1);
    brain->transformNN(a1);
    csf->transformNN(a1);
  }

  // edt
  int i;
  csf->incrementBorder();

  /*
  for(i=0;i<csf->N;i++)
    if (csf->voxel(i) == 1)
      csf->voxel(i) = 2;
  */

  for(i=0;i<csf->N;i++)
    if (csf->voxel(i) == 0) csf->voxel(i) = 1;

  Volume<int> *csfedt = csf->edt();
  Volume<float> *csfedt2;

  csfedt2 = new Volume<float>(csfedt->W,csfedt->H,csfedt->D);
  for(i=0;i<csfedt->N;i++) {
    csfedt2->voxel(i) = (float) sqrt(csfedt->voxel(i));
    csfedt->voxel(i) = (int) csfedt2->voxel(i);
  }

  if (verbose)
    cout << "[edt]\n";

  if (debugprefix!=NULL) {
    sprintf(x,"%s-edt.scn",debugprefix);
    csfedt->writeSCN(x);
  }
  delete csfedt;

  // align brain
  printf("brain={%d,%d,%d}, csfedt2={%d,%d,%d}, orig={%d,%d,%d}\n",
	 brain->W,brain->H,brain->D,
	 csfedt2->W,csfedt2->H,csfedt2->D,
	 orig->W,orig->H,orig->D);

  trans.identity();
  planez = orig->bestSlice2(brain, csfedt2,  
			    (int) (10000.0 / (orig->dx * orig->dx)));
  orig->planeFit2( planez, brain, csfedt2, trans, ntrans, out, verbose);

  delete brain;
  delete csfedt2;

  // fix Z rotation
  R3 v[3];
  T4 ar;
  float za;
  
  v[0].set(0,0,0);
  v[1].set(0,-1,0);
  v[0] = trans.apply(v[0]);
  v[1] = trans.apply(v[1]);
  v[1] -= v[0];
  v[1].Z = 0.0;
  v[1].normalize();
  v[2].set(0,-1,0);
  za = v[2].inner(v[1]);
  if (za > 1.0) za = 1.0;
  za = 180.0 * acos( za ) / M_PI;
  if (v[0].X > 0.0) za = -za;
  //cout << "za = " << za << endl;

  ar.zrot(za, orig->W/2, orig->H/2, orig->D/2);
  trans *= ar;

  T[2] = Timestamp::now();
  delete orig;
  tseg = T[1]-T[0];
  tali = T[2]-T[1];

  if (verbose) {
    cout << "output plane: Z=" << planez << " ";
    trans.printLine();
  }
}


char *namestrip(const char *src) {
  static char n[1000];
  char *p;

  strcpy(n,src);
  p = strchr(n,'.');
  if (p!=NULL) *p=0;
  return n;
}

T4 & randomTrans(int cx, int cy, int cz) {

  T4 tmp;
  static T4 trans;
  int i,j;
  float k;

  trans.identity();
  tmp.identity();

  for(i=0;i<6;i++) {
    j = rand() % 6;
    k = 2.0 * (((rand() % 200) / 100.0) - 1.0);
    
    //    printf("j=%d k=%f\n",j,k);

    switch(j) {
    case 0: tmp.xrot(3*k,cx,cy,cz); break;
    case 1: tmp.yrot(3*k,cx,cy,cz); break;
    case 2: tmp.zrot(3*k,cx,cy,cz); break;
    case 3: tmp.translate(k,0,0); break;
    case 4: tmp.translate(0,k,0); break;
    case 5: tmp.translate(0,0,k); break;
    }

    trans *= tmp;
  }
  return(trans);
}

Volume<int> * toSagittal(Volume<int> *src, 
			 VolumeOrientation orientation) 
{
  Volume<int> *dest;
  int i,j,k;

  if (src->dx != src->dy || src->dy != src->dz) {
    cerr << "ERROR: toSagittal requires isotropic data.\n";
    return NULL;
  }

  switch(orientation) {

  case SagittalOrientation:
    return(new Volume<int>(src)); // just make a copy

  case AxialOrientation:

    dest = new Volume<int>( src->H, src->D, src->W,
			    src->dx, src->dx, src->dx );
    if (!dest->ok()) return dest;

    for(k=0;k<src->D;k++)
      for(j=0;j<src->H;j++)
	for(i=0;i<src->W;i++)
	  dest->voxel(src->H - j - 1,src->D - k - 1,i) = src->voxel(i,j,k);

    return dest;

  case CoronalOrientation:

    dest = new Volume<int>( src->D, src->H, src->W,
			    src->dx, src->dx, src->dx );
    if (!dest->ok()) return dest;

    // todo: check directions
    for(k=0;k<src->D;k++)
      for(j=0;j<src->H;j++)
	for(i=0;i<src->W;i++)
	  dest->voxel(src->D - k - 1,j,i) = src->voxel(i,j,k);

    return dest;

  default:
    return NULL;
  }
}

void align3(Volume<int> *input,
	    const char *out, 
	    const char *debugprefix,
	    VolumeOrientation orientation,
	    int &ntrans,
	    T4 &trans, 
	    int &planez,
	    double &tseg,
	    double &tali,
	    bool verbose,
	    int maxdelta)
{
  Timestamp T[2];
  T[0] = Timestamp::now();

  Volume<int> *orig = input->isometricInterpolation();

  if (verbose) {
    cout << "[interp] ";
    fflush(stdout);
  }

  // reorient if needed (broken, to be fixed)

  if (orientation!=SagittalOrientation) {
    T4 a1 = sagTrans(orientation, orig);
    orig->transformLI(a1);
  }

  int v1;
  Triangle v2,v3;
  v1 = orig->volkauFirst();
  
  if (verbose) {
    cout << "[volkau1] ";
    fflush(stdout);
  }

  v2 = orig->volkauSecond(v1);  

  if (verbose) {
    cout << "[volkau2] ";
    fflush(stdout);
  }

  v3 = orig->volkauThird(v2,maxdelta);

  if (verbose) {
    cout << "[volkau3] ";
    fflush(stdout);
  }

  // find transform for the plane defined by the triangle v3
  R3 n1,n2,rot,ab,ac;
  T4 r1,r2,t1;
  double ang;

  ab = v3.B - v3.A;
  ac = v3.C - v3.A;
  n1 = ab.cross(ac);
  n2.set(0,0,1);
  n1.normalize();
  n2.normalize();
  rot = n1.cross(n2);
  rot.normalize();
  ang = acos(n1.inner(n2)) * 180.0 / M_PI;

  /*
  cout << "\nang = " << ang << endl;
  v3.A.print();
  v3.B.print();
  v3.C.print();
  */

  r1.axisrot(-ang,rot);
  r2.axisrot(ang,rot);
  R3 c,d;
  c.set(orig->W/2,orig->H/2,orig->planeZ(orig->W/2,orig->H/2,v3.A,v3.B,v3.C));
  d = r2.apply(c);
  t1.translate(d.X-orig->W/2,d.Y-orig->H/2,0.0);
  t1 *= r1;

  T[1] = Timestamp::now();

  tseg = 0;
  tali = T[1] - T[0];
  planez = (int) d.Z;
  trans = t1;
  ntrans = 3;

  if (debugprefix!=NULL) {
    int i,j,k,dv;
    char name[512];
    dv=orig->maximum() / 2;
    for(i=0;i<orig->W;i++)
      for(j=0;j<orig->H;j++) {
	k = (int) (orig->planeZ(i,j,v3.A,v3.B,v3.C));
	if (orig->valid(i,j,k)) 
	  orig->voxel(i,j,k) += dv;
      }
    sprintf(name,"%s-volkau.scn",debugprefix);
    orig->writeSCN(name,16,true);
  }

  if (verbose)
    cout << endl;

  delete orig;

}


} //end MSP namespace


