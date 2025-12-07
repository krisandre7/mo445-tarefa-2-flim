
#include "msp_align.h"

namespace MSP{

void usage() {
  cerr << endl;
  cerr << "usage: align [options] input-scn output-scn\n\n";
  cerr << "Optional parameters:\n\n";
  cerr << " -v               : verbose operation\n";
  cerr << " -hs factor       : scale volume by factor for segmentation (default = 1.0).\n";
  cerr << " -a number        : algorithm selection:\n";
  cerr << "                      1 - morphology + darkest plane (default)\n";
  cerr << "                      2 - CSF distance-based score\n";
  cerr << "                      3 - Volkau'06 (delta=3)\n";
  cerr << "                      4 - Volkau'06 (delta=10)\n";
  cerr << " -vs value        : interpolate image to given isotropic voxel size (mm).\n";
  cerr << "                      (default: interpolate to minimum voxel size)\n";
  cerr << " -d prefix        : save intermediary images with given prefix.\n";
  cerr << " -sag / -sagittal : input is in sagittal orientation (default)\n";
  cerr << " -cor / -coronal  : input is in coronal orientation\n";
  cerr << " -axi / -axial    : input is in axial orientation\n";
  cerr << " -nopad           : transform without padding (default: pad to fit)\n";
  cerr << " -draw mode       : draw plane on output (default: do not draw)\n";
  cerr << "                       mode=0 : solid plane\n";
  cerr << "                       mode=1 : translucid plane\n";
  cerr << "                       mode=2 : 2x2 checkers plane\n";
  cerr << "                       mode=3 : 4x4 checkers plane\n";
  cerr << "                       mode=4 : 8x8 checkers plane\n";
  cerr << " -more in1,out1,in2,out2,...  : transform in-volumes into out-volumes\n";
  cerr << "                                with the same alignment transform.\n";
  cerr << endl;
}

void imgAlign(const char *in, 
	      const char *out, 
	      T4 &itrans, 
	      VolumeOrientation orientation,
	      int iw,int ih,int id, 
	      float idx, float idy, float idz,
	      float vs, bool verbose, bool pad)
{
  Volume<int> *img, *img2, *img3, *img4;

  img = new Volume<int>(in);
  if (!img->ok()) {
    cerr << "** error loading " << in << ", skipping\n";
    delete img;
    return;
  }

  if (img->W != iw || img->H != ih || img->D != id) {
    cerr << "** dimensions do not match for " << in << ", skipping\n";
    delete img;
    return;
  }
  
  if (verbose) cout << "loaded " << in << endl;

  img->dx = idx;
  img->dy = idy;
  img->dz = idz;
  cout << "interpolating " << in << endl;

  img2 = img->interpolate(vs,vs,vs);
  delete img;
  
  if (orientation != SagittalOrientation) {
    if (verbose) cout << "reorienting...\n";	
    img4 = toSagittal(img2, orientation);
    delete img2;
    img2 = img4;
    img4 = NULL;
  }
  
  if (verbose) cout << "applying transform...\n";
  if (pad) {
    img3 = img2->paddedTransformLI(itrans);
    delete img2;
    img2 = img3;
  } else {
    img2->transformLI(itrans);
  }
  if (verbose) cout << "writing " << out << endl;
  img2->writeSCN(out,-1,false);
  delete img2;
}


int msp_align(char *command){
  int argc=0;
  char *argv[1024];
  char *aux;

  aux = strtok(command, " \t\n");
  while(aux!=NULL){
    argv[argc] = strdup(aux);
    argc++;
    aux = strtok(NULL, " \t\n");
  }

//int main(int argc, char **argv) {
  char *input=NULL, *output=NULL, *debug=NULL, *more=NULL;
  bool verbose = false, pad = true;
  float scale = 1.0f;
  int algo=1;
  int i,j,k;
  float vs=-1.0;
  VolumeOrientation orientation = SagittalOrientation;
  int draw = -1;

  int   iw,ih,id;
  float idx,idy,idz;

  for(i=1;i<argc;i++) {
    if (!strcmp(argv[i],"-v")) { verbose=true; continue; }
    if (i<argc-1 && !strcmp(argv[i],"-hs")) { scale=atof(argv[++i]); continue; }
    if (i<argc-1 && !strcmp(argv[i],"-more")) { more=strdup(argv[++i]); continue; }
    if (i<argc-1 && !strcmp(argv[i],"-d")) { debug=argv[++i]; continue; }
    if (i<argc-1 && !strcmp(argv[i],"-a")) { algo=atoi(argv[++i]); continue; }
    if (i<argc-1 && !strcmp(argv[i],"-vs")) { vs=atof(argv[++i]); continue; }
    if (i<argc-1 && !strcmp(argv[i],"-draw")) { draw=atoi(argv[++i]); continue; }
    if (!strcmp(argv[i],"-nopad")) {
      pad = false;
      continue;
    }
    if (!strcmp(argv[i],"-sag") || !strcmp(argv[i],"-sagittal")) {
      orientation = SagittalOrientation;
      continue;
    }
    if (!strcmp(argv[i],"-cor") || !strcmp(argv[i],"-coronal")) {
      orientation = CoronalOrientation;
      continue;
    }
    if (!strcmp(argv[i],"-axi") || !strcmp(argv[i],"-axial")) {
      orientation = AxialOrientation;
      continue;
    }

    if (input==NULL)
      input = argv[i];
    else
      output = argv[i];
  }

  if (algo<1 || algo>4) {
    usage();
    return 1;
  }

  if (output==NULL) {
    usage();
    return 1;
  }

  if (verbose) cout << "loading " << input << endl;
  Volume<int> *inp = new Volume<int>(input);

  if (!inp->ok()) {
    cerr << "error loading " << input << endl;
    return 2;
  }

  iw = inp->W;
  ih = inp->H;
  id = inp->D;
  idx = inp->dx;
  idy = inp->dy;
  idz = inp->dz;

  if (verbose) cout << "interpolating\n";

  Volume<int> *interp;
  if (vs <= 0.0) {
    interp = inp->isometricInterpolation();
    vs = interp->dx;
  } else
    interp = inp->interpolate(vs,vs,vs);
  delete inp;

  if (!interp->ok()) {
    cerr << "error: unable to interpolate\n";
    return 2;
  }

  if (verbose) cout << "voxel size = " << vs << " mm\n";

  int ntrans, pz;
  T4 trans;
  double t1,t2;
  
  if (orientation != SagittalOrientation) {
    Volume<int> *tmp = interp;

    if (verbose)
      cout << "reorienting...\n";

    interp = toSagittal(tmp, orientation);
    delete tmp;
  }

  if (verbose) cout << "aligning with algorithm " << algo << " (scale=" << scale << ")...\n";

  switch(algo) {
  case 1:
    align(interp,NULL,debug,SagittalOrientation,ntrans,trans,pz,t1,t2,verbose,scale);
    break;
  case 2:
    align2(interp,NULL,debug,SagittalOrientation,ntrans,trans,pz,t1,t2,verbose,scale);
    break;
  case 3:
    align3(interp,NULL,debug,SagittalOrientation,ntrans,trans,pz,t1,t2,verbose,3);
    break;
  case 4:
    align3(interp,NULL,debug,SagittalOrientation,ntrans,trans,pz,t1,t2,verbose,10);

    break;
  default: return 99; // impossible
  }

  T4 itrans, tz;
  R3 P;

  itrans = trans;
  itrans.invert();
  tz.translate(0,0,(interp->D/2)-pz);
  itrans *= tz;

  if (verbose) cout << "inverse transform...\n";
  if (pad) {
    Volume<int> * tmp;
    tmp = interp->paddedTransformLI(itrans);
    delete interp;
    interp = tmp;
  } else {
    interp->transformLI(itrans);
  }

  int val;
  switch(draw) {
  case 0: // solid plane
    val = interp->maximum();
    k = interp->D / 2;
    for(j=0;j<interp->H;j++)
      for(i=0;i<interp->W;i++)
	interp->voxel(i,j,k) = val;
    break;
  case 1: // translucid plane
    val = interp->maximum() / 2;
    k = interp->D / 2;
    for(j=0;j<interp->H;j++)
      for(i=0;i<interp->W;i++)
	interp->voxel(i,j,k) = val + interp->voxel(i,j,k);
    break;
  case 2: // interleaved plane 2x2
    val = interp->maximum();
    k = interp->D / 2;
    for(j=0;j<interp->H;j++)
      for(i=0;i<interp->W;i++)
	if ( (i+j) % 2 )
	  interp->voxel(i,j,k) = val;
    break;
  case 3: // interleaved plane 4x4
    val = interp->maximum();
    k = interp->D / 2;
    for(j=0;j<interp->H;j++)
      for(i=0;i<interp->W;i++)
	if ( ((i/2)+(j/2)) % 2 )
	  interp->voxel(i,j,k) = val;
    break;
  case 4: // interleaved plane 8x8
    val = interp->maximum();
    k = interp->D / 2;
    for(j=0;j<interp->H;j++)
      for(i=0;i<interp->W;i++)
	if ( ((i/4)+(j/4)) % 2 )
	  interp->voxel(i,j,k) = val;
    break;
  }

  if (verbose) cout << "writing " << output << endl;
  interp->writeSCN(output,-1,false);

  if (verbose) {
    cout << "Time segment = " << t1 << " secs.\n";
    cout << "Time align   = " << t2 << " secs.\n";
    cout << "Time total   = " << (t1+t2) << " secs.\n";
    cout << "Iterations   = " << ntrans << "\n";
  }
  
  delete interp;

  if (more!=NULL) {

    char *p;

    if (verbose)
      cout << "processing -more volumes:\n";

    p = strtok(more, ",");

    char *in, *out;
    while(p!=NULL) {
      in = strdup(p);
      p = strtok(NULL, ",");
      if (p==NULL) { 
	free(in); 
	break; 
      }
      out = strdup(p);

      imgAlign(in,out,
	       itrans,orientation,
	       iw,ih,id,
	       idx,idy,idz,
	       vs,verbose,pad);
      p = strtok(NULL, ",");
      free(in);
      free(out);
    }
    free(more);
  }

  //-----------------------
  for(i=0; i<argc; i++)
    free(argv[i]);

  return 0;
}


} //end MSP namespace


