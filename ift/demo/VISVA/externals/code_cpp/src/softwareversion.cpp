
#include "softwareversion.h"


SoftwareVersion :: SoftwareVersion(int major,
				   int minor,
				   int revision){
  this->major = major;
  this->minor = minor;
  this->revision = revision;
}

SoftwareVersion :: SoftwareVersion(  SoftwareVersion& ver){
  this->major = ver.major;
  this->minor = ver.minor;
  this->revision = ver.revision;
}

void SoftwareVersion :: GetVersion(char *ver){
  sprintf(ver,"%d.%d.%d",major,minor,revision);
}

int SoftwareVersion :: GetMajor(){
  return major;
}

int SoftwareVersion :: GetMinor(){
  return minor;
}

int SoftwareVersion :: GetRevision(){
  return revision;
}


