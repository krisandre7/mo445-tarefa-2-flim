
#ifndef _SOFTWAREVERSION_H_
#define _SOFTWAREVERSION_H_

extern "C" {
#include "oldift.h"
}

class SoftwareVersion {
public:
  SoftwareVersion(int major,
		  int minor,
		  int revision);
  SoftwareVersion(  SoftwareVersion& ver);
  void GetVersion(char *ver);
  int  GetMajor();
  int  GetMinor();
  int  GetRevision();
private:
  int major;
  int minor;
  int revision;
};

#endif



