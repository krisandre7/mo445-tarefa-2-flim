#ifndef msp_timestamp_h
#define msp_timestamp_h 1

#include <sys/time.h>

namespace MSP{

//! A timestamp for measuring time intervals
class Timestamp {
 public:
  //! Constructor, creates a timestamp with values (sec,usec) for
  //! seconds and microseconds elapsed since last midnight
  Timestamp(int sec, int usec) { S=sec; U=usec; }

  //! Constructor, creates a timestamp with the current time
  Timestamp() { (*this) = Timestamp::now(); }

  //! Assignment operator
  Timestamp & operator=(Timestamp &t) { S = t.S; U=t.U; return(*this); }

  //! Subtraction, returns the number of seconds elapsed between this
  //! timestamp and t.
  double operator-(Timestamp &t) {
    int msec;
    double sec;
    msec = 1000*(S-t.S);
    msec += (U/1000 - t.U/1000);
    sec = (double) (msec / 1000.0);
    return sec;
  }

  //! Returns the current timestamp
  static Timestamp & now() { 
    static Timestamp t(0,0);
    struct timeval tv;
    gettimeofday(&tv,0);
    t.S = (int) tv.tv_sec;
    t.U = (int) tv.tv_usec;
    return(t);
  }

 private:
  int S,U;

};

} //end MSP namespace

#endif

