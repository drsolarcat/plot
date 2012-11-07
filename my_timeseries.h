
#ifndef MY_TIMESERIES_H
#define MY_TIMESERIES_H

//project headers
#include "my_data.h"
#include "my_time.h"

namespace My {
  class Timeseries : public Data {
    public:
      Timeseries& readFile(std::string, std::string);
      Timeseries& filter(Time, Time);
  };
}

namespace My {
  class Timeseries1D : public Timeseries {
    public:
      Timeseries1D(Timeseries, int);
      Timeseries1D& filter(std::string, double);
  };
}

#endif

