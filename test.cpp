
// project headers
#include "my_data.h"
#include "my_timeseries.h"
// standard headers
#include <iostream>

int main(int argc, char* argv[]) {

  My::Timeseries ts;
//  ts.readFile("../data/ACE_SWEPAM_Data.txt", "ydhms");
  ts.readFile("../data/ace_epam_240.dat", "ymdhms");

  My::Time beginTime("2008-02-01 10:00:00");
  My::Time endTime("2008-02-05 18:40:00");

  ts.filter(beginTime, endTime);

  std::cout << ts.col(0)(0) << " " << ts.col(1)(1) << std::endl;
  std::cout << ts.cols().size() << std::endl;
}

