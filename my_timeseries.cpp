
// project headers
#include "my_timeseries.h"
#include "my_data.h"
// library headers
#include <eigen3/Eigen/Dense>
// standard headers
#include <string>

My::Timeseries& My::Timeseries::readFile(std::string dataFilePath,
                                         std::string format) {
  My::Data::readFile(dataFilePath);

  _data.insert(_data.begin(), Eigen::VectorXd::Zero(_data[0].size()));
  if (format == "ydhms") {
    for (int i = 0; i < _data[0].size(); i++) {
      My::Time currentTime(_data[1](i), _data[2](i), _data[3](i), _data[4](i),
                           _data[5](i));
      _data[0](i) = currentTime.unixtime();
    }
    _data.erase(_data.begin()+1, _data.begin()+6);
  } else if (format == "ymdhms") {
    for (int i = 0; i < _data[0].size(); i++) {
      My::Time currentTime(_data[1](i), _data[2](i), _data[3](i), _data[4](i),
                           _data[5](i), _data[6](i));
      _data[0](i) = currentTime.unixtime();
    }
    _data.erase(_data.begin()+1, _data.begin()+7);
  }

  return *this;
}

My::Timeseries& My::Timeseries::filter(My::Time beginTime, My::Time endTime) {
  int beginIndex, endIndex; // indices for minimum and maximum limits of the
                            // data interval
  My::Time currentTime; // Time object for storing of the current time

  // iterate through data array
  for (int i = 0; i < _data[0].size(); i++) {
    // initialize current Time object
    currentTime = My::Time((int)_data[0](i), "unix");
    // searching for lower index
    if (currentTime < beginTime) beginIndex = i;
    // searching for upper index
    if (currentTime > endTime) {
      endIndex = i;
      break; // we are already above the upper limit
    }
  } // end of iteration through data
  beginIndex++;
  endIndex--;

  // erase data outside desired time interval
  for (int i = 0; i < _data.size(); i++) {
    _data[i].head(endIndex-beginIndex+1) = _data[i].segment(beginIndex, endIndex-beginIndex+1);
    _data[i].conservativeResize(endIndex-beginIndex+1);
  }

  return *this; // chained method
}

