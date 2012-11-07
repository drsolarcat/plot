
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
    _data[i].head(endIndex-beginIndex+1) =
      _data[i].segment(beginIndex, endIndex-beginIndex+1);
    _data[i].conservativeResize(endIndex-beginIndex+1);
  }

  return *this; // chained method
}

My::Timeseries1D::Timeseries1D(My::Timeseries ts, int col) {
  _data.push_back(ts.col(0));
  _data.push_back(ts.col(col));
}

My::Timeseries1D& My::Timeseries1D::filter(std::string logic, double val) {
  int f;
  if (logic == "<=" || logic == "lte") {
    f = 1;
  } else if (logic == "<" || logic == "lt") {
    f = 2;
  } else if (logic == ">" || logic == "gt") {
    f = 3;
  } else if (logic == ">=" || logic == "gte") {
    f = 4;
  } else {
    f = 0;
  }
  int i = 0;
  if (f != 0) {
    while (true) {
      if ((f == 1 & _data[1](i) <= val) | (f == 2 & _data[1](i) <  val) |
          (f == 3 & _data[1](i) >  val) | (f == 4 & _data[1](i) >= val)) {
        i++;
      } else {
        _data[0].segment(i, _data[0].size()-i-1) =
          _data[0].segment(i+1, _data[0].size()-i-1);
        _data[0].conservativeResize(_data[0].size()-1);
        _data[1].segment(i, _data[1].size()-i-1) =
          _data[1].segment(i+1, _data[1].size()-i-1);
        _data[1].conservativeResize(_data[1].size()-1);
      }
      if (i >= _data[0].size()) break;
    }
  }
}

