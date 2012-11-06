
#ifndef MY_DATA_H
#define MY_DATA_H

// library headers
#include <eigen3/Eigen/Dense>
// standard headers
#include <string>
#include <vector>

namespace My {
  class Data {
    protected:
      std::vector<Eigen::VectorXd> _data;
    public:
      Data& readFile(std::string);
      const std::vector<Eigen::VectorXd>& cols() const {return _data;}
      const Eigen::VectorXd& col(int i) const {return _data[i];}
  };
}

#endif

