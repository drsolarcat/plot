
// project headers
#include "my_data.h"
// library headers
#include <eigen3/Eigen/Dense>
// standard headers
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

My::Data& My::Data::readFile(std::string dataFilePath) {

  /* variables */

  std::ifstream dataFileStream; // stream from a file
  std::string dataFileLine; // a single line from the file as a string
  // a sample line from the file, to be used for counting the columns
  std::string dataFileSampleLine;
  std::istringstream dataFileLineStream; // a stream from the line from the file
  int nRows = 0; // file rows counter
  int nColumns = 0; // file column counter

  /* count the number of rows */

  // open data file as a stream
  dataFileStream.open(dataFilePath.c_str());
  // check if the file was opened correctly
  if (dataFileStream.is_open()) {
    // iterate through the file lines
    while (dataFileStream.good()) {
      // get line from the file as a string
      getline(dataFileStream, dataFileLine);
      if (!dataFileLine.empty() && // check that it's not empty
          dataFileLine[0] != '#') // and not commented out
      {
        // save one sample line from the file to count file columns later
        if (nRows == 0) dataFileSampleLine = dataFileLine;
        nRows++; // +1 line
      }
    }
  }

  /* count the number of columns */

  // use the sample line of data as a source for the string stream
  dataFileLineStream.str(dataFileSampleLine);
  double sampleData; // sample data
  // clean data vector, if it was filled previously
  _data.clear();
  // iterate through columns of data
  while (dataFileLineStream >> sampleData) {
    nColumns++; // +1 column
    _data.push_back(Eigen::VectorXd::Zero(nRows)); // add a data vector
  }

  /* read the data */

  // go to the beginning of the file again
  dataFileStream.clear();
  dataFileStream.seekg(0, std::ios::beg);
  // check if the file was opened correctly
  if (dataFileStream.is_open()) {
    int i = 0; // number of file line
    // start iterating through data file line, one timestamp at a time
    while (dataFileStream.good()) {
      // get line from the file as a string
      getline(dataFileStream, dataFileLine);
      // check if the line contains actually data
      if (!dataFileLine.empty() && // check that it's not empty
          dataFileLine[0] != '#') // and not commented out
      {
        // return the stream iterator to the beginning of the line
        dataFileLineStream.clear();
        // use the next line of data as a source for the string stream
        dataFileLineStream.str(dataFileLine);
        // parse the data from the stream of line of the data file
        for (int k = 0; k < nColumns; k++) dataFileLineStream >> _data[k](i);
        i++; // +1 line
      }
    } // end of iteration through the lines of the data file
  }
}

