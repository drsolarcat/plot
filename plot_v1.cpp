/*
Spacecraft data plotter

version 1.0

Alexey Isavnin
*/

// project headers
#include "my_time.h"
#include "data.h"
// library headers
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/regex.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <python2.7/Python.h>
#include <python2.7/site-packages/numpy/core/include/numpy/arrayobject.h>
#include <eigen3/Eigen/Dense>
// standard headers
#include <string>
#include <iostream>
#include <typeinfo>


// used for handling of command line arguments
namespace po = boost::program_options;
// used for iterating through nested sets of arguments
namespace pt = boost::property_tree;

int main(int argc, char* argv[]) {

  // program description
  po::options_description desc("Allowed options");
  desc.add_options()
    ("version,v", "print version")
    ("help,h", "produce this help message")
    ("start-date,s", po::value<std::string>()->composing(),
     "start date in the format \"yyyy-mm-dd HH:MM:SS\"")
    ("end-date,e", po::value<std::string>()->composing(),
     "end date in the format \"yyyy-mm-dd HH:MM:SS\"")
    ("ace", "use ACE data")
    ("wind", "use WIND data")
    ("stereo-a", "use STEREO-A data")
    ("stereo-b", "use STEREO-B data")
    ("data,d", po::value<std::string>()->composing(),
     "data to be plotted")
  ;

  po::variables_map vm; // variables map
  // associate command line arguments with the variables map
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // output program version
  if (vm.count("version")) {
    std::cout << "1.0" << std::endl;
    return 1;
  }

  // output the help message
  if (vm.count("help") || vm.size() == 0) {
    std::cout << desc << "\n";
    return 1;
  }

  // assign start and end datetimes
  My::Time dateStart, dateEnd;
  if (vm.count("start-date")) {
    dateStart = My::Time(vm["start-date"].as<std::string>());
  } else {
    std::cout << "start date is not set" << std::endl;
    return 0;
  }
  if (vm.count("end-date")) {
    dateEnd = My::Time(vm["end-date"].as<std::string>());
  } else {
    std::cout << "end date is not set" << std::endl;
    return 0;
  }

  // construct a data tree of parameters to be plotted
  pt::ptree dataTree;
  if (vm.count("data")) {
    std::stringstream dataStringStream;
    dataStringStream << "[" <<
      boost::regex_replace(vm["data"].as<std::string>(),
                           boost::regex("([a-zA-Z/]+)"), "\"$1\"",
                           boost::match_default | boost::format_all) << "]";
    pt::read_json(dataStringStream, dataTree);
  } else {
    std::cout << "data selection is not set" << std::endl;
    return 0;
  }

  std::string spacecraft;
  std::string dataDir("../icme/res");
  std::ostringstream dataPathStream;
  std::string dataPath;
  dataPathStream << dataDir << '/';
  if (vm.count("ace")) {
    spacecraft = "ACE";
    dataPathStream << "ace_240.dat";
  } else if (vm.count("wind")) {
    spacecraft = "WIND";
    dataPathStream << "wind_240.dat";
  } else if (vm.count("stereo-a")) {
    spacecraft = "STA";
    dataPathStream << "stereo_a_240.dat";
  } else if (vm.count("stereo-b")) {
    spacecraft = "STB";
    dataPathStream << "stereo_b_240.dat";
  } else {
    std::cout << "spacecraft is not set" << std::endl;
    return 0;
  }

  dataPath = dataPathStream.str();
  Data data;
  data.readFile(dataPath, dateStart, dateEnd);
  double Te = 130000; // K
  double kb = 1.380648e-23; // J/K
  double mu0 = 1.256637e-6; // N/A^2
  std::map<std::string, Eigen::MatrixXd> dataMap;

  boost::tokenizer<boost::char_separator<char> >
    tk(vm["data"].as<std::string>(),
       boost::char_separator<char>(",[] "));
  BOOST_FOREACH (const std::string& t, tk) {

    if (t == "B") {
      dataMap[t] = data.cols().B;
    } else if (t == "Bx" || t == "Br") {
      dataMap[t] = data.cols().Bx;
    } else if (t == "By" || t == "Bt") {
      dataMap[t] = data.cols().By;
    } else if (t == "Bz" || t == "Bn") {
      dataMap[t] = data.cols().Bz;
    } else if (t == "Vp") {
      dataMap[t] = data.cols().Vp;
    } else if (t == "Vx" || t == "Vr") {
      dataMap[t] = data.cols().Vx;
    } else if (t == "Vy" || t == "Vt") {
      dataMap[t] = data.cols().Vy;
    } else if (t == "Vz" || t == "Vn") {
      dataMap[t] = data.cols().Vz;
    } else if (t == "Pth") {
      dataMap[t] = data.cols().Pth;
    } else if (t == "Np") {
      dataMap[t] = data.cols().Np;
    } else if (t == "Tp") {
      dataMap[t] = data.cols().Tp;
    } else if (t == "Vth") {
      dataMap[t] = data.cols().Vth;
    } else if (t == "beta") {
      dataMap[t] = data.cols().beta;
    } else if (t == "Pt") {
      dataMap[t] = data.cols().B.array().pow(2).matrix()/2/mu0+
        data.cols().Np*kb*Te+
        (data.cols().Np.array()*0.96*kb*data.cols().Tp.array()).matrix()+
        (data.cols().Np.array()*0.04*kb*data.cols().Tp.array()*4).matrix();
    } else if (t == "thetaB" or t == "phiB") {

      Eigen::VectorXd theta(data.rows().size());
      Eigen::VectorXd phi(data.rows().size());
      Eigen::VectorXd Bxy = (data.cols().By.array().pow(2)+
                      data.cols().Bx.array().pow(2)).sqrt().matrix();

      for (int i = 0; i < data.rows().size(); i++) {
        phi(i) = atan2(data.cols().By(i), data.cols().Bx(i));
        phi(i) = (phi(i) < 0 ? phi(i)+M_PI : phi(i));
        phi(i) = phi(i)*180/M_PI;
        theta(i) = atan(data.cols().Bz(i)/Bxy(i));
        theta(i) = theta(i)*180/M_PI;
      }

      if (t == "thetaB") {
        dataMap[t] = theta;
      } else if (t == "phiB") {
        dataMap[t] = phi;
      }
    } else if (t == "PA") {
      std::ifstream dataFileStream; // stream from the file with data
      std::string dataFileLine; // a single line from a file as a string
      std::istringstream dataFileLineStream; // a stream from a line from a file
      My::Time currentTime; // Time object for storing current time for comparison
      std::string dataFilePath = "../data/ace_epam_240.dat";
      int N = ceil((dateEnd.unixtime()-dateStart.unixtime())/240)+1;

      dataMap[t] = Eigen::VectorXd::Zero(20*N+20);
      dataMap[t](0) = 0+4.5;
      dataMap[t](1) = 9+4.5;
      dataMap[t](2) = 18+4.5;
      dataMap[t](3) = 27+4.5;
      dataMap[t](4) = 36+4.5;
      dataMap[t](5) = 45+4.5;
      dataMap[t](6) = 54+4.5;
      dataMap[t](7) = 63+4.5;
      dataMap[t](8) = 72+4.5;
      dataMap[t](9) = 81+4.5;
      dataMap[t](10) = 90+4.5;
      dataMap[t](11) = 99+4.5;
      dataMap[t](12) = 108+4.5;
      dataMap[t](13) = 117+4.5;
      dataMap[t](14) = 126+4.5;
      dataMap[t](15) = 135+4.5;
      dataMap[t](16) = 144+4.5;
      dataMap[t](17) = 153+4.5;
      dataMap[t](18) = 162+4.5;
      dataMap[t](19) = 171+4.5;

      int i = 0;

      // open data file as a stream
      dataFileStream.open(dataFilePath.c_str());
      // check if the file was opened correctly
      if (dataFileStream.is_open()) {
        // start iterating through data file line, one timestamp at a time
        while (dataFileStream.good()) {
          // get line from the file as a string
          getline(dataFileStream, dataFileLine);
          // check if the line contains actually data
          if (!dataFileLine.empty() && // check that it's not empty
              dataFileLine[0] != '#') // and not commented out
          {
            // clear string stream from the previus line of data
            dataFileLineStream.clear();
            // use the next line of data as a source for the string stream
            dataFileLineStream.str(dataFileLine);
            int year, month, day, hour, minute, second;
            double pa0d, pa9d, pa18d, pa27d, pa36d, pa45d, pa54d, pa63d, pa72d,
                   pa81d, pa90d, pa99d, pa108d, pa117d, pa126d, pa135d, pa144d,
                   pa153d, pa162d, pa171d;
            // parse the data from the stream of line of the data file
            dataFileLineStream >> year >> month >> day >> hour >> minute >>
              second >> pa0d >> pa9d >> pa18d >> pa27d >> pa36d >> pa45d >>
              pa54d >> pa63d >> pa72d >> pa81d >> pa90d >> pa99d >> pa108d >>
              pa117d >> pa126d >> pa135d >> pa144d >> pa153d >> pa162d >>
              pa171d;
            // initialize current Time object with time data
            currentTime = My::Time(year, month, day, hour, minute, second);
            if (currentTime < dateStart) { // before the minimum time limit
              continue; // miss it
            } else {
              dataMap[t](20+i) = pa0d;
              dataMap[t](20+N+i) = pa9d;
              dataMap[t](20+2*N+i) = pa18d;
              dataMap[t](20+3*N+i) = pa27d;
              dataMap[t](20+4*N+i) = pa36d;
              dataMap[t](20+5*N+i) = pa45d;
              dataMap[t](20+6*N+i) = pa54d;
              dataMap[t](20+7*N+i) = pa63d;
              dataMap[t](20+8*N+i) = pa72d;
              dataMap[t](20+9*N+i) = pa81d;
              dataMap[t](20+10*N+i) = pa90d;
              dataMap[t](20+11*N+i) = pa99d;
              dataMap[t](20+12*N+i) = pa108d;
              dataMap[t](20+13*N+i) = pa117d;
              dataMap[t](20+14*N+i) = pa126d;
              dataMap[t](20+15*N+i) = pa135d;
              dataMap[t](20+16*N+i) = pa144d;
              dataMap[t](20+17*N+i) = pa153d;
              dataMap[t](20+18*N+i) = pa162d;
              dataMap[t](20+19*N+i) = pa171d;
              i++;
            }
            if (currentTime > dateEnd) break; // after the maximum time limit
          }
        } // end of iteration through the lines of the data file
      }
    } else if (t == "AE") {
      std::ifstream dataFileStream; // stream from the file with data
      std::string dataFileLine; // a single line from a file as a string
      std::istringstream dataFileLineStream; // a stream from a line from a file
      My::Time currentTime; // Time object for storing current time for comparison
      std::string dataFilePath = "../data/ae_240.dat";
      int N = ceil((dateEnd.unixtime()-dateStart.unixtime())/240)+1;

      dataMap[t] = Eigen::VectorXd::Zero(N);

      int i = 0;

      My::Time dateStartAE = dateStart;
      My::Time dateEndAE = dateEnd;

      double dt = abs(1.5e9/data.cols().Vx.array().mean()); //s
      dateStartAE.add(dt, "second");
      dateEndAE.add(dt, "second");

      // open data file as a stream
      dataFileStream.open(dataFilePath.c_str());
      // check if the file was opened correctly
      if (dataFileStream.is_open()) {
        // start iterating through data file line, one timestamp at a time
        while (dataFileStream.good()) {
          // get line from the file as a string
          getline(dataFileStream, dataFileLine);
          // check if the line contains actually data
          if (!dataFileLine.empty() && // check that it's not empty
              dataFileLine[0] != '#') // and not commented out
          {
            // clear string stream from the previus line of data
            dataFileLineStream.clear();
            // use the next line of data as a source for the string stream
            dataFileLineStream.str(dataFileLine);
            int year, month, day, hour, minute, second;
            double ae;
            // parse the data from the stream of line of the data file
            dataFileLineStream >> year >> month >> day >> hour >> minute >>
              second >> ae;
            // initialize current Time object with time data
            currentTime = My::Time(year, month, day, hour, minute, second);
            if (currentTime < dateStartAE) { // before the minimum time limit
              continue; // miss it
            } else {
              dataMap[t](i) = ae;
              i++;
            }
            if (currentTime > dateEndAE) break; // after the maximum time limit
          }
        } // end of iteration through the lines of the data file
      }
    } else if (t == "sVp") {
      dataMap[t] = (data.cols().Vp.array()-data.cols().Vp.array().mean()).matrix();
    } else if (t == "sVx") {
      dataMap[t] = (data.cols().Vx.array()-data.cols().Vx.array().mean()).matrix();
    } else if (t == "sVy") {
      dataMap[t] = (data.cols().Vy.array()-data.cols().Vy.array().mean()).matrix();
    } else if (t == "sVz") {
      dataMap[t] = (data.cols().Vz.array()-data.cols().Vz.array().mean()).matrix();
    } else if (t == "He/p") {
      std::ifstream dataFileStream; // stream from the file with data
      std::string dataFileLine; // a single line from a file as a string
      std::istringstream dataFileLineStream; // a stream from a line from a file
      My::Time currentTime; // Time object for storing current time for comparison
      std::string dataFilePath = "../data/ACE_SWEPAM_Data.txt";

      int N = ceil((dateEnd.unixtime()-dateStart.unixtime())/3600)+10;

      dataMap[t] = Eigen::VectorXd::Zero(N);
      dataMap["t_He/p"] = Eigen::VectorXd::Zero(N);

      int i = 0;

      // open data file as a stream
      dataFileStream.open(dataFilePath.c_str());
      // check if the file was opened correctly
      if (dataFileStream.is_open()) {
        // start iterating through data file line, one timestamp at a time
        while (dataFileStream.good()) {
          // get line from the file as a string
          getline(dataFileStream, dataFileLine);
          // check if the line contains actually data
          if (!dataFileLine.empty() && // check that it's not empty
              dataFileLine[0] != '#') // and not commented out
          {
            // clear string stream from the previus line of data
            dataFileLineStream.clear();
            // use the next line of data as a source for the string stream
            dataFileLineStream.str(dataFileLine);
            double s;
            int year, doy, hour, minute, second;
            double v;
            // parse the data from the stream of line of the data file
            dataFileLineStream >> year >> doy >> hour >> minute >> s >> v;
            // initialize current Time object with time data
            second = s;
            currentTime = My::Time(year, doy, hour, minute, second);
            if (currentTime < dateStart) { // before the minimum time limit
              continue; // miss it
            } else {
              if (v > 0) {
                dataMap[t](i) = v;
                dataMap["t_He/p"](i) = currentTime.unixtime();
                i++;
              }
            }
            if (currentTime > dateEnd) break; // after the maximum time limit
          }
        } // end of iteration through the lines of the data file
      }
      dataMap[t].conservativeResize(i,1);
      dataMap["t_He/p"].conservativeResize(i,1);
    }
  }

  // launch the python interpreter
  Py_Initialize();
  // this macro is defined be NumPy and must be included
  import_array1(-1);
  // update module search path
  PyObject *sys_module = PyImport_ImportModule("sys");
  PyObject *sys_dict = PyModule_GetDict(sys_module);
  PyObject *sys_path = PyMapping_GetItemString(sys_dict, "path");
  PyObject *add_value = PyString_FromString("./");
  PyList_Append(sys_path, add_value);
  // remove temporary references
  Py_DECREF(add_value);
  Py_DECREF(sys_path);
  Py_DECREF(sys_module);
  // assign python and dictionary objects
  PyObject* python_module = PyImport_ImportModule("plot"); // python module
  // python dictionary
  PyObject* python_dictionary = PyModule_GetDict(python_module);

  PyObject *pArgs, *func; // pointers to the python objects

  // initialize the shape arrays
  npy_intp pDataDim[] = {data.cols().B.size()};
  npy_intp pDataDimPA[] = {20+20*data.cols().B.size()};
  npy_intp pDataDimHe[] = {dataMap["He/p"].size()};

  int nArguments = dataTree.count("");

  pArgs = PyTuple_New(nArguments); // initialize the arguments tuple

  // set year
  PyTuple_SetItem(pArgs, 0,
    PyArray_SimpleNewFromData(1, pDataDim, PyArray_INT,
      const_cast<int*>(data.cols().year.data())));

  // set month
  PyTuple_SetItem(pArgs, 1,
    PyArray_SimpleNewFromData(1, pDataDim, PyArray_INT,
      const_cast<int*>(data.cols().month.data())));

  // set day
  PyTuple_SetItem(pArgs, 2,
    PyArray_SimpleNewFromData(1, pDataDim, PyArray_INT,
      const_cast<int*>(data.cols().day.data())));

  // set hour
  PyTuple_SetItem(pArgs, 3,
    PyArray_SimpleNewFromData(1, pDataDim, PyArray_INT,
      const_cast<int*>(data.cols().hour.data())));

  // set minute
  PyTuple_SetItem(pArgs, 4,
    PyArray_SimpleNewFromData(1, pDataDim, PyArray_INT,
      const_cast<int*>(data.cols().minute.data())));

  // set second
  PyTuple_SetItem(pArgs, 5,
    PyArray_SimpleNewFromData(1, pDataDim, PyArray_INT,
      const_cast<int*>(data.cols().second.data())));

  // set number of panels
  PyTuple_SetItem(pArgs, 6, PyInt_FromLong(dataTree.count("")));

  int i = 7;

  BOOST_FOREACH(const pt::ptree::value_type &child, dataTree.get_child("")) {
    int nNodes = child.second.count("");
    PyObject *pArg;

    if (nNodes > 0) {
      pArg = PyTuple_New(nNodes);
      BOOST_FOREACH(const pt::ptree::value_type &node, child.second.get_child("")) {
        PyTuple_SetItem(pArgs, i++,
                        PyString_FromString(node.second.data().c_str()));
        PyTuple_SetItem(pArgs, i++,
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(dataMap[node.second.data()].data())));
      }



      PyTuple_SetItem(pArgs, i++, PyInt_FromLong(nNodes));
      BOOST_FOREACH(const pt::ptree::value_type &node, child.second.get_child("")) {
        PyTuple_SetItem(pArgs, i++,
                        PyString_FromString(node.second.data().c_str()));
        PyTuple_SetItem(pArgs, i++,
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(dataMap[node.second.data()].data())));
      }
    } else {
      pArg = PyTuple_New(1);




      PyTuple_SetItem(pArgs, i++, PyInt_FromLong(1));
      PyTuple_SetItem(pArgs, i++,
                      PyString_FromString(child.second.data().c_str()));
      if (child.second.data() == "PA") {
        PyTuple_SetItem(pArgs, i++,
          PyArray_SimpleNewFromData(1, pDataDimPA, PyArray_DOUBLE,
            const_cast<double*>(dataMap[child.second.data()].data())));
      } else if (child.second.data() == "He/p") {
        PyTuple_SetItem(pArgs, i++,
          PyArray_SimpleNewFromData(1, pDataDimHe, PyArray_DOUBLE,
            const_cast<double*>(dataMap["t_He/p"].data())));
        PyTuple_SetItem(pArgs, i++,
          PyArray_SimpleNewFromData(1, pDataDimHe, PyArray_DOUBLE,
            const_cast<double*>(dataMap["He/p"].data())));
      } else {
        PyTuple_SetItem(pArgs, i++,
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(dataMap[child.second.data()].data())));
      }

    }
  }

  // initialize and call the python function
  func = PyDict_GetItemString(python_dictionary, "plot");
  if (PyCallable_Check(func)) {
      PyObject_CallObject(func, pArgs);
  } else {
      PyErr_Print();
  }
}

