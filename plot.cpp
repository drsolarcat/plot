
// project headers
#include "my_time.h"
#include "my_timeseries.h"
// library headers
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/regex.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <python2.7/Python.h>
#include <python2.7/site-packages/numpy/core/include/numpy/arrayobject.h>
//#include <python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h>
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
//    ("ace", "use ACE data")
//    ("wind", "use WIND data")
//    ("stereo-a", "use STEREO-A data")
//    ("stereo-b", "use STEREO-B data")
    ("data,d", po::value<std::string>()->composing(),
     "data to be plotted")
  ;

  po::variables_map vm; // variables map
  // associate command line arguments with the variables map
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // output program version
  if (vm.count("version")) {
    std::cout << "2.0" << std::endl;
    return 1;
  }

  // output the help message
  if (vm.count("help") || vm.size() == 0) {
    std::cout << desc << "\n";
    return 1;
  }

  // assign start and end datetimes
  My::Time beginTime, endTime;
  if (vm.count("start-date")) {
    beginTime = My::Time(vm["start-date"].as<std::string>());
  } else {
    std::cout << "start date is not set" << std::endl;
    return 0;
  }
  if (vm.count("end-date")) {
    endTime = My::Time(vm["end-date"].as<std::string>());
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
                           boost::regex("([a-zA-Z0-9/><]+)"), "\"$1\"",
                           boost::match_default | boost::format_all) << "]";
    pt::read_json(dataStringStream, dataTree);
  } else {
    std::cout << "data selection is not set" << std::endl;
    return 0;
  }

  double Te = 130000; // K
  double kb = 1.380648e-23; // J/K
  double mu0 = 1.256637e-6; // N/A^2

  My::Timeseries tsMain, tsPA, tsAE, tsQ;
  My::Timeseries1D tsHe;
  tsMain.readFile("../icme/res/ace_240.dat", "ymdhms").filter(beginTime, endTime);
  tsPA.readFile("../data/ace_epam_240.dat", "ymdhms").filter(beginTime, endTime);
  tsAE.readFile("../data/ae_240.dat", "ymdhms").filter(beginTime, endTime);
  tsHe.readFile("../data/ACE_SWEPAM_Data.txt", "ydhms").filter(beginTime, endTime);
  tsHe.filter(">=", 0);
  tsQ.readFile("../data/ACE_SWICS_Data.txt", "ydhms").filter(beginTime, endTime);
  My::Timeseries1D tsO7O6(tsQ, 1), tsFe(tsQ, 2);
  tsO7O6.filter(">=", 0);
  tsFe.filter(">=", 0);

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
  PyErr_Print();
  // python dictionary
  PyObject* python_dictionary = PyModule_GetDict(python_module);
  PyObject *pArgs, *func; // pointers to the python objects

  boost::tokenizer<boost::char_separator<char> >
    tk(vm["data"].as<std::string>(),
       boost::char_separator<char>(",[] "));
  std::map<std::string,PyObject*> dictMap;
  Eigen::VectorXd Pt, Pb, thetaB, phiB, PAy, PAz;
  BOOST_FOREACH (const std::string& t, tk) {
    dictMap[t] = PyDict_New();
    PyDict_SetItemString(dictMap[t], "name", PyString_FromString(t.c_str()));
    if (t == "B" || t == "Bx" || t == "Br" || t == "By" || t == "Bt" ||
        t == "Bz" || t == "Bn") {
      npy_intp pDataDim[] = {tsMain.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PyDict_SetItemString(dictMap[t], "unit", PyString_FromString("nT"));
      if (t == "B") {
        npy_intp pDataDim[] = {tsMain.col(1).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(1).data())));
      } else if (t == "Bx" || t == "Br") {
        npy_intp pDataDim[] = {tsMain.col(2).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(2).data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("red"));
      } else if (t == "By" || t == "Bt") {
        npy_intp pDataDim[] = {tsMain.col(3).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(3).data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("green"));
      } else if (t == "Bz" || t == "Bn") {
        npy_intp pDataDim[] = {tsMain.col(4).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(4).data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("blue"));
      }
    } else if (t == "Vp" || t == "Vx" || t == "Vr" || t == "Vy" || t == "Vt" ||
               t == "Vz" || t == "Vn" || t == "Vth") {
      npy_intp pDataDim[] = {tsMain.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      if (t == "Vp") {
        npy_intp pDataDim[] = {tsMain.col(5).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(5).data())));
      } else if (t == "Vx" || t == "Vr") {
        npy_intp pDataDim[] = {tsMain.col(6).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(6).data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("red"));
      } else if (t == "Vy" || t == "Vt") {
        npy_intp pDataDim[] = {tsMain.col(7).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(7).data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("green"));
      } else if (t == "Vz" || t == "Vn") {
        npy_intp pDataDim[] = {tsMain.col(8).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(8).data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("blue"));
      } else if (t == "Vth") {
        npy_intp pDataDim[] = {tsMain.col(12).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(12).data())));
      }
    } else if (t == "Pth" || t == "Pt" || t == "Pb") {
      npy_intp pDataDim[] = {tsMain.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      if (t == "Pth") {
        npy_intp pDataDim[] = {tsMain.col(9).size()};
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(tsMain.col(9).data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("red"));
      } else if (t == "Pt") {
        npy_intp pDataDim[] = {tsMain.col(10).size()};
        Pt = ((tsMain.col(1)*1e-9).array().pow(2).matrix()/2/mu0+
             (tsMain.col(10)*1e6)*kb*Te+
             ((tsMain.col(10)*1e6).array()*0.96*kb*tsMain.col(11).array()).matrix()+
             ((tsMain.col(10)*1e6).array()*0.04*kb*tsMain.col(11).array()*4).matrix())*1e9;
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(Pt.data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("green"));
      } else if (t == "Pb") {
        Pb = (tsMain.col(1)*1e-9).array().pow(2).matrix()/2/mu0*1e9;
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(Pb.data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("blue"));
      }
    } else if (t == "Np") {
      npy_intp pDataDim[] = {tsMain.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(10).data())));
    } else if (t == "Tp") {
      npy_intp pDataDim[] = {tsMain.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(11).data())));
    } else if (t == "beta") {
      npy_intp pDataDim[] = {tsMain.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(13).data())));
    } else if (t == "thetaB" || t == "phiB") {
      npy_intp pDataDim[] = {tsMain.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      Eigen::VectorXd Bxy = (tsMain.col(3).array().pow(2)+
                             tsMain.col(2).array().pow(2)).sqrt().matrix();
      if (t == "thetaB") {
        thetaB = Eigen::VectorXd::Zero(tsMain.col(2).size());
        for (int i = 0; i < tsMain.col(2).size(); i++) {
          thetaB(i) = atan(tsMain.col(4)(i)/Bxy(i));
          thetaB(i) = thetaB(i)*180/M_PI;
        }
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(thetaB.data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("red"));
      } else if (t == "phiB") {
        phiB = Eigen::VectorXd::Zero(tsMain.col(2).size());
        for (int i = 0; i < tsMain.col(2).size(); i++) {
          phiB(i) = atan2(tsMain.col(3)(i), tsMain.col(2)(i));
          phiB(i) = (phiB(i) < 0 ? phiB(i)+M_PI : phiB(i));
          phiB(i) = phiB(i)*180/M_PI;
        }
        PyDict_SetItemString(dictMap[t], "y",
          PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
            const_cast<double*>(phiB.data())));
        PyDict_SetItemString(dictMap[t], "color", PyString_FromString("green"));
      }
    } else if (t == "PA") {
      npy_intp pDataDim[] = {tsPA.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsPA.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PAy = Eigen::VectorXd(20);
      PAy << 4.5,13.5,22.5,31.5,40.5,49.5,58.5,67.5,76.5,85.5,94.5,103.5,112.5,
             121.5,130.5,139.5,148.5,157.5,166.5,175.5;
      PAz = Eigen::VectorXd(tsPA.col(0).size()*20);
      for (int i = 1; i <= 20; i++) {
        PAz.segment((i-1)*tsPA.col(0).size(),tsPA.col(0).size()) = tsPA.col(i);
      }
      npy_intp pDataYDim[] = {20};
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataYDim, PyArray_DOUBLE,
          const_cast<double*>(PAy.data())));
      npy_intp pDataZDim[] = {tsPA.col(0).size()*20};
      PyDict_SetItemString(dictMap[t], "z",
        PyArray_SimpleNewFromData(1, pDataZDim, PyArray_DOUBLE,
          const_cast<double*>(PAz.data())));
    } else if (t == "AE") {
      npy_intp pDataDim[] = {tsAE.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsAE.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsAE.col(1).data())));
    } else if (t == "He/p") {
      npy_intp pDataDim[] = {tsHe.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsHe.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsHe.col(1).data())));
      PyDict_SetItemString(dictMap[t], "marker", PyString_FromString("+"));
      PyDict_SetItemString(dictMap[t], "color", PyString_FromString("red"));
    } else if (t == "O7/O6") {
      npy_intp pDataDim[] = {tsO7O6.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsO7O6.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsO7O6.col(1).data())));
      PyDict_SetItemString(dictMap[t], "marker", PyString_FromString("+"));
      PyDict_SetItemString(dictMap[t], "color", PyString_FromString("green"));
    } else if (t == "<Q>Fe") {
      npy_intp pDataDim[] = {tsFe.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsFe.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsFe.col(1).data())));
      PyDict_SetItemString(dictMap[t], "marker", PyString_FromString("+"));
    }
  }

  std::vector<PyObject*> tupleVec;
  BOOST_FOREACH(const pt::ptree::value_type &child, dataTree.get_child("")) {
    int nNodes = child.second.count("");
    tupleVec.push_back(PyTuple_New(nNodes == 0 ? 1 : nNodes));
    if (nNodes == 0) {
      PyTuple_SetItem(tupleVec.back(), 0, dictMap[child.second.data()]);
    } else {
      int i = 0;
      BOOST_FOREACH(const pt::ptree::value_type &node, child.second.get_child("")) {
        PyTuple_SetItem(tupleVec.back(), i, dictMap[node.second.data()]);
        i++;
      }
    }
  }

  pArgs = PyTuple_New(tupleVec.size()); // initialize the arguments tuple
  for (int i = 0; i < tupleVec.size(); i++) {
    PyTuple_SetItem(pArgs, i, tupleVec[i]);
  }

  // initialize and call the python function
  func = PyDict_GetItemString(python_dictionary, "plot");
  if (PyCallable_Check(func)) {
      PyObject_CallObject(func, pArgs);
  } else {
      PyErr_Print();
  }

//  Py_Finalize();
}

