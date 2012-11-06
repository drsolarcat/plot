
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
                           boost::regex("([a-zA-Z/]+)"), "\"$1\"",
                           boost::match_default | boost::format_all) << "]";
    pt::read_json(dataStringStream, dataTree);
  } else {
    std::cout << "data selection is not set" << std::endl;
    return 0;
  }

  My::Timeseries tsMain, tsPA, tsHe;
  tsMain.readFile("../icme/res/ace_240.dat", "ymdhms").filter(beginTime, endTime);
//  tsPA.readFile("../data/ace_epam_240.dat", "ymdhms").filter(beginTime, endTime);
//  tsHe.readFile("../data/ACE_SWEPAM_Data.txt", "ydhms").filter(beginTime, endTime);

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

  boost::tokenizer<boost::char_separator<char> >
    tk(vm["data"].as<std::string>(),
       boost::char_separator<char>(",[] "));
  std::map<std::string,PyObject*> dictMap;
  BOOST_FOREACH (const std::string& t, tk) {
    if (t == "B" || t == "Bx" || t == "Br" || t == "By" || t == "Bt" ||
        t == "Bz" || t == "Bn") {
      dictMap[t] = PyDict_New();
      npy_intp pDataDim[] = {tsMain.col(0).size()};
      PyDict_SetItemString(dictMap[t], "t",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(0).data())));
      PyDict_SetItemString(dictMap[t], "factor", PyFloat_FromDouble(1));
    }
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
    } else if (t == "By" || t == "Bt") {
      npy_intp pDataDim[] = {tsMain.col(3).size()};
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(3).data())));
    } else if (t == "Bz" || t == "Bn") {
      npy_intp pDataDim[] = {tsMain.col(4).size()};
      PyDict_SetItemString(dictMap[t], "y",
        PyArray_SimpleNewFromData(1, pDataDim, PyArray_DOUBLE,
          const_cast<double*>(tsMain.col(4).data())));
    }
  }

  std::cout << "dictionaries created" << std::endl;

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

  std::cout << "tuples created" << std::endl;

  pArgs = PyTuple_New(tupleVec.size()); // initialize the arguments tuple
  for (int i = 0; i < tupleVec.size(); i++) {
    PyTuple_SetItem(pArgs, i, tupleVec[i]);
  }

  std::cout << "argument tuple created" << std::endl;

  // initialize and call the python function
  func = PyDict_GetItemString(python_dictionary, "plot");
  if (PyCallable_Check(func)) {
      PyObject_CallObject(func, pArgs);
  } else {
      PyErr_Print();
  }

//  Py_Finalize();
}

