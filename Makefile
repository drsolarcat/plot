
# include paths
ifeq ($(USERNAME),isavnin)
	CPLUS_INCLUDE_PATH=/home/isavnin/usr/local/include:/home/isavnin/usr/local/lib
	LIBRARY_PATH=/home/isavnin/usr/local/lib
	LD_LIBRARY_PATH=/home/isavnin/usr/local/lib
else
  CPLUS_INCLUDE_PATH=/usr/lib
endif

# export the paths
export CPLUS_INCLUDE_PATH LIBRARY_PATH

# names of executables
PROGRAM = plot
TEST = test

# sources
CXXSOURCES = my_time.cpp my_data.cpp my_timeseries.cpp
CXXOBJECTS = $(CXXSOURCES:.cpp=.o)

#flags
CXX = g++
#GSL = -lgsl -lgslcblas -lm
BOOST = -lboost_program_options -lboost_regex
PYTHON = -lpython2.7
#CXFORM = -lcxform
#LOG4CPLUS = -llog4cplus
CXXFLAGS = -O2 -Wl,-rpath,$(LD_LIBRARY_PATH),-rpath-link,$(LD_LIBRARY_PATH) -Wno-write-strings

all: $(PROGRAM) $(TEST)

$(PROGRAM): $(CXXOBJECTS) $(PROGRAM).o
	$(CXX) -o $@ $@.o $(CXXOBJECTS) $(BOOST) $(PYTHON) $(CXXFLAGS)

$(PROGRAM).o: plot.cpp
	$(CXX) -c -o plot.o plot.cpp $(BOOST) $(CXXFLAGS)

$(TEST): $(CXXOBJECTS) $(TEST).o
	$(CXX) -o $@ $@.o $(CXXOBJECTS) $(BOOST) $(PYTHON) $(CXXFLAGS)

$(TEST).o: test.cpp
	$(CXX) -c -o test.o test.cpp $(BOOST) $(CXXFLAGS)

my_time.o: my_time.h my_time.cpp
	$(CXX) -c -o my_time.o my_time.cpp $(CXXFLAGS)

my_data.o: my_data.h my_data.cpp
	$(CXX) -c -o my_data.o my_data.cpp $(CXXFLAGS)

my_timeseries.o: my_timeseries.h my_timeseries.cpp
	$(CXX) -c -o my_timeseries.o my_timeseries.cpp $(CXXFLAGS)

clean:
	$(RM) -f $(CXXOBJECTS) $(PROGRAM).o $(PROGRAM) $(TEST).o $(TEST)

run:
	./$(PROGRAM)

