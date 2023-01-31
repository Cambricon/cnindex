ifeq ($(NEUWARE_HOME),)
	NEUWARE_HOME=/usr/local/neuware
endif

DEBUG = 0

CXX         = g++ -std=c++11
CXXCPP      = g++ -std=c++11 -E
CPPFLAGS    = -DFINTEGER=int
CXXFLAGS    = -fPIC -Wno-format-truncation -Wno-sign-compare -O3 -mpopcnt -msse4 -mavx2 -I./include -I$(NEUWARE_HOME)/include 
LDFLAGS     = -L/usr/lib/x86_64-linux-gnu -L$(NEUWARE_HOME)/lib64 -lcnnl_extra -lcnnl -lcnrt -lcndrv -lpthread

LIB_NAME    = cnindex

HEADERS     = $(wildcard include/*.h)
SRC         = $(wildcard src/*.cpp src/utils/*.cpp)
TESTS_SRC   = $(filter-out tests/common.cpp, $(wildcard tests/*.cpp))
OBJ         = $(SRC:.cpp=.o)
TESTS_OBJ   = $(TESTS_SRC:.cpp=.o)
TESTS_BIN   = $(TESTS_OBJ:.o=)
DIRS = lib

ifeq ($(DEBUG), 1)
  CXXFLAGS += -g
else
  STRIP_CMD = -s
endif

############################
# Building

all: ${DIRS} lib$(LIB_NAME).so $(TESTS_BIN) 

$(DIRS):
	@if [ ! -e $@ ]; then mkdir $@; fi

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

lib$(LIB_NAME).so: $(OBJ)
	$(CXX) -shared $(LDFLAGS) $(STRIP_CMD) -Wl,-soname,$@ -o lib/$@ $^

$(TESTS_BIN): %: %.o lib/lib$(LIB_NAME).so
	$(CXX) -o $@ $^ $(LDFLAGS) -L.

.PHONY:clean
clean:
	rm -rf lib
	rm -f $(OBJ) $(TESTS_OBJ) $(TESTS_BIN)

