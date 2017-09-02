UNAME := $(shell uname)
ifeq ($(UNAME),Linux)
	SHAREDLIB_EXT := so
	SHAREDLIB_FLAG := -shared
endif
ifeq ($(UNAME),Darwin)
	SHAREDLIB_EXT := dylib
	SHAREDLIB_FLAG := -dynamiclib
endif

LIB := libgraph.$(SHAREDLIB_EXT)

CC := $(CXX)
CXXFLAGS := -std=c++11 -Wall -O3 -Wshadow -fPIC

SWEEPCUT_SRCS = ../sweepcut.cpp sweepcut_test.cpp readData.cpp
ACLPAGERANK_SRCS = ../aclpagerank.cpp aclpagerank_test.cpp readData.cpp
MQI_SRCS = ../MQI.cpp MQI_test.cpp ../maxflow.cpp ../MQI_c_interface.cpp readData.cpp
PPR_PATH_SRCS = ../ppr_path.cpp ppr_path_test.cpp readData.cpp
ACLPAGERANK_WEIGHTED_SRCS = ../aclpagerank_weighted.cpp aclpagerank_weighted_test.cpp readData.cpp
PROXL1PRACCEL_SRCS = ../proxl1PRaccel.cpp proxl1PRaccel_test.cpp readData.cpp

SWEEPCUT_OBJS = $(SWEEPCUT_SRCS:%.cpp=%.o)
ACLPAGERANK_OBJS = $(ACLPAGERANK_SRCS:%.cpp=%.o)
MQI_OBJS = $(MQI_SRCS:%.cpp=%.o)
PPR_PATH_OBJS = $(PPR_PATH_SRCS:%.cpp=%.o)
ACLPAGERANK_WEIGHTED_OBJS = $(ACLPAGERANK_WEIGHTED_SRCS:%.cpp=%.o)
PROXL1PRACCEL_OBJS = $(PROXL1PRACCEL_SRCS:%.cpp=%.o)

LIBOBJS = ../sweepcut.o ../aclpagerank.o ../MQI_c_interface.o ../ppr_path.o ../aclpagerank_weighted.o ../proxl1PRaccel.o

all: sweepcut aclpagerank MQI ppr_path aclpagerank_weighted proxl1PRaccel $(LIB)

sweepcut: $(SWEEPCUT_OBJS)
	$(CC) $(SWEEPCUT_OBJS) -o sweepcut
aclpagerank: $(ACLPAGERANK_OBJS)
	$(CC) $(ACLPAGERANK_OBJS) -o aclpagerank
MQI: $(MQI_OBJS)
	$(CC) $(MQI_OBJS) -o MQI
ppr_path: $(PPR_PATH_OBJS)
	$(CC) $(PPR_PATH_OBJS) -o ppr_path
aclpagerank_weighted: $(ACLPAGERANK_WEIGHTED_OBJS)
	$(CC) $(ACLPAGERANK_WEIGHTED_OBJS) -o aclpagerank_weighted
proxl1PRaccel: $(PROXL1PRACCEL_OBJS)
	$(CC) $(PROXL1PRACCEL_OBJS) -o proxl1PRaccel

$(LIB): $(LIBOBJS)
	$(CC) $(SHAREDLIB_FLAG) -o $(LIB) $(LIBOBJS)


TARGET := sweepcut aclpagerank MQI ppr_path aclpagerank_weighted proxl1PRaccel
OBJS := $(SWEEPCUT_OBJS) $(ACLPAGERANK_OBJS) $(MQI_OBJS) $(PPR_PATH_OBJS) $(ACLPAGERANK_WEIGHTED_OBJS) $(PROXL1PRACCEL_OBJS)

test:
	./sweepcut
	./aclpagerank
	./MQI
	./ppr_path
	./aclpagerank_weighted
	./proxl1PRaccel

.PHONY: all clean test

clean:
	$(RM) $(TARGET) $(LIB) $(OBJS)
