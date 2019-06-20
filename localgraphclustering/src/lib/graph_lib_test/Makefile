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
CXXFLAGS := -g -std=c++11 -Wall -Wshadow -fPIC

SWEEPCUT_SRCS = ../sweepcut.cpp sweepcut_test.cpp readData.cpp
ACLPAGERANK_SRCS = ../aclpagerank.cpp aclpagerank_test.cpp readData.cpp
MQI_SRCS = ../MQI.cpp MQI_test.cpp ../MQI_c_interface.cpp readData.cpp
MQI_WEIGHTED_SRCS = ../MQI_weighted.cpp ../MQI_weighted_c_interface.cpp
PPR_PATH_SRCS = ../ppr_path.cpp ppr_path_test.cpp readData.cpp
ACLPAGERANK_WEIGHTED_SRCS = ../aclpagerank_weighted.cpp aclpagerank_weighted_test.cpp readData.cpp
PROXL1PRACCEL_SRCS = ../proxl1PRaccel.cpp proxl1PRaccel_test.cpp readData.cpp
PROXL1PRRAND_SRCS = ../proxl1PRrand.cpp proxl1PRrand_test.cpp
DENSEST_SUBGRAPH_SRCS = ../densest_subgraph.cpp densest_subgraph_test.cpp ../densest_subgraph_c_interface.cpp readData.cpp ../graph_check.cpp
CRD_SRCS = ../capacity_releasing_diffusion.cpp ../unit_flow.cpp crd_test.cpp ../crd_c_interface.cpp readData.cpp
SL_SRCS = ../SimpleLocal.cpp ../SimpleLocal_c_interface.cpp readData.cpp SimpleLocal_test.cpp
SL_WEIGHTED_SRCS = ../SimpleLocal_weighted.cpp ../SimpleLocal_weighted_c_interface.cpp
TC_SRCS = ../triangleclusters.cpp
SS_SRCS = ../set_scores.cpp

SWEEPCUT_OBJS = $(SWEEPCUT_SRCS:%.cpp=%.o)
ACLPAGERANK_OBJS = $(ACLPAGERANK_SRCS:%.cpp=%.o)
MQI_OBJS = $(MQI_SRCS:%.cpp=%.o)
MQI_WEIGHTED_OBJS = $(MQI_WEIGHTED_SRCS:%.cpp=%.o)
PPR_PATH_OBJS = $(PPR_PATH_SRCS:%.cpp=%.o)
ACLPAGERANK_WEIGHTED_OBJS = $(ACLPAGERANK_WEIGHTED_SRCS:%.cpp=%.o)
PROXL1PRACCEL_OBJS = $(PROXL1PRACCEL_SRCS:%.cpp=%.o)
PROXL1PRRAND_OBJS = $(PROXL1PRRAND_SRCS:%.cpp=%.o)
DENSEST_SUBGRAPH_OBJS = $(DENSEST_SUBGRAPH_SRCS:%.cpp=%.o)
CRD_OBJS = $(CRD_SRCS:%.cpp=%.o)
SL_OBJS = $(SL_SRCS:%.cpp=%.o)
SL_WEIGHTED_OBJS = $(SL_WEIGHTED_SRCS:%.cpp=%.o)
TC_OBJS = $(TC_SRCS:%.cpp=%.o)
SS_OBJS = $(SS_SRCS:%.cpp=%.o)

LIBOBJS = ../sweepcut.o ../aclpagerank.o ../MQI.o ../MQI_weighted.o ../MQI_c_interface.o ../MQI_weighted_c_interface.o ../ppr_path.o \
../aclpagerank_weighted.o ../proxl1PRaccel.o ../proxl1PRrand.o ../set_scores.o ../densest_subgraph_c_interface.o \
../capacity_releasing_diffusion.o ../unit_flow.o ../crd_c_interface.o ../SimpleLocal.o ../SimpleLocal_c_interface.o \
../SimpleLocal_weighted.o ../SimpleLocal_weighted_c_interface.o ../triangleclusters.o

all: sweepcut aclpagerank MQI ppr_path aclpagerank_weighted proxl1PRaccel densest_subgraph capacity_releasing_diffusion SimpleLocal $(LIB)

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
proxl1PRrand: $(PROXL1PRRAND_OBJS)
	$(CC) $(CXXFLAGS) $(PROXL1PRRAND_SRCS) -o proxl1PRrand
densest_subgraph: $(DENSEST_SUBGRAPH_OBJS)
	$(CC) $(DENSEST_SUBGRAPH_OBJS) -o densest_subgraph
capacity_releasing_diffusion: $(CRD_OBJS)
	$(CC) $(CRD_OBJS) -o capacity_releasing_diffusion
SimpleLocal: $(SL_OBJS)
	$(CC) $(SL_OBJS) -o SimpleLocal

$(LIB): $(LIBOBJS)
	$(CC) $(SHAREDLIB_FLAG) -o $(LIB) $(LIBOBJS)


TARGET := sweepcut aclpagerank MQI ppr_path aclpagerank_weighted proxl1PRaccel proxl1PRrand densest_subgraph SimpleLocal capacity_releasing_diffusion set_scores
OBJS := $(SWEEPCUT_OBJS) $(ACLPAGERANK_OBJS) $(MQI_OBJS) $(MQI_WEIGHTED_OBJS) $(PPR_PATH_OBJS) $(ACLPAGERANK_WEIGHTED_OBJS) $(PROXL1PRACCEL_OBJS) $(PROXL1PRRAND_OBJS) $(DENSEST_SUBGRAPH_OBJS) $(CRD_OBJS) $(SL_OBJS) $(SL_WEIGHTED_OBJS) $(TC_OBJS) $(SS_OBJS)

test:
	./sweepcut
	./aclpagerank
	./MQI
	./ppr_path
	./aclpagerank_weighted
	./proxl1PRaccel
	./proxl1PRrand
	./densest_subgraph
	./capacity_releasing_diffusion
	./SimpleLocal

.PHONY: all clean test

clean:
	$(RM) $(TARGET) $(LIB) $(OBJS)
