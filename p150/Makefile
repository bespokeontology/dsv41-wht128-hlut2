# Pure C++20 TT-Metal Makefile for 2x Tenstorrent Blackhole P150A Worker (FABRIC_1D TP2)
HERE          := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT          := $(abspath $(HERE)/..)
OUT           ?= $(ROOT)/bin
OBJ           ?= $(ROOT)/build/p150
TT_METAL_HOME ?= /opt/tt-metal
CXX           ?= clang++

CXXFLAGS := -std=c++20 -O3 -mavx2 -mf16c -mfma -fPIC -DNTEST -DFMT_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL
ifdef P150_MAX_HOT_L1_SHARDS
CXXFLAGS += -DP150_MAX_HOT_L1_SHARDS=$(P150_MAX_HOT_L1_SHARDS)
endif
CXXFLAGS += -I$(TT_METAL_HOME)/include -I$(TT_METAL_HOME)/include/ttnn -I$(ROOT)/src

ifdef TT_EXTRA_CXXFLAGS
CXXFLAGS += $(TT_EXTRA_CXXFLAGS)
endif

LDLIBS := -L$(TT_METAL_HOME)/lib -Wl,-rpath,$(TT_METAL_HOME)/lib
LDLIBS += -Wl,--no-undefined -l:_ttnncpp.so -ltt_metal -ltt_stl -pthread -lrt

ifdef TT_EXTRA_LDLIBS
LDLIBS += $(TT_EXTRA_LDLIBS)
endif

all: $(OUT)/p150_dsv41_worker

$(OBJ) $(OUT):
	mkdir -p $@

$(OBJ)/p150_dsv41_engine.o: $(HERE)/p150_dsv41_engine.cpp $(HERE)/p150_dsv41_engine.hpp $(HERE)/dsv41_p150_l1.hpp $(HERE)/dsv41_p150_shm.hpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ)/p150_dsv41_worker.o: $(HERE)/p150_dsv41_worker.cpp $(HERE)/p150_dsv41_engine.hpp $(HERE)/dsv41_p150_l1.hpp $(HERE)/dsv41_p150_shm.hpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/p150_dsv41_worker: $(OBJ)/p150_dsv41_engine.o $(OBJ)/p150_dsv41_worker.o | $(OUT)
	$(CXX) -o $@.new $^ $(LDLIBS) && mv -f $@.new $@

clean:
	rm -rf $(OBJ) $(OUT)/p150_dsv41_worker

.PHONY: all clean
