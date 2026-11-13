# Pure-C++17 / HIP (gfx906) Makefile for 3x AMD Instinct MI50 Runtime & :8042 HTTP/SSE Server
HERE        := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT        := $(abspath $(HERE)/..)
ROCM_PATH   ?= /opt/rocm
HIPCC       ?= $(ROCM_PATH)/bin/hipcc
GCC_DIR     ?= /usr/lib/gcc/x86_64-linux-gnu/11
CXX         ?= g++

BUILD_DIR   ?= $(ROOT)/build/mi50
BIN_DIR     ?= $(ROOT)/bin

CXXFLAGS    := -O3 -std=c++17 -march=znver2 -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -D__HIP_PLATFORM_AMD__ -I$(ROCM_PATH)/include -I$(ROOT)/src -I$(ROOT)/p150
HIPFLAGS    := --gcc-install-dir=$(GCC_DIR) -O3 -std=c++17 --offload-arch=gfx906 -fno-gpu-rdc
LDFLAGS     := -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lpcre2-8 -licuuc -licui18n -fopenmp -pthread -lrt -ldl

OBJS := \
	$(BUILD_DIR)/dsv41_tokenizer_dsml.o \
	$(BUILD_DIR)/dsv41_engram_cpu.o \
	$(BUILD_DIR)/dsv41_mi50_kernels.o \
	$(BUILD_DIR)/dsv41_runtime.o \
	$(BUILD_DIR)/dsv41_fe.o

all: $(BIN_DIR)/dsv41_fe

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

$(BUILD_DIR)/dsv41_tokenizer_dsml.o: $(HERE)/dsv41_tokenizer_dsml.cpp $(HERE)/dsv41_tokenizer_dsml.hpp $(HERE)/dsv41_json.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/dsv41_engram_cpu.o: $(HERE)/dsv41_engram_cpu.cpp $(HERE)/dsv41_engram_cpu.hpp $(ROOT)/src/safetensors_mmap.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/dsv41_mi50_kernels.o: $(HERE)/dsv41_mi50_kernels.hip $(HERE)/dsv41_mi50_kernels.hpp | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

$(BUILD_DIR)/dsv41_runtime.o: $(HERE)/dsv41_runtime.cpp $(HERE)/dsv41_runtime.hpp $(HERE)/dsv41_kolmogorov_tree_verifier.hpp $(HERE)/dsv41_tree_spec.hpp $(HERE)/dsv41_think_loop_guard.hpp $(ROOT)/src/dsv41_rarebal.hpp $(HERE)/dsv41_mi50_kernels.hpp $(HERE)/dsv41_engram_cpu.hpp $(HERE)/dsv41_tokenizer_dsml.hpp $(ROOT)/p150/dsv41_p150_shm.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/dsv41_fe.o: $(HERE)/dsv41_fe.cpp $(HERE)/dsv41_runtime.hpp $(HERE)/dsv41_think_loop_guard.hpp $(HERE)/dsv41_live_progress.hpp $(HERE)/dsv41_tokenizer_dsml.hpp $(HERE)/dsv41_json.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/dsv41_fe: $(OBJS) | $(BIN_DIR)
	$(HIPCC) --gcc-install-dir=$(GCC_DIR) -O3 -std=c++17 --offload-arch=gfx906 $(OBJS) $(LDFLAGS) -o $@.new && mv -f $@.new $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)/dsv41_fe

.PHONY: all clean
