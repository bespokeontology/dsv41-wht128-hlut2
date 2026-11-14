# Top-Level Pure-C++ Makefile for DeepSeek-V4.1-Flash-1017 WHT128-HLUT2 (Wave 8 Canonical)
# Lead Tier: 2x Tenstorrent Blackhole P150A (p150/) + 3x AMD Instinct MI50 (mi50/) + Offline Tools (tools/)
# 100% Pure C++17 / gfx906 HIP / TT-Metal C++20 — Zero Python Anywhere.

ROOT          := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BIN_DIR       := $(ROOT)/bin
BUILD_DIR     := $(ROOT)/build
CXX           ?= g++
ROCM_PATH     ?= /opt/rocm
HIPCC         ?= $(ROCM_PATH)/bin/hipcc
GCC_DIR       ?= /usr/lib/gcc/x86_64-linux-gnu/11
TT_METAL_HOME ?= /opt/tt-metal

TOOL_CXXFLAGS := -O3 -std=c++17 -mavx2 -mfma -mf16c -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter -D__HIP_PLATFORM_AMD__ -I$(ROCM_PATH)/include -I$(ROOT)/src -I$(ROOT)/mi50 -I$(ROOT)/p150
HIP_CXXFLAGS  := --gcc-install-dir=$(GCC_DIR) -O3 -std=c++17 --offload-arch=gfx906 -fno-gpu-rdc
MI50_LDFLAGS  := -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lpcre2-8 -licuuc -licui18n -fopenmp -pthread -lrt -ldl
TOK_LDFLAGS   := -lpcre2-8 -licuuc -licui18n -fopenmp -pthread

TOOLS_BINS := \
	$(BIN_DIR)/pack_dsv41_hlut2 \
	$(BIN_DIR)/repack_hot_experts_rarebal \
	$(BIN_DIR)/harvest_rarebal_wave8 \
	$(BIN_DIR)/bench_hlut2 \
	$(BIN_DIR)/verify_wave5_all \
	$(BIN_DIR)/verify_wave6_math \
	$(BIN_DIR)/verify_wave7_opt \
	$(BIN_DIR)/verify_wave8_opt \
	$(BIN_DIR)/test_think_loop_guard \
	$(BIN_DIR)/test_anthropic_messages_fe

all: mi50 tools p150_if_present

$(BIN_DIR) $(BUILD_DIR)/tools $(BUILD_DIR)/mi50:
	mkdir -p $@

tools: mi50 $(TOOLS_BINS)

mi50:
	$(MAKE) -C $(ROOT)/mi50 BIN_DIR=$(BIN_DIR) BUILD_DIR=$(BUILD_DIR)/mi50 ROCM_PATH=$(ROCM_PATH)

p150:
	$(MAKE) -f $(ROOT)/p150/Makefile OUT=$(BIN_DIR) OBJ=$(BUILD_DIR)/p150 TT_METAL_HOME=$(TT_METAL_HOME)

p150_if_present:
	@if [ -d "$(TT_METAL_HOME)/include" ]; then \
		$(MAKE) -f $(ROOT)/p150/Makefile OUT=$(BIN_DIR) OBJ=$(BUILD_DIR)/p150 TT_METAL_HOME=$(TT_METAL_HOME); \
	else \
		echo "[INFO] TT_METAL_HOME ($(TT_METAL_HOME)) not found; skip optional p150 host link (set TT_METAL_HOME to build p150_dsv41_worker)"; \
	fi

$(BIN_DIR)/pack_dsv41_hlut2: $(ROOT)/tools/pack_dsv41_hlut2.cpp $(ROOT)/src/dsv41_pack_format.hpp $(ROOT)/src/hlut2_pack.hpp | $(BIN_DIR)
	$(CXX) $(TOOL_CXXFLAGS) $< -o $@

$(BIN_DIR)/repack_hot_experts_rarebal: $(ROOT)/tools/repack_hot_experts_rarebal.cpp $(ROOT)/tools/repack_rarebal_helpers.hpp $(ROOT)/src/hlut2_pack.hpp | $(BIN_DIR)
	$(CXX) $(TOOL_CXXFLAGS) $< -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lrocblas -o $@

$(BIN_DIR)/bench_hlut2: $(ROOT)/tools/bench_hlut2.cpp $(ROOT)/src/hlut2_pack.hpp | $(BIN_DIR)
	$(CXX) $(TOOL_CXXFLAGS) $< -o $@

$(BIN_DIR)/verify_wave5_all: $(ROOT)/tools/verify_wave5_all.cpp $(ROOT)/src/hlut2_pack.hpp | $(BIN_DIR)
	$(CXX) $(TOOL_CXXFLAGS) $< -o $@

$(BUILD_DIR)/tools/dsv41_tokenizer_dsml.o: $(ROOT)/mi50/dsv41_tokenizer_dsml.cpp $(ROOT)/mi50/dsv41_tokenizer_dsml.hpp $(ROOT)/mi50/dsv41_json.hpp | $(BUILD_DIR)/tools
	$(CXX) $(TOOL_CXXFLAGS) -c $< -o $@

$(BIN_DIR)/verify_wave6_math: $(ROOT)/tools/verify_wave6_math.cpp $(BUILD_DIR)/tools/dsv41_tokenizer_dsml.o | $(BIN_DIR)
	$(CXX) $(TOOL_CXXFLAGS) $^ $(TOK_LDFLAGS) -o $@

$(BIN_DIR)/verify_wave7_opt: $(ROOT)/tools/verify_wave7_opt.cpp $(ROOT)/p150/dsv41_p150_l1.hpp $(ROOT)/mi50/dsv41_kolmogorov_tree_verifier.hpp $(ROOT)/mi50/dsv41_tree_spec.hpp $(ROOT)/src/hlut2_pack.hpp | $(BIN_DIR)
	$(CXX) $(TOOL_CXXFLAGS) $< -o $@

$(BUILD_DIR)/mi50/verify_wave8_opt.o: $(ROOT)/tools/verify_wave8_opt.cpp $(ROOT)/p150/dsv41_p150_l1.hpp $(ROOT)/p150/dsv41_p150_shm.hpp $(ROOT)/mi50/dsv41_kolmogorov_tree_verifier.hpp $(ROOT)/mi50/dsv41_tree_spec.hpp $(ROOT)/mi50/dsv41_mi50_kernels.hpp | $(BUILD_DIR)/mi50
	$(HIPCC) -x hip $(HIP_CXXFLAGS) -Wall -Wextra -Werror -c $< -o $@

$(BIN_DIR)/verify_wave8_opt: mi50 $(BUILD_DIR)/mi50/verify_wave8_opt.o | $(BIN_DIR)
	$(HIPCC) --gcc-install-dir=$(GCC_DIR) -O3 -std=c++17 --offload-arch=gfx906 $(BUILD_DIR)/mi50/verify_wave8_opt.o $(BUILD_DIR)/mi50/dsv41_mi50_kernels.o -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -pthread -lrt -o $@

$(BUILD_DIR)/tools/harvest_rarebal_wave8.o: $(ROOT)/tools/harvest_rarebal_wave8.cpp $(ROOT)/mi50/dsv41_runtime.hpp | $(BUILD_DIR)/tools
	$(CXX) $(TOOL_CXXFLAGS) -Werror -c $< -o $@

$(BIN_DIR)/harvest_rarebal_wave8: mi50 $(BUILD_DIR)/tools/harvest_rarebal_wave8.o | $(BIN_DIR)
	$(HIPCC) --gcc-install-dir=$(GCC_DIR) -O3 -std=c++17 --offload-arch=gfx906 $(BUILD_DIR)/tools/harvest_rarebal_wave8.o $(BUILD_DIR)/mi50/dsv41_runtime.o $(BUILD_DIR)/mi50/dsv41_mi50_kernels.o $(BUILD_DIR)/mi50/dsv41_engram_cpu.o $(BUILD_DIR)/mi50/dsv41_tokenizer_dsml.o $(MI50_LDFLAGS) -o $@

$(BIN_DIR)/test_anthropic_messages_fe: $(ROOT)/tools/test_anthropic_messages_fe.cpp $(ROOT)/mi50/dsv41_anthropic_messages.hpp $(BUILD_DIR)/tools/dsv41_tokenizer_dsml.o | $(BIN_DIR)
	$(CXX) $(TOOL_CXXFLAGS) -Werror $< $(BUILD_DIR)/tools/dsv41_tokenizer_dsml.o $(TOK_LDFLAGS) -o $@

$(BIN_DIR)/test_think_loop_guard: $(ROOT)/mi50/dsv41_think_loop_guard.hpp | $(BIN_DIR)
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Werror -x c++ -DDSV41_THINK_GUARD_SELFTEST $< -o $@

check: mi50 $(TOOLS_BINS) audit-gfx906
	$(BIN_DIR)/verify_wave5_all
	$(BIN_DIR)/verify_wave6_math
	$(BIN_DIR)/verify_wave7_opt
	$(BIN_DIR)/test_think_loop_guard
	$(BIN_DIR)/test_anthropic_messages_fe
	$(BIN_DIR)/harvest_rarebal_wave8 --link-check
	$(BIN_DIR)/dsv41_fe --link-check

check-gpu: $(BIN_DIR)/verify_wave8_opt
	$(BIN_DIR)/verify_wave8_opt

audit-gfx906: | $(BUILD_DIR)/tools
	$(HIPCC) $(HIP_CXXFLAGS) -S $(ROOT)/mi50/dsv41_mi50_kernels.hip -o $(BUILD_DIR)/tools/wave8_gfx906.s
	$(HIPCC) -x hip $(HIP_CXXFLAGS) -S $(ROOT)/tools/verify_wave8_opt.cpp -o $(BUILD_DIR)/tools/verify_wave8_gfx906.s
	@echo "Total runtime gfx906 kernels: $$(grep -cE 'ScratchSize:[[:space:]]+' $(BUILD_DIR)/tools/wave8_gfx906.s)"
	@echo "Zero-scratch runtime kernels: $$(grep -cE 'ScratchSize:[[:space:]]+0$$' $(BUILD_DIR)/tools/wave8_gfx906.s)"
	@echo "Total verify8 gfx906 kernels: $$(grep -cE 'ScratchSize:[[:space:]]+' $(BUILD_DIR)/tools/verify_wave8_gfx906.s)"
	@echo "Zero-scratch verify8 kernels: $$(grep -cE 'ScratchSize:[[:space:]]+0$$' $(BUILD_DIR)/tools/verify_wave8_gfx906.s)"
	@echo "Device trig instructions (v_sin_f32/v_cos_f32): $$(grep -cE 'v_sin_f32|v_cos_f32' $(BUILD_DIR)/tools/wave8_gfx906.s $(BUILD_DIR)/tools/verify_wave8_gfx906.s | awk -F: '{s+=$$NF} END {print s+0}')"
	@test "$$(grep -cE 'ScratchSize:[[:space:]]+' $(BUILD_DIR)/tools/wave8_gfx906.s)" -eq "$$(grep -cE 'ScratchSize:[[:space:]]+0$$' $(BUILD_DIR)/tools/wave8_gfx906.s)"
	@test "$$(grep -cE 'ScratchSize:[[:space:]]+' $(BUILD_DIR)/tools/verify_wave8_gfx906.s)" -eq "$$(grep -cE 'ScratchSize:[[:space:]]+0$$' $(BUILD_DIR)/tools/verify_wave8_gfx906.s)"
	@! grep -qE "v_sin_f32|v_cos_f32" $(BUILD_DIR)/tools/wave8_gfx906.s $(BUILD_DIR)/tools/verify_wave8_gfx906.s
	@! grep -rn "hipDeviceEnablePeerAccess" $(ROOT)/mi50 $(ROOT)/src $(ROOT)/p150 $(ROOT)/tools
	@test "$$(find $(ROOT) -name '*.py' -print | wc -l)" -eq 0

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all tools mi50 p150 p150_if_present check check-gpu audit-gfx906 clean
