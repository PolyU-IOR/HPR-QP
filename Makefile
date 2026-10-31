

.PHONY: all cpu gpu shared auto-build auto-shared auto-test auto-install auto-clean run run-gpu \
	test test-cpu gpu-test install dataset dataset-gpu \
	clean clean-cpu gpu-clean rebuild gpu-rebuild fresh gpu-fresh help \
	build build-shared solve gpu-build gpu-build-shared gpu-solve \
	test-shared gpu-test-shared install-shared gpu-install-shared \
	_announce-auto-build _announce-auto-shared _announce-auto-test _announce-auto-install _announce-auto-clean _require-cuda

.DEFAULT_GOAL := all

CC ?= gcc
CXX ?= g++
AR ?= ar
RANLIB ?= ranlib
BUILD_DIR ?= build
GPU_BUILD_DIR ?= build-cuda
SHARED_BUILD_DIR ?= build-shared
GPU_SHARED_BUILD_DIR ?= build-cuda-shared
MODEL ?= model.mps
DATASET_DIR ?= .
RESULT_DIR ?= results
INSTALL_PREFIX ?= $(CURDIR)/install
GPU_INSTALL_PREFIX ?= $(INSTALL_PREFIX)-gpu
PYTHON ?= python3
CUDA_ROOT ?=
CUDA_HOME ?=
CUDA_PATH ?=
CUDA_NVCC ?=
GPU_SM ?=
ARGS ?=
V ?= 0
SHOW_WARNINGS ?= 0

CUDA_ENV_ROOT := $(strip $(if $(CUDA_ROOT),$(CUDA_ROOT),$(if $(CUDA_HOME),$(CUDA_HOME),$(CUDA_PATH))))
CUDA_AUTO_ROOT := $(strip $(shell \
	if [ -n "$(CUDA_ENV_ROOT)" ]; then \
		candidate="$(CUDA_ENV_ROOT)"; \
		if [ -x "$$candidate/bin/nvcc" ] && [ -f "$$candidate/include/cuda.h" ]; then \
			printf '%s\n' "$$candidate"; \
		fi; \
	else \
		{ \
			command -v nvcc 2>/dev/null || true; \
			for candidate in /usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*; do \
				[ -x "$$candidate/bin/nvcc" ] || continue; \
				printf '%s/bin/nvcc\n' "$$candidate"; \
			done; \
		} | while IFS= read -r candidate; do \
			[ -n "$$candidate" ] || continue; \
			resolved=$$(readlink -f "$$candidate" 2>/dev/null || printf '%s\n' "$$candidate"); \
			root=$$(dirname $$(dirname "$$resolved")); \
			[ -x "$$root/bin/nvcc" ] || continue; \
			[ -f "$$root/include/cuda.h" ] || continue; \
			version=$$("$$root/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1); \
			[ -n "$$version" ] || continue; \
			printf '%s\t%s\n' "$$version" "$$root"; \
		done | sort -u -t "$$(printf '\t')" -k1,1V -k2,2 | tail -n 1 | awk -F '\t' '{print $$2}'; \
	fi))

ifeq ($(strip $(CUDA_NVCC)),)
CUDA_NVCC := $(if $(CUDA_AUTO_ROOT),$(CUDA_AUTO_ROOT)/bin/nvcc)
endif

CUDA_TOOLKIT_ROOT := $(strip $(if $(CUDA_ENV_ROOT),$(CUDA_ENV_ROOT),$(if $(CUDA_AUTO_ROOT),$(CUDA_AUTO_ROOT),$(shell \
	if [ -n "$(CUDA_NVCC)" ]; then \
		resolved=$$(readlink -f "$(CUDA_NVCC)" 2>/dev/null || printf '%s\n' "$(CUDA_NVCC)"); \
		dirname $$(dirname "$$resolved"); \
	fi))))
CUDA_TOOLKIT_TARGET_DIR := $(firstword $(wildcard $(CUDA_TOOLKIT_ROOT)/targets/*))
CUDA_LIB_DIR := $(if $(wildcard $(CUDA_TOOLKIT_ROOT)/lib64),$(CUDA_TOOLKIT_ROOT)/lib64,$(CUDA_TOOLKIT_TARGET_DIR)/lib)
CUDA_STUB_DIR := $(CUDA_TOOLKIT_TARGET_DIR)/lib/stubs
CUDA_SEARCH_HINT := PATH, /usr/local/cuda*, /opt/cuda*, CUDA_ROOT, CUDA_HOME, CUDA_PATH

NVIDIA_SMI := $(shell command -v nvidia-smi 2>/dev/null)
DETECTED_CC := $(strip $(shell \
	if [ -n "$(NVIDIA_SMI)" ]; then \
		nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '; \
	fi))
ifneq ($(DETECTED_CC),)
ifneq ($(DETECTED_CC),N/A)
DETECTED_SM := $(subst .,,$(DETECTED_CC))
endif
endif
ifeq ($(strip $(GPU_SM)),)
GPU_SM := $(if $(strip $(DETECTED_SM)),$(DETECTED_SM),90)
endif
CUDA_ARCH := -arch=sm_$(GPU_SM)

COMMON_CPPFLAGS := -I"$(CURDIR)/include" -I"$(CURDIR)/src/shared" -I"$(CURDIR)/src/gpu" -I"$(CURDIR)/tests" '-DHPRQP_SOURCE_DIR="$(CURDIR)"'
CPU_CPPFLAGS := $(COMMON_CPPFLAGS) -DHPRQP_HAVE_CUDA=0
GPU_CPPFLAGS := $(COMMON_CPPFLAGS) -DHPRQP_HAVE_CUDA=1 $(if $(CUDA_TOOLKIT_ROOT),-I$(CUDA_TOOLKIT_ROOT)/include)
WARNING_FLAGS := $(if $(filter 1,$(SHOW_WARNINGS)),-Wall -Wextra -Wpedantic,-w)
COMMON_CFLAGS := -O2 -DNDEBUG $(WARNING_FLAGS) -std=c11 -fPIC
COMMON_CXXFLAGS := -O2 -DNDEBUG $(WARNING_FLAGS) -std=c++17 -fPIC
COMMON_NVCCFLAGS := -O2 -DNDEBUG $(if $(filter 1,$(SHOW_WARNINGS)),,-w) --std=c++17 $(CUDA_ARCH) -Xcompiler -fPIC $(if $(filter 1,$(SHOW_WARNINGS)),,-Xcompiler -w)
CPU_LINK_LIBS := -lm
GPU_LINK_LIBS := -lm $(if $(CUDA_LIB_DIR),-L$(CUDA_LIB_DIR)) $(if $(wildcard $(CUDA_STUB_DIR)),-L$(CUDA_STUB_DIR)) -lcudart -lcusparse -lcublas
GPU_RUN_ENV := LD_LIBRARY_PATH=$(CUDA_LIB_DIR)$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))

ifeq ($(V),1)
Q :=
else
Q := @
endif

define map_objects
$(addprefix $(1)/obj/,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(2)))))
endef

define basename_noext
$(notdir $(basename $(1)))
endef

HPRQP_SHARED_SOURCES := \
	src/shared/binding.c \
	src/shared/build.c \
	src/shared/mps_qp.cpp \
	src/shared/mps_reader.cpp \
	src/shared/parallel.c \
	src/shared/solver.c \
	src/shared/solver_backend.c \
	src/shared/sparse.c \
	src/shared/vector.c

HPRQP_CPU_SOURCES := \
	src/cpu/control.c \
	src/cpu/eigs.c \
	src/cpu/iteration.c \
	src/cpu/residuals.c \
	src/cpu/scale.c \
	src/cpu/workspace.c

HPRQP_GPU_STUB_SOURCES := \
	src/gpu/backend_stub.c \
	src/gpu/solver_stub.c

HPRQP_GPU_CUDA_SOURCES := \
	src/gpu/context.cu \
	src/gpu/workspace.cu \
	src/gpu/scale.cu \
	src/gpu/spmv.cu \
	src/gpu/eigs.cu \
	src/gpu/control.cu \
	src/gpu/residuals.cu \
	src/gpu/qcqp.cu \
	src/gpu/iteration.cu \
	src/gpu/solver.cu

CPU_TEST_SOURCES := \
	tests/test_sparse_helpers.c \
	tests/test_mps_reader.c \
	tests/test_builder_preprocess.c \
	tests/test_binding_api.c \
	tests/cpu/test_workspace.c \
	tests/cpu/test_scale.c \
	tests/cpu/test_residuals.c \
	tests/cpu/test_solver.c \
	tests/cpu/test_iteration.c \
	tests/cpu/test_parallel.c \
	tests/cpu/test_eigs.c \
	tests/cpu/test_control.c \
	tests/gpu/test_gpu_dispatch.c

GPU_TEST_SOURCES := \
	tests/gpu/test_binding_api_gpu.c \
	tests/gpu/test_gpu_workspace.c \
	tests/gpu/test_gpu_scale.c \
	tests/gpu/test_gpu_residuals.c \
	tests/gpu/test_gpu_eigs.c \
	tests/gpu/test_gpu_qcqp.c \
	tests/gpu/test_gpu_control.c \
	tests/gpu/test_gpu_iteration.cpp \
	tests/gpu/test_gpu_solver.c

GPU_EXAMPLE_SOURCES := \
	examples/gpu/compare_gpu_unchecked_path.c \
	examples/gpu/trace_restart_compare.c

CPU_CORE_OBJECTS := $(call map_objects,$(BUILD_DIR),$(HPRQP_SHARED_SOURCES) $(HPRQP_CPU_SOURCES) $(HPRQP_GPU_STUB_SOURCES))
GPU_CORE_OBJECTS := $(call map_objects,$(GPU_BUILD_DIR),$(HPRQP_SHARED_SOURCES) $(HPRQP_CPU_SOURCES) $(HPRQP_GPU_CUDA_SOURCES))
CPU_TEST_SUPPORT_OBJECT := $(call map_objects,$(BUILD_DIR),tests/test_support.c)
GPU_TEST_SUPPORT_OBJECT := $(call map_objects,$(GPU_BUILD_DIR),tests/test_support.c)

CPU_LIB := $(BUILD_DIR)/libhprqp.a
GPU_LIB := $(GPU_BUILD_DIR)/libhprqp.a
CPU_TEST_SUPPORT_LIB := $(BUILD_DIR)/libhprqp_test_support.a
GPU_TEST_SUPPORT_LIB := $(GPU_BUILD_DIR)/libhprqp_test_support.a
CPU_SHARED_LIB := $(SHARED_BUILD_DIR)/libhprqp.so
GPU_SHARED_LIB := $(GPU_SHARED_BUILD_DIR)/libhprqp.so

CPU_TEST_BINARIES := $(foreach src,$(CPU_TEST_SOURCES),$(BUILD_DIR)/$(call basename_noext,$(src)))
GPU_TEST_BINARIES := $(foreach src,$(GPU_TEST_SOURCES),$(GPU_BUILD_DIR)/$(call basename_noext,$(src)))
GPU_EXAMPLE_BINARIES := $(foreach src,$(GPU_EXAMPLE_SOURCES),$(GPU_BUILD_DIR)/$(call basename_noext,$(src)))
CPU_SOLVER_BINARY := $(BUILD_DIR)/solve_qp
GPU_SOLVER_BINARY := $(GPU_BUILD_DIR)/solve_qp

AUTO_SHARED_TARGETS := build-shared
AUTO_BUILD_TARGETS := build
AUTO_TEST_TARGETS := test-cpu
AUTO_INSTALL_TARGETS := install-shared
AUTO_CLEAN_DIRS := $(BUILD_DIR) $(SHARED_BUILD_DIR)
ifneq ($(strip $(CUDA_NVCC)),)
AUTO_SHARED_TARGETS += gpu-build-shared
AUTO_BUILD_TARGETS += gpu-build gpu-build-shared
AUTO_TEST_TARGETS += gpu-test
AUTO_INSTALL_TARGETS += gpu-install-shared
AUTO_CLEAN_DIRS += $(GPU_BUILD_DIR) $(GPU_SHARED_BUILD_DIR)
endif

all: auto-build

cpu: build

gpu: gpu-build

shared: auto-shared

run: solve

run-gpu: gpu-solve

fresh: rebuild

gpu-fresh: gpu-rebuild

help:
	@printf '%s\n' \
	  'Common targets:' \
	  '  make                  Build CPU targets by default, or CPU+GPU targets plus shared libs when CUDA is available' \
	  '  make shared           Build CPU shared libs by default, or CPU+GPU shared libs when CUDA is available' \
	  '  make test             Run CPU tests by default, or CPU+GPU tests when CUDA is available' \
	  '  make install          Install CPU shared artifacts by default, or CPU+GPU shared artifacts when CUDA is available' \
	  '  make clean            Remove default build directories for the current machine' \
	  '  make build            Build the CPU solver and tests only' \
	  '  make build-shared     Build the shared CPU library for wrappers' \
	  '  make gpu-build        Build the CUDA solver, tests, and GPU examples only' \
	  '  make gpu-build-shared Build the shared CUDA library for wrappers' \
	  '  make solve            Run solve_qp on MODEL using the CPU build' \
	  '  make gpu-solve        Run solve_qp on MODEL using the GPU build' \
	  '  make test-cpu         Run the CPU test suite only' \
	  '  make test-shared      Run the CPU test suite after building the shared CPU library' \
	  '  make gpu-test         Run the CUDA test suite' \
	  '  make gpu-test-shared  Run the CUDA test suite after building the shared CUDA library' \
	  '  make install-shared   Install the CPU shared library and headers' \
	  '  make gpu-install-shared Install the CUDA shared library and headers' \
	  '  make dataset          Run solve_qp on all .mps files in DATASET_DIR' \
	  '  make dataset-gpu      Run solve_qp on all .mps files in DATASET_DIR with GPU enabled' \
	  '  make clean-cpu        Remove the CPU build and CPU shared-build directories' \
	  '  make gpu-clean        Remove the CUDA build and CUDA shared-build directories' \
	  '  make rebuild          Same as make clean && make' \
	  '  make gpu-rebuild      Clean and rebuild the CUDA tree' \
	  '' \
	  'Short aliases:' \
	  '  make cpu             Same as make build' \
	  '  make gpu             Same as make gpu-build' \
	  '  make shared          Same as make auto-shared' \
	  '  make run             Same as make solve' \
	  '  make run-gpu         Same as make gpu-solve' \
	  '  make fresh           Same as make rebuild' \
	  '  make gpu-fresh       Same as make gpu-rebuild' \
	  '' \
	  'Variables:' \
	  '  BUILD_DIR=dir          Build directory (default: build)' \
	  '  GPU_BUILD_DIR=dir      CUDA build directory (default: build-cuda)' \
	  '  SHARED_BUILD_DIR=dir   Shared CPU build directory (default: build-shared)' \
	  '  GPU_SHARED_BUILD_DIR=dir Shared CUDA build directory (default: build-cuda-shared)' \
	  '  INSTALL_PREFIX=dir     Install prefix for CPU shared installs and default auto-install' \
	  '  GPU_INSTALL_PREFIX=dir Install prefix for CUDA shared installs in auto-install (default: INSTALL_PREFIX-gpu)' \
	  '  MODEL=path             MPS file for make solve (default: model.mps)' \
	  '  DATASET_DIR=path       Folder of .mps files for dataset targets' \
	  '  RESULT_DIR=path        Output folder for dataset CSV/log files' \
	  '  CUDA_ROOT=path         Override the auto-detected CUDA toolkit for gpu-* targets' \
	  '  CUDA_HOME=path         Alternative CUDA toolkit hint used by auto-detection' \
	  '  CUDA_PATH=path         Alternative CUDA toolkit hint used by auto-detection' \
	  '  CUDA_NVCC=path         Override the auto-detected nvcc for gpu-* targets' \
	  '  GPU_SM=arch            Override the detected GPU compute capability (for example: GPU_SM=90)' \
	  '  V=1                    Show raw compile and link commands' \
	  '  SHOW_WARNINGS=1        Enable compiler warning output' \
	  '  ARGS="..."             Extra arguments passed to solve_qp or dataset runner' \
	  '' \
	  'Current CUDA auto-detection:' \
	  '  CUDA_TOOLKIT_ROOT: $(if $(CUDA_TOOLKIT_ROOT),$(CUDA_TOOLKIT_ROOT),<not found>)' \
	  '  CUDA_NVCC:         $(if $(CUDA_NVCC),$(CUDA_NVCC),<not found>)' \
	  '  CUDA_ARCH:         $(CUDA_ARCH)' \
	  '' \
	  'This Makefile builds directly with gcc/g++/nvcc and does not require CMake.'

_announce-auto-build:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; building CPU and GPU targets, including shared libraries.'; \
		printf '  CUDA toolkit root: %s\n' '$(CUDA_TOOLKIT_ROOT)'; \
		printf '  CUDA compiler: %s\n' '$(CUDA_NVCC)'; \
		printf '  CUDA arch: %s\n' '$(CUDA_ARCH)'; \
	else \
		echo 'CUDA toolkit not detected; building CPU targets only.'; \
		printf '  Searched: %s\n' '$(CUDA_SEARCH_HINT)'; \
		if [ -n "$(CUDA_ENV_ROOT)" ]; then \
			printf '  Rejected CUDA root: %s (expected bin/nvcc and include/cuda.h)\n' '$(CUDA_ENV_ROOT)'; \
		fi; \
		echo '  Hint: set CUDA_ROOT=/path/to/cuda or CUDA_HOME=/path/to/cuda for GPU builds.'; \
	fi

_announce-auto-shared:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; building CPU and GPU shared libraries.'; \
		printf '  CUDA toolkit root: %s\n' '$(CUDA_TOOLKIT_ROOT)'; \
		printf '  CUDA compiler: %s\n' '$(CUDA_NVCC)'; \
		printf '  CUDA arch: %s\n' '$(CUDA_ARCH)'; \
	else \
		echo 'CUDA toolkit not detected; building the CPU shared library only.'; \
		printf '  Searched: %s\n' '$(CUDA_SEARCH_HINT)'; \
		if [ -n "$(CUDA_ENV_ROOT)" ]; then \
			printf '  Rejected CUDA root: %s (expected bin/nvcc and include/cuda.h)\n' '$(CUDA_ENV_ROOT)'; \
		fi; \
		echo '  Hint: set CUDA_ROOT=/path/to/cuda or CUDA_HOME=/path/to/cuda for GPU builds.'; \
	fi

_announce-auto-test:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; running CPU and GPU test suites.'; \
		printf '  CUDA toolkit root: %s\n' '$(CUDA_TOOLKIT_ROOT)'; \
		printf '  CUDA compiler: %s\n' '$(CUDA_NVCC)'; \
		printf '  CUDA arch: %s\n' '$(CUDA_ARCH)'; \
	else \
		echo 'CUDA toolkit not detected; running the CPU test suite only.'; \
		printf '  Searched: %s\n' '$(CUDA_SEARCH_HINT)'; \
		if [ -n "$(CUDA_ENV_ROOT)" ]; then \
			printf '  Rejected CUDA root: %s (expected bin/nvcc and include/cuda.h)\n' '$(CUDA_ENV_ROOT)'; \
		fi; \
		echo '  Hint: set CUDA_ROOT=/path/to/cuda or CUDA_HOME=/path/to/cuda for GPU builds.'; \
	fi

_announce-auto-install:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; installing CPU shared artifacts to $(INSTALL_PREFIX) and CUDA shared artifacts to $(GPU_INSTALL_PREFIX).'; \
		printf '  CUDA toolkit root: %s\n' '$(CUDA_TOOLKIT_ROOT)'; \
		printf '  CUDA compiler: %s\n' '$(CUDA_NVCC)'; \
		printf '  CUDA arch: %s\n' '$(CUDA_ARCH)'; \
	else \
		echo 'CUDA toolkit not detected; installing CPU shared artifacts only to $(INSTALL_PREFIX).'; \
		printf '  Searched: %s\n' '$(CUDA_SEARCH_HINT)'; \
		if [ -n "$(CUDA_ENV_ROOT)" ]; then \
			printf '  Rejected CUDA root: %s (expected bin/nvcc and include/cuda.h)\n' '$(CUDA_ENV_ROOT)'; \
		fi; \
		echo '  Hint: set CUDA_ROOT=/path/to/cuda or CUDA_HOME=/path/to/cuda for GPU builds.'; \
	fi

_announce-auto-clean:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; removing CPU and GPU build directories, including shared-build directories.'; \
		printf '  CUDA toolkit root: %s\n' '$(CUDA_TOOLKIT_ROOT)'; \
		printf '  CUDA compiler: %s\n' '$(CUDA_NVCC)'; \
	else \
		echo 'CUDA toolkit not detected; removing CPU build directories, including the CPU shared-build directory.'; \
		printf '  Searched: %s\n' '$(CUDA_SEARCH_HINT)'; \
		if [ -n "$(CUDA_ENV_ROOT)" ]; then \
			printf '  Rejected CUDA root: %s (expected bin/nvcc and include/cuda.h)\n' '$(CUDA_ENV_ROOT)'; \
		fi; \
		echo '  Hint: set CUDA_ROOT=/path/to/cuda or CUDA_HOME=/path/to/cuda for GPU builds.'; \
	fi

_require-cuda:
	@if [ -z "$(CUDA_NVCC)" ] || [ ! -x "$(CUDA_NVCC)" ]; then \
		echo 'error: unable to find nvcc; set CUDA_ROOT=/path/to/cuda, CUDA_HOME=/path/to/cuda, or CUDA_NVCC=/path/to/nvcc'; \
		exit 1; \
	fi
	@if [ -z "$(CUDA_TOOLKIT_ROOT)" ] || [ ! -f "$(CUDA_TOOLKIT_ROOT)/include/cuda.h" ]; then \
		echo 'error: CUDA_ROOT/CUDA_HOME/CUDA_PATH must point to a CUDA toolkit containing include/cuda.h'; \
		exit 1; \
	fi
	@printf 'Using CUDA compiler: %s\n' '$(CUDA_NVCC)'
	@printf 'Using CUDA toolkit root: %s\n' '$(CUDA_TOOLKIT_ROOT)'
	@printf 'Using CUDA arch: %s\n' '$(CUDA_ARCH)'

$(BUILD_DIR)/obj/%.o: %.c
	@mkdir -p $(dir $@)
	$(Q)$(CC) $(CPU_CPPFLAGS) $(COMMON_CFLAGS) -c $< -o $@

$(BUILD_DIR)/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(Q)$(CXX) $(CPU_CPPFLAGS) $(COMMON_CXXFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/obj/%.o: %.c
	@mkdir -p $(dir $@)
	$(Q)$(CC) $(GPU_CPPFLAGS) $(COMMON_CFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(Q)$(CXX) $(GPU_CPPFLAGS) $(COMMON_CXXFLAGS) -c $< -o $@

$(GPU_BUILD_DIR)/obj/%.o: %.cu | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) $(GPU_CPPFLAGS) $(COMMON_NVCCFLAGS) -c $< -o $@

$(CPU_LIB): $(CPU_CORE_OBJECTS)
	@mkdir -p $(dir $@)
	$(Q)$(AR) rcs $@ $^
	$(Q)$(RANLIB) $@

$(GPU_LIB): $(GPU_CORE_OBJECTS) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(AR) rcs $@ $^
	$(Q)$(RANLIB) $@

$(CPU_TEST_SUPPORT_LIB): $(CPU_TEST_SUPPORT_OBJECT)
	@mkdir -p $(dir $@)
	$(Q)$(AR) rcs $@ $^
	$(Q)$(RANLIB) $@

$(GPU_TEST_SUPPORT_LIB): $(GPU_TEST_SUPPORT_OBJECT) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(AR) rcs $@ $^
	$(Q)$(RANLIB) $@

$(CPU_SHARED_LIB): $(CPU_CORE_OBJECTS)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -shared -o $@ $^ $(CPU_LINK_LIBS)

$(GPU_SHARED_LIB): $(GPU_CORE_OBJECTS) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -shared $(COMMON_NVCCFLAGS) -o $@ $^ $(GPU_LINK_LIBS)

$(CPU_SOLVER_BINARY): $(call map_objects,$(BUILD_DIR),examples/solve_qp.c) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),examples/solve_qp.c) $(CPU_LIB) $(CPU_LINK_LIBS)

$(GPU_SOLVER_BINARY): $(call map_objects,$(GPU_BUILD_DIR),examples/solve_qp.c) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),examples/solve_qp.c) $(GPU_LIB) $(GPU_LINK_LIBS)

$(BUILD_DIR)/test_sparse_helpers: $(call map_objects,$(BUILD_DIR),tests/test_sparse_helpers.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/test_sparse_helpers.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_mps_reader: $(call map_objects,$(BUILD_DIR),tests/test_mps_reader.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/test_mps_reader.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_builder_preprocess: $(call map_objects,$(BUILD_DIR),tests/test_builder_preprocess.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/test_builder_preprocess.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_binding_api: $(call map_objects,$(BUILD_DIR),tests/test_binding_api.c) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/test_binding_api.c) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_workspace: $(call map_objects,$(BUILD_DIR),tests/cpu/test_workspace.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/cpu/test_workspace.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_scale: $(call map_objects,$(BUILD_DIR),tests/cpu/test_scale.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/cpu/test_scale.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_residuals: $(call map_objects,$(BUILD_DIR),tests/cpu/test_residuals.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/cpu/test_residuals.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_solver: $(call map_objects,$(BUILD_DIR),tests/cpu/test_solver.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/cpu/test_solver.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_iteration: $(call map_objects,$(BUILD_DIR),tests/cpu/test_iteration.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/cpu/test_iteration.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_parallel: $(call map_objects,$(BUILD_DIR),tests/cpu/test_parallel.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/cpu/test_parallel.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_eigs: $(call map_objects,$(BUILD_DIR),tests/cpu/test_eigs.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/cpu/test_eigs.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_control: $(call map_objects,$(BUILD_DIR),tests/cpu/test_control.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/cpu/test_control.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(BUILD_DIR)/test_gpu_dispatch: $(call map_objects,$(BUILD_DIR),tests/gpu/test_gpu_dispatch.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB)
	@mkdir -p $(dir $@)
	$(Q)$(CXX) -o $@ $(call map_objects,$(BUILD_DIR),tests/gpu/test_gpu_dispatch.c) $(CPU_TEST_SUPPORT_LIB) $(CPU_LIB) $(CPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_binding_api_gpu: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_binding_api_gpu.c) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_binding_api_gpu.c) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_gpu_workspace: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_workspace.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_workspace.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_gpu_scale: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_scale.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_scale.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_gpu_residuals: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_residuals.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_residuals.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_gpu_eigs: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_eigs.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_eigs.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_gpu_qcqp: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_qcqp.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_qcqp.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_gpu_control: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_control.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_control.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_gpu_iteration: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_iteration.cpp) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_iteration.cpp) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/test_gpu_solver: $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_solver.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),tests/gpu/test_gpu_solver.c) $(GPU_TEST_SUPPORT_LIB) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/compare_gpu_unchecked_path: $(call map_objects,$(GPU_BUILD_DIR),examples/gpu/compare_gpu_unchecked_path.c) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),examples/gpu/compare_gpu_unchecked_path.c) $(GPU_LIB) $(GPU_LINK_LIBS)

$(GPU_BUILD_DIR)/trace_restart_compare: $(call map_objects,$(GPU_BUILD_DIR),examples/gpu/trace_restart_compare.c) $(GPU_LIB) | _require-cuda
	@mkdir -p $(dir $@)
	$(Q)$(CUDA_NVCC) -o $@ $(call map_objects,$(GPU_BUILD_DIR),examples/gpu/trace_restart_compare.c) $(GPU_LIB) $(GPU_LINK_LIBS)

build: $(CPU_LIB) $(CPU_TEST_SUPPORT_LIB) $(CPU_SOLVER_BINARY) $(CPU_TEST_BINARIES)

build-shared: $(CPU_SHARED_LIB)

gpu-build: _require-cuda $(GPU_LIB) $(GPU_TEST_SUPPORT_LIB) $(GPU_SOLVER_BINARY) $(GPU_TEST_BINARIES) $(GPU_EXAMPLE_BINARIES)

gpu-build-shared: _require-cuda $(GPU_SHARED_LIB)

auto-build: _announce-auto-build $(AUTO_BUILD_TARGETS)

auto-shared: _announce-auto-shared $(AUTO_SHARED_TARGETS)

test: auto-test

test-cpu: build
	@set -e; for test_bin in $(CPU_TEST_BINARIES); do \
		printf 'Running %s\n' "$$test_bin"; \
		"$$test_bin"; \
	done

test-shared: build-shared test-cpu

gpu-test: gpu-build
	@set -e; for test_bin in $(GPU_TEST_BINARIES); do \
		printf 'Running %s\n' "$$test_bin"; \
		env $(GPU_RUN_ENV) "$$test_bin"; \
	done

gpu-test-shared: gpu-build-shared gpu-test

auto-test: _announce-auto-test $(AUTO_TEST_TARGETS)

install: auto-install

install-shared: build-shared
	@mkdir -p $(INSTALL_PREFIX)/lib $(INSTALL_PREFIX)/include
	cp $(CPU_SHARED_LIB) $(INSTALL_PREFIX)/lib/
	cp -R include/. $(INSTALL_PREFIX)/include/

gpu-install-shared: gpu-build-shared
	@mkdir -p $(GPU_INSTALL_PREFIX)/lib $(GPU_INSTALL_PREFIX)/include
	cp $(GPU_SHARED_LIB) $(GPU_INSTALL_PREFIX)/lib/
	cp -R include/. $(GPU_INSTALL_PREFIX)/include/

auto-install: _announce-auto-install $(AUTO_INSTALL_TARGETS)

auto-clean: _announce-auto-clean
	rm -rf $(AUTO_CLEAN_DIRS)

solve: build
	./$(CPU_SOLVER_BINARY) $(MODEL) $(ARGS)

gpu-solve: gpu-build
	env $(GPU_RUN_ENV) ./$(GPU_SOLVER_BINARY) $(MODEL) --use-gpu 1 $(ARGS)

dataset: build
	$(PYTHON) scripts/run_dataset.py $(DATASET_DIR) $(RESULT_DIR) --build-dir $(BUILD_DIR) $(ARGS)

dataset-gpu: gpu-build
	env $(GPU_RUN_ENV) $(PYTHON) scripts/run_dataset.py $(DATASET_DIR) $(RESULT_DIR) --build-dir $(GPU_BUILD_DIR) --use-gpu 1 $(ARGS)

clean: auto-clean

clean-cpu:
	rm -rf $(BUILD_DIR) $(SHARED_BUILD_DIR)

gpu-clean:
	rm -rf $(GPU_BUILD_DIR) $(GPU_SHARED_BUILD_DIR)

rebuild: clean auto-build

gpu-rebuild: gpu-clean gpu-build
