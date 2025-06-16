
.PHONY: build_cpu_execs build_gpu_execs build_lib_execs all clean

SRC_DIR = src
BIN_DIR = bin
# DEPENDENCIES = $(wildcard $(SRC_DIR)/*.h)



# CPU targets

CXX = g++
CXXFLAGS = -g

CPU_DEPENDENCIES = $(addprefix $(SRC_DIR)/, utils.h bitonic_sort_cpu.h)
# CPU_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CPU_SOURCES = $(addprefix $(SRC_DIR)/, bitonic_sort_cpu_measure.cpp bitonic_sort_cpu_test.cpp)
CPU_EXEC_NAMES = $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%,$(CPU_SOURCES))
OPTIMIZED_CPU_EXECS = $(foreach OPTIMIZATION,0 1 2 3,$(addsuffix $(OPTIMIZATION),$(CPU_EXEC_NAMES)))


build_cpu_execs: $(OPTIMIZED_CPU_EXECS)


$(filter %0,$(OPTIMIZED_CPU_EXECS)): $(BIN_DIR)/%0: $(SRC_DIR)/%.cpp
$(filter %1,$(OPTIMIZED_CPU_EXECS)): $(BIN_DIR)/%1: $(SRC_DIR)/%.cpp
$(filter %2,$(OPTIMIZED_CPU_EXECS)): $(BIN_DIR)/%2: $(SRC_DIR)/%.cpp
$(filter %3,$(OPTIMIZED_CPU_EXECS)): $(BIN_DIR)/%3: $(SRC_DIR)/%.cpp

$(filter %0,$(OPTIMIZED_CPU_EXECS)): CXXFLAGS += -O0
$(filter %1,$(OPTIMIZED_CPU_EXECS)): CXXFLAGS += -O1
$(filter %2,$(OPTIMIZED_CPU_EXECS)): CXXFLAGS += -O2
$(filter %3,$(OPTIMIZED_CPU_EXECS)): CXXFLAGS += -O3


$(OPTIMIZED_CPU_EXECS): $(CPU_DEPENDENCIES) | $(BIN_DIR)

$(OPTIMIZED_CPU_EXECS):
	$(LINK.cpp) $^ $(LDLIBS) -o $@



# GPU targets

GPU_DEPENDENCIES = $(addprefix $(SRC_DIR)/, utils.h bitonic_sort_gpu.h)
# GPU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
GPU_SOURCES = $(addprefix $(SRC_DIR)/, bitonic_sort_gpu_measure.cu bitonic_sort_gpu_test.cu)
GPU_LIBS_SOURCES = $(addprefix $(SRC_DIR)/, thrust_sort.cu cub_merge_sort.cu cub_radix_sort.cu)
GPU_EXEC_NAMES = $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(GPU_SOURCES))
LIB_EXEC_NAMES = $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(GPU_LIBS_SOURCES))

$(GPU_EXEC_NAMES) $(LIB_EXEC_NAMES): $(BIN_DIR)/%: $(SRC_DIR)/%.cu $(GPU_DEPENDENCIES) | $(BIN_DIR)
	nvcc --generate-line-info $< -o $@

build_gpu_execs: $(GPU_EXEC_NAMES)

build_lib_execs: $(LIB_EXEC_NAMES)



# Common targets

$(BIN_DIR):
	mkdir -p $(BIN_DIR)


all: build_cpu_execs build_gpu_execs build_lib_execs

clean:
	$(RM) -r $(BIN_DIR)

