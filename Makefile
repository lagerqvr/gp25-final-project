# Default build configuration (override by creating make.inc)
NVCC ?= nvcc
ARCH ?= sm_80
HOST_CXX ?= clang++

BIN_DIR := bin
TARGET := $(BIN_DIR)/hashhat
HOST_TARGET := $(BIN_DIR)/hashhat_host
SRC_HOST := src/host/main.cu

CXXFLAGS := -std=c++17 -O2 -Isrc
NVCCFLAGS := -std=c++17 -O2 -arch=$(ARCH) -Isrc
HOST_ONLY_FLAGS := -x c++ -nocudainc -nocudalib
LDFLAGS := -lncurses

.PHONY: all cuda host run clean help

all: cuda

cuda: $(TARGET)

$(TARGET): $(SRC_HOST)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC_HOST) $(LDFLAGS)

host: $(HOST_TARGET)

$(HOST_TARGET): $(SRC_HOST)
	@mkdir -p $(BIN_DIR)
	$(HOST_CXX) $(CXXFLAGS) $(HOST_ONLY_FLAGS) -o $@ $(SRC_HOST) $(LDFLAGS)

run: cuda
	$(TARGET) --help

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "make cuda       # build GPU binary (requires nvcc)"
	@echo "make host       # build host-only CLI parser (no CUDA, e.g., macOS)"
	@echo "make run        # build GPU binary and show --help"
	@echo "make clean      # remove build outputs"

-include make.inc
