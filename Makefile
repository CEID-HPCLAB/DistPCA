# Detect architecture
ARCH := $(shell uname -m)

SRC_DIR := src
BUILD_DIR := build

# X86_64 architecture with Intel MKL
ifeq ($(ARCH),x86_64)

ifndef MKLROOT
$(error MKLROOT is not set. Please run: source /opt/intel/oneapi/setvars.sh)
endif

MKL_ROOT = $(MKLROOT)
MKL_LIBROOT = $(MKL_ROOT)/lib/intel64
MKL_INCROOT = $(MKL_ROOT)/include

# Use mpicxx for MPI support
COMP = mpicxx -mkl -Wall -mavx2 -mfma -march=native
CCOMP = mpicc
CFLAGS = -O3 -std=c++11 -mavx2 -mfma -march=native -I$(MKL_INCROOT) -DUSE_MPI
CFLAGS_C = -O3 -fPIE -I$(MKL_INCROOT) -DUSE_MPI

MKL_LIB = -Wl,--start-group $(MKL_LIBROOT)/libmkl_intel_lp64.a $(MKL_LIBROOT)/libmkl_intel_thread.a $(MKL_LIBROOT)/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

CPP_SOURCES = $(SRC_DIR)/terapca.cpp $(SRC_DIR)/utilities.cpp $(SRC_DIR)/methods.cpp
C_SOURCES = $(SRC_DIR)/gaussian.c $(SRC_DIR)/gennorm.c $(SRC_DIR)/io.c
EXE = $(BUILD_DIR)/TeraPCA_MPI.exe

CPP_OBJECTS = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
C_OBJECTS = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJECTS = $(CPP_OBJECTS) $(C_OBJECTS)

# Create build directory before building
$(EXE): $(BUILD_DIR) $(OBJECTS)
	$(COMP) $(OBJECTS) -o $@ $(MKL_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(COMP) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CCOMP) $(CFLAGS_C) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)


# ARM64 architecture with OpenBLAS and OpenMP
else ifeq ($(ARCH),arm64)

OPENBLAS_ROOT = /opt/homebrew/opt/openblas
OPENBLAS_INC = $(OPENBLAS_ROOT)/include
OPENBLAS_LIB = $(OPENBLAS_ROOT)/lib

OPENMP_ROOT = /opt/homebrew/opt/libomp
OPENMP_INC = $(OPENMP_ROOT)/include
OPENMP_LIB = $(OPENMP_ROOT)/lib

# MPI paths
MPI_ROOT = /opt/homebrew/opt/open-mpi
MPI_INC = $(MPI_ROOT)/include
MPI_LIB = $(MPI_ROOT)/lib

COMP = mpicxx
CCOMP = mpicc
CFLAGS = -Wall -O3 -std=c++11 -I$(OPENBLAS_INC) -I$(OPENMP_INC) -I$(MPI_INC) -Xpreprocessor -fopenmp -DUSE_MPI
CFLAGS_C = -Wall -O3 -I$(OPENBLAS_INC) -I$(OPENMP_INC) -I$(MPI_INC) -Xpreprocessor -fopenmp -DUSE_MPI
LDFLAGS = -L$(OPENBLAS_LIB) -lopenblas -L$(OPENMP_LIB) -lomp -L$(MPI_LIB) -lmpi -lpthread -lm

CPP_SOURCES = $(SRC_DIR)/terapca.cpp $(SRC_DIR)/utilities.cpp $(SRC_DIR)/methods.cpp
C_SOURCES = $(SRC_DIR)/gaussian.c $(SRC_DIR)/gennorm.c $(SRC_DIR)/io.c
EXE = $(BUILD_DIR)/TeraPCA_MPI.exe

CPP_OBJECTS = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
C_OBJECTS = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJECTS = $(CPP_OBJECTS) $(C_OBJECTS)

# Create build directory before building
$(EXE): $(BUILD_DIR) $(OBJECTS)
	$(COMP) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(COMP) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CCOMP) $(CFLAGS_C) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

else
$(error Unsupported architecture: $(ARCH))
endif

.PHONY: all clean
all: $(EXE)