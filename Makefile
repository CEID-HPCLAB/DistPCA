# Detect architecture and OS
ARCH := $(shell uname -m)
OS   := $(shell uname -s)

SRC_DIR := src
BUILD_DIR := build

CPP_SOURCES = $(SRC_DIR)/distpca.cpp $(SRC_DIR)/utilities.cpp $(SRC_DIR)/methods.cpp
C_SOURCES   = $(SRC_DIR)/gaussian.c $(SRC_DIR)/gennorm.c $(SRC_DIR)/io.c
EXE = $(BUILD_DIR)/DistPCA.exe

CPP_OBJECTS = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
C_OBJECTS   = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJECTS = $(CPP_OBJECTS) $(C_OBJECTS)

# Linux x86_64 : Intel oneAPI + MKL
ifeq ($(OS),Linux)

ifndef MKLROOT
$(error MKLROOT is not set. Please run: source /opt/intel/oneapi/setvars.sh)
endif

MKL_LIBROOT = $(MKLROOT)/lib/intel64
MKL_INCROOT = $(MKLROOT)/include

COMP  = mpicxx -mkl -mavx2 -mfma -march=native
CCOMP = mpicc
CFLAGS   = -O3 -std=c++11 -mavx2 -mfma -march=native -I$(MKL_INCROOT) -DUSE_MPI
CFLAGS_C = -O3 -fPIE -I$(MKL_INCROOT) -DUSE_MPI
LDLIBS   = -Wl,--start-group $(MKL_LIBROOT)/libmkl_intel_lp64.a $(MKL_LIBROOT)/libmkl_intel_thread.a $(MKL_LIBROOT)/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl


# macOS (Apple Silicon or Intel) : Homebrew OpenBLAS + libomp + open-mpi
#   brew install open-mpi openblas libomp
else ifeq ($(OS),Darwin)

OPENBLAS := $(shell brew --prefix openblas 2>/dev/null)
LIBOMP   := $(shell brew --prefix libomp   2>/dev/null)

ifeq ($(OPENBLAS),)
$(error OpenBLAS not found. Run: brew install open-mpi openblas libomp)
endif

COMP  = mpicxx
CCOMP = mpicc
CFLAGS   = -O3 -std=c++11 -DUSE_MPI -I$(OPENBLAS)/include -I$(LIBOMP)/include \
           -Xpreprocessor -fopenmp -Wno-vla-extension -Wno-vla-cxx-extension
CFLAGS_C = -O3 -DUSE_MPI -I$(OPENBLAS)/include -I$(LIBOMP)/include \
           -Xpreprocessor -fopenmp
LDLIBS   = -L$(OPENBLAS)/lib -lopenblas -L$(LIBOMP)/lib -lomp -lpthread -lm

else
$(error Unsupported OS: $(OS))
endif

# Common rules
$(EXE): $(BUILD_DIR) $(OBJECTS)
	$(COMP) $(OBJECTS) -o $@ $(LDLIBS)
	@echo ""
	@echo "Build successful! Executable: $@"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(COMP) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CCOMP) $(CFLAGS_C) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
all: $(EXE)