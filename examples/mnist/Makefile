GNN_HOME=../..

include $(GNN_HOME)/make_common
USE_GPU = 0

ifeq ($(USE_GPU), 1)
	lib_dir := $(GNN_HOME)/build/lib
	CXXFLAGS += -DUSE_GPU
else
	lib_dir := $(GNN_HOME)/build_cpuonly/lib
endif

gnn_lib := $(lib_dir)/libgnn.a

include_dirs := $(CUDA_HOME)/include $(MKL_ROOT)/include $(GNN_HOME)/include include
CXXFLAGS += $(addprefix -I,$(include_dirs))

all: build/mnist

build/mnist: mnist.cpp $(gnn_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lgnn $(LDFLAGS)

clean:
	rm -rf build
