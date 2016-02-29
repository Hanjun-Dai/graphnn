GNN_HOME=../..

include $(GNN_HOME)/make_common

lib_dir := $(GNN_HOME)/build/lib
gnn_lib := $(lib_dir)/libgnn.a

include_dirs := $(CUDA_HOME)/include $(MKL_ROOT)/include $(GNN_HOME)/include/matrix $(GNN_HOME)/include/graphnn
CXXFLAGS += $(addprefix -I,$(include_dirs))

all: build/mnist

build/mnist: mnist.cpp $(gnn_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lgnn $(LDFLAGS)

clean:
	rm -rf build
