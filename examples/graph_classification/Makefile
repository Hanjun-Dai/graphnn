GNN_HOME=../..

include $(GNN_HOME)/make_common

lib_dir := $(GNN_HOME)/build/lib
gnn_lib := $(lib_dir)/libgnn.a

include_dirs := $(CUDA_HOME)/include $(MKL_ROOT)/include $(GNN_HOME)/include include
CXXFLAGS += $(addprefix -I,$(include_dirs)) -DUSE_GPU
   
all: build/kernel_mean_field build/kernel_loopy_bp

build/%: src/%.cpp $(gnn_lib) ./include/*
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp, %.a $^) -L$(lib_dir) -lgnn $(LDFLAGS)

clean:
	rm -rf build
