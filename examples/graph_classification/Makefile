GNN_HOME=../..

include $(GNN_HOME)/make_common

lib_dir := $(GNN_HOME)/build/lib
gnn_lib := $(lib_dir)/libgnn.a

   
all: build/kernel_mean_field build/kernel_loopy_bp

include_dirs := $(CUDA_HOME)/include $(MKL_ROOT)/include $(GNN_HOME)/include/matrix $(GNN_HOME)/include/graphnn ./include
CXXFLAGS += $(addprefix -I,$(include_dirs))

build/%: src/%.cpp $(gnn_lib) ./include/*
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp, %.a $^) -L$(lib_dir) -lgnn $(LDFLAGS)

clean:
	rm -rf build
