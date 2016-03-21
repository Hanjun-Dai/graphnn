include make_common

include_dirs := $(CUDA_HOME)/include $(MKL_ROOT)/include include/matrix include/graphnn
CXXFLAGS += $(addprefix -I,$(include_dirs))
NVCCFLAGS += $(addprefix -I,$(include_dirs))
NVCCFLAGS += -std=c++11 --use_fast_math

cu_files := $(shell $(FIND) src/ -name "*.cu" -printf "%P\n")
cpp_files := $(shell $(FIND) src/ -name "*.cpp" -printf "%P\n")

cu_obj_files := $(subst .cu,.o,$(cu_files))
cxx_obj_files := $(subst .cpp,.o,$(cpp_files))

obj_build_root := build/objs

objs := $(addprefix $(obj_build_root)/cuda/,$(cu_obj_files)) $(addprefix $(obj_build_root)/cxx/,$(cxx_obj_files))
DEPS := ${objs:.o=.d}

lib_dir := build/lib
gnn_lib := $(lib_dir)/libgnn.a

all: $(gnn_lib)

$(gnn_lib): $(objs)
	$(dir_guard)
	ar rcs $@ $(objs)

$(obj_build_root)/cuda/%.o: src/%.cu
	$(dir_guard)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} -odir $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@
		
$(obj_build_root)/cxx/%.o: src/%.cpp
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $(filter %.cpp, $^)

clean:
	rm -rf build

-include $(DEPS)
