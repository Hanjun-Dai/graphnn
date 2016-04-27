include make_common

build_root = build

include_dirs = $(CUDA_HOME)/include $(MKL_ROOT)/include include/matrix include/graphnn
CXXFLAGS += $(addprefix -I,$(include_dirs))
NVCCFLAGS += $(addprefix -I,$(include_dirs))
NVCCFLAGS += -std=c++11 --use_fast_math

cu_files = $(shell $(FIND) src/ -name "*.cu" -printf "%P\n")
cpp_files = $(shell $(FIND) src/ -name "*.cpp" -printf "%P\n")
cu_obj_files = $(subst .cu,.o,$(cu_files))
cxx_obj_files = $(subst .cpp,.o,$(cpp_files))
obj_build_root = $(build_root)/objs
objs = $(addprefix $(obj_build_root)/cuda/,$(cu_obj_files)) $(addprefix $(obj_build_root)/cxx/,$(cxx_obj_files))
DEPS = ${objs:.o=.d}

lib_dir = $(build_root)/lib
gnn_lib = $(lib_dir)/libgnn.a

test_src = $(shell $(FIND) test/ -name "*.cpp" -printf "%P\n")
test_prog = $(subst .cpp,,$(test_src))
test_build_root = $(build_root)/test
test_target = $(addprefix $(test_build_root)/,$(test_prog))
DEPS += ${test_target:=.d}

all: $(gnn_lib) $(test_target)

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

$(test_build_root)/%: test/%.cpp $(gnn_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -o $@ $(filter %.cpp, $^) -L$(lib_dir) -lgnn $(LDFLAGS)

clean:
	rm -rf build

-include $(DEPS)
