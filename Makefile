include core_makefile

app_files := unit_test.cpp
apps += $(subst .cpp,,$(app_files))
app_path := $(GNN_HOME)/build/app

all: $(gnn_objs) $(addprefix $(app_path)/,$(apps))

$(app_path)/%: $(GNN_HOME)/app/%.cpp $(gnn_objs) $(gnn_include)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS)

clean:
	rm $(mat_obj_path)/* $(graphnn_obj_path)/* $(app_path)/*
