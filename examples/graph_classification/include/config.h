#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <unistd.h>
#include <cstring>

typedef float Dtype;

struct cfg
{
    static int dev_id, iter; 
    static int max_lv, fp_len;
    static unsigned n_hidden, conv_size;
    static unsigned batch_size; 
    static unsigned max_epoch; 
    static unsigned test_interval; 
    static unsigned report_interval; 
    static unsigned save_interval; 
    static int node_dim, num_class;
    static Dtype lr;
    static Dtype l2_penalty; 
    static Dtype momentum; 
    static const char *train_idx_file, *test_idx_file, *graph_file, *save_dir; 
    
    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {             
		    if (strcmp(argv[i], "-lr") == 0)
		        lr = atof(argv[i + 1]);                                  
            if (strcmp(argv[i], "-cur_iter") == 0)
                iter = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-hidden") == 0)
			    n_hidden = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-lv") == 0)
				max_lv = atoi(argv[i + 1]);
        	if (strcmp(argv[i], "-conv") == 0)
				conv_size = atoi(argv[i + 1]);
        	if (strcmp(argv[i], "-fp") == 0)
				fp_len = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-b") == 0)
    			batch_size = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-maxe") == 0)
	       		max_epoch = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-int_test") == 0)
    			test_interval = atoi(argv[i + 1]);
    	   	if (strcmp(argv[i], "-int_report") == 0)
    			report_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-int_save") == 0)
    			save_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-l2") == 0)
    			l2_penalty = atof(argv[i + 1]);
    		if (strcmp(argv[i], "-m") == 0)
    			momentum = atof(argv[i + 1]);	
    		if (strcmp(argv[i], "-svdir") == 0)
    			save_dir = argv[i + 1];
    		if (strcmp(argv[i], "-graph") == 0)
				graph_file = argv[i + 1];
			if (strcmp(argv[i], "-train_idx") == 0)
				train_idx_file = argv[i + 1];
			if (strcmp(argv[i], "-test_idx") == 0)
				test_idx_file = argv[i + 1];
            if (strcmp(argv[i], "-device") == 0)
    			dev_id = atoi(argv[i + 1]);
        }

        std::cerr << "n_hidden = " << n_hidden << std::endl;
		std::cerr << "max level = " << max_lv << std::endl;
    	std::cerr << "conv size = " << conv_size << std::endl;
    	std::cerr << "fp len = " << fp_len << std::endl;
        std::cerr << "batch_size = " << batch_size << std::endl;
        std::cerr << "max_epoch = " << max_epoch << std::endl;
    	std::cerr << "test_interval = " << test_interval << std::endl;
    	std::cerr << "report_interval = " << report_interval << std::endl;
    	std::cerr << "save_interval = " << save_interval << std::endl;
    	std::cerr << "lr = " << lr << std::endl;
    	std::cerr << "l2_penalty = " << l2_penalty << std::endl;
    	std::cerr << "momentum = " << momentum << std::endl;
    	std::cerr << "init iter = " << iter << std::endl;	
        std::cerr << "device id = " << dev_id << std::endl;    
    }
};

int cfg::dev_id = 0;
int cfg::node_dim = 0;
int cfg::num_class = 0;
int cfg::iter = 0;
int cfg::max_lv = 4;
unsigned cfg::conv_size = 20;
int cfg::fp_len = 512;

unsigned cfg::n_hidden = 100;
unsigned cfg::batch_size = 50;
unsigned cfg::max_epoch = 200;
unsigned cfg::test_interval = 10000;
unsigned cfg::report_interval = 100;
unsigned cfg::save_interval = 50000;
Dtype cfg::lr = 0.0005;
Dtype cfg::l2_penalty = 0;
Dtype cfg::momentum = 0;
const char* cfg::train_idx_file = nullptr;
const char* cfg::test_idx_file = nullptr;
const char* cfg::graph_file = nullptr;
const char* cfg::save_dir = "./saved";

#endif