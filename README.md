# graphnn

#### Document

(Doxygen)
http://www.cc.gatech.edu/~hdai8/graphnn/html/annotated.html 

#### Prerequisites

Tested under Ubuntu 14.04, 16.04 and Mac OSX 10.12.6

##### Download and install cuda from https://developer.nvidia.com/cuda-toolkit

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    
  in .bashrc, add the following path (suppose you installed to the default path)
  
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    
##### Download and install intel mkl

  in .bashrc, add the following path
  
    source {path_to_your_intel_root/name_of_parallel_tool_box}/bin/psxevars.sh
    export MKL_ROOT={path_to_your_intel_root}/mkl
    
##### Install cppformat (now called fmtlib)

    check https://github.com/fmtlib/fmt for help

#### Build static library

    cp make_common.example make_common
    modify configurations in make_common file
    make
    
#### Run example

##### Run mnist

    cd examples/mnist
    make
    ./run_exp.sh

##### Run graph classification

    cd examples/graph_classification
    make
    ./local_run.sh
    
    The 5 datasets under the data/ folder are commonly used in graph kernel. 
    
#### Reference

```bibtex
@article{dai2016discriminative,
  title={Discriminative Embeddings of Latent Variable Models for Structured Data},
  author={Dai, Hanjun and Dai, Bo and Song, Le},
  journal={arXiv preprint arXiv:1603.05629},
  year={2016}
}
```
