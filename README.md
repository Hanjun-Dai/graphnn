# graphnn

#### Prerequisites

Tested under Ubuntu 14.04

##### Download and install cuda from https://developer.nvidia.com/cuda-toolkit

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    
  in .bashrc, add the following path
  
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    
##### Download and install intel mkl

  in .bashrc, add the following path
  
    source {path_to_your_intel_root/name_of_parallel_tool_box}/bin/psxevars.sh
    export MKL_ROOT={path_to_your_intel_root}/mkl
    
##### Install cppformat

    check https://github.com/cppformat/cppformat for help
  
##### Install Spiral-wht

    wget http://www.ece.cmu.edu/~spiral/software/spiral-wht-1.8.tgz
    tar -zxvf spiral-wht-1.8.tgz
    cd spiral-wht-1.8
    ./configure
    make
    make install
    
#### Build package

    edit core_makefile
    set GNN_HOME to the root folder of this package
    make
    
#### Run example

##### Run mnist

    Download and uncompress the data from http://yann.lecun.com/exdb/mnist/
    setup the path in run_exp.sh under the package root folder
    ./run_exp.sh
    
    
