## Dec. 22, 2017 update: pytorch version of structure2vec

For people who prefer python, here is the pytorch implementation of s2v: 

https://github.com/Hanjun-Dai/pytorch_structure2vec

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


#### Docker
  Dockerfile contains all the required installations (including Intel MKL and TBB) above. Only additional requirement is to provide `NVIDIA*.run` script that will load the same NVIDIA driver of host into the target. Then to build the container, execute:

    docker build -t "graphnn:test" .

  To run it:

    docker run --runtime=nvidia graphnn:test bash

  If above command fails for a reason, refer to https://github.com/NVIDIA/nvidia-docker. If no error occurs, you can simply follow the below instructions and execute them in the container without failure.
#### Build static library

    cp make_common.example make_common
    modify configurations in make_common file
    make -j8
    
#### Run example

##### Run mnist

    cd examples/mnist
    make
    ./run.sh

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
