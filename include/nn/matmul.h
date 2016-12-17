#ifndef MATMUL_H
#define MATMUL_H

#include "util/gnn_macros.h"
#include "nn/factor.h"

namespace gnn
{

template<typename mode, typename Dtype>
class MatMul : public Factor
{
public:
	MatMul(std::string _name);

};

}

#endif